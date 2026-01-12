// SSMU.cpp (DEADLOCK-FIXED + DEPTH-INCREASED + JT-safe)
//
// What’s fixed vs your original:
// 1) projection: produce B and C interleaved (C arrives early)
// 2) join: drain H1 first (VEC_D tokens) into local buffer, then read C_i, then write DDR + HC
// 3) final_output: remove the hidden constraint "VEC_D % JT == 0".
//    - Old code read HC_in in tiles of JT=16 unconditionally -> if VEC_D not multiple of JT => read count mismatch => deadlock.
//    - New code reads exactly N*VEC_D HC packets (no more, no less), regardless of VEC_D.
//
// Also: significantly increased FIFO depths + force BRAM FIFO for the heavy streams.
//
// Notes:
// - Keeps DDR transaction counts: writes HUGE_LEN entries to both C_ddr and H1_ddr
// - Output order preserved: HC packets remain in (i-major, j-minor) order
// - W_B/W_C/W_delta are const (read-only)

#include "ssmu.h"
#include <hls_math.h>

#ifndef __SYNTHESIS__
#include <cstdio>
#endif

#ifndef __SYNTHESIS__
  #define DUT_PRINTF(...) do { std::printf(__VA_ARGS__); std::fflush(stdout); } while(0)
#else
  #define DUT_PRINTF(...) do {} while(0)
#endif

#ifndef USE_FIXED_ACC
#define USE_FIXED_ACC 1
#endif

#if USE_FIXED_ACC
typedef ap_fixed<32, 10> ACC_T;
#else
typedef float ACC_T;
#endif

// ------------------------------------------------------------
// Vector accessors
// ------------------------------------------------------------
static inline DTYPE vget(const DTYPE_VEC &v, int idx) {
#pragma HLS INLINE
    return v[(unsigned)idx];
}
static inline void vset(DTYPE_VEC &v, int idx, DTYPE val) {
#pragma HLS INLINE
    v[(unsigned)idx] = val;
}

// ------------------------------------------------------------
// activations
// ------------------------------------------------------------
static inline DTYPE silu_elem(DTYPE a) {
#pragma HLS INLINE
    float x    = (float)a;
    float expv = hls::expf(-x);
    float sig  = 1.0f / (1.0f + expv);
    return (DTYPE)(x * sig);
}

static inline DTYPE softplus_from_float(float x) {
#pragma HLS INLINE
    float y;
    if (x > 0.0f) y = x + hls::logf(1.0f + hls::expf(-x));
    else          y = hls::logf(1.0f + hls::expf(x));
    return (DTYPE)y;
}

// ============================================================
// dup for VEC_D tokens (safe small)
// ============================================================
static void dup_vecD_stream_local(hls::stream<DTYPE_VEC>& in,
                                  hls::stream<DTYPE_VEC>& out1,
                                  hls::stream<DTYPE_VEC>& out2) {
#pragma HLS INLINE off
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
    }
}

// ============================================================
// Part 1: conv1d + SiLU
// ============================================================
static void conv1d_silu_stream_local(hls::stream<DTYPE_VEC>& X_in,
                                     hls::stream<DTYPE>& kernel_in,
                                     hls::stream<DTYPE_VEC>& X_gate_out,
                                     hls::stream<DTYPE_VEC>& X_ssm_out) {
#pragma HLS INLINE off

    static DTYPE line_buffer[K-1][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_s2p impl=bram

    DTYPE kernel_buffer[K];
#pragma HLS ARRAY_PARTITION variable=kernel_buffer complete

    for (int i = 0; i < K; ++i) {
#pragma HLS PIPELINE II=1
        kernel_buffer[i] = kernel_in.read();
    }

    DTYPE_VEC X_buffer[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buffer type=ram_s2p impl=bram

    // read X, output gate, buffer for conv
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC xv = X_in.read();
        X_buffer[i] = xv;

        DTYPE_VEC gate_out;
        for (int k = 0; k < VEC_FACTOR; ++k) {
#pragma HLS UNROLL
            vset(gate_out, k, silu_elem(vget(xv, k)));
        }
        X_gate_out.write(gate_out);
    }

    // clear line buffer
    for (int i = 0; i < K-1; ++i)
        for (int k = 0; k < VEC_FACTOR; ++k)
            line_buffer[i][k] = 0;

    // conv1d
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=2
        DTYPE_VEC in_vec = X_buffer[i];

        DTYPE window[K][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=window complete dim=2

        for (int j = 0; j < K-1; ++j)
            for (int k = 0; k < VEC_FACTOR; ++k)
                window[j][k] = line_buffer[j][k];

        for (int k = 0; k < VEC_FACTOR; ++k)
            window[K-1][k] = vget(in_vec, k);

        for (int j = K-2; j > 0; --j)
            for (int k = 0; k < VEC_FACTOR; ++k)
                line_buffer[j][k] = line_buffer[j-1][k];

        for (int k = 0; k < VEC_FACTOR; ++k)
            line_buffer[0][k] = vget(in_vec, k);

        DTYPE_VEC conv_out;
        for (int lane = 0; lane < VEC_FACTOR; ++lane) {
#pragma HLS UNROLL
            float sum = 0.0f;
            for (int kk = 0; kk < K; ++kk) {
                sum += (float)kernel_buffer[kk] * (float)window[kk][lane];
            }
            vset(conv_out, lane, (DTYPE)sum);
        }

        DTYPE_VEC ssm_out;
        for (int k = 0; k < VEC_FACTOR; ++k) {
#pragma HLS UNROLL
            vset(ssm_out, k, silu_elem(vget(conv_out, k)));
        }
        X_ssm_out.write(ssm_out);
    }
}

// ============================================================
// Part 2: projections (delta) then (B,C) interleaved per i
// ============================================================
static void projection_streams_local(hls::stream<DTYPE_VEC>& X_ssm_in,
                                     const DTYPE_VEC W_B[N][VEC_D],
                                     const DTYPE_VEC W_C[N][VEC_D],
                                     const DTYPE_VEC W_delta[VEC_D][VEC_D],
                                     hls::stream<DTYPE_VEC>& B_out_N,
                                     hls::stream<DTYPE_VEC>& C_out_N,
                                     hls::stream<DTYPE_VEC>& delta_out_A,
                                     hls::stream<DTYPE_VEC>& delta_out_B) {
#pragma HLS INLINE off

    const int J_TILE = 8;

    DTYPE_VEC X_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=bram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_ssm_in.read();
    }

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] delta: start (produce %d)\n", VEC_D);
#endif
    // ---- delta (produce VEC_D) ----
    for (int i = 0; i < VEC_D; ++i) {
        ACC_T acc[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            acc[l] = (ACC_T)0;
        }

        for (int jt = 0; jt < VEC_D; jt += J_TILE) {
#pragma HLS PIPELINE II=8
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                X_tile[jj] = X_buf[jt + jj];
            }
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                DTYPE_VEC w = W_delta[i][jt + jj];
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    acc[l] += (ACC_T)vget(X_tile[jj], l) * (ACC_T)vget(w, l);
                }
            }
        }

        DTYPE_VEC delta_vec;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(delta_vec, l, softplus_from_float((float)acc[l]));
        }
        delta_out_A.write(delta_vec);
        delta_out_B.write(delta_vec);

#ifndef __SYNTHESIS__
        if ((i & 15) == 0) DUT_PRINTF("[DUT] delta: i=%d/%d\n", i, VEC_D);
#endif
    }

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] B+C: start (produce %d each)\n", N);
#endif
    // ---- B and C interleaved per i ----
    for (int i = 0; i < N; ++i) {
        ACC_T accB[VEC_FACTOR];
        ACC_T accC[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accB complete dim=1
#pragma HLS ARRAY_PARTITION variable=accC complete dim=1
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            accB[l] = (ACC_T)0;
            accC[l] = (ACC_T)0;
        }

        for (int jt = 0; jt < VEC_D; jt += J_TILE) {
#pragma HLS PIPELINE II=8
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                X_tile[jj] = X_buf[jt + jj];
            }

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                DTYPE_VEC wB = W_B[i][jt + jj];
                DTYPE_VEC wC = W_C[i][jt + jj];
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    ACC_T x = (ACC_T)vget(X_tile[jj], l);
                    accB[l] += x * (ACC_T)vget(wB, l);
                    accC[l] += x * (ACC_T)vget(wC, l);
                }
            }
        }

        DTYPE_VEC outB, outC;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(outB, l, (DTYPE)accB[l]);
            vset(outC, l, (DTYPE)accC[l]);
        }
        B_out_N.write(outB);
        C_out_N.write(outC);

#ifndef __SYNTHESIS__
        if ((i & 511) == 0) DUT_PRINTF("[DUT] B+C: i=%d/%d\n", i, N);
#endif
    }
}

// ============================================================
// Part 3: A -> ddA (HUGE_LEN)
// ============================================================
static void A_to_ddA_stream_local(hls::stream<DTYPE_VEC>& A_in,
                                  hls::stream<DTYPE_VEC>& delta_in,
                                  hls::stream<DTYPE_VEC>& ddA_out) {
#pragma HLS INLINE off

    DTYPE_VEC delta_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=delta_buf type=ram_s2p impl=bram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        delta_buf[j] = delta_in.read();
    }

    for (int i = 0; i < N; ++i) {
        DTYPE_VEC A_vec = A_in.read();
        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
            DTYPE_VEC dlt = delta_buf[j];
            DTYPE_VEC ddA;
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                float v = hls::expf((float)vget(A_vec, l) * (float)vget(dlt, l));
                vset(ddA, l, (DTYPE)v);
            }
            ddA_out.write(ddA);
        }
    }
}

// ============================================================
// Part 3b: B -> dB (HUGE_LEN)
// ============================================================
static void B_to_dB_stream_local(hls::stream<DTYPE_VEC>& B_in,
                                 hls::stream<DTYPE_VEC>& delta_in,
                                 hls::stream<DTYPE_VEC>& dB_out) {
#pragma HLS INLINE off

    DTYPE_VEC delta_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=delta_buf type=ram_s2p impl=bram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        delta_buf[j] = delta_in.read();
    }

    for (int i = 0; i < N; ++i) {
        DTYPE_VEC Bv = B_in.read();
        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
            DTYPE_VEC dlt = delta_buf[j];
            DTYPE_VEC outv;
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                vset(outv, l, (DTYPE)((ACC_T)vget(Bv, l) * (ACC_T)vget(dlt, l)));
            }
            dB_out.write(outv);
        }
    }
}

// ============================================================
// Part 4: Update H (HUGE_LEN)
// ============================================================
static void update_H_stream_local(hls::stream<DTYPE_VEC>& ddA_in,
                                  hls::stream<DTYPE_VEC>& dX_in,
                                  hls::stream<DTYPE_VEC>& dB_in,
                                  hls::stream<DTYPE_VEC>& H0_in,
                                  hls::stream<DTYPE_VEC>& H1_out) {
#pragma HLS INLINE off

    DTYPE_VEC dX_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=dX_buf type=ram_s2p impl=bram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        dX_buf[j] = dX_in.read();
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
            DTYPE_VEC H0v = H0_in.read();
            DTYPE_VEC ddA = ddA_in.read();
            DTYPE_VEC dBv = dB_in.read();
            DTYPE_VEC dx  = dX_buf[j];

            DTYPE_VEC H1v;
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACC_T v = (ACC_T)vget(H0v, l) * (ACC_T)vget(ddA, l)
                        + (ACC_T)vget(dBv, l) * (ACC_T)vget(dx,  l);
                vset(H1v, l, (DTYPE)v);
            }
            H1_out.write(H1v);
        }
    }
}

// ============================================================
// Join + replicate C + write DDR in lockstep with H1
// drain H1 first -> read C -> write
// ============================================================
static void join_replicateC_writeDDR_local(hls::stream<DTYPE_VEC>& H1_in,
                                          hls::stream<DTYPE_VEC>& C_per_i_in,
                                          DTYPE_VEC* C_ddr,
                                          DTYPE_VEC* H1_ddr,
                                          hls::stream<HC_Packet>& HC_out) {
#pragma HLS INLINE off

    DTYPE_VEC H1_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=H1_buf type=ram_s2p impl=bram

    for (int i = 0; i < N; ++i) {
        // Drain H1 first (VEC_D tokens for this i)
        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
            H1_buf[j] = H1_in.read();
        }

        // Now get C_i (one token per i)
        DTYPE_VEC Ci = C_per_i_in.read();

        // Write DDR + stream HC packets
        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
            int idx = i * VEC_D + j;

            DTYPE_VEC H1v = H1_buf[j];

            C_ddr[idx]  = Ci;
            H1_ddr[idx] = H1v;

            HC_Packet p;
            p.c  = Ci;
            p.h1 = H1v;
            HC_out.write(p);
        }
    }
}

// ============================================================
// Part 5: final output (consumes HC stream, outputs VEC_D)
// DEADLOCK FIX: consume exactly N*VEC_D tokens from HC_in (no JT multiple requirement)
// ============================================================
static void final_output_stream_safe_local(hls::stream<DTYPE_VEC>& X_gate_in,
                                          hls::stream<HC_Packet>& HC_in,
                                          hls::stream<DTYPE_VEC>& out) {
#pragma HLS INLINE off

    // Read X_gate
    DTYPE_VEC X_gate[VEC_D];
#pragma HLS BIND_STORAGE variable=X_gate type=ram_s2p impl=bram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_gate[j] = X_gate_in.read();
    }

    // Acc init
    DTYPE_VEC acc[VEC_D];
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=bram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC z;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(z, l, (DTYPE)0);
        }
        acc[j] = z;
    }

    // Consume exactly N * VEC_D HC packets
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
            HC_Packet p = HC_in.read();
            DTYPE_VEC a = acc[j];

            DTYPE_VEC newv;
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACC_T base = (ACC_T)vget(a,     l);
                ACC_T addt = (ACC_T)vget(p.h1, l) * (ACC_T)vget(p.c, l);
                vset(newv, l, (DTYPE)(base + addt));
            }
            acc[j] = newv;
        }
    }

    // output = X_gate + acc
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC a = acc[j];
        DTYPE_VEC x = X_gate[j];
        DTYPE_VEC outv;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(outv, l, (DTYPE)((ACC_T)vget(x, l) + (ACC_T)vget(a, l)));
        }
        out.write(outv);
    }
}

// ============================================================
// TOP: SSMU  (MUST match ssmu.h signature: weights are const)
// ============================================================
void SSMU(hls::stream<DTYPE>& kernel_in,
          hls::stream<DTYPE_VEC>& A_in,
          const DTYPE_VEC W_B[N][VEC_D],
          const DTYPE_VEC W_C[N][VEC_D],
          const DTYPE_VEC W_delta[VEC_D][VEC_D],
          hls::stream<DTYPE_VEC>& X_in,
          hls::stream<DTYPE_VEC>& H0_in,
          DTYPE_VEC* C_ddr,
          DTYPE_VEC* H1_ddr,
          hls::stream<DTYPE_VEC>& out) {

#pragma HLS INTERFACE ap_fifo port=kernel_in
#pragma HLS INTERFACE ap_fifo port=A_in
#pragma HLS INTERFACE ap_fifo port=X_in
#pragma HLS INTERFACE ap_fifo port=H0_in
#pragma HLS INTERFACE ap_fifo port=out

#pragma HLS INTERFACE m_axi port=C_ddr  offset=slave bundle=gmem0 depth=HUGE_LEN
#pragma HLS INTERFACE m_axi port=H1_ddr offset=slave bundle=gmem1 depth=HUGE_LEN
#pragma HLS INTERFACE s_axilite port=C_ddr  bundle=control
#pragma HLS INTERFACE s_axilite port=H1_ddr bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

// ---- external ports: VERY large to reduce inlet backpressure in cosim ----
#pragma HLS STREAM variable=kernel_in depth=1024
#pragma HLS STREAM variable=A_in      depth=8192
#pragma HLS STREAM variable=X_in      depth=8192
#pragma HLS STREAM variable=H0_in     depth=8192
#pragma HLS STREAM variable=out       depth=8192

#pragma HLS DATAFLOW

    hls::stream<DTYPE_VEC> X_gate_stream("X_gate_stream");
    hls::stream<DTYPE_VEC> X_ssm_stream ("X_ssm_stream");
    hls::stream<DTYPE_VEC> X_ssm_proj_stream("X_ssm_proj_stream");
    hls::stream<DTYPE_VEC> X_ssm_upd_stream ("X_ssm_upd_stream");

    hls::stream<DTYPE_VEC> B_stream_N("B_stream_N");
    hls::stream<DTYPE_VEC> C_stream_N("C_stream_N");

    hls::stream<DTYPE_VEC> delta_A("delta_A");
    hls::stream<DTYPE_VEC> delta_B("delta_B");

    hls::stream<DTYPE_VEC> ddA_stream("ddA_stream");
    hls::stream<DTYPE_VEC> dB_stream ("dB_stream");
    hls::stream<DTYPE_VEC> H1_stream ("H1_stream");

    hls::stream<HC_Packet> HC_to_final("HC_to_final");

// ---- internal FIFOs: push high to survive rate mismatch in cosim ----
#pragma HLS STREAM variable=X_gate_stream     depth=8192
#pragma HLS STREAM variable=X_ssm_stream      depth=8192
#pragma HLS STREAM variable=X_ssm_proj_stream depth=8192
#pragma HLS STREAM variable=X_ssm_upd_stream  depth=8192

#pragma HLS STREAM variable=delta_A           depth=8192
#pragma HLS STREAM variable=delta_B           depth=8192

#pragma HLS STREAM variable=B_stream_N        depth=16384
#pragma HLS STREAM variable=C_stream_N        depth=16384

#pragma HLS STREAM variable=ddA_stream        depth=262144
#pragma HLS STREAM variable=dB_stream         depth=262144
#pragma HLS STREAM variable=H1_stream         depth=524288
#pragma HLS STREAM variable=HC_to_final       depth=524288

// Force BRAM FIFO for heavy streams (avoid tiny LUT FIFO causing premature backpressure)
#pragma HLS BIND_STORAGE variable=ddA_stream   type=fifo impl=bram
#pragma HLS BIND_STORAGE variable=dB_stream    type=fifo impl=bram
#pragma HLS BIND_STORAGE variable=H1_stream    type=fifo impl=bram
#pragma HLS BIND_STORAGE variable=HC_to_final  type=fifo impl=bram

    conv1d_silu_stream_local(X_in, kernel_in, X_gate_stream, X_ssm_stream);
    dup_vecD_stream_local(X_ssm_stream, X_ssm_proj_stream, X_ssm_upd_stream);

    projection_streams_local(X_ssm_proj_stream, W_B, W_C, W_delta,
                             B_stream_N, C_stream_N, delta_A, delta_B);

    A_to_ddA_stream_local(A_in, delta_A, ddA_stream);
    B_to_dB_stream_local(B_stream_N, delta_B, dB_stream);
    update_H_stream_local(ddA_stream, X_ssm_upd_stream, dB_stream, H0_in, H1_stream);

    join_replicateC_writeDDR_local(H1_stream, C_stream_N, C_ddr, H1_ddr, HC_to_final);

    // JT-safe final
    final_output_stream_safe_local(X_gate_stream, HC_to_final, out);
}
