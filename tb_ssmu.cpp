// tb_ssmu_golden_out.cpp (COSIM SAFE: NO std::thread; deterministic; no Bad TV due to threading)
// - Uses ::expf/::logf for MinGW/clang compatibility
// - Keeps H0 consistent (stores exact H0 sent to DUT)
// - SPEEDUP: float mirror W_delta + per-i WB/WC mirroring
// - COSIM SAFE: no concurrency; rely on DUT out FIFO depth >= VEC_D (set out depth >= VEC_D in DUT)
//   -> With the fixed DUT (C_per_i + lockstep join), out is produced at the end, so no backpressure deadlock.

#include "ssmu.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <hls_stream.h>

#ifndef HUGE_LEN
#define HUGE_LEN (N * VEC_D)
#endif

// ============================================================
// Helpers (LCG random) - deterministic
// ============================================================
static inline unsigned lcg_next(unsigned &s) {
    s = 1664525u * s + 1013904223u;
    return s;
}
static inline float frand(unsigned &s, float lo=-1.0f, float hi=1.0f) {
    unsigned r = lcg_next(s);
    float t = (r / 4294967295.0f);
    return lo + (hi - lo) * t;
}

// ============================================================
// DTYPE_VEC helpers
// ============================================================
static inline DTYPE vget_tb(const DTYPE_VEC &v, int idx) { return v[(unsigned)idx]; }

static inline bool is_finite_vec(const DTYPE_VEC &v) {
    for (int i = 0; i < VEC_FACTOR; i++) {
        float x = (float)v[i];
        if (!std::isfinite(x)) return false;
    }
    return true;
}

// Quantize like DUT datatype (DTYPE cast)
static inline float qf(float x) { return (float)((DTYPE)x); }

// Match DUT: silu using float expf
static inline float silu_f(float x) {
    float expv = ::expf(-x);
    float sig  = 1.0f / (1.0f + expv);
    return x * sig;
}

// Match DUT: stable softplus in float
static inline float softplus_f(float x) {
    if (x > 0.0f) return x + ::logf(1.0f + ::expf(-x));
    else          return ::logf(1.0f + ::expf(x));
}

static float vec_max_abs_err(const DTYPE_VEC &a, const DTYPE_VEC &b) {
    float m = 0.0f;
    for (int l = 0; l < VEC_FACTOR; ++l) {
        float ea = (float)vget_tb(a, l);
        float eb = (float)vget_tb(b, l);
        m = std::max(m, std::fabs(ea - eb));
    }
    return m;
}

// ============================================================
// Golden OUT only (FULL N) - uses EXACT SAME H0 that TB pushed
// SPEEDUP: use float mirror for W_delta and per-i row mirror for W_B/W_C
// ============================================================
static void golden_out_only_streaming(
    const float kernel[K],
    const float X_in_f[VEC_D][VEC_FACTOR],
    const float A_in_f[N][VEC_FACTOR],
    const float H0_in_f[N][VEC_D][VEC_FACTOR],
    const DTYPE_VEC W_B[N][VEC_D],
    const DTYPE_VEC W_C[N][VEC_D],
    const float     Wdelta_f[VEC_D][VEC_D][VEC_FACTOR],
    float out_gold[VEC_D][VEC_FACTOR]
) {
    float X_gate[VEC_D][VEC_FACTOR];
    float X_ssm [VEC_D][VEC_FACTOR];

    // gate = silu(X)
    for (int j = 0; j < VEC_D; ++j) {
        for (int l = 0; l < VEC_FACTOR; ++l) {
            X_gate[j][l] = qf(silu_f(X_in_f[j][l]));
        }
    }

    // conv1d + silu
    float linebuf[K-1][VEC_FACTOR];
    for (int t = 0; t < K-1; ++t)
        for (int l = 0; l < VEC_FACTOR; ++l)
            linebuf[t][l] = 0.0f;

    for (int j = 0; j < VEC_D; ++j) {
        float window[K][VEC_FACTOR];

        for (int t = 0; t < K-1; ++t)
            for (int l = 0; l < VEC_FACTOR; ++l)
                window[t][l] = linebuf[t][l];

        for (int l = 0; l < VEC_FACTOR; ++l)
            window[K-1][l] = X_in_f[j][l];

        for (int t = K-2; t > 0; --t)
            for (int l = 0; l < VEC_FACTOR; ++l)
                linebuf[t][l] = linebuf[t-1][l];

        for (int l = 0; l < VEC_FACTOR; ++l)
            linebuf[0][l] = X_in_f[j][l];

        for (int l = 0; l < VEC_FACTOR; ++l) {
            float sum = 0.0f;
            for (int kk = 0; kk < K; ++kk) sum += kernel[kk] * window[kk][l];
            float conv_q = qf(sum);
            X_ssm[j][l]  = qf(silu_f(conv_q));
        }
    }

    // delta = softplus( X_ssm * W_delta )
    float delta[VEC_D][VEC_FACTOR];
    for (int i = 0; i < VEC_D; ++i) {
        for (int l = 0; l < VEC_FACTOR; ++l) {
            float accv = 0.0f;
            for (int j = 0; j < VEC_D; ++j) {
                accv += X_ssm[j][l] * Wdelta_f[i][j][l];
            }
            delta[i][l] = qf(softplus_f(accv));
        }
    }

    // acc over j,l
    float acc[VEC_D][VEC_FACTOR];
    for (int j = 0; j < VEC_D; ++j)
        for (int l = 0; l < VEC_FACTOR; ++l)
            acc[j][l] = 0.0f;

    for (int i = 0; i < N; ++i) {
        float B_i[VEC_FACTOR];
        float C_i[VEC_FACTOR];

        float WB_row[VEC_D][VEC_FACTOR];
        float WC_row[VEC_D][VEC_FACTOR];
        for (int j = 0; j < VEC_D; ++j) {
            for (int l = 0; l < VEC_FACTOR; ++l) {
                WB_row[j][l] = (float)W_B[i][j][l];
                WC_row[j][l] = (float)W_C[i][j][l];
            }
        }

        for (int l = 0; l < VEC_FACTOR; ++l) { B_i[l] = 0.0f; C_i[l] = 0.0f; }

        for (int j = 0; j < VEC_D; ++j) {
            for (int l = 0; l < VEC_FACTOR; ++l) {
                float x = X_ssm[j][l];
                B_i[l] += x * WB_row[j][l];
                C_i[l] += x * WC_row[j][l];
            }
        }

        for (int l = 0; l < VEC_FACTOR; ++l) {
            B_i[l] = qf(B_i[l]);
            C_i[l] = qf(C_i[l]);
        }

        for (int j = 0; j < VEC_D; ++j) {
            for (int l = 0; l < VEC_FACTOR; ++l) {
                float h0  = qf(H0_in_f[i][j][l]);
                float ddA = qf(::expf(A_in_f[i][l] * delta[j][l]));
                float dB  = qf(B_i[l] * delta[j][l]);
                float H1  = qf(h0 * ddA + dB * X_ssm[j][l]);

                acc[j][l] = qf(acc[j][l] + H1 * C_i[l]);
            }
        }

        if ((i & 8191) == 0) {
            std::printf("[golden] i=%d/%d\n", i, N);
            std::fflush(stdout);
        }
    }

    for (int j = 0; j < VEC_D; ++j)
        for (int l = 0; l < VEC_FACTOR; ++l)
            out_gold[j][l] = qf(X_gate[j][l] + acc[j][l]);
}

int main() {
    std::printf("PARAM: DIM=%d N=%d VEC_FACTOR=%d VEC_D=%d HUGE_LEN=%d K=%d\n",
                DIM, N, VEC_FACTOR, VEC_D, HUGE_LEN, K);
    std::fflush(stdout);

    hls::stream<DTYPE>     kernel_in("kernel_in");
    hls::stream<DTYPE_VEC> A_in("A_in");
    hls::stream<DTYPE_VEC> X_in("X_in");
    hls::stream<DTYPE_VEC> H0_in("H0_in");
    hls::stream<DTYPE_VEC> out("out");

    static DTYPE_VEC W_B[N][VEC_D];
    static DTYPE_VEC W_C[N][VEC_D];
    static DTYPE_VEC W_delta[VEC_D][VEC_D];

    static float Wdelta_f[VEC_D][VEC_D][VEC_FACTOR];

    static DTYPE_VEC C_ddr[HUGE_LEN];
    static DTYPE_VEC H1_ddr[HUGE_LEN];

    static float A_f[N][VEC_FACTOR];
    static float X_f[VEC_D][VEC_FACTOR];
    static float kernel_f[K];
    static float H0_f[N][VEC_D][VEC_FACTOR];

    unsigned seed = 1;

    std::printf("[TB] init weights...\n"); std::fflush(stdout);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < VEC_D; j++) {
            DTYPE_VEC vb, vc;
            for (int l = 0; l < VEC_FACTOR; l++) {
                float fb = frand(seed, -1.0f, 1.0f) * 0.05f;
                float fc = frand(seed, -1.0f, 1.0f) * 0.05f;
                vb[l] = (DTYPE)fb;
                vc[l] = (DTYPE)fc;
            }
            W_B[i][j] = vb;
            W_C[i][j] = vc;
        }
        if ((i & 1023) == 0) {
            std::printf("[TB]   weights row i=%d/%d\n", i, N);
            std::fflush(stdout);
        }
    }

    std::printf("[TB] init W_delta...\n"); std::fflush(stdout);
    for (int i = 0; i < VEC_D; i++) {
        for (int j = 0; j < VEC_D; j++) {
            DTYPE_VEC vd;
            for (int l = 0; l < VEC_FACTOR; l++) {
                float fd = frand(seed, -1.0f, 1.0f) * 0.02f;
                vd[l] = (DTYPE)fd;
            }
            W_delta[i][j] = vd;
            for (int l = 0; l < VEC_FACTOR; l++) {
                Wdelta_f[i][j][l] = (float)vd[l];
            }
        }
        if ((i & 63) == 0) {
            std::printf("[TB]   W_delta i=%d/%d\n", i, VEC_D);
            std::fflush(stdout);
        }
    }

    std::printf("[TB] push kernel...\n"); std::fflush(stdout);
    for (int i = 0; i < K; i++) {
        float x = frand(seed, -0.5f, 0.5f);
        kernel_f[i] = x;
        kernel_in.write((DTYPE)x);
    }

    std::printf("[TB] push X...\n"); std::fflush(stdout);
    for (int j = 0; j < VEC_D; j++) {
        DTYPE_VEC v;
        for (int l = 0; l < VEC_FACTOR; l++) {
            float x = frand(seed, -1.0f, 1.0f) * 0.5f;
            v[l] = (DTYPE)x;
            X_f[j][l] = (float)v[l];
        }
        X_in.write(v);
    }

    std::printf("[TB] push A...\n"); std::fflush(stdout);
    for (int i = 0; i < N; i++) {
        DTYPE_VEC v;
        for (int l = 0; l < VEC_FACTOR; l++) {
            float x = frand(seed, -1.0f, 1.0f) * 0.1f;
            v[l] = (DTYPE)x;
            A_f[i][l] = (float)v[l];
        }
        A_in.write(v);
        if ((i & 1023) == 0) {
            std::printf("[TB]   A i=%d/%d\n", i, N);
            std::fflush(stdout);
        }
    }

    std::printf("[TB] push H0 (N*VEC_D=%d)...\n", N * VEC_D); std::fflush(stdout);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < VEC_D; j++) {
            DTYPE_VEC v;
            for (int l = 0; l < VEC_FACTOR; l++) {
                float x = frand(seed, -1.0f, 1.0f) * 0.1f;
                v[l] = (DTYPE)x;
                H0_f[i][j][l] = (float)v[l];
            }
            H0_in.write(v);
        }
        if ((i & 255) == 0) {
            std::printf("[TB]   H0 i=%d/%d\n", i, N);
            std::fflush(stdout);
        }
    }

    std::printf("[TB] CALL DUT...\n"); std::fflush(stdout);
    SSMU(kernel_in, A_in, W_B, W_C, W_delta, X_in, H0_in, C_ddr, H1_ddr, out);
    std::printf("[TB] DUT returned.\n"); std::fflush(stdout);

    // Drain out AFTER DUT returns (COSIM-safe; requires out FIFO depth >= VEC_D)
    static DTYPE_VEC out_dut_buf[VEC_D];
    for (int j = 0; j < VEC_D; ++j) {
        out_dut_buf[j] = out.read();
    }
    const int out_cnt = VEC_D;

    std::printf("[TB] out drained: out_cnt=%d (expected %d)\n", out_cnt, VEC_D);
    std::fflush(stdout);

    // DDR sanity sample
    int nonzero_C = 0, nonzero_H1 = 0;
    for (int t = 0; t < 64; t++) {
        int idx = (t * 9973) % HUGE_LEN;
        float sC = 0.f, sH = 0.f;
        for (int k = 0; k < VEC_FACTOR; k++) {
            sC += std::fabs((float)C_ddr[idx][k]);
            sH += std::fabs((float)H1_ddr[idx][k]);
        }
        if (sC > 1e-6f) nonzero_C++;
        if (sH > 1e-6f) nonzero_H1++;
    }
    std::printf("[TB] DDR sanity: C_nonzero_samples=%d, H1_nonzero_samples=%d\n", nonzero_C, nonzero_H1);
    std::fflush(stdout);

    std::printf("[TB] compute golden OUT (FULL N=%d)...\n", N); std::fflush(stdout);
    static float out_gold[VEC_D][VEC_FACTOR];
    golden_out_only_streaming(kernel_f, X_f, A_f, H0_f, W_B, W_C, Wdelta_f, out_gold);

    std::printf("[TB] compare OUT...\n"); std::fflush(stdout);

    const float TOL_OUT = 2e-2f;
    float max_err_out = 0.0f;
    bool ok = true;

    for (int j = 0; j < VEC_D; j++) {
        DTYPE_VEC dut = out_dut_buf[j];

        if (!is_finite_vec(dut)) {
            std::printf("[TB] FAIL: out has NaN/Inf at j=%d\n", j);
            ok = false;
            break;
        }

        DTYPE_VEC gold;
        for (int l = 0; l < VEC_FACTOR; ++l) gold[l] = (DTYPE)out_gold[j][l];

        float e = vec_max_abs_err(dut, gold);
        max_err_out = std::max(max_err_out, e);
    }

    std::printf("[TB] OUT summary: out_cnt=%d (expected %d), MAX_ERR_OUT=%.6g (TOL=%.3g)\n",
                out_cnt, VEC_D, max_err_out, TOL_OUT);

    if (!ok) { std::printf("[TB] FAIL\n"); return 1; }
    if (max_err_out > TOL_OUT) { std::printf("[TB] FAIL: tolerance exceeded\n"); return 1; }

    std::printf("[TB] PASS\n");
    return 0;
}
