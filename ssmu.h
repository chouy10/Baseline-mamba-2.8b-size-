#ifndef __UCI_EECS_SSMU_HEADER_20260106_BIG__
#define __UCI_EECS_SSMU_HEADER_20260106_BIG__

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <hls_vector.h>

// =============================================================
// Optional debug banner
// =============================================================
#ifndef SSMU_HEADER_DEBUG
#define SSMU_HEADER_DEBUG 0
#endif

#if SSMU_HEADER_DEBUG
#warning ">>> ssmu.h INCLUDED (FULL SIZE MODE) <<<"
#endif

// =============================================================
// Model sizes (FULL checkpoint sizes ONLY)
// DIM=2560, N=5120, VEC_FACTOR=16 => VEC_D=160
// =============================================================
#ifndef DIM
#define DIM 2560
#endif

#ifndef N
#define N 5120
#endif

#ifndef VEC_FACTOR
#define VEC_FACTOR 16
#endif

#ifndef K
#define K 4
#endif

// =============================================================
// Derived sizes
// =============================================================
#ifndef VEC_D
#define VEC_D (DIM / VEC_FACTOR)
#endif

#ifndef HUGE_LEN
#define HUGE_LEN (N * VEC_D)
#endif

// =============================================================
// Types
// =============================================================
typedef ap_fixed<16,4> DTYPE;
typedef hls::vector<DTYPE, VEC_FACTOR> DTYPE_VEC;

// =============================================================
// Joined packet type (H1 + C)
// NOTE: Defined ONLY here (no cpp redefinition)
// =============================================================
struct HC_Packet {
    DTYPE_VEC h1;
    DTYPE_VEC c;
};

// =============================================================
// constexpr mirrors
// =============================================================
static constexpr int DIM_C        = DIM;
static constexpr int N_C          = N;
static constexpr int K_C          = K;
static constexpr int VEC_FACTOR_C = VEC_FACTOR;
static constexpr int VEC_D_C      = VEC_D;
static constexpr int HUGE_LEN_C   = HUGE_LEN;

// =============================================================
// Compile-time safety checks
// =============================================================
static_assert(VEC_FACTOR_C > 0, "VEC_FACTOR must be > 0");
static_assert(DIM_C > 0,        "DIM must be > 0");
static_assert((DIM_C % VEC_FACTOR_C) == 0, "DIM must be divisible by VEC_FACTOR");

// Your design assumes tiles of 8/16 in multiple places; keep these assertions
static_assert((VEC_D_C % 8)  == 0, "VEC_D must be multiple of 8");
static_assert((VEC_D_C % 16) == 0, "VEC_D must be multiple of 16");

static_assert(HUGE_LEN_C > 0, "HUGE_LEN must be > 0");

// =============================================================
// Function prototypes
// =============================================================

// duplicator
void dup_vecD_stream(
    hls::stream<DTYPE_VEC>& in,
    hls::stream<DTYPE_VEC>& out1,
    hls::stream<DTYPE_VEC>& out2
);

// Part 1
void conv1d_silu_stream(
    hls::stream<DTYPE_VEC>& X_in,
    hls::stream<DTYPE>& kernel_in,
    hls::stream<DTYPE_VEC>& X_gate_out,
    hls::stream<DTYPE_VEC>& X_ssm_out
);

// Part 2 (weights are read-only => const improves interface stability)
void projection_streams(
    hls::stream<DTYPE_VEC>& X_ssm_in,
    const DTYPE_VEC W_B[N][VEC_D],
    const DTYPE_VEC W_C[N][VEC_D],
    const DTYPE_VEC W_delta[VEC_D][VEC_D],
    hls::stream<DTYPE_VEC>& B_out,
    hls::stream<DTYPE_VEC>& C_out,
    hls::stream<DTYPE_VEC>& delta_out_A,
    hls::stream<DTYPE_VEC>& delta_out_B
);

// Part 3
void A_to_ddA_stream(
    hls::stream<DTYPE_VEC>& A_in,
    hls::stream<DTYPE_VEC>& delta_in,
    hls::stream<DTYPE_VEC>& ddA_out
);

// Part 3b
void B_to_dB_stream(
    hls::stream<DTYPE_VEC>& B_in,
    hls::stream<DTYPE_VEC>& delta_in,
    hls::stream<DTYPE_VEC>& dB_out
);

// Part 4
void update_H_stream(
    hls::stream<DTYPE_VEC>& ddA_in,
    hls::stream<DTYPE_VEC>& dX_in,
    hls::stream<DTYPE_VEC>& dB_in,
    hls::stream<DTYPE_VEC>& H0_in,
    hls::stream<DTYPE_VEC>& H1_out
);

// Part 5 (JOIN / ZIP version)
void final_output_stream_tiled(
    hls::stream<DTYPE_VEC>& X_gate_in,
    hls::stream<HC_Packet>& HC_in,
    hls::stream<DTYPE_VEC>& out
);

// TOP (weights are read-only => const improves interface stability)
void SSMU(
    hls::stream<DTYPE>& kernel_in,
    hls::stream<DTYPE_VEC>& A_in,
    const DTYPE_VEC W_B[N][VEC_D],
    const DTYPE_VEC W_C[N][VEC_D],
    const DTYPE_VEC W_delta[VEC_D][VEC_D],
    hls::stream<DTYPE_VEC>& X_in,
    hls::stream<DTYPE_VEC>& H0_in,
    DTYPE_VEC* C_ddr,      // length = HUGE_LEN
    DTYPE_VEC* H1_ddr,     // length = HUGE_LEN
    hls::stream<DTYPE_VEC>& out
);

#endif // __UCI_EECS_SSMU_HEADER_20260106_BIG__
