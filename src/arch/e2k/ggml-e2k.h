
#include "ggml-e2kintrin.h"
#include "../../ggml-f16c.h"

#ifdef __cplusplus
extern "C"
{
#endif
/* Functions */
#define GGML_FP16_TO_FP32 ggml_cvt_fp16_to_fp32
#define GGML_FP32_TO_FP16 ggml_cvt_fp32_to_fp16
#define ARCH_VEC_DOT_Q8_0_Q8_0 __e2k_vec_dot_q8_0_q8_0
/* Constants */
#define _MAS4_ 0x0F0F0F0F0F0F0F0FLL
#define VF32MAX 0x7F7FFFFF7F7FFFFFLL //  FLT_MAX x 2
#define VF32MIN 0xFF7FFFFFFF7FFFFFLL // -FLT_MAX x 2

/*
    Sets vectors size and data types for specied e2k vers.
*/
#if __e2k_v__ >= 5
// modern quad registers (AVX compatible)
# define QK4_L 1
# define QK8_L 2
# define QS_L (32 / 4)
# define QS_H (32 / 8)
# define VF32_C 4
# define VF16_C 8
#else
// standart double registers (SSE compatible)
# define QK4_L 2
# define QK8_L 4
# define QS_L (32 / 2)
# define QS_H (32 / 4)
# define VF32_C 2
# define VF16_C 4
#endif

/* bitmask type */
typedef unsigned int __msk32_t;
typedef union {
    struct { __vd m[QK8_L]; };
    struct { __vd ml[QK4_L], mh[QK4_L]; };
} type_umsk_256;

typedef struct {
    ggml_fp16_t d;   // delta
    __vd qs[QK4_L]; // nibbles / quants
} vd_q4_0;

typedef struct {
    ggml_fp16_t d;    // delta
    __msk32_t qh[1]; // 5-th bit of quants
    __vd qs[QK4_L]; // nibbles / quants
} vd_q5_0;

typedef struct {
    ggml_fp16_t d;  // delta
    __vd qs[QK8_L]; // quants
} vd_q8_0;

typedef struct {
    ggml_fp16_t d, m;// delta, min
    __vd qs[QK4_L]; // nibbles / quants
} vd_q4_1;

typedef struct {
    ggml_fp16_t d, m; // delta, min
    __msk32_t qh[1]; // 5-th bit of quants
    __vd qs[QK4_L]; // nibbles / quants
} vd_q5_1;

typedef struct {
    float d, s;     // delta, d * sum(qs[i])
    __vd qs[QK8_L]; // quants
} vd_q8_1;
/* <==== end block ====^ */


/* 
  Unpack 32-bit in to 256-bit mask of byte signs.

  Based on Intel solution:
  https://stackoverflow.com/questions/35589189/unpacking-a-bitfield-inverse-of-movmskb

  @param m    (mask to unpack)
  @param cut  (mask to highlight only the bits you need)
*/
__E2K_INLINE type_umsk_256 unpack_msk32(__msk32_t m, const __vd cut)
{
/*  Example bitmask:
    01010010 01000010 ~ hi{31:16}
    10010110 10110101 ~ lo{15:0}
  =>
    0x00ff00ff0000ff00 0x00ff00000000ff00 ~ hi{3:2}
    0xff0000ff00ffff00 0xff00ffff00ff00ff ~ lo{1:0}
*/
#define _SHMSK_ 0x8040201008040201LL
    type_umsk_256 dst;
    int j;

#if __e2k_v__ >= 5
    const __vd qsd = __builtin_e2k_qppackdl(m, m),
               shm = __builtin_e2k_qppackdl(_SHMSK_, _SHMSK_);

    dst.m[0] = __builtin_e2k_qpshufb(qsd,qsd,
               __builtin_e2k_qppackdl(0x0101010101010101LL, 0x0LL));
    dst.m[1] = __builtin_e2k_qpshufb(qsd,qsd,
               __builtin_e2k_qppackdl(0x0303030303030303LL, 0x0202020202020202LL));
#else
    const __vd shm = _SHMSK_;

#pragma unroll
    for (j = 0; j < QK8_L; j++, m >>= 8)
        dst.m[j] = __builtin_e2k_pshufb(0, m, 0x0);
#endif
#undef _SHMSK_
#pragma unroll
    for (j = 0; j < QK8_L; j++) {
        dst.m[j] = __e2k_vbitw(and, dst.m[j], shm),
        dst.m[j] =  __e2k_vcmp(eqb, dst.m[j], shm),
        dst.m[j] = __e2k_vbitw(and, dst.m[j], cut);
    }
    return dst;
}



/* 
  Vec Dot Functions
*/
__E2K_INLINE float
__e2k_vec_dot_q8_0_q8_0(
    const int nb,
    const vd_q8_0 * restrict x,
    const vd_q8_0 * restrict y
) {
    float sumf = 0.0f;
    int i;

#pragma loop count(1000)
    for (i = 0; i < nb; i++)
    {
        float fm = ggml_cvt_fp16_to_fp32(x[i].d) * ggml_cvt_fp16_to_fp32(y[i].d);
        int sumi, j;
/*
    Wide instructions per iteration:

    E2K_V5+ :  6
    E2K_V3+ :  8
    E2K_V2  : 14
*/
       __vd vs[QK8_L];

#pragma unroll
        for (j = 0; j < QK8_L; j++) {
            vs[j] = __e2k_vhsat_i16_i32(
                    __e2k_vmadd_i8_i16(x[i].qs[j], y[i].qs[j]));
        }
#if   __e2k_v__ >= 5
        type_union_128 sat = {
            .__v2di = __e2k_varith(addw, vs[0], vs[1])
        };
        sumi = sat.i.i0 + sat.i.i1 + sat.i.i2 + sat.i.i3;
#else
        type_union_64 sat1, sat2;
        
        sat1.l0 = __e2k_varith(addw, vs[0], vs[1]);
        sat2.l0 = __e2k_varith(addw, vs[2], vs[3]);

        sumi = sat1.i.i0 + sat1.i.i1 + sat2.i.i0 + sat2.i.i1;
#endif
        sumf += fm * sumi;
    }
    return sumf;
}


#define __E2K_QN 4
#define __E2K_QS_I 0
#include "tmpl_vec_dot_q4_i_q8_i.c"
/*
    Wide instructions per iteration:

    E2K_V5+ :  8
    E2K_V3+ : 10
    E2K_V2  : 15

    Compiler flags: lcc (1.27.06) -O4 -ffast
*/
#define ARCH_VEC_DOT_Q4_0_Q8_0 __e2k_vec_dot_q4_0_q8_0
#undef __E2K_QS_I

#define __E2K_QS_I 1
#include "tmpl_vec_dot_q4_i_q8_i.c"
/*
    Wide instructions per iteration:

    E2K_V5+ : 8
    E2K_V3+ : 10
    E2K_V2+ : 15

    Compiler flags: lcc (1.27.06) -O4 -ffast
*/
#define ARCH_VEC_DOT_Q4_1_Q8_1 __e2k_vec_dot_q4_1_q8_1
#undef __E2K_QS_I
#undef __E2K_QN


#define __E2K_QN 5
#define __E2K_QS_I 0
#include "tmpl_vec_dot_q4_i_q8_i.c"
/*
    Wide instructions per iteration:

    E2K_V5+ :  9
    E2K_V3+ : 13
    E2K_V2  : 18

    Compiler flags: lcc (1.27.06) -O4 -ffast
*/
#define ARCH_VEC_DOT_Q5_0_Q8_0 __e2k_vec_dot_q5_0_q8_0
#undef __E2K_QS_I

#define __E2K_QS_I 1
#include "tmpl_vec_dot_q4_i_q8_i.c"
/*
    Wide instructions per iteration:

    E2K_V5+ :  8
    E2K_V3+ : 13
    E2K_V2  : 18

    Compiler flags: lcc (1.27.06) -O4 -ffast
*/
#define ARCH_VEC_DOT_Q5_1_Q8_1 __e2k_vec_dot_q5_1_q8_1
#undef __E2K_QS_I
#undef __E2K_QN

#define __E2K_PS 16
#include "tmpl_vec_dot_fPS.c"
/*
    Wide instructions per iteration:

    E2K_V5+ : 10 + 4
    E2K_V4  : 10
    E2K_V2+ : 4

    Compiler flags: lcc (1.27.0) -O4 -ffast
*/
#define ARCH_VEC_DOT_F16 __e2k_vec_dot_f16
#undef __E2K_PS

#define __E2K_PS 32
#include "tmpl_vec_dot_fPS.c"
/*
    Wide instructions per iteration:

    E2K_V2+: 4

    Compiler flags: lcc (1.27.06) -O4 -ffast
*/
#define ARCH_VEC_DOT_F32 __e2k_vec_dot_f32
#undef __E2K_PS


/*
  Quantize rows
*/
#define __E2K_QN 4
#include "tmpl_quantize_row_qN_0.c"
#include "tmpl_quantize_row_qN_1.c"
/*
    Wide instructions per iteration:

    E2K_V5+ : 17
    E2K_V4  : 28
    E2K_V2+ : 33

    Compiler flags: lcc (1.26.18) -O4 -ffast
*/
#define ARCH_QUANTIZE_ROW_Q4_0(n,x,y) __e2k_quantize_row_q4_0(n, (const __vd*)&x[0], y)
/*
    Wide instructions per iteration:

    E2K_V5+ : 22
    E2K_V2+ : 40

    Compiler flags: lcc (1.27.06) -O4 -ffast
*/
#define ARCH_QUANTIZE_ROW_Q4_1(n,x,y) __e2k_quantize_row_q4_1(n, (const __vd*)&x[0], y)
#undef __E2K_QN

#define __E2K_QN 5
#include "tmpl_quantize_row_qN_0.c"
#include "tmpl_quantize_row_qN_1.c"
/*
    Wide instructions per iteration:

    E2K_V5+ : 18
    E2K_V4  : 29
    E2K_V2+ : 34

    Compiler flags: lcc (1.26.18) -O4 -ffast
*/
#define ARCH_QUANTIZE_ROW_Q5_0(n,x,y) __e2k_quantize_row_q5_0(n, (const __vd*)&x[0], y)
/*
    Wide instructions per iteration:

    E2K_V5+ : 25
    E2K_V4  : 35
    E2K_V2+ : 39

    Compiler flags: lcc (1.27.06) -O4 -ffast
*/
#define ARCH_QUANTIZE_ROW_Q5_1(n,x,y) __e2k_quantize_row_q5_1(n, (const __vd*)&x[0], y)
#undef __E2K_QN

#define __E2K_Q8_I 0
#include "tmpl_quantize_row_q8_i.c"
/*
    Wide instructions per iteration:

    E2K_V5+ : 12
    E2K_V3+ : 21
    E2K_V2  : 30

    Compiler flags: lcc (1.26.18) -O4 -ffast
*/
#define ARCH_QUANTIZE_ROW_Q8_0(n,x,y) __e2k_quantize_row_q8_0(n, (const __vd*)&x[0], y)
#undef __E2K_Q8_I


#define __E2K_Q8_I 1
#include "tmpl_quantize_row_q8_i.c"
/*
    Wide instructions per iteration:

    E2K_V5+ : 14
    E2K_V3+ : 25
    E2K_V2+ : 34

    Compiler flags: lcc (1.26.18) -O4 -ffast
*/
#define ARCH_QUANTIZE_ROW_Q8_1(n,x,y) __e2k_quantize_row_q8_1(n, (const __vd*)&x[0], y)
#undef __E2K_Q8_I

#ifdef __cplusplus
}
#endif
