
#include "ggml-e2kintrin.h"

#ifdef __cplusplus
extern "C"
{
#endif
/* Functions */
#define ARCH_VEC_DOT_Q8_0_Q8_0 __e2k_vec_dot_q8_0_q8_0
/* Constants */
#define _MAS4_ 0x0F0F0F0F0F0F0F0FLL

/*
    Sets vectors size and data types for specied e2k vers.
*/
#if __e2k_v__ >= 5
// modern quad registers (AVX compatible)
# define QK4_L 1
# define QK8_L 2
# define QS_L (32 / 4)
# define QS_H (32 / 8)
#else
// standart double registers (SSE compatible)
# define QK4_L 2
# define QK8_L 4
# define QS_L (32 / 2)
# define QS_H (32 / 4)
#endif

typedef struct {
    __f16_t d;      // delta
    __vd qs[QK4_L]; // nibbles / quants
} vd_q4_0;

typedef struct {
    __f16_t d;      // delta
    __msk32_t qh[1];// 5-th bit of quants
    __vd qs[QK4_L]; // nibbles / quants
} vd_q5_0;

typedef struct {
    __f16_t d;      // delta
    __vd qs[QK8_L]; // quants
} vd_q8_0;

typedef struct {
    __f16_t d, m;   // delta, min
    __vd qs[QK4_L]; // nibbles / quants
} vd_q4_1;

typedef struct {
    __f16_t d, m;   // delta, min
    __msk32_t qh[1];// 5-th bit of quants
    __vd qs[QK4_L]; // nibbles / quants
} vd_q5_1;

typedef struct {
    float d, s;     // delta, d * sum(qs[i])
    __vd qs[QK8_L]; // quants
} vd_q8_1;
/* <==== end block ====^ */


__E2K_INLINE float
__e2k_vec_dot_q8_0_q8_0(const int nb, const void * restrict _x, const void * restrict _y)
{
    const vd_q8_0 * restrict x = (const vd_q8_0 * restrict)_x;
    const vd_q8_0 * restrict y = (const vd_q8_0 * restrict)_y;

    float sumf = 0.0f;
    int i;

#pragma loop count(1000)
    for (i = 0; i < nb; i++)
    {
        float fm = __e2k_cvt_f16_f32(x[i].d) * __e2k_cvt_f16_f32(y[i].d);
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


#define __E2K_QS_I 0
#include "tmpl_vec_dot_q4_i_q8_i.c"
/*
    Wide instructions per iteration:

    E2K_V5+ :  5
    E2K_V3+ :  8
    E2K_V2  : 13

    Compiler flags: lcc (1.26.18) -O4 -ffast
*/
#define ARCH_VEC_DOT_Q4_0_Q8_0 __e2k_vec_dot_q4_0_q8_0
#undef __E2K_QS_I


#define __E2K_QS_I 1
#include "tmpl_vec_dot_q4_i_q8_i.c"
/*
    Wide instructions per iteration:

    E2K_V5+ : 8
    E2K_V3+ : 8
    E2K_V2+ : 13

    Compiler flags: lcc (1.26.18) -O4 -ffast
*/
#define ARCH_VEC_DOT_Q4_1_Q8_1 __e2k_vec_dot_q4_1_q8_1
#undef __E2K_QS_I


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
#define ARCH_QUANTIZE_ROW_Q4_0 __e2k_quantize_row_q4_0
/*
    Wide instructions per iteration:

    E2K_V5+ : 22
    E2K_V2+ : 40

    Compiler flags: lcc (1.27.06) -O4 -ffast
*/
#define ARCH_QUANTIZE_ROW_Q4_1 __e2k_quantize_row_q4_1
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
#define ARCH_QUANTIZE_ROW_Q5_0 __e2k_quantize_row_q5_0
/*
    Wide instructions per iteration:

    E2K_V5+ : 25
    E2K_V4  : 35
    E2K_V2+ : 39

    Compiler flags: lcc (1.27.06) -O4 -ffast
*/
#define ARCH_QUANTIZE_ROW_Q5_1 __e2k_quantize_row_q5_1
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
#define ARCH_QUANTIZE_ROW_Q8_0 __e2k_quantize_row_q8_0
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
#define ARCH_QUANTIZE_ROW_Q8_1 __e2k_quantize_row_q8_1
#undef __E2K_Q8_I

#ifdef __cplusplus
}
#endif
