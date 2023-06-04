
#include "ggml-e2kintrin.h"

#ifdef __cplusplus
extern "C"
{
#endif
/* Functions */
#define ARCH_VEC_DOT_Q8_0_Q8_0 __e2k_vec_dot_q8_0_q8_0
#define ARCH_QUANTIZE_ROW_Q4_0 __e2k_quantize_row_q4_0
#define ARCH_QUANTIZE_ROW_Q4_1 __e2k_quantize_row_q4_1
/* Constants */
#define _MAS4_ 0x0F0F0F0F0F0F0F0FLL
#define VF_MAX 0x7F7FFFFF7F7FFFFFLL //  FLT_MAX x 2
#define VF_MIN 0xFF7FFFFFFF7FFFFFLL // -FLT_MAX x 2

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
    __vd qs[QK8_L]; // quants
} vd_q8_0;

typedef struct {
    __f16_t d, m;   // delta, min
    __vd qs[QK4_L]; // nibbles / quants
} vd_q4_1;

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
#define ARCH_VEC_DOT_Q4_0_Q8_0 __e2k_vec_dot_q4_1_q8_1
#undef __E2K_QS_I


/*
  Quantize rows
*/
__E2K_INLINE void
__e2k_quantize_row_q4_0(const int nb, const void * restrict _x, void * restrict _y)
{
    const __vd * restrict x = (const __vd * restrict)_x;
       vd_q4_0 * restrict y = (   vd_q4_0 * restrict)_y;

    int i, iq;

#define _FEIH_ 0x4108000041080000LL // [8.5, 8.5]
#if __e2k_v__ >= 5
    const __vd rnd = __builtin_e2k_qppackdl(_FEIH_, _FEIH_),
             v4max = __builtin_e2k_qppackdl(_MAS4_, _MAS4_);
#else
    const __vd rnd = _FEIH_,
             v4max = _MAS4_;
#endif
#undef _FEIH_

#pragma loop count(1000)
    for (i = 0, iq = 0; i < nb; iq += QS_L, i++)
    {
        int j, k;
/*
    Wide instructions per iteration:

    E2K_V5+ : 15
    E2K_V4  : 27
    E2K_V2+ : 33

    Compiler flags: lcc -O4 -ffast
*/
        __vd abx[QS_L], max[QS_H], vx[QS_L], x_l[QS_H], x_h[QS_H];

#pragma unroll
        for (j = 0; j < QS_L; j++) {
            // all `vx` values writes in registers, not to stack
            abx[j] = __e2k_vabs_f32((vx[j] = x[iq + j]));
        }
        /* Compares absolute `x` and merging vectors at `abs` and non-abs via mask  */
#pragma unroll
        for (j = 0, k = 0; j < QS_L; k++, j += 2) {
            // compares (l)ess or (e)qual (x1 <= x0)
            __vd m = __e2k_vfcmp(les, abx[j+1], abx[j]);
            abx[j] = __e2k_vmerge(abx[j], abx[j+1], m);
            max[k] = __e2k_vmerge( vx[j],  vx[j+1], m);
        }
#pragma unroll
        for (j = 0, k = 0; j < QS_L; k += 2, j += 4) {
            __vd m = __e2k_vfcmp(les, abx[j+2], abx[j]);
            abx[j] = __e2k_vmerge(abx[j], abx[j+2], m);
            max[k] = __e2k_vmerge(max[k], max[k+1], m);
        }
#pragma unroll
        for (j = 0, k = 0; j < QS_L; k += 4, j += 8) {
            __vd m = __e2k_vfcmp(les, abx[j+4], abx[j]);
            abx[j] = __e2k_vmerge(abx[j], abx[j+4], m);
            max[k] = __e2k_vmerge(max[k], max[k+2], m);
        }

#if __e2k_v__ >= 5
        type_union_128 fvd = { .__v2di = max[0] },
                       fcm = { .__v2di = abx[0] };

        __di a = fvd.l.l0, ua = fcm.l.l0,
             b = fvd.l.l1, ub = fcm.l.l1;
#else
        type_union_64 fvd, fcm;

        __vd a = max[0], ua = abx[0],
             b = max[4], ub = abx[8];
#endif
        bool hasDx = (a != 0 || b != 0);
        float dmax = 0.0f;
        float dmul = hasDx ? 1.0f : 0.0f;

        if ( hasDx ) {
            a  = __builtin_e2k_pmerge(a, b, __builtin_e2k_pfcmples(ub, ua)),
            ua = __builtin_e2k_pandd (a, _FABS_);

#if __e2k_v__ >= 5
            fvd.l.l0 = a, fcm.l.l0 = ua;
#else
            fvd.l0 = a, fcm.l0 = ua;
#endif
            dmax = fcm.f.f1 <= fcm.f.f0 ? fvd.f.f0 : fvd.f.f1,
            dmax /= -8,
            dmul /= dmax;
        }
        y[i].d = __e2k_cvt_f32_f16(dmax);

#if __e2k_v__ >= 5
        fcm.f.f0 = fcm.f.f1 = fcm.f.f2 = fcm.f.f3 = dmul;

        const __vd am = fcm.__v2di;
#else
        fcm.f.f0 = fcm.f.f1 = dmul;

        const __vd am = fcm.l0;
#endif

#pragma unroll
        for (j = 0, k = QS_H; j < QS_H; j++, k++) {
            // Convert f32 -> i32 ;; x * (1.0 / d) + 8.5
            x_l[j] = __e2k_vcon_f32i(__e2k_vmul_add_f32(vx[j], am, rnd)),
            x_h[j] = __e2k_vcon_f32i(__e2k_vmul_add_f32(vx[k], am, rnd));
        }
#pragma unroll
        for (j = 0, k = 0; j < QS_H; j += 4, k++) {
            __vd  xh, xl;

            xh = __e2k_vpack_i32_i8(x_h[j], x_h[j+1], x_h[j+2], x_h[j+3]);
            xl = __e2k_vpack_i32_i8(x_l[j], x_l[j+1], x_l[j+2], x_l[j+3]);

            xh = __e2k_vmerge(v4max, xh, __e2k_vcmp(gtb, v4max, xh));
            xl = __e2k_vmerge(v4max, xl, __e2k_vcmp(gtb, v4max, xl));

            xh = __e2k_vbitw(and, xh, v4max);

            y[i].qs[k] = __e2k_vbitw(or, xl, __e2k_vshift(lld, xh, 4));
        }
    }
}

__E2K_INLINE void
__e2k_quantize_row_q4_1(const int nb, const void * restrict _x, void * restrict _y)
{
    const __vd * restrict x = (const __vd * restrict)_x;
       vd_q4_1 * restrict y = (   vd_q4_1 * restrict)_y;

    int i, iq;

#define _FZEH_ 0x3F0000003F000000LL // [0.5, 0.5]
#if __e2k_v__ >= 5
    const __vd vad = __builtin_e2k_qppackdl(_FZEH_, _FZEH_),
             v4max = __builtin_e2k_qppackdl(_MAS4_, _MAS4_),
             vfmax = __builtin_e2k_qppackdl(VF_MAX, VF_MAX),
             vfmin = __builtin_e2k_qppackdl(VF_MIN, VF_MIN);
#else
    const __vd vad = _FZEH_,
             v4max = _MAS4_,
             vfmax = VF_MAX,
             vfmin = VF_MIN;
#endif
#undef _FZEH_

#pragma loop count(1000)
    for (i = 0, iq = 0; i < nb; iq += QS_L, i++)
    {
        int j, k;
/*
    Wide instructions per iteration:

    E2K_V5+ : 16
    E2K_V4  : 29
    E2K_V3  : 34
    E2K_V2  : 37

    Compiler flags: lcc -O4 -ffast
*/
        __vd amax[QS_L], amin[QS_L], vx[QS_L], x_l[QS_H], x_h[QS_H];

#pragma unroll
        for (j = 0; j < QS_L; j++) {
            // all `vx` values writes in registers, not to stack
              vx[j] = x[iq + j], 
            amax[j] = __e2k_vmax_f32(vx[j], vfmin),
            amin[j] = __e2k_vmin_f32(vx[j], vfmax);
        }
#pragma unroll
        for (j = 0; j < QS_L; j += 2) {
            amax[j] = __e2k_vmax_f32(amax[j], amax[j+1]),
            amin[j] = __e2k_vmin_f32(amin[j], amin[j+1]);
        }
#pragma unroll
        for (j = 0; j < QS_L; j += 4) {
            amax[j] = __e2k_vmax_f32(amax[j], amax[j+2]),
            amin[j] = __e2k_vmin_f32(amin[j], amin[j+2]);
        }
#pragma unroll
        for (j = 0; j < QS_L; j += 8){
            amax[j] = __e2k_vmax_f32(amax[j], amax[j+4]),
            amin[j] = __e2k_vmin_f32(amin[j], amin[j+4]);
        }
#if __e2k_v__ >= 5
        type_union_128 fcmax = { .__v2di = amax[0] },
                       fcmin = { .__v2di = amin[0] };

        __di amx = fcmax.l.l0, bmx = fcmax.l.l1;
        __di ain = fcmin.l.l0, bin = fcmin.l.l1;
#else
        type_union_64 fcmax, fcmin;

        __di amx = amax[0], bmx = amax[8];
        __di ain = amin[0], bin = amin[8];
#endif
        amx = __builtin_e2k_pfmaxs(amx, bmx);
        ain = __builtin_e2k_pfmins(ain, bin);

        bool hasDx = (amx != 0 || ain != 0);
        float dmax = 0.0f, dmin = 0.0f;
        float dmul = hasDx ? 1.0f : 0.0f;

        if ( hasDx ) {
#if __e2k_v__ >= 5
            fcmax.l.l0 = amx, fcmin.l.l0 = ain;
#else
            fcmax.l0 = amx, fcmin.l0 = ain;
#endif
            dmin = fcmin.f.f0 < fcmin.f.f1 ? fcmin.f.f0 : fcmin.f.f1;
            dmax = fcmax.f.f0 > fcmax.f.f1 ? fcmax.f.f0 : fcmax.f.f1;

            dmax = (dmax - dmin) / 0xF,
            dmul /= dmax;
        }
        y[i].d = __e2k_cvt_f32_f16(dmax);
        y[i].m = __e2k_cvt_f32_f16(dmin);

#if __e2k_v__ >= 5
        fcmax.f.f0 = fcmax.f.f1 = fcmax.f.f2 = fcmax.f.f3 = dmul;
        fcmin.f.f0 = fcmin.f.f1 = fcmin.f.f2 = fcmin.f.f3 = dmin;

        const __vd am = fcmax.__v2di,
                   as = fcmin.__v2di;
#else
        fcmax.f.f0 = fcmax.f.f1 = dmul;
        fcmin.f.f0 = fcmin.f.f1 = dmin;

        const __vd am = fcmax.l0,
                   as = fcmin.l0;
#endif

#pragma unroll
        for (j = 0, k = QS_H; j < QS_H; j++, k++) {
            // Convert f32 -> i32
            x_l[j] = __e2k_vcon_f32i( // (x - min) * ((max - min) / 0xF) + 0.5
                     __e2k_vmul_add_f32(__e2k_vsub_f32(vx[j], as), am, vad)),
            x_h[j] = __e2k_vcon_f32i(
                     __e2k_vmul_add_f32(__e2k_vsub_f32(vx[k], as), am, vad));
        }
#pragma unroll
        for (j = 0, k = 0; j < QS_H; j += 4, k++) {
            __vd  xh, xl;

            xh = __e2k_vpack_i32_i8(x_h[j], x_h[j+1], x_h[j+2], x_h[j+3]);
            xl = __e2k_vpack_i32_i8(x_l[j], x_l[j+1], x_l[j+2], x_l[j+3]);

            xh = __e2k_vmerge(v4max, xh, __e2k_vcmp(gtb, v4max, xh));
            xl = __e2k_vmerge(v4max, xl, __e2k_vcmp(gtb, v4max, xl));

            xh = __e2k_vbitw(and, xh, v4max);

            y[i].qs[k] = __e2k_vbitw(or, xl, __e2k_vshift(lld, xh, 4));
        }
    }
}


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
