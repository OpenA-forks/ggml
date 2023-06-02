
#include "ggml-e2kintrin.h"

#ifdef __cplusplus
extern "C"
{
#endif
/* Functions */
#define ARCH_VEC_DOT_Q4_0_Q8_0 __e2k_vec_dot_q4_0_q8_0
#define ARCH_VEC_DOT_Q8_0_Q8_0 __e2k_vec_dot_q8_0_q8_0
#define ARCH_QUANTIZE_ROW_Q4_0 __e2k_quantize_row_q4_0
#define ARCH_QUANTIZE_ROW_Q8_0 __e2k_quantize_row_q8_0
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
    float d;        // delta
    __vd qs[QK4_L]; // nibbles / quants
} vd_q4_0;

typedef struct {
    float d;         // delta
    __vd qs[QK8_L]; // quants
} vd_q8_0;
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
        float fm = x[i].d * y[i].d;
        int sumi, j;
/*
    Wide instructions per iteration:

    E2K_V5+ :  4
    E2K_V3+ :  6
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


__E2K_INLINE float
__e2k_vec_dot_q4_0_q8_0(const int nb, const void * restrict _x, const void * restrict _y)
{
/*
    0x1DE2 - is max posible value for each line (0xF*0xFF + 0xF*0xFF),
             we can safely sum of 16b vec 4 or 8 times (0x1DE2 * 8 => 0xEF10)

    We also don't need to keep the correct order of the sum,
    we only need to take care of the correct multiplication of bytes in vec.
*/
    const vd_q4_0 * restrict x = (const vd_q4_0 * restrict)_x;
    const vd_q8_0 * restrict y = (const vd_q8_0 * restrict)_y;

    float sumf = 0.0;
    int i;

#define _BAIS_ 0x0808080808080808LL
#if __e2k_v__ >= 5
    const __vd mas4 = __builtin_e2k_qppackdl(_MAS4_, _MAS4_),
               bais = __builtin_e2k_qppackdl(_BAIS_, _BAIS_);
#else
    const __vd mas4 = _MAS4_,
               bais = _BAIS_;
#endif
#undef _BAIS_

#pragma loop count(1000)
    for (i = 0; i < nb; i++)
    {
        float fm = x[i].d * y[i].d;
        int sumi, j, k;
/*
    Wide instructions per iteration:

    E2K_V5+ :  5
    E2K_V3+ :  8
    E2K_V2  : 13
*/
       __vd vs[QK4_L];

#pragma unroll
        for (j = 0, k = QK4_L; j < QK4_L; j++, k++) {

            __vd y_l = y[i].qs[j],
                 y_h = y[i].qs[k],
                 x_l = x[i].qs[j],
                 x_h = __e2k_vshift(rlh, x_l, 4);
            // Unpack nibbles into individual bytes
            x_l = __e2k_vbitw(and, x_l, mas4); // 0x1E   -> 0x0E
            x_h = __e2k_vbitw(and, x_h, mas4); // 0xF1F2 -> 0x0F1F -> 0x0F0F
            // Move each one in [ -8 .. +7 ] interval:
            x_l = __e2k_varith(subsb, x_l, bais);
            x_h = __e2k_varith(subsb, x_h, bais);
            // Perform multiplication and create 16-bit values
            y_l = __e2k_vmadd_i8_i16(x_l, y_l);
            y_h = __e2k_vmadd_i8_i16(x_h, y_h);
            // Sum overflow is impossible: 0xF * 0xFF * 4 => 0x7788
            vs[j] = __e2k_varith(addh, y_l, y_h);
        }
#if   __e2k_v__ >= 5
        type_union_128 sat = {
            .__v2di = __e2k_vhsat_i16_i32(vs[0])
        };
        sumi = sat.i.i0 + sat.i.i1 + sat.i.i2 + sat.i.i3;
#else
        type_union_64 sat = {
            .l0 = __e2k_vhsat_i16_i32(__e2k_varith(addh, vs[0], vs[1]))
        };
        sumi = sat.i.i0 + sat.i.i1;
#endif
        sumf += fm * sumi;
    }
    return sumf;
}

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
        y[i].d = dmax;

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
__e2k_quantize_row_q8_0(const int nb, const float * restrict _x, void * restrict _y)
{
    const __vd * restrict x = (const __vd * restrict)_x;
       vd_q8_0 * restrict y = (   vd_q8_0 * restrict)_y;

    int i, iq;

#pragma loop count(1000)
    for (i = 0, iq = 0; i < nb; i++, iq += QS_L)
    {
        int j, k;
/*
    Wide instructions per iteration:

    E2K_V5+ : 11
    E2K_V4  : 16
    E2K_V2+ : 19

    Compiler flags: lcc -O4 -ffast
*/
        __vd umx[QS_L], vx[QS_L];

#pragma unroll
        for (j = 0; j < QS_L; j++) {
            // all `vx` values writes in registers, not to stack
            umx[j] = __e2k_vabs_f32((vx[j] = x[iq + j]));
        }
#pragma unroll
        for (j = 0; j < QS_L; j += 2)
            umx[j] = __e2k_vmax_f32(umx[j], umx[j+1]);
#pragma unroll
        for (j = 0; j < QS_L; j += 4)
            umx[j] = __e2k_vmax_f32(umx[j], umx[j+2]);
#pragma unroll
        for (j = 0; j < QS_L; j += 8)
            umx[j] = __e2k_vmax_f32(umx[j], umx[j+4]);

#if __e2k_v__ >= 5
        type_union_128 fcm = { .__v2di = umx[0] };

        __di a = fcm.l.l0, b = fcm.l.l1;
#else
        type_union_64 fcm;

        __vd a = umx[0], b = umx[8];
#endif
        bool hasDx = (a != 0 || b != 0);
        float dmax = 0.0f;
        float dmul = hasDx ? 1.0f : 0.0f;

        if ( hasDx ) {
            a = __builtin_e2k_pfmaxs(a, b);

#if __e2k_v__ >= 5
            fcm.l.l0 = a;
#else
            fcm.l0 = a;
#endif
            dmax = fcm.f.f1 <= fcm.f.f0 ? fcm.f.f0 : fcm.f.f1,
            dmax /= 0x7F, //((1 << 7) - 1)
            dmul /= dmax;
        }
        y[i].d = dmax;

#if __e2k_v__ >= 5
        fcm.f.f0 = fcm.f.f1 = fcm.f.f2 = fcm.f.f3 = dmul;

#pragma unroll
        for (j = 0; j < QS_L; j++)
            vx[j] = __builtin_e2k_qpfmuls(vx[j], fcm.__v2di);
#else
        fcm.f.f0 = fcm.f.f1 = dmul;

#pragma unroll
        for (j = 0; j < QS_L; j++)
            vx[j] = __builtin_e2k_pfmuls(vx[j], fcm.l0);
#endif
#pragma unroll
        for (j = 0, k = 0; j < QK8_L; j++, k += 4)
        {
            __vd x0 = __e2k_vcon_f32i(__e2k_vround_f32(vx[j+0])),
                 x1 = __e2k_vcon_f32i(__e2k_vround_f32(vx[j+1])),
                 x2 = __e2k_vcon_f32i(__e2k_vround_f32(vx[j+2])),
                 x3 = __e2k_vcon_f32i(__e2k_vround_f32(vx[j+3]));

            y[i].qs[j] = __e2k_vpack_i32_i8(x0, x1, x2, x3);
        }
    }
}

#ifdef __cplusplus
}
#endif
