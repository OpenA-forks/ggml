
#ifndef __E2K_QN
# define __E2K_QN 5
#endif

#define __E2K_CONST(a,i,b)             a##i##b
#define __E2K_TEMPL(a,i,b) __E2K_CONST(a, i, b)
#define __E2K_QN_T         __E2K_TEMPL(vd_q,__E2K_QN, _0)

/*
  Tamplate function for quantize_row_q4_0/q5_0
*/
__E2K_INLINE void __E2K_TEMPL(__e2k_quantize_row_q, __E2K_QN, _0)(
    const int nb,
    const __vd * restrict x,
    __E2K_QN_T * restrict y
) {

#if __E2K_QN == 5
# define _FEIH_ 0x4184000041840000LL // [16.5, 16.5]
# define _QSMX_ 0x1f1f1f1f1f1f1f1fLL
# define _EI_   -16
#else
# define _FEIH_ 0x4108000041080000LL // [8.5, 8.5]
# define _QSMX_ _MAS4_
# define _EI_   -8
#endif

#if __e2k_v__ >= 5
    const __vd rnd = __builtin_e2k_qppackdl(_FEIH_, _FEIH_),
             vqmax = __builtin_e2k_qppackdl(_QSMX_, _QSMX_),
             v4max = __builtin_e2k_qppackdl(_MAS4_, _MAS4_);
#else
    const __vd rnd = _FEIH_,
             vqmax = _QSMX_,
             v4max = _MAS4_;
#endif
#undef _FEIH_
#undef _QSMX_

    int i, iq;

#pragma loop count(1000)
    for (i = 0, iq = 0; i < nb; iq += QS_L, i++)
    {
        int j, k, c;

        __vd abx[QS_L], max[QS_H], vx[QS_L];

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

        __di a = max[0], ua = abx[0],
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
            dmax /= _EI_,
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

#if __E2K_QN == 5
        __di mq_h[2], mq_l[2];
#endif
#pragma unroll
        for (j = 0; j < QS_L; j++)
            // Convert f32 -> i32 ;; x * (1.0 / d) + 8.5
            vx[j] = __e2k_vcon_f32i(__e2k_vfmadd_f32(vx[j], am, rnd));
#pragma unroll
        for (c = 0, j = 0, k = QS_H; c < QK4_L; c++, j += 4, k += 4) {
            __vd  xl, xh;

            xl = __e2k_vpack_i32_i8(vx[j], vx[j+1], vx[j+2], vx[j+3]);
            xh = __e2k_vpack_i32_i8(vx[k], vx[k+1], vx[k+2], vx[k+3]);

            xl = __e2k_vmerge(vqmax, xl, __e2k_vcmp(gtb, vqmax, xl));
            xh = __e2k_vmerge(vqmax, xh, __e2k_vcmp(gtb, vqmax, xh));

#if __E2K_QN == 5
            // shift target bit in to sign of byte
            __vd ml = __e2k_vshift(lld, xl, 3), // 0001 0001 << 3
                 mh = __e2k_vshift(lld, xh, 3); // 1000 1000

# if __e2k_v__ >= 5
            type_union_128 loq = { .__v2di = ml },
                           hiq = { .__v2di = mh };

            mq_l[0] = loq.l.l0, mq_h[0] = hiq.l.l0,
            mq_l[1] = loq.l.l1, mq_h[1] = hiq.l.l1,
# else
            mq_l[c] = ml, mq_h[c] = mh,
# endif
            xl = __e2k_vbitw(and, xl, v4max);
#endif
            xh = __e2k_vbitw(and, xh, v4max);

            y[i].qs[c] = __e2k_vbitw(or, xl, __e2k_vshift(lld, xh, 4));
        }
#if __E2K_QN == 5
        // create 16-bit mask from bytes signs
        __msk32_t ql = __builtin_e2k_pmovmskb(mq_l[1], mq_l[0]),
                  qh = __builtin_e2k_pmovmskb(mq_h[1], mq_h[0]);

        y[i].qh[0] = ql | (qh << 16);
#endif
#undef _EI_
    }
}
#undef __E2K_CONST
#undef __E2K_TEMPL
#undef __E2K_QN_T
