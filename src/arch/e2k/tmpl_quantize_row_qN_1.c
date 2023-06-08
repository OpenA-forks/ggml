
#ifndef __E2K_QN
# define __E2K_QN 5
#endif

#define __E2K_CONST(a,i,b)             a##i##b
#define __E2K_TEMPL(a,i,b) __E2K_CONST(a, i, b)
#define __E2K_QN_T         __E2K_TEMPL(vd_q,__E2K_QN, _1)

/*
  Tamplate function for quantize_row_q4_1/q5_1
*/
__E2K_INLINE void __E2K_TEMPL(__e2k_quantize_row_q, __E2K_QN, _1)(
    const int nb,
    const __vd * restrict x,
    __E2K_QN_T * restrict y
) {

#if __E2K_QN == 5
# define _EI_   0x1F
#else
# define _EI_   0xF
#endif

#define VF_MAX 0x7F7FFFFF7F7FFFFFLL //  FLT_MAX x 2
#define VF_MIN 0xFF7FFFFFFF7FFFFFLL // -FLT_MAX x 2
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
#undef VF_MAX
#undef VF_MIN

    int i, iq;

#pragma loop count(1000)
    for (i = 0, iq = 0; i < nb; iq += QS_L, i++)
    {
        int j, k, c;

        __vd amax[QS_L], amin[QS_L], vx[QS_L];

#pragma unroll
        for (j = 0; j < QS_L; j++) {
            // all `vx` values writes in registers, not to stack
              vx[j] = x[iq + j], 
            amax[j] = __e2k_vmax_f32(x[iq + j], vfmin),
            amin[j] = __e2k_vmin_f32(x[iq + j], vfmax);
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

            dmax = (dmax - dmin) / _EI_,
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

#if __E2K_QN == 5
        __di mq_h[2], mq_l[2];
#endif
#pragma unroll
        for (j = 0; j < QS_L; j++) {
            // Convert f32 -> i32
            vx[j] = __e2k_vcon_f32i( // (x - min) * ((max - min) / 0xF) + 0.5
                    __e2k_vmul_add_f32(__e2k_vsub_f32(vx[j], as), am, vad));
        }
#pragma unroll
        for (c = 0, j = 0, k = QS_H; c < QK4_L; c++, j += 4, k += 4) {

            __vd xl = __e2k_vpack_i32_i8(vx[j], vx[j+1], vx[j+2], vx[j+3]),
                 xh = __e2k_vpack_i32_i8(vx[k], vx[k+1], vx[k+2], vx[k+3]);

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
#else
            xl = __e2k_vmerge(v4max, xl, __e2k_vcmp(gtb, v4max, xl));
            xh = __e2k_vmerge(v4max, xh, __e2k_vcmp(gtb, v4max, xh));
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
