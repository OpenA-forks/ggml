
#ifndef __E2K_Q8_I
# define __E2K_Q8_I 1
#endif

#define __E2K_CONST(a,b) a##b
#define __E2K_TEMPL(x,n) __E2K_CONST(x, n)
#define __E2K_Q8_T       __E2K_TEMPL(vd_q8_, __E2K_Q8_I)

/*
  Tamplate function for quantize_row_q8_0/q8_1
*/
__E2K_INLINE void __E2K_TEMPL(__e2k_quantize_row_q8_, __E2K_Q8_I)(
    const int nb,
    const void * restrict _x,
          void * restrict _y
) {
    const __vd * restrict x = (const __vd * restrict)_x;
    __E2K_Q8_T * restrict y = (__E2K_Q8_T * restrict)_y;

    int i, iq;

#pragma loop count(1000)
    for (i = 0, iq = 0; i < nb; i++, iq += QS_L)
    {
        int j, k;

        __vd amax[QS_L], vx[QS_L];

#pragma unroll
        for (j = 0; j < QS_L; j++) {
            // all `vx` values writes in registers, not to stack
            amax[j] = __e2k_vabs_f32((vx[j] = x[iq + j]));
        }
#pragma unroll
        for (j = 0; j < QS_L; j += 2)
            amax[j] = __e2k_vmax_f32(amax[j], amax[j+1]);
#pragma unroll
        for (j = 0; j < QS_L; j += 4)
            amax[j] = __e2k_vmax_f32(amax[j], amax[j+2]);
#pragma unroll
        for (j = 0; j < QS_L; j += 8)
            amax[j] = __e2k_vmax_f32(amax[j], amax[j+4]);

#if __e2k_v__ >= 5
        type_union_128 fcmax = { .__v2di = amax[0] };

        __di amx = fcmax.l.l0, bmx = fcmax.l.l1;
#else
        type_union_64 fcmax;

        __di amx = amax[0], bmx = amax[8];
#endif
        amx = __builtin_e2k_pfmaxs(amx, bmx);

        bool hasDx = (amx != 0);
        float dmax = 0.0f;
        float dmul = hasDx ? 1.0f : 0.0f;

        if ( hasDx ) {
#if __e2k_v__ >= 5
            fcmax.l.l0 = amx;
#else
            fcmax.l0 = amx;
#endif
            dmax = fcmax.f.f0 > fcmax.f.f1 ? fcmax.f.f0 : fcmax.f.f1,
            dmax /= 0xF,
            dmul /= dmax;
        }
        y[i].d = dmax;

#if __e2k_v__ >= 5
        fcmax.f.f0 = fcmax.f.f1 = fcmax.f.f2 = fcmax.f.f3 = dmul;

        const __vd am = fcmax.__v2di;
#else
        fcmax.f.f0 = fcmax.f.f1 = dmul;

        const __vd am = fcmax.l0;
#endif
#pragma unroll
        for (j = 0; j < QS_L; j++)
            vx[j] = __e2k_varith(fmuls, vx[j], am);

#if __E2K_Q8_I == 1
        __vd vs[QK8_L]; int sumi;
#endif
#pragma unroll
        for (j = 0, k = 0; k < QK8_L; k++, j += 4)
        {
            __vd x0 = __e2k_vcon_f32i(__e2k_vround_f32(vx[j+0])),
                 x1 = __e2k_vcon_f32i(__e2k_vround_f32(vx[j+1])),
                 x2 = __e2k_vcon_f32i(__e2k_vround_f32(vx[j+2])),
                 x3 = __e2k_vcon_f32i(__e2k_vround_f32(vx[j+3]));

            y[i].qs[k] = __e2k_vpack_i32_i8(x0, x1, x2, x3);

#if __E2K_Q8_I == 1
            x0 = __e2k_varith(addw, x0, x1);
            x2 = __e2k_varith(addw, x2, x3);

            vs[k] = __e2k_varith(addw, x0, x2);
#endif
        }

#if __E2K_Q8_I == 1
# if   __e2k_v__ >= 5
        type_union_128 sat = {
            .__v2di = __e2k_varith(addw, vs[0], vs[1])
        };
        sumi = sat.i.i0 + sat.i.i1 + sat.i.i2 + sat.i.i3;
# else
        type_union_64 sat1, sat2;

        sat1.l0 = __e2k_varith(addw, vs[0], vs[1]);
        sat2.l0 = __e2k_varith(addw, vs[2], vs[3]);

        sumi = sat1.i.i0 + sat1.i.i1 + sat2.i.i0 + sat2.i.i1;
# endif
        y[i].s = dmax * sumi;
#endif
    }
}

#undef __E2K_CONST
#undef __E2K_TEMPL
#undef __E2K_Q8_T
