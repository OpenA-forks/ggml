
#ifndef __E2K_PS
# define __E2K_PS 32
#endif

#define __E2K_CONST(a,i,b)             a##i##b
#define __E2K_TEMPL(a,i,b) __E2K_CONST(a, i, b)
#define __E2K_QN_T         __E2K_TEMPL(__f,__E2K_PS, _t)

/*
  Tamplate function for vec_dot_f16/f32
*/
__E2K_INLINE float __E2K_TEMPL(__e2k_vec_dot_f, __E2K_PS, p)(
    const int n,
    const __E2K_QN_T * restrict x,
          __E2K_QN_T * restrict y
) {
    float sumf = 0.0;
    int i, k;

#if __E2K_PS == 32
# define VF_L VF32_C
#else
# define VF_L VF16_C
#endif

    const int ev = (n % VF_L),
              nb = (n - ev);

    const __vd * restrict vx = (const __vd * restrict)x;
    const __vd * restrict vy = (const __vd * restrict)y;

#if __E2K_PS == 32
    float d0 = ev != 0 ? x[n + 0] * y[n + 0] : 0;
# if __e2k_v__ >= 5
    float d1 = ev >= 2 ? x[n + 1] * y[n + 1] : 0;
    float d2 = ev == 3 ? x[n + 2] * y[n + 2] : 0;
# endif
#endif

#pragma loop count (1000)
    for (i = 0, k = 0; i < nb; i += VF_L, k++)
    {
#if __E2K_PS == 32
        __vd vsum = __e2k_varith(fmuls, vx[k], vy[k]);
#else
        __vd vsum = __e2k_vhsat_f16_f32(vx[k], vy[k]);
#endif
#if __e2k_v__ >= 5
        type_union_128 sat = { .__v2di = vsum };

        sumf += sat.f.f0 + sat.f.f1 + sat.f.f2 + sat.f.f3;
#else
        type_union_64 sat = { .l0 = vsum };

        sumf += sat.f.f0 + sat.f.f1;
#endif
    }

#if __E2K_PS == 16
# pragma loop count (VF16_C - 1)
    for (; i < n; i++)
        sumf += __e2k_cvt_f16_f32(x[i]) * __e2k_cvt_f16_f32(y[i]);
#else
    sumf += d0;
# if __e2k_v__ >= 5
    sumf += d1 + d2;
# endif
#endif
    return sumf;
}
#undef __E2K_CONST
#undef __E2K_TEMPL
#undef __E2K_QN_T
