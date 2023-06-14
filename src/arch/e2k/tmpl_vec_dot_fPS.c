
#ifndef __E2K_PS
# define __E2K_PS 32
#endif

#define __E2K_CONST(a,i,b)             a##i##b
#define __E2K_TEMPL(a,i,b) __E2K_CONST(a, i, b)
#define __E2K_QN_T         __E2K_TEMPL(ggml_fp,__E2K_PS, _t)
#define __vmadd_ps         __E2K_TEMPL(__e2k_vmadd_f,__E2K_PS,)

/*
  Tamplate function for vec_dot_f16/f32
*/
double __E2K_TEMPL(__e2k_vec_dot, _f, __E2K_PS)(
    const int n,
    const __E2K_QN_T * restrict x,
    const __E2K_QN_T * restrict y
) {
const __vd zero = __e2k_vset(0);
      __vd vsum = zero, mask, venx, veny;

    int i, k; 

#if __E2K_PS == 32
# define VF_L VF32_C
#else
# define VF_L VF16_C
#endif

    unsigned mod = n % VF_L;
    const int nb = n - mod;

    const __vd * restrict vx = (const __vd *)&x[0];
    const __vd * restrict vy = (const __vd *)&y[0];

#pragma loop count (1000)
    for (i = 0, k = 0; i < nb; i += VF_L, k++)
    {
        vsum = __vmadd_ps(vx[k], vy[k], vsum);
        venx = vx[k + 1];
        veny = vy[k + 1];
    }
    if (mod != 0) {
        unsigned sb = __E2K_PS * mod;
#if __e2k_v__ >= 5
        __di lo = sb >= 64 ? MAXUL >> (sb - 64) : MAXUL,
             hi = sb <  64 ? MAXUL >> (64 - sb) : 0;

        mask = __builtin_e2k_qppackdl(lo, hi);
#else
        mask = MAXUL >> sb;
#endif
        venx = __vmadd_ps(
            __e2k_vmerge(venx, zero, mask),
            __e2k_vmerge(veny, zero, mask), vsum);
    }
#if __e2k_v__ >= 5
    type_union_128 sat = { .__v2di = vsum };

    return (double)sat.f.f0 + (double)sat.f.f1 +
           (double)sat.f.f2 + (double)sat.f.f3;
#else
    type_union_64 sat = { .l0 = vsum };

    return (double)sat.f.f0 + (double)sat.f.f1;
#endif
#undef VF_L
}
#undef __E2K_CONST
#undef __E2K_TEMPL
#undef __E2K_QN_T
