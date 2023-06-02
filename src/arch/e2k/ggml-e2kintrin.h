
#include <e2kintrin.h>

#ifndef restrict
#define restrict __restrict
#endif

/**
    E2K - the general purpose Little-endian VLIW architecture.

    Major futures of e2k (especialy for GGML):

    1. 512b wide instruction~bundles executes by one clock.
       Typical operations is RISC-like - `ldw, adds, pfmuls, etc`.
       Bundle can contain large constants and use them with commands bypassing registers.

    2. 192 procedure registers, common to float/int/vector op's.
       Register allocation occurs by two register windows -
       the first window sets the static local registers for the current procedure ~ {%r0...%r20}
       the second sets the moving window inside the static window ~ {%b[0]...%b[10]}
       and changes its relative position.

 Note1: The reg. windows sets inside procedure and can be resized,
        but current base (position) controlled by hardware and cannot be changed

 Note2: Original reg. size is 64b (+ 16b extension for f64 multiplys).
        64b `%dr0` splits into `%r1,%r0` (`hi,lo`), 16b ext cannot be used separatly

 Note3: E2Kv5+ has 128b registers (I'm not sure if they were just grouped in 192x64b -> 96x128b)
        128b `%xr0` also splits into `%dr1,%dr0` and `%r3,%r2,%r1,%r0`

    3. Array Prefetch Buffer (APB) - this is special unit for fetch data from memory or cache
       No matter how well the vliw bundles are packaged, or how many on per iteration,
       is unpredicteble how many clock cycles we have wait for each `ld` commands.
       The array prefetch unit allows you to order to asynchronously pull data
       from memory into his buffer (by setup the address, size and data size of array at runtime).
       Preparation may take some time, but then in loop we just `mov` data from buffer without memory access.
       The buffer will fill up as the data is pullout.

 Note4: e2k_v < 5 required memory alignment for fast array data acess via APB unit.
        (future versions are not needed, but recommend).

 Conclusion: So, when optimizations for e2k arch, we do not need to abuse vectorization and loop unrolling,
             only to reduce the code and make more efficient use of available registers,
             which even having such a large number (100+) is still useful to save for procedure transitions.

*/
#define __e2k_v__ __iset__


#define _ONES_  0x0001000100010001LL
#define _FABS_ ~0x8000000080000000LL


/*
    Setup universal vector size, data types and command groups
    for different ISA versions.
*/
#if __e2k_v__ >= 5
/*
    Modern Quad registers (AVX compatible)

    E2Kv5 ~ AVX compatible
    E2Kv6 ~ FMA compatible
*/
typedef __v2di __vd;

# define __e2k_vsub_f32      __builtin_e2k_qpfsubs
# define __e2k_vmin_f32      __builtin_e2k_qpfmins
# define __e2k_vmax_f32      __builtin_e2k_qpfmaxs
# define __e2k_vsign_i8      __builtin_e2k_qpsignb
# define __e2k_vmadd_u8_i16  __builtin_e2k_qpmaddubsh
# define __e2k_vhadd_i32     __builtin_e2k_qphaddw
# define __e2k_vpadd_i32     __builtin_e2k_qpaddw
# define __e2k_vmerge        __builtin_e2k_qpmerge

/* Horisontal sum i16x8 vector pairs with i32x4 vector output */
# define __e2k_vhsat_i16_i32(s1) __builtin_e2k_qpmaddh(s1,\
                                 __builtin_e2k_qppackdl(_ONES_,_ONES_))
/* Remove sign from f32x4 vector numbers */
# define __e2k_vabs_f32(s1) __builtin_e2k_qpand(s1,\
                            __builtin_e2k_qppackdl(_FABS_,_FABS_))
/* Rounding of f32x4 vector (nearest) */
# define __e2k_vround_f32(s1) __builtin_e2k_qpfstoifs(_TOIF_RC_NEAREST,s1)
/* Converting f32x4 vector to i32x4 */
# define __e2k_vcon_f32i    __builtin_e2k_qpfstoistr
/* Float multiply+add of f32x4 vectors.
 * the compiler replace it if target arch support FMA instructions */
# define __e2k_vmul_add_f32(m1,m2,a3) __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(m1,m2), a3)

/* Bitwise operations:
 * and  - `&`
 * andn - `~`
 * or   - `|`
 * xor  - `^`
*/
# define __e2k_vbitw(PAT, s1, s2) __builtin_e2k_qp##PAT(s1, s2)

/* Shifts i64x2,i32x4,i16x8,i8x16 vectors:
 * l{l,a,c}{d,w,h,b} - `<<`
 * r{l,a,c}{d,w,h,b} - `>>`
 * _|
 * _{logic,arithmetic,cycled}
*/
# define __e2k_vshift(PAT, s1, s2) __builtin_e2k_qps##PAT(s1, s2)

/* Compares i64x2,i32x4,i16x8,i8x16 vectors:
 * gt{d,w,h,b} - `>`
 * eq{d,w,h,b} - `==`
*/
# define __e2k_vcmp(PAT, s1, s2) __builtin_e2k_qpcmp##PAT(s1, s2)

/* Compares two f32x4:
 * les - `<=`
 * lts - `<`
 * eqs - `==`
 * neqs - `!=`
*/
# define __e2k_vfcmp(PAT, s1, s2) __builtin_e2k_qpfcmp##PAT(s1, s2)

/* Pack high parts of operands into one striped vector
    B{15..8} ~ A{15..8} -> {A15 B14 ... A1 B0} */
# define __e2k_vnpck_Hb(b,a) \
         __builtin_e2k_qppermb(b,a, \
           __builtin_e2k_qppackdl(0x1F0F1E0E1D0D1C0CLL, 0x1B0B1A0A19091808LL))

/* Pack low parts of operands into one striped vector
    B{7..0} ~ A{7..0} -> {A15 B14 ... A1 B0} */
# define __e2k_vnpck_Lb(b,a) \
         __builtin_e2k_qppermb(b,a,\
           __builtin_e2k_qppackdl(0x1707160615051404LL, 0x1303120211011000LL))


#else // __e2k_v__ <= 4
/*
    Standart double registers (SSE compatible)

    E2Kv3 ~ SSSE3 compatible
    E2Kv4 ~ SSE4(1,2,a) compatible
*/
typedef __di __vd;

# if __e2k_v__ >= 3
#  define __e2k_vsign_i8      __builtin_e2k_psignb
#  define __e2k_vmadd_u8_i16  __builtin_e2k_pmaddubsh
#  define __e2k_vhadd_i32     __builtin_e2k_phaddw
/* Rounding of f32x2 vector (nearest) */
#  define __e2k_vround_f32(s1) __builtin_e2k_pfstoifs(_TOIF_RC_NEAREST,s1)
# else
// `mul_add` without vec expansion
#  define __e2k_vmaddn_i16(xe, xo, ye, yo) \
          __builtin_e2k_paddh(             \
             __builtin_e2k_pmullh(xe, ye), \
             __builtin_e2k_pmullh(xo, yo))

/* Rounding of f32x2 vector (nearest) */
__E2K_INLINE __di
__e2k_vround_f32(__di s1) {
    __di dst =__builtin_e2k_pistofs(
              __builtin_e2k_pfstois(s1 & _FABS_));
    return dst | (s1 & ~_FABS_);
}

# endif

/* Horisontal sum i16x4 vector pairs with i32x2 vector output */
# define __e2k_vhsat_i16_i32(s1) __builtin_e2k_pmaddh(s1, _ONES_)
// B{7..4} ~ A{7..4} -> {A7 B6 ... A1 B0}
# define __e2k_vnpck_Hb  __builtin_e2k_punpckhbh
// B{3..0} ~ A{3..0} -> {A7 B6 ... A1 B0} 
# define __e2k_vnpck_Lb  __builtin_e2k_punpcklbh
# define __e2k_vpadd_i32 __builtin_e2k_paddw
# define __e2k_vmerge    __builtin_e2k_pmerge
# define __e2k_vsub_f32  __builtin_e2k_pfsubs
# define __e2k_vmin_f32  __builtin_e2k_pfmins
# define __e2k_vmax_f32  __builtin_e2k_pfmaxs
/* Remove sign from f32x2 vector numbers */
# define __e2k_vabs_f32(s1) __builtin_e2k_pandd(s1, _FABS_)
/* Float multiply+add of f32x2 vectors.
 * the compiler will decide for itself whether it
 * is worth replacing this with `fmul_add` instruction */
# define __e2k_vmul_add_f32(m1,m2,a3) __builtin_e2k_pfadds(__builtin_e2k_pfmuls(m1,m2), a3)
/* Converting f32x2 vector to i32x2 */
# define __e2k_vcon_f32i  __builtin_e2k_pfstoistr

/* Bitwise operations:
 * and  - `&`
 * andn - `~`
 * or   - `|`
 * xor  - `^`
*/
# define __e2k_vbitw(_P_, s1, s2) __builtin_e2k_p##_P_##d(s1, s2)

/* Shifts i64x1,i32x2,i16x4,i8x8 vectors:
 * l{l,a,c}{d,w,h,b} - `<<`
 * r{l,a,c}{d,w,h,b} - `>>`
 * _|
 * _{logic,arithmetic,cycled}
*/
# define __e2k_vshift(PAT, s1, s2) __builtin_e2k_ps##PAT(s1, s2)

/* Compares i64x1,i32x2,i16x4,i8x8 vectors:
 * gt{d,w,h,b} - `>`
 * eq{d,w,h,b} - `==`
*/
# define __e2k_vcmp(PAT, s1, s2) __builtin_e2k_pcmp##PAT(s1, s2)

/* Compares two f32x2:
 * les - `<=`
 * lts - `<`
 * eqs - `==`
 * neqs - `!=`
*/
# define __e2k_vfcmp(PAT, s1, s2) __builtin_e2k_pfcmp##PAT(s1, s2)

/*
    Splits i8x8 vector by even/odd bytes in to two i16x4:

    1) logic      (0x0F8F << 8) => 0x8F00
    2) arithmetic (0x8F00 >> 8) => 0x800F
*/
# define __e2k_vspliteo_i8_i16(src1, dst_e, dst_o) \
    (dst_o = __builtin_e2k_psrah(               \
             __builtin_e2k_psllh(src1, 8), 8)), \
    (dst_e = __builtin_e2k_psrah(src1, 8))

#endif // __e2k_v__ 


__E2K_INLINE __vd
__e2k_vmadd_i8_i16(__vd sx, __vd sy) {

#if __e2k_v__ >= 3
    // Sign the values of the y vectors
    sy = __e2k_vsign_i8(sy, sx);
    // Sign the values of the x vectors
    sx = __e2k_vsign_i8(sx, sx);
    // Perform multiplication and create 16-bit values
    return __e2k_vmadd_u8_i16(sx, sy);

#else
    __vd x_e, x_o, y_e, y_o;

    __e2k_vspliteo_i8_i16(sx, x_e, x_o);
    __e2k_vspliteo_i8_i16(sy, y_e, y_o);

    return __e2k_vmaddn_i16(x_e, x_o, y_e, y_o);
#endif
}

__E2K_INLINE __vd
__e2k_vpack_i32_i8(__vd s1, __vd s2, __vd s3, __vd s4) {
#if __e2k_v__ >= 5
    type_union_128 v0, v1, v2, v3, dst;

    v0.__v2di = s1, v2.__v2di = s3;
    v1.__v2di = s2, v3.__v2di = s4;

    dst.c.c0 = v0.i.i0, dst.c.c4 = v1.i.i0, dst.c.c8  = v2.i.i0, dst.c.c12 = v3.i.i0,
    dst.c.c1 = v0.i.i1, dst.c.c5 = v1.i.i1, dst.c.c9  = v2.i.i1, dst.c.c13 = v3.i.i1,
    dst.c.c2 = v0.i.i2, dst.c.c6 = v1.i.i2, dst.c.c10 = v2.i.i2, dst.c.c14 = v3.i.i2,
    dst.c.c3 = v0.i.i3, dst.c.c7 = v1.i.i3, dst.c.c11 = v2.i.i3, dst.c.c15 = v3.i.i3;

    return dst.__v2di;

#else
    __vd dst0, dst1;

    dst0 = __builtin_e2k_pshufb(s2, s1, 0x808080800C080400LL);
    dst1 = __builtin_e2k_pshufb(s4, s3, 0x0C08040080808080LL);
    return __builtin_e2k_pord(dst0, dst1);
#endif
}
