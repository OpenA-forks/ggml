

#include "ggml-e2kintrin.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define ARCH_VEC_DOT_Q4_0_Q8_0 __e2k_vec_dot_q4_0_q8_0
#define ARCH_VEC_DOT_Q8_0_Q8_0 __e2k_vec_dot_q8_0_q8_0
#define ARCH_QUANTIZE_ROW_Q4_0 __e2k_quantize_row_q4_0


/*
    Sets vectors size and data types for specied e2k vers.
*/
#if __e2k_v__ >= 5
// modern quad registers (AVX compatible)
# define QK4_0L 1
# define QK8_0L 2
#else
// standart double registers (SSE compatible)
# define QK4_0L 2
# define QK8_0L 4
#endif

typedef struct {
    float d;         // delta
    __vd qs[QK4_0L]; // nibbles / quants
} vd_q4_0;

typedef struct {
    float d;         // delta
    __vd qs[QK8_0L]; // quants
} vd_q8_0;
/* <==== end block ====^ */


__E2K_INLINE float
__e2k_vec_dot_q8_0_q8_0(const int nc, const void * restrict vx, const void * restrict vy)
{
    const vd_q8_0 * restrict x = (const vd_q8_0 * restrict)vx;
    const vd_q8_0 * restrict y = (const vd_q8_0 * restrict)vy;

    float sumf = 0.0;
    int i;

#pragma loop count(1000)
    for (i = 0; i < nc; i++)
    {
        float fm = x[i].d * y[i].d;
        int sumi = 0, j = 0;
        __vd sumvd;
/*
    Wide instructions per iteration:

    E2K_V5+ :  4
    E2K_V3+ :  6
    E2K_V1+ : 14
*/
        for (; j < QK8_0L; j++) {
            __vd dx = x[i].qs[j],
                 dy = y[i].qs[j];

            __vd sh = __e2k_vhsat_i16_i32(
                      __e2k_vmadd_i8_i16(dx, dy) );
              sumvd = __e2k_vpadd_i32(sumvd, sh);
        }
#if   __e2k_v__ >= 5
        type_union_128 sat = {
            .__v2di = sumvd
        };
        sumi += sat.i.i0 + sat.i.i1 + sat.i.i2 + sat.i.i3;
#else
        type_union_64 sat = {
            .l0 = sumvd
        };
        sumi += sat.i.i0 + sat.i.i1;
#endif
        sumf += fm * sumi;
    }
    return sumf;
}


__E2K_INLINE float
__e2k_vec_dot_q4_0_q8_0(const int nc, const void * restrict vx, const void * restrict vy)
{
/*
    0x1DE2 - is max posible value for each line (0xF*0xFF + 0xF*0xFF)

    we can safely sum of 16b vec 4 or 8 times (0x1DE2 * 8 => 0xEF10)
*/
    const vd_q4_0 * restrict x = (const vd_q4_0 * restrict)vx;
    const vd_q8_0 * restrict y = (const vd_q8_0 * restrict)vy;

    float sumf = 0.0;
    int i;

#define _MASK_ 0x0F0F0F0F0F0F0F0FLL
#define _BAIS_ 0x0808080808080808LL

#pragma loop count(1000)
    for (i = 0; i < nc; i++)
    {
        float fm = x[i].d * y[i].d;
        int sumi = 0;
/*
    Wide instructions per iteration:

    E2K_V5+ :  9
    E2K_V3+ :  9
    E2K_V1+ : 13
*/
#if   __e2k_v__ >= 5
        type_union_128 sat;

        const __vd mask = __builtin_e2k_qppackdl(_MASK_, _MASK_);
        const __vd bais = __builtin_e2k_qppackdl(_BAIS_, _BAIS_);

        // Loads 8b arrays as 128b vectors
        __vd dx0 = x[i].qs[0], x_lo, ax0,
             dy0 = y[i].qs[0], x_hi, ax1,
             dy1 = y[i].qs[1], dot0, dot1;

        // Unpack nibbles into individual bytes
        x_lo = __builtin_e2k_qpand( dx0,     mask); // 0xFF   -> 0x0F
        x_hi = __builtin_e2k_qpand(                 // 0xFFFF -> 0x0FFF -> 0x0F0F
               __builtin_e2k_qpsrlh(dx0, 4), mask);
        // Pack "hi" and "lo" bytes in to vectors, ordered by dy[0-1] values
        ax0 = __e2k_vnpck_Hb(x_hi, x_lo);
        ax1 = __e2k_vnpck_Lb(x_hi, x_lo);
        // Move each one in [ -8 .. +7 ] interval:
        ax0 = __builtin_e2k_qpsubb(ax0, bais);
        ax1 = __builtin_e2k_qpsubb(ax1, bais);
        // Perform multiplication and create 16-bit values
        dot0 = __e2k_vmadd_i8_i16(ax0, dy0);
        dot1 = __e2k_vmadd_i8_i16(ax1, dy1);
        // Sum overflow is impossible: 0xF * 0xFF * 4 => 0x7788
        dot0 = __builtin_e2k_qpaddh(dot0, dot1);
        // Reduce result to 4 int32_t
        sat.__v2di = __e2k_vhsat_i16_i32(dot0);

        sumi += sat.i.i0 + sat.i.i1 + sat.i.i2 + sat.i.i3;

#else //__e2k_v__ <= 4
        type_union_64 sat;

        // Loads 8b arrays as 64b long numbers
        __vd sh0, dy0 = y[i].qs[0], dx0 = x[i].qs[0],
             sh1, dy1 = y[i].qs[1], dx1 = x[i].qs[1],
             sh2, dy2 = y[i].qs[2], x0_l, x1_l,
             sh3, dy3 = y[i].qs[3], x0_h, x1_h;

# if __e2k_v__ >= 3 // -- generates 10 wide commands and 9 clocks per iteration

        __vd ax0, ax1, ax2, ax3;

        // Split "x" 4b values in to 8b "lo" and "hi" values
        x0_l = __builtin_e2k_pandd(dx0,      _MASK_);
        x0_h = __builtin_e2k_psrlh(dx0, 4) & _MASK_;
        x1_l = __builtin_e2k_pandd(dx1,      _MASK_); // 0xFF   -> 0x0F
        x1_h = __builtin_e2k_psrlh(dx1, 4) & _MASK_;  // 0xFFFF -> 0x0FFF -> 0x0F0F
        // Pack "hi" and "lo" values in to vectors, ordered by dy[0-3] values
        ax0 = __builtin_e2k_punpckhbh(x0_h, x0_l);
        ax1 = __builtin_e2k_punpcklbh(x0_h, x0_l);
        ax2 = __builtin_e2k_punpckhbh(x1_h, x1_l);
        ax3 = __builtin_e2k_punpcklbh(x1_h, x1_l);
        // Offset into [ -8 .. +7 ] interval.
        ax0 = __builtin_e2k_psubb(ax0, _BAIS_);
        ax1 = __builtin_e2k_psubb(ax1, _BAIS_);
        ax2 = __builtin_e2k_psubb(ax2, _BAIS_);
        ax3 = __builtin_e2k_psubb(ax3, _BAIS_);
        // Multiply + saturation of x,y values with 16-bit output
        sh0 = __e2k_vmadd_i8_i16(ax0, dy0);
        sh1 = __e2k_vmadd_i8_i16(ax1, dy1);
        sh2 = __e2k_vmadd_i8_i16(ax2, dy2); // {x0 x1 ...}, {y0 y1 ...} -> (x0*y0 + x1*y1), ...
        sh3 = __e2k_vmadd_i8_i16(ax3, dy3);

# else // __generic__

# undef  _BAIS_
# define _BAIS_ 0x0008000800080008LL

        __vd y0_h, y1_h, y2_h, y3_h, x2_h, x3_h, 
             y0_l, y1_l, y2_l, y3_l, x2_l, x3_l;

        // Converts every 4b to u16
        x0_h = __builtin_e2k_psrlh(dx0 , 12);
        x0_l = __builtin_e2k_psrlh(dx0 & 0x0F000F000F000F00LL, 8);
        x1_h = __builtin_e2k_psrlh(dx0 & 0x00F000F000F000F0LL, 4);
        x1_l = __builtin_e2k_pandd(dx0 , 0x000F000F000F000FLL);

        x2_h = __builtin_e2k_psrlh(dx1 , 12);
        x2_l = __builtin_e2k_psrlh(dx1 & 0x0F000F000F000F00LL, 8);
        x3_h = __builtin_e2k_psrlh(dx1 & 0x00F000F000F000F0LL, 4);
        x3_l = __builtin_e2k_pandd(dx1 , 0x000F000F000F000FLL);

        // u16x4 - i8x4 => i16x4
        x0_h = __builtin_e2k_psubsh(x0_h,_BAIS_);
        x0_l = __builtin_e2k_psubsh(x0_l,_BAIS_);
        x1_h = __builtin_e2k_psubsh(x1_h,_BAIS_);
        x1_l = __builtin_e2k_psubsh(x1_l,_BAIS_);

        x2_h = __builtin_e2k_psubsh(x2_h,_BAIS_);
        x2_l = __builtin_e2k_psubsh(x2_l,_BAIS_);
        x3_h = __builtin_e2k_psubsh(x3_h,_BAIS_);
        x3_l = __builtin_e2k_psubsh(x3_l,_BAIS_);

        __e2k_vspliteo_i8_i16(dy0, y0_h, y0_l);
        __e2k_vspliteo_i8_i16(dy1, y1_h, y1_l);
        __e2k_vspliteo_i8_i16(dy2, y2_h, y2_l);
        __e2k_vspliteo_i8_i16(dy3, y3_h, y3_l);

/*
    Original 4b × 8b algorithm:

    0xFEDC * 0x1A2A3A4A  => (0xE * 0x1A + 0xF * 0x2A) +
                            (0xC * 0x3A + 0xD * 0x4A)

    Vectorised 16b × 16b perform:

    0x000E000C * 0x001A003A +
    0x000F000D * 0x002A004A    =>  (0x000E * 0x001A + 0x000F * 0x002A) +
                                   (0x000C * 0x003A + 0x000E * 0x001A)
*/
        sh0 = __e2k_vmaddn_i16(x0_l, x0_h, y0_h, y0_l);
        sh1 = __e2k_vmaddn_i16(x1_l, x1_h, y1_h, y1_l);
        sh2 = __e2k_vmaddn_i16(x2_l, x2_h, y2_h, y2_l);
        sh3 = __e2k_vmaddn_i16(x3_l, x3_h, y3_h, y3_l);

# endif //__generic__

        // Sums x 2: (0x1DE2 * 2) => 0x3BC4
        sh0 = __builtin_e2k_paddh(sh0, sh1);
        sh2 = __builtin_e2k_paddh(sh2, sh3);
        // Sums x 4: (0x1DE2 * 2) + (0x1DE2 * 2) => 0x7788
        sh0 = __builtin_e2k_paddh(sh0, sh2);
        // Reduce result to 2 x i32
        sat.l0 = __e2k_vhsat_i16_i32(sh0);

        sumi += sat.i.i0 + sat.i.i1;
#endif
        sumf += fm * sumi;
    }
#undef _MASK_
#undef _BAIS_
    return sumf;
}

__E2K_INLINE void
__e2k_quantize_row_q4_0(const int nb, const void * restrict _x, void * restrict _y)
{
    const __di * restrict x = (const __di * restrict)_x;
       vd_q4_0 * restrict y = (   vd_q4_0 * restrict)_y;
/*
   Optimized for 64-bit vectors only,
   because it's not needed constants packing
   and special packed bitwise op's.
*/
    int i, iq;

#define  QS_L (32 / 2)
#define  QS_H (32 / 4)
#define _MIN4_ 0x0000000F0000000FLL
#define _MAX8_ 0x000000FF000000FFLL

#pragma loop count(1000)
    for (i = 0, iq = 0; i < nb; iq += QS_L, i++)
    {
        int j, k;
/*
    Wide instructions per iteration:

    E2K_V2+ : 32
*/
        __di ad[2], am, vx[QS_L];

#pragma unroll
        for (j = 0, k = 0; k < 2; j += 8, k++) {
/*
    Find maximum delta in vectors:

  a0  ;  a1  
------+------   b0 <= a0 ; b1 <= a1  @  FFFF ; 0000
  b0  ;  b1                         ~>   a0  ;  b1
------+------   c0 <= a0 ; c1 <= b1  @  0000 ; FFFF
  c0  ;  c1                         ~>   c0  ;  b1
*/
        __di a = (vx[k + 0] = x[iq + j + 0]), b = (vx[k + 1] = x[iq + j + 1]), m0,
             c = (vx[k + 2] = x[iq + j + 2]), p = (vx[k + 3] = x[iq + j + 3]), m1,
             e = (vx[k + 4] = x[iq + j + 4]), q = (vx[k + 5] = x[iq + j + 5]), m2,
             y = (vx[k + 6] = x[iq + j + 6]), g = (vx[k + 7] = x[iq + j + 7]), m3;

            m0 = __builtin_e2k_pfcmples(b & _MABS_, a & _MABS_);
            m1 = __builtin_e2k_pfcmples(p & _MABS_, c & _MABS_);
            m2 = __builtin_e2k_pfcmples(q & _MABS_, e & _MABS_);
            m3 = __builtin_e2k_pfcmples(g & _MABS_, y & _MABS_);

            a = (a & m0) | (b & ~m0), e = (e & m2) | (q & ~m2);
            p = (c & m1) | (p & ~m1), g = (y & m3) | (g & ~m3);

            m0 = __builtin_e2k_pfcmples(p & _MABS_, a & _MABS_);
            m2 = __builtin_e2k_pfcmples(g & _MABS_, e & _MABS_);

            a = (a & m0) | (p & ~m0),
            g = (e & m2) | (g & ~m2);

            m0 = __builtin_e2k_pfcmples(g & _MABS_, a & _MABS_);

            ad[k] = (a & m0) | (g & ~m0);
        }
        // no delta
        if (ad[1] == 0 && ad[0] == 0) {
            y[i].d = 0.0f;
#if __e2k_v__ >= 5
            type_union_128 sat = { .l = { .l0 = 0x8888888888888888LL, .l1 = 0x8888888888888888LL } };
            y[i].qs[0] = sat.__v2di;
#else
            y[i].qs[0] = y[i].qs[1] = 0x8888888888888888LL;
#endif
            continue;
        }
        am  = __builtin_e2k_pfcmples(ad[1] & _MABS_,ad[0] & _MABS_);
                      /* convert signed max */
        type_union_64 fvd = { .l0 = (ad[0] & am) | (ad[1] & ~am) },
                      /* convert unsigned max */
                      fcm = { .l0 = fvd.l0 & _MABS_ };
                     /* uf0 => uf1 ? sf0 : sf1 */
        float d = (fcm.f.f1 <= fcm.f.f0 ? fvd.f.f0 : fvd.f.f1) / -8;

        fcm.f.f0 = fcm.f.f1 = 1.0f / d;
        fvd.f.f0 = fvd.f.f1 = 8.5f;

        __di qs[QS_H], qs0, qs1;

#pragma unroll
        for (j = 0, k = QS_H; j < QS_H; j++, k++)
        {
            __di x_lo, x_hi, ml, mh;

        /* Example what we doo in this cycle

           0x0000000A0000000B
           0x0000000F0000000E (<< 4) 0x000000F0000000E0  
                              (  OR) 0x000000FA000000EB  
        */
            x_lo = __builtin_e2k_pfadds(// xf0 * (1.0 / d) + 8.5
                   __builtin_e2k_pfmuls(vx[j], fcm.l0), fvd.l0);
            x_hi = __builtin_e2k_pfadds(
                   __builtin_e2k_pfmuls(vx[k], fcm.l0), fvd.l0);
            // Convert f32 -> i32
            x_lo = __builtin_e2k_pfstoistr(x_lo);
            x_hi = __builtin_e2k_pfstoistr(x_hi);
            // 0xF > x  -> mask
            ml  =  __builtin_e2k_pcmpgtw(_MIN4_, x_lo);
            mh  =  __builtin_e2k_pcmpgtw(_MIN4_, x_hi);
            // (x & mask) | (min4 & ~m)
            x_lo = __builtin_e2k_pmerge(_MIN4_, x_lo & _MAX8_, ml);
            x_hi = __builtin_e2k_pmerge(_MIN4_, x_hi & _MIN4_, mh);
            // lo | (hi << 4)
            qs[j] = x_lo | __builtin_e2k_psllw(x_hi, 4);
        }
        qs[0] = __builtin_e2k_psllw(qs[0], 8);
        qs[2] = __builtin_e2k_psllw(qs[2], 8);

        qs[4] = __builtin_e2k_psllw(qs[4], 8);
        qs[6] = __builtin_e2k_psllw(qs[6], 8);

        qs0   = __builtin_e2k_pshufb(
//          0x0000F0F10000E0E1  0x0000F2F30000E2E3
             (qs[0]  |  qs[1]),  (qs[2]  |  qs[3]),
            0x0D090C0805010400LL
        );
        qs1   = __builtin_e2k_pshufb(
//          0x0000F4F50000E4E5  0x0000F6F70000E6E7
             (qs[4]  |  qs[5]),  (qs[6]  |  qs[7]),
            0x0D090C0805010400LL
        );
        y[i].d = d;
#if __e2k_v__ >= 5
        type_union_128 sat = { .l = { .l0 = qs0, .l1 = qs1 } };
        y[i].qs[0] = sat.__v2di;
#else
        y[i].qs[0] = qs0, y[i].qs[1] = qs1;
#endif
#undef  QS_L
#undef  QS_H
#undef _MIN8_
#undef _MIN4_
    }
}

#ifdef __cplusplus
}
#endif
