
/* F16C format is not supported by many arch's.
   this functions implement converting fp32 <--> fp16
   as lite as possible.
*/
typedef unsigned short ggml_fp16_t;
typedef float          ggml_fp32_t;
typedef union {
    unsigned u;
    float f;
} __attribute__ ((__may_alias__)) ggml_bits_fp;

#define _Exp_  0x70000000u
#define _Inf_  0xFF000000u
#define _F0d5_ 0x3F000000u // 0.5f
#define _SiGn_ 0x80000000u

/*
    Combined method for convert fp16 -> fp32 

    normal(inf/nan) :~ from e2kbuiltin
    denormalized    :~ https://github.com/Maratyszcza/FP16
*/
static inline ggml_fp32_t ggml_cvt_fp16_to_fp32(unsigned h)
{
    const unsigned w = h << 16;
    const unsigned twos = (w + w);
    const unsigned sign = (w & _SiGn_);
    const bool is_lestn = (twos < (_SiGn_>> 4)); // less than null

    ggml_bits_fp normal, denorm, scale = { .u = _F0d5_ };

#if 1 // this varian used more independent and low-cost op's
    unsigned mant = ((h & 0x03ff) << 13),
              exp = ((h & 0x7c00));
              exp += /*exp == 0x7c00 ? 0x30000 :*/ 0x1C000; // decomment if need checks inf/nan
    normal.u = (exp << 13) | mant; // normal ~ A (e2kbuiltin)
#else
    normal.u = (twos >> 4 ) + _Exp_ , normal.f *= 0x1.0p-112f; // normal ~ B (Maratyszcza)
#endif
    denorm.u = (twos >> 17) | _F0d5_, denorm.f -= scale.f; // denormalized

    normal.u = (is_lestn ? denorm.u : normal.u) | sign;

    return normal.f;
}

static inline ggml_fp16_t ggml_cvt_fp32_to_fp16(float f)
{
    ggml_bits_fp base = { .f = f },
                 bias;

    unsigned twos = base.u + base.u,
             sign = base.u & _SiGn_;

    bias.u  = (twos & _Inf_) >> 1;
    base.u &= (~_SiGn_); // fabs()

    if (bias.u < 0x38800000u) {
        bias.u = 0x38800000u;
    }
    base.f *= 0x1.0p+112f, //0x77800000u  ~ inf
    base.f *= 0x1.0p-110f, //0x08800000U  ~ zero
    bias.f += base.f;

    const unsigned exp = (bias.u >> 13) & 0x7C00;
    const unsigned mant = bias.u & 0xFFF;

    return (sign >> 16) | (twos > _Inf_ ? 0x7E00 : exp + mant);
}

#undef _Inf_
#undef _Exp_
#undef _F0d5_
#undef _SiGn_
