#ifndef GOLDILOCKS
#define GOLDILOCKS

#include <stdint.h> // uint64_t
#include <string>   // string
#include <gmpxx.h>
#include <iostream> // string

#define USE_MONTGOMERY 0
#define GOLDILOCKS_DEBUG 0

#define GOLDILOCKS_PRIME 0xFFFFFFFF00000001ULL

class Goldilocks
{
public:
    typedef struct
    {
        uint64_t fe;
    } Element;

private:
    static const Element Q;
    static const Element MM;
    static const Element CQ;
    static const Element R2;

    static const Element ZERO;
    static const Element ONE;
    static const Element NEGONE;

public:
    static inline const Element &zero() { return ZERO; };
    static inline void zero(Element &result) { result.fe = ZERO.fe; };

    static inline const Element &one() { return ONE; };
    static inline void one(Element &result) { result.fe = ONE.fe; };

    static inline const Element &negone() { return NEGONE; };
    static inline void negone(Element &result) { result.fe = NEGONE.fe; };

    static Element fromU64(uint64_t in1);
    static void fromU64(Element &result, uint64_t in1);

    static Element fromS32(int32_t in1);
    static void fromS32(Element &result, int32_t in1);

    static uint64_t toU64(const Element &in1);
    static void toU64(uint64_t &result, const Element &in1);

    static int32_t toS32(const Element &in1);
    static void toS32(int32_t &result, const Element &in1);

    static std::string toString(const Element &in1, int radix = 10);
    static void toString(std::string &result, const Element &in1, int radix = 10);

    static Element fromString(const std::string &in1, int radix = 10);
    static void fromString(Element &result, const std::string &in1, int radix = 10);

    static Element add(const Element &in1, const Element &in2);
    static void add(Element &result, const Element &in1, const Element &in2);

    static Element sub(const Element &in1, const Element &in2);
    static void sub(Element &result, const Element &in1, const Element &in2);

    static Element mul(const Element &in1, const Element &in2);
    static void mul(Element &result, const Element &in1, const Element &in2);

    static Element div(const Element &in1, const Element &in2) { return mul(in1, inv(in2)); };
    static void div(Element &result, const Element &in1, const Element &in2) { mul(result, in1, inv(in2)); };

    static Element square(const Element &in1) { return mul(in1, in1); };
    static void square(Element &result, const Element &in1) { return mul(result, in1, in1); };

    static Element neg(const Element &in1) { return sub(Goldilocks::zero(), in1); };
    static void neg(Element &result, const Element &in1) { return sub(result, Goldilocks::zero(), in1); };

    static bool isZero(const Element &in1) { return equal(in1, Goldilocks::zero()); };
    static bool isOne(const Element &in1) { return equal(in1, Goldilocks::one()); };
    static bool isNegone(const Element &in1) { return equal(in1, Goldilocks::negone()); };

    static bool equal(const Element &in1, const Element &in2) { return Goldilocks::toU64(in1) == Goldilocks::toU64(in2); }

    static Element inv(const Element &in1);
    static void inv(Element &result, const Element &in1);

    static Element mulScalar(const Element &base, const uint64_t &scalar);
    static void mulScalar(Element &result, const Element &base, const uint64_t &scalar);

    static Element exp(Element base, uint64_t exp);
    static void exp(Element &result, Element base, uint64_t exps);
};

inline Goldilocks::Element operator+(const Goldilocks::Element &in1, const Goldilocks::Element &in2)
{
    return Goldilocks::add(in1, in2);
}
inline Goldilocks::Element operator+(const Goldilocks::Element &in1)
{
    return in1;
}
inline Goldilocks::Element operator*(const Goldilocks::Element &in1, const Goldilocks::Element &in2)
{
    return Goldilocks::mul(in1, in2);
}
inline Goldilocks::Element operator-(const Goldilocks::Element &in1, const Goldilocks::Element &in2)
{
    return Goldilocks::sub(in1, in2);
}
inline Goldilocks::Element operator-(const Goldilocks::Element &in1)
{
    return Goldilocks::neg(in1);
}
inline Goldilocks::Element operator/(const Goldilocks::Element &in1, const Goldilocks::Element &in2)
{
    return Goldilocks::div(in1, in2);
}

inline bool operator==(const Goldilocks::Element &in1, const Goldilocks::Element &in2)
{
#if USE_MONTGOMERY == 1
    // Convert to montgomery
#endif
    return Goldilocks::toU64(in1) == Goldilocks::toU64(in2);
}

inline std::string Goldilocks::toString(const Element &in1, int radix)
{
    std::string result;
    Goldilocks::toString(result, in1, radix);
    return result;
}

inline void Goldilocks::toString(std::string &result, const Element &in1, int radix)
{
#if USE_MONTGOMERY == 1
    // Convert to montgomery
#endif
    mpz_class aux = Goldilocks::toU64(in1);
    result = aux.get_str(radix);
}

inline uint64_t Goldilocks::toU64(const Element &in1)
{
    uint64_t res;
    Goldilocks::toU64(res, in1);
    return res;
};
inline void Goldilocks::toU64(uint64_t &result, const Element &in1)
{
#if USE_MONTGOMERY == 1
    // Convert to montgomery
#endif
    result = in1.fe % GOLDILOCKS_PRIME;
};

inline Goldilocks::Element Goldilocks::fromU64(uint64_t in1)
{
    Goldilocks::Element res;
    Goldilocks::fromU64(res, in1);
    return res;
}

inline void Goldilocks::fromU64(Element &result, uint64_t in1)
{
#if USE_MONTGOMERY == 1
    // Convert to montgomery
#endif
    result.fe = in1;
}

inline Goldilocks::Element Goldilocks::fromS32(int32_t in1)
{
    Goldilocks::Element res;
    Goldilocks::fromS32(res, in1);
    return res;
}

inline void Goldilocks::fromS32(Element &result, int32_t in1)
{

    uint64_t aux;
    (in1 < 0) ? aux = static_cast<uint64_t>(in1) + GOLDILOCKS_PRIME : aux = static_cast<uint64_t>(in1);

#if USE_MONTGOMERY == 1
    // Convert to montgomery
#endif
    result = {aux};
}

inline int32_t Goldilocks::toS32(const Element &in1)
{
    int32_t res;
    Goldilocks::toS32(res, in1);
    return res;
}

/* Converts a field element into a signed 32bits integer */
/* Precondition:  Goldilocks::Element < 2^31 */
inline void Goldilocks::toS32(int32_t &result, const Element &in1)
{
#if USE_MONTGOMERY == 1
    // Convert from montgomery
#endif
    mpz_class out = Goldilocks::toU64(in1);

    mpz_class maxInt(0x7FFFFFFF);
    mpz_class minInt = (uint64_t)GOLDILOCKS_PRIME - 0x80000000;

    if (out > maxInt)
    {
        mpz_class onegative = (uint64_t)GOLDILOCKS_PRIME - out;
        if (out > minInt)
        {
            result = -onegative.get_si();
        }
        else
        {
            std::cerr << "Error: Goldilocks::toS32 accessing a non-32bit value: " << Goldilocks::toString(in1, 16) << " out=" << out.get_str(16) << " minInt=" << minInt.get_str(16) << " maxInt=" << maxInt.get_str(16) << std::endl;
            exit(-1);
        }
    }
    else
    {
        result = out.get_si();
    }
}

inline Goldilocks::Element Goldilocks::fromString(const std::string &in1, int radix)
{
    Goldilocks::Element result;
    Goldilocks::fromString(result, in1, radix);
    return result;
};
inline void Goldilocks::fromString(Element &result, const std::string &in1, int radix)
{
    mpz_class aux(in1, radix);
    aux = (aux + (uint64_t)GOLDILOCKS_PRIME) % (uint64_t)GOLDILOCKS_PRIME;
#if USE_MONTGOMERY == 1
    // Convert to montgomery
#endif
    result.fe = aux.get_ui();
};

inline Goldilocks::Element Goldilocks::add(const Element &in1, const Element &in2)
{
    Goldilocks::Element result;
    Goldilocks::add(result, in1, in2);
    return result;
}

inline void Goldilocks::add(Element &result, const Element &in1, const Element &in2)
{
    __asm__("xor   %%r10, %%r10\n\t"
            "mov   %1, %0\n\t"
            "add   %2, %0\n\t"
            "cmovc %3, %%r10\n\t"
            "add   %%r10, %0\n\t"
            : "=&a"(result.fe)
            : "r"(in1), "r"(in2), "m"(CQ)
            : "%r10");

#if GOLDILOCKS_DEBUG == 1 && USE_MONTGOMERY == 0
    result.fe = result.fe % GOLDILOCKS_PRIME;
#endif
}

inline Goldilocks::Element Goldilocks::sub(const Element &in1, const Element &in2)
{
    Goldilocks::Element result;
    Goldilocks::sub(result, in1, in2);
    return result;
}

inline void Goldilocks::sub(Element &result, const Element &in1, const Element &in2)
{
    __asm__("mov   %1, %0\n\t"
            "sub   %2, %0\n\t"
            "jnc  1f\n\t"
            "add   %3, %0\n\t"
            "1: \n\t"
            : "=&a"(result.fe)
            : "r"(in1), "r"(in2), "m"(Q)
            :);
#if GOLDILOCKS_DEBUG == 1 && USE_MONTGOMERY == 0
    result.fe = result.fe % GOLDILOCKS_PRIME;
#endif
}

inline Goldilocks::Element Goldilocks::mul(const Element &in1, const Element &in2)
{
    Goldilocks::Element result;
    Goldilocks::mul(result, in1, in2);
    return result;
}

inline void Goldilocks::mul(Element &result, const Element &in1, const Element &in2)
{
#if USE_MONTGOMERY == 1
    __asm__("xor   %%r10, %%r10\n\t"
            "mov   %1, %%rax\n\t"
            "mul   %2\n\t"
            "mov   %%rdx, %%r8\n\t"
            "mov   %%rax, %%r9\n\t"
            "mulq   %3\n\t"
            "mulq   %4\n\t"
            "add    %%r9, %%rax\n\t"
            "adc    %%r8, %%rdx\n\t"
            "cmovc %5, %%r10\n\t"
            "add   %%r10, %%rdx\n\t"
            : "=&d"(result.fe)
            : "r"(in1), "r"(in2), "m"(MM), "m"(Q), "m"(CQ)
            : "%rax", "%r8", "%r9", "%r10");
#else
    __asm__(
        "mulq   %2\n\t"
        "divq   %3\n\t"
        : "=&d"(result.fe)
        : "a"(in1), "rm"(in2), "rm"((uint64_t)GOLDILOCKS_PRIME));
#endif
#if GOLDILOCKS_DEBUG == 1 && USE_MONTGOMERY == 0
    result.fe = result.fe % GOLDILOCKS_PRIME;
#endif
}
inline Goldilocks::Element Goldilocks::inv(const Element &in1)
{
    Goldilocks::Element result;
    Goldilocks::inv(result, in1);
    return result;
};

// TODO: Review and optimize inv imlementation
inline void Goldilocks::inv(Element &result, const Element &in1)
{
    if (Goldilocks::isZero(in1))
    {
        std::cerr << "Error: Goldilocks::inv called with zero" << std::endl;
        exit(-1);
    }
    u_int64_t t = 0;
    u_int64_t r = GOLDILOCKS_PRIME;
    u_int64_t newt = 1;

    u_int64_t newr = Goldilocks::toU64(in1);
    Element q;
    Element aux1;
    Element aux2;
    while (newr != 0)
    {
        q = {r / newr};
        aux1 = {t};
        aux2 = {newt};
        t = Goldilocks::toU64(aux2);
        newt = Goldilocks::toU64(Goldilocks::sub(aux1, Goldilocks::mul(q, aux2)));
        aux1 = {r};
        aux2 = {newr};
        r = Goldilocks::toU64(aux2);
        newr = Goldilocks::toU64(Goldilocks::sub(aux1, Goldilocks::mul(q, aux2)));
    }

    result = {t};
#if GOLDILOCKS_DEBUG == 1 && USE_MONTGOMERY == 0
    result.fe = result.fe % GOLDILOCKS_PRIME;
#endif
};

inline Goldilocks::Element Goldilocks::mulScalar(const Element &base, const uint64_t &scalar)
{
    Goldilocks::Element result;
    Goldilocks::mulScalar(result, base, scalar);
    return result;
};
inline void Goldilocks::mulScalar(Element &result, const Element &base, const uint64_t &scalar)
{
    Element eScalar = fromU64(scalar);
    mul(result, base, eScalar);
};

inline Goldilocks::Element Goldilocks::exp(Element base, uint64_t exp)
{
    Goldilocks::Element result;
    Goldilocks::exp(result, base, exp);
    return result;
};

inline void Goldilocks::exp(Element &result, Element base, uint64_t exp)
{
    result = Goldilocks::one();

    for (;;)
    {
        if (exp & 1)
            mul(result, result, base);
        exp >>= 1;
        if (!exp)
            break;
        mul(base, base, base);
    }
};
#endif // GOLDILOCKS