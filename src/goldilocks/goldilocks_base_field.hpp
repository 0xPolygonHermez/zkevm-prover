#ifndef GOLDILOCKS
#define GOLDILOCKS

#include <stdint.h> // uint64_t
#include <string>   // string
#include <gmpxx.h>
#include <iostream> // string
#include "omp.h"

#define USE_MONTGOMERY 0
#define GOLDILOCKS_DEBUG 0
#define GOLDILOCKS_NUM_ROOTS 33
#define GOLDILOCKS_PRIME 0xFFFFFFFF00000001ULL

// TODO: ROOTS of UNITY: https://github.com/hermeznetwork/starkpil/blob/e990d99d0936ec3de751ed927af98fe816d72ede/circuitsGL/fft.circom#L17
class Goldilocks
{
public:
    typedef struct
    {
        uint64_t fe;
    } Element;

private:
    static const Element ZR;
    static const Element Q;
    static const Element MM;
    static const Element CQ;
    static const Element R2;
    static const Element TWO32;

    static const Element ZERO;
    static const Element ONE;
    static const Element NEGONE;
    static const Element SHIFT;
    static const Element W[GOLDILOCKS_NUM_ROOTS];

public:
    static inline uint64_t to_montgomery(const uint64_t &in1);
    static inline uint64_t from_montgomery(const uint64_t &in1);

    static inline const Element &zero() { return ZERO; };
    static inline void zero(Element &result) { result.fe = ZERO.fe; };

    static inline const Element &one() { return ONE; };
    static inline void one(Element &result) { result.fe = ONE.fe; };

    static inline const Element &negone() { return NEGONE; };
    static inline void negone(Element &result) { result.fe = NEGONE.fe; };

    static inline const Element &shift() { return SHIFT; };
    static inline void shift(Element &result) { result.fe = SHIFT.fe; };

    static inline const Element &w(uint64_t i) { return W[i]; };
    static inline void w(Element &result, uint64_t i) { result.fe = W[i].fe; };

    static Element fromU64(uint64_t in1);
    static void fromU64(Element &result, uint64_t in1);

    static Element fromS64(int64_t in1);
    static void fromS64(Element &result, int64_t in1);

    static Element fromS32(int32_t in1);
    static void fromS32(Element &result, int32_t in1);

    static uint64_t toU64(const Element &in1);
    static void toU64(uint64_t &result, const Element &in1);

    static int64_t toS64(const Element &in1);
    static void toS64(int64_t &result, const Element &in1);

    static int32_t toS32(const Element &in1);
    static void toS32(int32_t &result, const Element &in1);

    static std::string toString(const Element &in1, int radix = 10);
    static void toString(std::string &result, const Element &in1, int radix = 10);
    static std::string toString(const Element *in1, const uint64_t size, int radix = 10);

    static Element fromString(const std::string &in1, int radix = 10);
    static void fromString(Element &result, const std::string &in1, int radix = 10);

    static Element fromScalar(const mpz_class &scalar);
    static void fromScalar(Element &result, const mpz_class &scalar);

    static Element add(const Element &in1, const Element &in2);
    static void add(Element &result, const Element &in1, const Element &in2);

    static Element sub(const Element &in1, const Element &in2);
    static void sub(Element &result, const Element &in1, const Element &in2);

    static Element mul(const Element &in1, const Element &in2);
    static void mul(Element &result, const Element &in1, const Element &in2);
    static void mul2(Element &result, const Element &in1, const Element &in2);

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

    static void copy(Element &dst, const Element &src) { dst.fe = src.fe; };
    static void copy(Element *dst, const Element *src) { dst->fe = src->fe; };
    static void parcpy(Element *dst, const Element *src, uint64_t size, int num_threads_copy = 64);

    static void batchInverse(Goldilocks::Element *res, Element *src, uint64_t size);
};

/*
    Operator Overloading
*/
inline Goldilocks::Element operator+(const Goldilocks::Element &in1, const Goldilocks::Element &in2) { return Goldilocks::add(in1, in2); }
inline Goldilocks::Element operator*(const Goldilocks::Element &in1, const Goldilocks::Element &in2) { return Goldilocks::mul(in1, in2); }
inline Goldilocks::Element operator-(const Goldilocks::Element &in1, const Goldilocks::Element &in2) { return Goldilocks::sub(in1, in2); }
inline Goldilocks::Element operator/(const Goldilocks::Element &in1, const Goldilocks::Element &in2) { return Goldilocks::div(in1, in2); }
inline bool operator==(const Goldilocks::Element &in1, const Goldilocks::Element &in2) { return Goldilocks::equal(in1, in2); }
inline Goldilocks::Element operator-(const Goldilocks::Element &in1) { return Goldilocks::neg(in1); }
inline Goldilocks::Element operator+(const Goldilocks::Element &in1) { return in1; }

inline std::string Goldilocks::toString(const Element &in1, int radix)
{
    std::string result;
    Goldilocks::toString(result, in1, radix);
    return result;
}

inline void Goldilocks::toString(std::string &result, const Element &in1, int radix)
{
    mpz_class aux = Goldilocks::toU64(in1);
    result = aux.get_str(radix);
}

inline std::string Goldilocks::toString(const Element *in1, const uint64_t size, int radix)
{
    std::string result = "";
    for (uint64_t i = 0; i < size; i++)
    {
        mpz_class aux = Goldilocks::toU64(in1[i]);
        result += std::to_string(i) + ": " + aux.get_str(radix) + "\n";
    }
    return result;
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
    result = Goldilocks::from_montgomery(in1.fe) % GOLDILOCKS_PRIME;
#else
    result = in1.fe % GOLDILOCKS_PRIME;
#endif
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
    result.fe = Goldilocks::to_montgomery(in1);
#else
    result.fe = in1;
#endif
}

inline Goldilocks::Element Goldilocks::fromS64(int64_t in1)
{
    Goldilocks::Element res;
    Goldilocks::fromS64(res, in1);
    return res;
}

inline void Goldilocks::fromS64(Element &result, int64_t in1)
{
    uint64_t aux;
    (in1 < 0) ? aux = static_cast<uint64_t>(in1) + GOLDILOCKS_PRIME : aux = static_cast<uint64_t>(in1);
#if USE_MONTGOMERY == 1
    result.fe = Goldilocks::to_montgomery(aux);
#else
    result.fe = aux;
#endif
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
    result.fe = Goldilocks::to_montgomery(aux);
#else
    result.fe = aux;
#endif
}

inline int64_t Goldilocks::toS64(const Element &in1)
{
    int64_t res;
    Goldilocks::toS64(res, in1);
    return res;
}

/* Converts a field element into a signed 64bits integer */
inline void Goldilocks::toS64(int64_t &result, const Element &in1)
{
    mpz_class out = Goldilocks::toU64(in1);

    mpz_class maxInt(((uint64_t)GOLDILOCKS_PRIME - 1) / 2);

    if (out > maxInt)
    {
        mpz_class onegative = (uint64_t)GOLDILOCKS_PRIME - out;
        result = -onegative.get_si();
    }
    else
    {
        result = out.get_si();
    }
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
    result.fe = Goldilocks::to_montgomery(aux.get_ui());
#else
    result.fe = aux.get_ui();
#endif
};

inline Goldilocks::Element Goldilocks::fromScalar(const mpz_class &scalar)
{
    Goldilocks::Element result;
    Goldilocks::fromScalar(result, scalar);
    return result;
};

inline void Goldilocks::fromScalar(Element &result, const mpz_class &scalar)
{
    mpz_class aux = (scalar + (uint64_t)GOLDILOCKS_PRIME) % (uint64_t)GOLDILOCKS_PRIME;
#if USE_MONTGOMERY == 1
    result.fe = Goldilocks::to_montgomery(aux.get_ui());
#else
    result.fe = aux.get_ui();
#endif
};

inline Goldilocks::Element Goldilocks::add(const Element &in1, const Element &in2)
{
    Goldilocks::Element result;
    Goldilocks::add(result, in1, in2);
    return result;
}

inline void Goldilocks::add(Element &result, const Element &in1, const Element &in2)
{
    uint64_t in_1 = in1.fe;
    uint64_t in_2 = in2.fe;
    __asm__("xor   %%r10, %%r10\n\t"
            "mov   %1, %0\n\t"
            "add   %2, %0\n\t"
            "cmovc %3, %%r10\n\t"
            "add   %%r10, %0\n\t"
            "jnc  1f\n\t"
            "add   %3, %0\n\t"
            "1: \n\t"
            : "=&a"(result.fe)
            : "r"(in_1), "r"(in_2), "m"(CQ), "m"(ZR)
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
    uint64_t in_1 = in1.fe;
    uint64_t in_2 = in2.fe;
    __asm__("xor   %%r10, %%r10\n\t"
            "mov   %1, %0\n\t"
            "sub   %2, %0\n\t"
            "cmovc %3, %%r10\n\t"
            "sub   %%r10, %0\n\t"
            "jnc  1f\n\t"
            "sub   %3, %0\n\t"
            "1: \n\t"
            : "=&a"(result.fe)
            : "r"(in_1), "r"(in_2), "m"(CQ), "m"(ZR)
            : "%r10");
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
            //"cmovnc %6, %%r10\n\t"
            //"add   %%r10, %0\n\t"
            "jnc  1f\n\t"
            "add   %5, %0\n\t"
            "1: \n\t"
            : "=&d"(result.fe)
            : "r"(in1.fe), "r"(in2.fe), "m"(MM), "m"(Q), "m"(CQ), "m"(ZR)
            : "%rax", "%r8", "%r9", "%r10");

#else
    __asm__("mov   %1, %0\n\t"
            "mul   %2\n\t"
            // "xor   %%rbx, %%rbx\n\t"
            "mov   %%edx, %%ebx\n\t"
            "sub   %4, %%rbx\n\t"
            "rol   $32, %%rdx\n\t"
            //"xor   %%rcx, %%rcx;\n\t"
            "mov   %%edx, %%ecx\n\t"
            "sub   %%rcx, %%rdx\n\t"
            "add   %4, %%rcx\n\t"
            "sub   %%rbx, %%rdx\n\t"
            //"mov   %3,%%r10 \n\t"
            "xor   %%rbx, %%rbx\n\t"
            "add   %%rdx, %0\n\t"
            "cmovc %3, %%rbx\n\t"
            "add   %%rbx, %0\n\t"
            // TODO: migrate to labels
            //"xor   %%rbx, %%rbx\n\t"
            //"sub   %%rcx, %0\n\t"
            //"cmovc %%r10, %%rbx\n\t"
            //"sub   %%rbx, %0\n\t"
            "sub   %%rcx, %0\n\t"
            "jnc  1f\n\t"
            "sub   %3, %0\n\t"
            "1: \n\t"
            : "=&a"(result.fe)
            : "r"(in1.fe), "r"(in2.fe), "m"(CQ), "m"(TWO32)
            : "%rbx", "%rcx", "%rdx");

#endif
#if GOLDILOCKS_DEBUG == 1 && USE_MONTGOMERY == 0
    result.fe = result.fe % GOLDILOCKS_PRIME;
#endif
}

inline void Goldilocks::mul2(Element &result, const Element &in1, const Element &in2)
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
            : "r"(in1.fe), "r"(in2.fe), "m"(MM), "m"(Q), "m"(CQ)
            : "%rax", "%r8", "%r9", "%r10");
#else
    __asm__(
        "mov   %1, %%rax\n\t"
        "mul   %2\n\t"
        "divq   %3\n\t"
        : "=&d"(result.fe)
        : "r"(in1.fe), "r"(in2.fe), "m"(Q)
        : "%rax");
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
        q = Goldilocks::fromU64(r / newr);
        aux1 = Goldilocks::fromU64(t);
        aux2 = Goldilocks::fromU64(newt);
        t = Goldilocks::toU64(aux2);
        newt = Goldilocks::toU64(Goldilocks::sub(aux1, Goldilocks::mul(q, aux2)));
        aux1 = Goldilocks::fromU64(r);
        aux2 = Goldilocks::fromU64(newr);
        r = Goldilocks::toU64(aux2);
        newr = Goldilocks::toU64(Goldilocks::sub(aux1, Goldilocks::mul(q, aux2)));
    }

    Goldilocks::fromU64(result, t);
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
/*
    Private functions (Montgomery)
*/
inline uint64_t Goldilocks::to_montgomery(const uint64_t &in1)
{
    uint64_t res;
    __asm__(
        "xor   %%r10, %%r10\n\t"
        "mov   %1, %%rax\n\t"
        "mulq   %5\n\t"
        "mov   %%rdx, %%r8\n\t"
        "mov   %%rax, %%r9\n\t"
        "mulq   %2\n\t"
        "mulq   %3\n\t"
        "add    %%r9, %%rax\n\t"
        "adc    %%r8, %%rdx\n\t"
        "cmovc %4, %%r10\n\t"
        "add   %%r10, %%rdx\n\t"
        : "=&d"(res)
        : "r"(in1), "m"(MM), "m"(Q), "m"(CQ), "m"(R2)
        : "%rax", "%r8", "%r9", "%r10");
    return res;
}
inline uint64_t Goldilocks::from_montgomery(const uint64_t &in1)
{
    uint64_t res;
    __asm__(
        "xor   %%r10, %%r10\n\t"
        "mov   %1, %%rax\n\t"
        "mov   %%rax, %%r9\n\t"
        "mulq   %2\n\t"
        "mulq   %3\n\t"
        "add    %%r9, %%rax\n\t"
        "adc    %%r10, %%rdx\n\t"
        "cmovc %4, %%r10\n\t"
        "add   %%r10, %%rdx\n\t"
        : "=&d"(res)
        : "r"(in1), "m"(MM), "m"(Q), "m"(CQ)
        : "%rax", "%r8", "%r9", "%r10");
    return res;
}
inline void Goldilocks::parcpy(Element *dst, const Element *src, uint64_t size, int num_threads_copy)
{
    uint64_t dim_total = size * sizeof(Goldilocks::Element);
    uint64_t dim_thread = dim_total / num_threads_copy;
    uint64_t components_thread = size / num_threads_copy;
    uint64_t dim_res = dim_total % num_threads_copy;

#pragma omp parallel num_threads(num_threads_copy) firstprivate(dim_thread)
    {
        int id = omp_get_thread_num();
        uint64_t offset = id * components_thread;
        if (id == num_threads_copy - 1)
        {
            dim_thread += dim_res;
        }
        std::memcpy(&dst[offset], &src[offset], dim_thread);
    }
}

#endif // GOLDILOCKS