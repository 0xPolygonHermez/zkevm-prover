#ifndef GOLDILOCKS
#define GOLDILOCKS

#include <stdint.h> // uint64_t
#include <cstdio>

class Goldilocks
{
public:
    typedef uint64_t Element;
    typedef uint64_t *pElement;

    static const uint64_t P;
    static const uint64_t Q;
    static const uint64_t MM;
    static const uint64_t CQ;
    static const uint64_t R2;

    // Montgomery Conversion
    static Element gl_tom(Element in1);
    static Element gl_fromm(Element in1);

    static Element add_gl(Element in1, Element in2);

    static Element gl_add_2(Element in1, Element in2);
    static Element gl_add(Element in1, Element in2);
    static void gl_add(Element &res, Element &in1, Element &in2);

    static Element gl_sub(Element in1, Element in2);

    static Element gl_mul(Element a, Element b);

    static Element gl_mmul2(Element in1, Element in2);
    static Element gl_mmul(Element in1, Element in2);

    static inline void print(pElement element)
    {

        printf("%lu\n", *element);
    }
};


inline Goldilocks::Element Goldilocks::add_gl(Element in1, Element in2)
{
    Element res = 0;
    if (__builtin_add_overflow(in1, in2, &res))
    {
        res += CQ;
    }
    return res;
}

inline Goldilocks::Element Goldilocks::gl_mul(Goldilocks::Element a, Goldilocks::Element b)
{
    Element r;
    Element q;
    __asm__(
        "mulq   %3\n\t"
        "divq   %4\n\t"
        : "=a"(r), "=&d"(q)
        : "a"(a), "rm"(b), "rm"(P));
    return q;
}

inline Goldilocks::Element Goldilocks::gl_tom(Goldilocks::Element in1)
{
    Element res;
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

inline Goldilocks::Element Goldilocks::gl_fromm(Goldilocks::Element in1)
{
    Element res;

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

inline Goldilocks::Element Goldilocks::gl_add_2(Goldilocks::Element in1, Goldilocks::Element in2)
{
    Element res;
    __asm__("mov   %1, %0\n\t"
            "add   %2, %0\n\t"
            "jnc  1f\n\t"
            "add   %3, %0\n\t"
            "1: \n\t"
            : "=&a"(res)
            : "r"(in1), "r"(in2), "m"(CQ)
            :);
    return res;
};

inline Goldilocks::Element Goldilocks::gl_add(Goldilocks::Element in1, Goldilocks::Element in2)
{
    Element res;
    __asm__("xor   %%r10, %%r10\n\t"
            "mov   %1, %0\n\t"
            "add   %2, %0\n\t"
            "cmovc %3, %%r10\n\t"
            "add   %%r10, %0\n\t"
            : "=&a"(res)
            : "r"(in1), "r"(in2), "m"(CQ)
            : "%r10");
    return res;
};

inline void Goldilocks::gl_add(Element &res, Element &in1, Element &in2)
{
    res = gl_add(in1, in2);
}

inline Goldilocks::Element Goldilocks::gl_sub(Goldilocks::Element in1, Goldilocks::Element in2)
{
    Element res;
    __asm__("mov   %1, %0\n\t"
            "sub   %2, %0\n\t"
            "jnc  1f\n\t"
            "add   %3, %0\n\t"
            "1: \n\t"
            : "=&a"(res)
            : "r"(in1), "r"(in2), "m"(Q)
            :);
    return res;
}

inline Goldilocks::Element Goldilocks::gl_mmul2(Goldilocks::Element in1, Goldilocks::Element in2)
{
    Element res;
    __asm__("mov   %1, %%rax\n\t"
            "mul   %2\n\t"
            "mov   %%rdx, %%r8\n\t"
            "mov   %%rax, %%r9\n\t"
            "mulq   %3\n\t"
            "mulq   %4\n\t"
            "add    %%r9, %%rax\n\t"
            "adc    %%r8, %%rdx\n\t"
            "jnc  1f\n\t"
            "add   %5, %%rdx\n\t"
            "1:"
            : "=&d"(res)
            : "r"(in1), "r"(in2), "m"(MM), "m"(Q), "m"(CQ)
            : "%rax", "%r8", "%r9");
    return res;
}

inline Goldilocks::Element Goldilocks::gl_mmul(Goldilocks::Element in1, Goldilocks::Element in2)
{
    Element res;
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
            : "=&d"(res)
            : "r"(in1), "r"(in2), "m"(MM), "m"(Q), "m"(CQ)
            : "%rax", "%r8", "%r9", "%r10");
    return res;
}

#endif // GOLDILOCKS
