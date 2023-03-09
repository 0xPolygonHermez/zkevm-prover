#ifndef MUL_Z_HPP
#define MUL_Z_HPP

#include "assert.h"
#include <sstream>
#include <gmp.h>
#include "fft.hpp"

template<typename Engine>
class MulZ {
    using FrElement = typename Engine::FrElement;

    Engine &E;

    FrElement Z1[4];
    FrElement Z2[4];
    FrElement Z3[4];
public:
    MulZ(Engine &E, FFT<typename Engine::Fr> *fft);

    inline
    tuple<typename Engine::FrElement, typename Engine::FrElement> mul2(const FrElement &a,
                                                                       const FrElement &b,
                                                                       const FrElement &ap,
                                                                       const FrElement &bp,
                                                                       int64_t p) {
        const FrElement a_b = E.fr.mul(a, b);
        const FrElement a_bp = E.fr.mul(a, bp);
        const FrElement ap_b = E.fr.mul(ap, b);
        const FrElement ap_bp = E.fr.mul(ap, bp);

        FrElement r = a_b;

        FrElement a0 = E.fr.add(a_bp, ap_b);
        FrElement a1 = ap_bp;

        FrElement rz = a0;
        if (p >= 0) {
            rz = E.fr.add(rz, E.fr.mul(Z1[p], a1));
        }

        return make_tuple(r, rz);
    }

    inline
    tuple<typename Engine::FrElement, typename Engine::FrElement> mul4(const FrElement &a,
                                                                       const FrElement &b,
                                                                       const FrElement &c,
                                                                       const FrElement &d,
                                                                       const FrElement &ap,
                                                                       const FrElement &bp,
                                                                       const FrElement &cp,
                                                                       const FrElement &dp,
                                                                       int64_t p) {
        const FrElement a_b = E.fr.mul(a, b);
        const FrElement a_bp = E.fr.mul(a, bp);
        const FrElement ap_b = E.fr.mul(ap, b);
        const FrElement ap_bp = E.fr.mul(ap, bp);

        const FrElement c_d = E.fr.mul(c, d);
        const FrElement c_dp = E.fr.mul(c, dp);
        const FrElement cp_d = E.fr.mul(cp, d);
        const FrElement cp_dp = E.fr.mul(cp, dp);

        FrElement r = E.fr.mul(a_b, c_d);

        FrElement a0 = E.fr.mul(ap_b, c_d);
        a0 = E.fr.add(a0, E.fr.mul(a_bp, c_d));
        a0 = E.fr.add(a0, E.fr.mul(a_b, cp_d));
        a0 = E.fr.add(a0, E.fr.mul(a_b, c_dp));

        FrElement a1 = E.fr.mul(ap_bp, c_d);
        a1 = E.fr.add(a1, E.fr.mul(ap_b, cp_d));
        a1 = E.fr.add(a1, E.fr.mul(ap_b, c_dp));
        a1 = E.fr.add(a1, E.fr.mul(a_bp, cp_d));
        a1 = E.fr.add(a1, E.fr.mul(a_bp, c_dp));
        a1 = E.fr.add(a1, E.fr.mul(a_b, cp_dp));

        FrElement a2 = E.fr.mul(a_bp, cp_dp);
        a2 = E.fr.add(a2, E.fr.mul(ap_b, cp_dp));
        a2 = E.fr.add(a2, E.fr.mul(ap_bp, c_dp));
        a2 = E.fr.add(a2, E.fr.mul(ap_bp, cp_d));

        FrElement a3 = E.fr.mul(ap_bp, cp_dp);

        FrElement rz = a0;
        if (p >= 0) {
            rz = E.fr.add(rz, E.fr.mul(Z1[p], a1));
            rz = E.fr.add(rz, E.fr.mul(Z2[p], a2));
            rz = E.fr.add(rz, E.fr.mul(Z3[p], a3));
        }

        return make_tuple(r, rz);
    }
};

#include "mul_z.c.hpp"

#endif
