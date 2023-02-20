#ifndef CPOLYNOMIAL_HPP
#define CPOLYNOMIAL_HPP

#include "polynomial.hpp"


template<typename Engine>
class CPolynomial {
    using FrElement = typename Engine::FrElement;
    using G1Point = typename Engine::G1Point;
    using G1PointAffine = typename Engine::G1PointAffine;

    Polynomial<Engine> **polynomials;
    Engine &E;

    int n;

public:
    CPolynomial(Engine &_E, int n);

    ~CPolynomial();

    void addPolynomial(int position, Polynomial<Engine> * polynomial);

    u_int64_t getDegree() const;

    Polynomial<Engine> * getPolynomial(FrElement *reservedBuffer) const;

    typename Engine::G1Point multiExponentiation(G1PointAffine *PTau) const;
};

#include "cpolynomial.c.hpp"

#endif
