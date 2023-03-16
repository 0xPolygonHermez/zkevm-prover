#ifndef POLYNOMIAL_HPP
#define POLYNOMIAL_HPP

#include "assert.h"
#include <sstream>
#include <gmp.h>
#include "fft.hpp"


template<typename Engine>
class Polynomial {
    using FrElement = typename Engine::FrElement;
    using G1Point = typename Engine::G1Point;

    bool createBuffer;
    u_int64_t length;
    u_int64_t degree;

    Engine &E;

    void initialize(u_int64_t length, u_int64_t blindLength = 0, bool createBuffer = true);

    static Polynomial<Engine>* computeLagrangePolynomial(u_int64_t i, FrElement xArr[], FrElement yArr[], u_int32_t length);
public:
    FrElement *coef;

    Polynomial(Engine &_E, u_int64_t length, u_int64_t blindLength = 0);

    Polynomial(Engine &_E, FrElement *reservedBuffer, u_int64_t length, u_int64_t blindLength = 0);

    // From coefficients
    static Polynomial<Engine>* fromPolynomial(Engine &_E, Polynomial<Engine> &polynomial, u_int64_t blindLength = 0);

    static Polynomial<Engine>* fromPolynomial(Engine &_E, Polynomial<Engine> &polynomial, FrElement *reservedBuffer, u_int64_t blindLength = 0);

    // From evaluations
    static Polynomial<Engine>* fromEvaluations(Engine &_E, FFT<typename Engine::Fr> *fft, FrElement *evaluations, u_int64_t length, u_int64_t blindLength = 0);

    static Polynomial<Engine>* fromEvaluations(Engine &_E, FFT<typename Engine::Fr> *fft, FrElement *evaluations, FrElement *reservedBuffer, u_int64_t length, u_int64_t blindLength = 0);

    ~Polynomial();

    void fixDegree();

    void fixDegreeFrom(uint64_t initial);

    bool isEqual(const Polynomial<Engine> &other) const;

    void blindCoefficients(FrElement blindingFactors[], u_int32_t length);

    typename Engine::FrElement getCoef(u_int64_t index) const;

    void setCoef(u_int64_t index, FrElement value);

    u_int64_t getLength() const;

    u_int64_t getDegree() const;

    inline typename Engine::FrElement evaluate(FrElement point) const {
        FrElement result = E.fr.zero();

        for (u_int64_t i = degree + 1; i > 0; i--) {
            result = E.fr.add(coef[i - 1], E.fr.mul(result, point));
        }
        return result;
    }

    typename Engine::FrElement fastEvaluate(FrElement point) const;

    void add(Polynomial<Engine> &polynomial);

    void sub(Polynomial<Engine> &polynomial);

    void mulScalar(FrElement &value);

    void addScalar(FrElement &value);

    void subScalar(FrElement &value);

    // Multiply current polynomial by the polynomial (X - value)
    void byXSubValue(FrElement &value);

    void byXNSubValue(int n, FrElement &value);

    // Euclidean division, returns reminder polygon
    Polynomial<Engine>* divBy(Polynomial<Engine> &polynomial);

    void divByMonic(uint32_t m, FrElement beta);

    Polynomial<Engine>* divByVanishing(uint32_t m, FrElement beta);

    Polynomial<Engine>* divByVanishing(FrElement *reservedBuffer, uint64_t m, FrElement beta);

    void fastDivByVanishing(FrElement *reservedBuffer, uint32_t m, FrElement beta);

    void divZh(u_int64_t domainSize, int extension = 4);

    void divByZerofier(u_int64_t n, FrElement beta);

    void byX();

    static Polynomial<Engine>* lagrangePolynomialInterpolation(FrElement xArr[], FrElement yArr[], u_int32_t length);

    static Polynomial<Engine>* zerofierPolynomial(FrElement xArr[], u_int32_t length);

    void print();
};

#include "polynomial.c.hpp"

#endif
