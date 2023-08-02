#ifndef EVALUATIONS_HPP
#define EVALUATIONS_HPP

#include "assert.h"
#include <sstream>
#include <gmp.h>
#include "fft.hpp"
#include "polynomial.hpp"

template<typename Engine>
class Evaluations {
    using FrElement = typename Engine::FrElement;

    bool createBuffer;
    u_int64_t length;

    Engine &E;

    void initialize(u_int64_t length, bool createBuffer = true);

public:
    FrElement *eval;

    Evaluations(Engine &_E, u_int64_t length);

    Evaluations(Engine &_E, FrElement *reservedBuffer, u_int64_t length);

    Evaluations(Engine &_E, FFT<typename Engine::Fr> *fft, Polynomial<Engine> &polynomial, u_int32_t extensionLength);

    Evaluations(Engine &_E, FFT<typename Engine::Fr> *fft, FrElement *reservedBuffer, Polynomial<Engine> &polynomial, u_int32_t extensionLength);

    ~Evaluations();

    FrElement getEvaluation(u_int64_t index) const;

    u_int64_t getLength() const;
};

#include "evaluations.c.hpp"

#endif