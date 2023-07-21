#include "polynomial.hpp"
#include "thread_utils.hpp"
#include "evaluations.hpp"
#include <math.h>
#include "logger.hpp"

using namespace CPlusPlusLogging;

template<typename Engine>
void Polynomial<Engine>::initialize(u_int64_t length, u_int64_t blindLength, bool createBuffer) {
    this->createBuffer = createBuffer;
    u_int64_t totalLength = length + blindLength;

    if(createBuffer) coef = new FrElement[totalLength];

    int nThreads = omp_get_max_threads() / 2;
    ThreadUtils::parset(coef, 0, totalLength * sizeof(FrElement), nThreads);
    this->length = totalLength;
    degree = 0;
}

template<typename Engine>
Polynomial<Engine>::Polynomial(Engine &_E, u_int64_t length, u_int64_t blindLength) : E(_E) {
    this->initialize(length, blindLength);
}

template<typename Engine>
Polynomial<Engine>::Polynomial(Engine &_E, FrElement *reservedBuffer, u_int64_t length, u_int64_t blindLength) : E(_E) {
    this->coef = reservedBuffer;
    this->initialize(length, blindLength, false);
}

template<typename Engine>
Polynomial<Engine> *
Polynomial<Engine>::fromPolynomial(Engine &_E, Polynomial<Engine> &polynomial, u_int64_t blindLength) {
    Polynomial<Engine> *newPol = new Polynomial<Engine>(_E, polynomial.length, blindLength);

    int nThreads = omp_get_max_threads() / 2;
    ThreadUtils::parcpy(newPol->coef, &polynomial.coef[0], polynomial.length * sizeof(FrElement), nThreads);
    newPol->fixDegree();

    return newPol;
}

template<typename Engine>
Polynomial<Engine> *
Polynomial<Engine>::fromPolynomial(Engine &_E, Polynomial<Engine> &polynomial, FrElement *reservedBuffer, u_int64_t blindLength) {
    Polynomial<Engine> *newPol = new Polynomial<Engine>(_E, reservedBuffer, polynomial.length, blindLength);

    int nThreads = omp_get_max_threads() / 2;
    ThreadUtils::parcpy(newPol->coef, &polynomial.coef[0], polynomial.length * sizeof(FrElement), nThreads);
    newPol->fixDegree();

    return newPol;
}

template<typename Engine>
Polynomial<Engine> *
Polynomial<Engine>::fromCoefficients(Engine &_E, FrElement* coefficients, u_int64_t length) {
    Polynomial<Engine> *newPol = new Polynomial<Engine>(_E, length);

    int nThreads = omp_get_max_threads() / 2;
    ThreadUtils::parcpy(newPol->coef, coefficients, length * sizeof(FrElement), nThreads);
    newPol->fixDegree();

    return newPol;
}

template<typename Engine>
Polynomial<Engine> *
Polynomial<Engine>::fromEvaluations(Engine &_E, FFT<typename Engine::Fr> *fft, FrElement *evaluations, u_int64_t length,
                                    u_int64_t blindLength) {
    Polynomial<Engine> *pol = new Polynomial<Engine>(_E, length, blindLength);

    int nThreads = omp_get_max_threads() / 2;
    ThreadUtils::parcpy(pol->coef, evaluations, length * sizeof(FrElement), nThreads);

    fft->ifft(pol->coef, length);

    pol->fixDegree();

    return pol;
}

template<typename Engine>
Polynomial<Engine> *
Polynomial<Engine>::fromEvaluations(Engine &_E, FFT<typename Engine::Fr> *fft, FrElement *evaluations, FrElement *reservedBuffer, u_int64_t length, u_int64_t blindLength) {
    Polynomial<Engine> *pol = new Polynomial<Engine>(_E, reservedBuffer, length, blindLength);

    int nThreads = omp_get_max_threads() / 2;
    ThreadUtils::parcpy(pol->coef, evaluations, length * sizeof(FrElement), nThreads);

    fft->ifft(pol->coef, length);

    pol->fixDegree();

    return pol;
}

template<typename Engine>
Polynomial<Engine>::~Polynomial() {
    if(this->createBuffer) {
        delete[] this->coef;
    }
}

template<typename Engine>
void Polynomial<Engine>::fixDegree() {
    u_int64_t degree;
    for (degree = length - 1; degree != 0 && this->E.fr.isZero(coef[degree]); degree--);
    this->degree = degree;
}

template<typename Engine>
void Polynomial<Engine>::fixDegreeFrom(u_int64_t initial) {
    u_int64_t degree;
    initial = std::max(initial, this->degree);
    initial = std::min(initial, this->length);

    for (degree = initial; degree != 0 && this->E.fr.isZero(coef[degree]); degree--);
    this->degree = degree;
}

template<typename Engine>
bool Polynomial<Engine>::isEqual(const Polynomial<Engine> &other) const {    
    if (degree != other.degree) {
        return false;
    }

    for (u_int64_t i = 0; i <= degree; i++) {
        if (!E.fr.eq(coef[i], other.coef[i])) {
            return false;
        }
    }
    return true;
}

template<typename Engine>
void Polynomial<Engine>::blindCoefficients(FrElement blindingFactors[], u_int32_t length) {
    const u_int32_t polLength = this->length;

    for (u_int32_t i = 0; i < length; i++) {
        coef[polLength - length + i] = E.fr.add(coef[polLength - length + i], blindingFactors[i]);
        coef[i] = E.fr.sub(coef[i], blindingFactors[i]);
    }
    fixDegree();
}

template<typename Engine>
typename Engine::FrElement Polynomial<Engine>::getCoef(u_int64_t index) const {
    if (index > length - 1) {
        return E.fr.zero();
    }
    return coef[index];
}

template<typename Engine>
void Polynomial<Engine>::setCoef(u_int64_t index, FrElement value) {
    if (index > length - 1) {
        throw std::runtime_error("Polynomial::setCoef: invalid index");
    }
    coef[index] = value;
    if (index > degree) {
        degree = index;
    } else if (index == degree && E.fr.isZero(value)) {
        fixDegreeFrom(index - 1);
    }
}

template<typename Engine>
u_int64_t Polynomial<Engine>::getLength() const {
    return length;
}

template<typename Engine>
u_int64_t Polynomial<Engine>::getDegree() const {
    return degree;
}

template<typename Engine>
typename Engine::FrElement Polynomial<Engine>::fastEvaluate(FrElement point) const {
    int nThreads = omp_get_max_threads();

    uint64_t nCoefs = this->degree + 1;
    uint64_t coefsThread = nCoefs / nThreads;
    uint64_t residualCoefs = nCoefs - coefsThread * nThreads;

    FrElement res[nThreads * 4];
    FrElement xN[nThreads * 4];

    xN[0] = E.fr.one();

    #pragma omp parallel for
    for (int i = 0; i < nThreads; i++) {
        res[i*4] = E.fr.zero();

        uint64_t nCoefs = i == (nThreads - 1) ? coefsThread + residualCoefs : coefsThread;
        for (u_int64_t j = nCoefs; j > 0; j--) {
            res[i*4] = E.fr.add(coef[(i * coefsThread) + j - 1], E.fr.mul(res[i*4], point));

            if (i == 0) xN[0] = E.fr.mul(xN[0], point);
        }
    }

    for (int i = 1; i < nThreads; i++) {
        res[0] = E.fr.add(res[0], E.fr.mul(xN[i - 1], res[i*4]));
        xN[i] = E.fr.mul(xN[i - 1], xN[0]);
    }

    return res[0];
}

template <typename Engine>
void Polynomial<Engine>::add(Polynomial<Engine> &polynomial)
{
    FrElement *newCoef = NULL;
    bool resize = polynomial.length > this->length;

    if (resize) {
        newCoef = new FrElement[polynomial.length];
    }

    u_int64_t thisDegree = this->degree;
    u_int64_t polyDegree = polynomial.degree;
    u_int64_t maxDegree = std::max(thisDegree, polyDegree);

#pragma omp parallel for
    for (u_int64_t i = 0; i <= maxDegree; i++)
    {
        FrElement a = i <= thisDegree ? this->coef[i] : E.fr.zero();
        FrElement b = i <= polyDegree ? polynomial.coef[i] : E.fr.zero();
        FrElement sum;
        E.fr.add(sum, a, b);

        if (resize)
        {
            newCoef[i] = sum;
        }
        else
        {
            this->coef[i] = sum;
        }
    }

    if (resize) {
        if(createBuffer) delete[] this->coef;
        this->coef = newCoef;
        this->length = polynomial.length;
    }

    fixDegreeFrom(maxDegree);
}

template<typename Engine>
void Polynomial<Engine>::sub(Polynomial<Engine> &polynomial) {
    FrElement *newCoef = NULL;
    bool resize = polynomial.length > this->length;

    if (resize)
    {
        newCoef = new FrElement[polynomial.length];
    }

    u_int64_t thisDegree = this->degree;
    u_int64_t polyDegree = polynomial.degree;
    u_int64_t maxDegree = std::max(thisDegree, polyDegree);

#pragma omp parallel for
    for (u_int64_t i = 0; i <= maxDegree; i++)
    {
        FrElement a = i <= thisDegree ? this->coef[i] : E.fr.zero();
        FrElement b = i <= polyDegree ? polynomial.coef[i] : E.fr.zero();
        FrElement sum;
        E.fr.sub(sum, a, b);

        if (resize)
        {
            newCoef[i] = sum;
        }
        else
        {
            this->coef[i] = sum;
        }
    }

    if (resize)
    {
        if (createBuffer) delete[] this->coef;
        this->coef = newCoef;
        this->length = polynomial.length;
    }

    fixDegreeFrom(maxDegree);
}

template<typename Engine>
void Polynomial<Engine>::mulScalar(FrElement &value) {
    #pragma omp parallel for
    for (u_int64_t i = 0; i <= this->degree; i++) {
        this->coef[i] = E.fr.mul(this->coef[i], value);
    }
}

template<typename Engine>
void Polynomial<Engine>::addScalar(FrElement &value) {
    FrElement currentValue = (0 == this->length) ? E.fr.zero() : this->coef[0];
    E.fr.add(this->coef[0], currentValue, value);
}

template<typename Engine>
void Polynomial<Engine>::subScalar(FrElement &value) {
    FrElement currentValue = (0 == this->length) ? E.fr.zero() : this->coef[0];
    E.fr.sub(this->coef[0], currentValue, value);
}

// Multiply current polynomial by the polynomial (X - value)
template<typename Engine>
void Polynomial<Engine>::byXSubValue(FrElement &value) {
    bool resize = !E.fr.eq(E.fr.zero(), this->coef[this->length - 1]);

    u_int64_t length = resize ? this->length + 1 : this->length;
    Polynomial<Engine> *pol = new Polynomial<Engine>(E, length);

    // Step 0: Set current coefficients to the new buffer shifted one position
    int nThreads = omp_get_max_threads() / 2;
    ThreadUtils::parcpy(&pol->coef[1], this->coef, (resize ? this->length : this->length - 1) * sizeof(FrElement),
                        nThreads);
    pol->fixDegree();

    // Step 1: multiply each coefficient by (-value)
    FrElement negValue = E.fr.neg(value);

    this->mulScalar(negValue);

    // Step 2: Add current polynomial to destination polynomial
    pol->add(*this);

    // Swap buffers
    if(this->createBuffer) delete[] this->coef;
    this->coef = pol->coef;
    this->length = pol->length;

    fixDegreeFrom(this->degree + 1);
}

// Multiply current polynomial by the polynomial (X - value)
template<typename Engine>
void Polynomial<Engine>::byXNSubValue(int n, FrElement &value) {
    const bool resize = !((this->length - n - 1) >= this->degree);

    u_int64_t length = resize ? this->length + n : this->length;
    Polynomial<Engine> *pol = new Polynomial<Engine>(E, length);

    // Step 0: Set current coefficients to the new buffer shifted one position
    int nThreads = omp_get_max_threads() / 2;
    ThreadUtils::parcpy(&pol->coef[n], this->coef, (this->degree + 1) * sizeof(FrElement), nThreads);
    pol->fixDegree();

    // Step 1: multiply each coefficient by value
    FrElement negValue = E.fr.neg(value);

    this->mulScalar(negValue);

    // Step 2: Add current polynomial to destination polynomial
    pol->add(*this);

    // Swap buffers
    if(this->createBuffer) delete[] this->coef;
    this->coef = pol->coef;
    this->length = pol->length;

    fixDegreeFrom(this->degree + n);
}

template<typename Engine>
void Polynomial<Engine>::divByXSubValue(FrElement &value) {
    Polynomial<Engine> *pol = new Polynomial<Engine>(E, degree + 1);

    pol->coef[degree -1] = this->coef[degree];
    for (u_int64_t i = this->degree - 2; i >= 0; i--)
    {
        pol->coef[i] = E.fr.add(this->coef[i + 1], E.fr.mul(value, pol->coef[i + 1]));
        if(i == 0) break;
    }

    if(!E.fr.eq(this->coef[0], E.fr.mul(E.fr.neg(value), pol->coef[0]))) {
        throw std::runtime_error("Polynomial does not divide");
    }

    if (createBuffer) delete[] this->coef;
    this->coef = pol->coef;

    this->fixDegreeFrom(degree);
}

template<typename Engine>
void Polynomial<Engine>::divZh(u_int64_t domainSize, int extension) {
#pragma omp parallel for
    for (u_int64_t i = 0; i < domainSize; i++) {
        E.fr.neg(this->coef[i], this->coef[i]);
    }

    int nThreads = pow(2, floor(log2(omp_get_max_threads())));
    uint64_t nElementsThread = domainSize / nThreads;

    if(domainSize < (uint64_t)nThreads) {
        nThreads = 1;
        nElementsThread = domainSize;
    }

    assert(domainSize == nElementsThread * nThreads);

    uint64_t nChunks = this->length / domainSize;

    for (uint64_t i = 0; i < nChunks - 1; i++) {
    #pragma omp parallel for
        for (int k = 0; k < nThreads; k++) {
            for (uint64_t j = 0; j < nElementsThread; j++) {
                int id = k;
                u_int64_t idxBase = id * nElementsThread + j;
                u_int64_t idx0 = idxBase + i * domainSize;
                u_int64_t idx1 = idxBase + (i + 1) * domainSize;
                E.fr.sub(coef[idx1], coef[idx0], coef[idx1]);

                if (i > (domainSize * (extension - 1) - extension)) {
                    if (!E.fr.isZero(coef[idx1])) {
                        throw std::runtime_error("Polynomial is not divisible");
                    }
                }
            }
        }
    }

    fixDegreeFrom(this->degree);
}

template<typename Engine>
void Polynomial<Engine>::divByZerofier(u_int64_t n, FrElement beta) {
    FrElement negOne;
    E.fr.neg(negOne, E.fr.one());

    FrElement invBeta;
    E.fr.inv(invBeta, beta);
    FrElement invBetaNeg;
    E.fr.neg(invBetaNeg, invBeta);

    bool isOne = E.fr.eq(E.fr.one(), invBetaNeg);
    bool isNegOne = E.fr.eq(negOne, invBetaNeg);

    if(!isOne) {
        #pragma omp parallel for
        for (u_int64_t i = 0; i < n; i++) {
            // If invBetaNeg === -1 we'll save a multiplication changing it by a neg function call
            if(isNegOne) {
                E.fr.neg(this->coef[i], this->coef[i]);
            } else {
                this->coef[i] = E.fr.mul(invBetaNeg, this->coef[i]);
            }
        }
    }

    int nThreads = pow(2, floor(log2(omp_get_max_threads())));
    nThreads = std::min(n, (u_int64_t)nThreads);

    uint64_t nElementsThread = n / nThreads;
    uint64_t nChunks = (this->length + n - 1) / n;

    isOne = E.fr.eq(E.fr.one(), invBeta);
    isNegOne = E.fr.eq(negOne, invBeta);

    
    u_int64_t threadIters = (n + nThreads*nElementsThread - 1) / (nThreads*nElementsThread);
    for(uint64_t l = 0; l < threadIters; ++l) {
        u_int64_t idxThreads0 = l * nThreads;
        u_int64_t idxThreads1 = std::min((l + 1) * nThreads, n);
        #pragma omp parallel for
        for (u_int64_t k = idxThreads0; k < idxThreads1; k++) {
            for (uint64_t i = 0; i < nChunks - 1; i++) {
                for (uint64_t j = 0; j < nElementsThread; j++) {
                    u_int64_t idxBase = k * nElementsThread + j;
                    u_int64_t idx0 = idxBase + i * n;
                    u_int64_t idx1 = idxBase + (i + 1) * n;

                    if(idx1 > this->degree) {
                        if (idx1 < this->length && !E.fr.isZero(coef[idx1])) {
                            throw std::runtime_error("Polynomial is not divisible");
                        }
                        break;
                    } 

                    FrElement element = E.fr.sub(coef[idx0], coef[idx1]);

                    // If invBeta === 1 we'll not do anything
                    if(!isOne) {
                        // If invBeta === -1 we'll save a multiplication changing it by a neg function call
                        if(isNegOne) {
                            E.fr.neg(element, element);
                        } else {
                            element = E.fr.mul(invBeta, element);
                        }
                    }

                    coef[idx1] = element;

                    // Check if polynomial is divisible by checking if n high coefficients are zero
                    if (idx1 > this->degree - n) {
                        if (!E.fr.isZero(element)) {
                            throw std::runtime_error("Polynomial is not divisible");
                        }
                    }
                }
            }
        }
    }
   

    fixDegreeFrom(this->degree);
}

template <typename Engine>
void Polynomial<Engine>::byX()
{
    int nThreads = omp_get_max_threads() / 2;

    bool resize = !E.fr.isZero(this->coef[this->length - 1]);
    if (resize)
    {
        FrElement *newCoef = new FrElement[this->length + 1];
        ThreadUtils::parcpy(&newCoef[1], &coef[0], length * sizeof(FrElement), nThreads);
        if (createBuffer) delete[] this->coef;
        coef = newCoef;
        this->length++;
    }
    else
    {
        memcpy(&coef[1], &coef[0], (length - 1) * sizeof(FrElement));
    }

    this->degree++;

    coef[0] = E.fr.zero();
}

template<typename Engine>
Polynomial<Engine> *
Polynomial<Engine>::lagrangePolynomialInterpolation(FrElement xArr[], FrElement yArr[], u_int32_t length) {
    Polynomial<Engine> *polynomial = computeLagrangePolynomial(0, xArr, yArr, length);

    for (u_int64_t i = 1; i < length; i++) {
        Polynomial<Engine> *polynomialI = computeLagrangePolynomial(i, xArr, yArr, length);
        polynomial->add(*polynomialI);
    }

    return polynomial;
}

template<typename Engine>
Polynomial<Engine> *
Polynomial<Engine>::computeLagrangePolynomial(u_int64_t i, FrElement xArr[], FrElement yArr[], u_int32_t length) {
    Engine &E = Engine::engine;
    Polynomial<Engine> *polynomial = new Polynomial<Engine>(E, length);

    if(length == 1) {
        polynomial->coef[0] = E.fr.one();
        polynomial->fixDegree();
    }

    bool first = true;
    for (u_int64_t j = 0; j < length; j++) {
        if (j == i) continue;

        if (first) {
            polynomial = new Polynomial<Engine>(E, length);
            polynomial->coef[0] = E.fr.neg(xArr[j]);
            polynomial->coef[1] = E.fr.one();
            polynomial->fixDegree();
            first = false;
        } else {
            polynomial->byXSubValue(xArr[j]);
        }
    }

    FrElement denominator = polynomial->fastEvaluate(xArr[i]);
    E.fr.inv(denominator, denominator);
    FrElement mulFactor = E.fr.mul(yArr[i], denominator);

    polynomial->mulScalar(mulFactor);

    return polynomial;
}

template<typename Engine>
Polynomial<Engine> *Polynomial<Engine>::zerofierPolynomial(FrElement xArr[], u_int32_t length) {
    Engine &E = Engine::engine;
    Polynomial<Engine> *polynomial = new Polynomial<Engine>(E, length + 1);

    if(length == 0) {
        polynomial->coef[0] = E.fr.one();
        polynomial->fixDegree();

        return polynomial;
    }
    // Build a zerofier polynomial with the following form:
    // zerofier(X) = (X-xArr[0])(X-xArr[1])...(X-xArr[n])
    E.fr.neg(polynomial->coef[0], xArr[0]);
    polynomial->coef[1] = E.fr.one();

    polynomial->fixDegree();

    for (u_int64_t i = 1; i < length; i++) {
        polynomial->byXSubValue(xArr[i]);
    }

    return polynomial;
}

template<typename Engine>
void Polynomial<Engine>::print() {
    std::ostringstream res;

    for (u_int64_t i = 0; i < this->length; i++) {
        FrElement c = coef[i];
        // if (!E.fr.eq(E.fr.zero(), c)) {
            res << " ";
            // if (E.fr.neg(c)) {
            //     res << " - ";
            // } else if (i != this->degree) {
            //     res << " + ";
            // }
            res << E.fr.toString(c);
            // if (i > 0) {
            //     if (i > 1) {
            //         res << "x^" << i;
            //     } else {
            //         res << "x";
            //     }
            // }
            res << ", ";
        // }
    }
    std::cout << res.str() << std::endl;
    //LOG_TRACE(res);
}
