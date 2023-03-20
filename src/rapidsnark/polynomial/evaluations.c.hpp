#include "evaluations.hpp"
#include "thread_utils.hpp"


template<typename Engine>
void Evaluations<Engine>::initialize(u_int64_t length, bool createBuffer) {
    this->createBuffer = createBuffer;
    if(createBuffer) {
        eval = new FrElement[length];
    }
    int nThreads = omp_get_max_threads() / 2;
    ThreadUtils::parset(eval, 0, length * sizeof(FrElement), nThreads);
    //memset(eval, 0, length * sizeof(FrElement));
    this->length = length;
}

template<typename Engine>
Evaluations<Engine>::Evaluations(Engine &_E, u_int64_t length) : E(_E) {
    this->initialize(length);
}

template<typename Engine>
Evaluations<Engine>::Evaluations(Engine &_E, FrElement *reservedBuffer, u_int64_t length) : E(_E) {
    this->eval = reservedBuffer;
    this->initialize(length, false);
}

//template<typename Engine>
//Evaluations<Engine>::fromEvaluations(Engine &_E, FrElement *evaluations, u_int64_t length) : E(_E) {
//    initialize(length);
//
//    int nThreads = omp_get_max_threads() / 2;
//    ThreadUtils::parcpy(eval,
//                                evaluations,
//                                length * sizeof(FrElement), nThreads);
//
//}

template<typename Engine>
Evaluations<Engine>::Evaluations(Engine &_E, FFT<typename Engine::Fr> *fft, Polynomial<Engine> &polynomial, u_int32_t extensionLength) : E(_E) {
    //Extend polynomial
    initialize(extensionLength);

    int nThreads = omp_get_max_threads() / 2;
    ThreadUtils::parcpy(eval, polynomial.coef, (polynomial.getDegree() + 1) * sizeof(FrElement), nThreads);

    //Coefficients to evaluations
    fft->fft(eval, extensionLength);
}

template<typename Engine>
Evaluations<Engine>::Evaluations(Engine &_E, FFT<typename Engine::Fr> *fft, FrElement *reservedBuffer, Polynomial<Engine> &polynomial, u_int32_t extensionLength) : E(_E) {
    this->eval = reservedBuffer;
    this->initialize(extensionLength, false);

    int nThreads = omp_get_max_threads() / 2;
    ThreadUtils::parcpy(eval, polynomial.coef, (polynomial.getDegree() + 1) * sizeof(FrElement), nThreads);

    fft->fft(eval, extensionLength);
}

template<typename Engine>
Evaluations<Engine>::~Evaluations() {
    if(createBuffer) {
        delete[] this->eval;
    }
}

template<typename Engine>
typename Engine::FrElement Evaluations<Engine>::getEvaluation(u_int64_t index) const {
    if (index > length - 1) {
        throw std::runtime_error("Evaluations::getEvaluation: invalid index");
    }
    return eval[index];
}

template<typename Engine>
u_int64_t Evaluations<Engine>::getLength() const {
    return length;
}