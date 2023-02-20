#ifndef FFLONK_PROVER_HPP
#define FFLONK_PROVER_HPP

#include <string>
#include <map>
#include "snark_proof.hpp"
#include "binfile_utils.hpp"
#include <gmp.h>
#include "fft.hpp"
#include "zkey_fflonk.hpp"
#include "polynomial/polynomial.hpp"
#include "polynomial/evaluations.hpp"
#include <nlohmann/json.hpp>
#include "mul_z.hpp"
#include "dump.hpp"
#include "keccak_256_transcript.hpp"

using json = nlohmann::json;
using namespace std::chrono;

namespace Fflonk {

    template<typename Engine>
    class FflonkProver {
        using FrElement = typename Engine::FrElement;
        using G1Point = typename Engine::G1Point;
        using G1PointAffine = typename Engine::G1PointAffine;

        Dump::Dump<Engine> *dump;

        struct ProcessingTime {
            std::string label;
            double duration;

            ProcessingTime(std::string label, double duration) : label(label), duration(duration) {}
        };

        std::vector <ProcessingTime> T1;
        std::vector <ProcessingTime> T2;

        Engine &E;
        FFT<typename Engine::Fr> *fft = NULL;
        MulZ<Engine> *mulZ;

        BinFileUtils::BinFile *fdZkey;

        Zkey::FflonkZkeyHeader *zkey;
        u_int32_t zkeyPower;
        std::string curveName;
        size_t sDomain;

        G1PointAffine *PTau;

        FrElement *buffWitness;
        FrElement *buffInternalWitness;

        FrElement *bigBufferBuffers;
        FrElement *bigBufferPolynomials;
        FrElement *bigBufferEvaluations;
        std::map<std::string, FrElement *> bufPtr;
        std::map<std::string, FrElement *> polPtr;
        std::map<std::string, FrElement *> evalPtr;

        std::map<std::string, u_int32_t *> mapBuffers;
        std::map<std::string, FrElement *> buffers;
        std::map<std::string, Polynomial<Engine> *> polynomials;
        std::map<std::string, Evaluations<Engine> *> evaluations;

        std::map <std::string, FrElement> toInverse;
        std::map <std::string, FrElement> challenges;
        std::map<std::string, FrElement *> roots;
        FrElement blindingFactors[10];

        Keccak256Transcript<Engine> *transcript;
        SnarkProof<Engine> *proof;
    public:
        FflonkProver(Engine &E);

        ~FflonkProver();

        std::tuple <json, json> prove(BinFileUtils::BinFile *fdZkey, FrElement *buffWitness);

        void calculateAdditions(BinFileUtils::BinFile *fdZkey);

        FrElement getWitness(u_int64_t idx);

        void round1();

        void round2();

        void round3();

        void round4();

        void round5();

        //ROUND 1 functions
        void computeWirePolynomials();

        void computeWirePolynomial(std::string polName, FrElement blindingFactors[]);

        void computeT0();

        void computeC1();

        //ROUND 2 functions
        void computeZ();

        void computeT1();

        void computeT2();

        void computeC2();

        //ROUND 4 functions
        void computeR0();

        void computeR1();

        void computeR2();

        void computeF();

        void computeZT();

        //ROUND 5 functions
        void computeL();

        void computeZTS2();

        void batchInverse(FrElement *elements, u_int64_t length);

        FrElement *polynomialFromMontgomery(Polynomial<Engine> *polynomial);

        FrElement getMontgomeryBatchedInverse();

        FrElement computeLiS0(u_int32_t i);

        FrElement computeLiS1(u_int32_t i);

        FrElement computeLiS2(u_int32_t i);

        G1Point multiExponentiation(Polynomial<Engine> *polynomial);

        G1Point multiExponentiation(Polynomial<Engine> *polynomial, u_int32_t nx, u_int64_t x[]);

        void printPol(std::string name, const Polynomial<Engine> *polynomial);

        void resetTimer(std::vector <ProcessingTime> &T);
        void takeTime(std::vector <ProcessingTime> &T, const std::string label);

        void printTimer(std::vector <ProcessingTime> &T);
    };
}

#include "fflonk_prover.c.hpp"

#endif
