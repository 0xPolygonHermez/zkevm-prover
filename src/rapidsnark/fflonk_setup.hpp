#ifndef FFLONK_SETUP_HPP
#define FFLONK_SETUP_HPP

#include <iostream>
#include <string.h>
#include "binfile_utils.hpp"
#include "binfile_writer.hpp"
#include <nlohmann/json.hpp>
// #include "compare_fe_fr.hpp"
#include <sodium.h>
#include "zkey_fflonk.hpp"
#include "polynomial/polynomial.hpp"
// #include "ntt_bn128.hpp"
#include <iostream>
#include <fstream>
#include <string>
// #include <gmpxx.h>
#include <cstdio>
#include <sys/stat.h>
#include "../rapidsnark/fflonk_setup.hpp"
#include <alt_bn128.hpp>
#include <fft.hpp>
#include "utils.hpp"
#include <alt_bn128.hpp>
#include "r1cs_binfile.hpp"
#include "r1cs_constraint_processor.hpp"
#include "fflonk_setup_settings.hpp"
#include "polynomial/cpolynomial.c.hpp"

using json = nlohmann::json;

using namespace std;

namespace Fflonk 
{
    struct CommitmentAndPolynomial {
        AltBn128::Engine::G1PointAffine commitment;
        Polynomial<AltBn128::Engine>* polynomial;
    };
    
    class FflonkSetup
    {
        using FrElement = typename AltBn128::Engine::FrElement;
        using G1Point = typename AltBn128::Engine::G1Point;
        using G1PointAffine = typename AltBn128::Engine::G1PointAffine;
        using G2PointAffine = typename AltBn128::Engine::G2PointAffine;

        AltBn128::Engine &E;

        // G1PointAffine *PTau;

        FFT<AltBn128::Engine::Fr> *fft = NULL;
        FflonkSetupSettings settings;

        map<string, Polynomial<AltBn128::Engine>*> polynomials;
        FrElement k1, k2;
        FrElement w3, w4, w8, wr;
        G1PointAffine* PTau;

        std::vector<R1cs::ConstraintCoefficients> plonkConstraints;
        std::vector<R1cs::AdditionCoefficients> plonkAdditions;

        void computeFFConstraints(BinFileUtils::BinFile &r1cs, R1cs::R1csHeader &r1csHeader);

        std::array<std::vector<R1cs::R1csConstraint>, 3> readR1csConstraint(R1cs::R1csHeader &r1csHeader, BinFileUtils::BinFile &r1cs);
        vector<R1cs::R1csConstraint> readR1csConstraintLC(R1cs::R1csHeader &r1csHeader, BinFileUtils::BinFile &r1cs);

        void computeK1K2();
        bool isIncluded(FrElement k, vector<FrElement> &kArr);
        FrElement computeW3();
        FrElement computeW4();
        FrElement computeW8();
        FrElement getOmegaCubicRoot();
        void writeZkeyFile(std::string &zkeyFilename, BinFileUtils::BinFile &ptauFile);
        void writeZkeyHeader(BinFileUtils::BinFileWriter &zkeyFile);
        void writeAdditions(BinFileUtils::BinFileWriter &zkeyFile);
        void writeWitnessMap(BinFileUtils::BinFileWriter &zkeyFile, uint32_t sectionNum, uint32_t posConstraint);
        void writeQMap(BinFileUtils::BinFileWriter &zkeyFile, uint32_t sectionNum, uint32_t posConstraint);
        void writeSigma(BinFileUtils::BinFileWriter &zkeyFile);
        void buildSigma(FrElement *sigma, FrElement w, unordered_map<uint64_t, FrElement> &lastSeen, unordered_map<uint64_t, uint64_t> &firstPos, uint64_t signalId, uint64_t idx);
        void writeLagrangePolynomials(BinFileUtils::BinFileWriter &zkeyFile);
        void writePtau(BinFileUtils::BinFileWriter &zkeyFile, BinFileUtils::BinFile &ptauFile);
        void writeC0(BinFileUtils::BinFileWriter &zkeyFile);
        void writeFflonkHeader(BinFileUtils::BinFileWriter &zkeyFile, BinFileUtils::BinFile &ptauFile);

        FrElement *polynomialFromMontgomery(Polynomial<AltBn128::Engine> *polynomial);
        G1Point multiExponentiation(Polynomial<AltBn128::Engine> *polynomial, u_int32_t nx, u_int64_t x[]);
        void scalar2bytes(mpz_class s, uint8_t (&bytes)[32]);

        void reset();

    public:
        FflonkSetup(AltBn128::Engine &_E) : E(_E) {};

        ~FflonkSetup();

        void generateZkey(std::string r1csFilename, std::string pTauFilename, std::string zkeyFilename);
    };
}

#endif
