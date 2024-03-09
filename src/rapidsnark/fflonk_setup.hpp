#ifndef FFLONK_SETUP_HPP
#define FFLONK_SETUP_HPP

#include <iostream>
#include <string.h>
#include <binfile_utils.hpp>
#include <binfile_writer.hpp>
#include <nlohmann/json.hpp>
#include "compare_fe_fr.hpp"
#include <sodium.h>
#include "zkey_fflonk.hpp"
#include "polynomial/polynomial.hpp"
#include "ntt_bn128.hpp"
#include <alt_bn128.hpp>
#include "fft.hpp"
#include "utils.hpp"


using json = nlohmann::json;

using namespace std;

namespace Fflonk 
{
    struct CommitmentAndPolynomial {
        AltBn128::Engine::G1PointAffine commitment;
        Polynomial<AltBn128::Engine>* polynomial;
    };
    
    struct FflonkSetupSettings {
        uint64_t nVars;
        uint64_t nPublics;
        uint64_t cirPower;
        uint64_t domainSize;
    };

    class FflonkSetup
    {
        using FrElement = typename AltBn128::Engine::FrElement;
        using G1Point = typename AltBn128::Engine::G1Point;
        using G1PointAffine = typename AltBn128::Engine::G1PointAffine;
        using G2PointAffine = typename AltBn128::Engine::G2PointAffine;

        AltBn128::Engine &E;

        FFT<AltBn128::Engine::Fr> *fft = NULL;
        NTT_AltBn128 *ntt;
        NTT_AltBn128 *nttExtended;

        G1PointAffine *PTau;

        // FflonkZkeyWriter::FflonkZkeyWriter* zkey;
        // FflonkInfo::FflonkInfo* fflonkInfo;

        // void parseShKey(json shKeyJson);
        // void parseFShKey(json shKeyJson);
        // void parsePolsNamesStageShKey(json shKeyJson);
        // void parseOmegasShKey(json shKeyJson);

        // FrElement* constPolsEvals;
        // uint64_t constPolsEvalsSize;
        // FrElement* constPolsEvalsExt;
        // uint64_t constPolsEvalsExtSize;
        // FrElement* constPolsCoefs;
        // uint64_t constPolsCoefsSize;

        // FrElement* x_n;
        // FrElement* x_2ns;

        // FrElement* polynomialFromMontgomery(Polynomial<AltBn128::Engine> *polynomial);

        // G1Point multiExponentiation(Polynomial<AltBn128::Engine> *polynomial, u_int32_t nx, u_int64_t x[]);

        // void computeFCommitments(FflonkZkeyWriter::FflonkZkeyWriter* zkey, uint64_t domainSize);

        // u_int32_t findDegree(FflonkZkeyWriter::FflonkZkeyWriter* zkey, u_int32_t fIndex, std::string name);

        // u_int32_t findPolId(FflonkZkeyWriter::FflonkZkeyWriter* zkey, u_int32_t stage, std::string name);

        // int find(std::string* arr, u_int32_t n, std::string x);

    public:
        FflonkSetup(AltBn128::Engine &_E) : E(_E) {};

        ~FflonkSetup();

        void generateZkey(std::string r1csFilename, std::string pTauFilename, std::string zkeyFilename);
    };
}

#endif
