#ifndef PILFFLONK_PROVER_HPP
#define PILFFLONK_PROVER_HPP

#include <iostream>
#include <string.h>
#include <binfile_utils.hpp>
#include <nlohmann/json.hpp>
#include "compare_fe_fr.hpp"
#include <sodium.h>
#include "zkey_pilfflonk.hpp"
#include "shplonk.hpp"
#include "polynomial/polynomial.hpp"
#include "fflonk_info.hpp"
#include "pilfflonk_transcript.hpp"
#include "chelpers/pilfflonk_steps.hpp"
#include "ntt_bn128.hpp"
#include <alt_bn128.hpp>
#include "fft.hpp"
#include "utils.hpp"
#include "witness/main.pilfflonk.hpp"

using json = nlohmann::json;

using namespace std;

namespace PilFflonk
{
    struct BinFilePolsData
    {
        u_int64_t n;
        u_int32_t nPols;
        string *names;
        AltBn128::FrElement *buffer;
    };

    class PilFflonkProver
    {
        using FrElement = typename AltBn128::Engine::FrElement;
        using G1Point = typename AltBn128::Engine::G1Point;
        using G1PointAffine = typename AltBn128::Engine::G1PointAffine;

        AltBn128::Engine &E;
        std::string curveName;

        std::unique_ptr<BinFileUtils::BinFile> zkeyBinFile;
        std::unique_ptr<BinFileUtils::BinFile> precomputedBinFile;

        FrElement *reservedMemoryPtr;
        uint64_t reservedMemorySize;

        NTT_AltBn128 *ntt;
        NTT_AltBn128 *nttExtended;

        PilFflonkZkey::PilFflonkZkey *zkey;

        u_int64_t N;
        u_int64_t NCoefs;
        u_int64_t NExt;

        u_int32_t nBits;
        u_int32_t nBitsCoefs;
        u_int32_t nBitsExt;

        u_int32_t extendBits;

        u_int32_t extendBitsTotal;

        FrElement challenges[5];

        u_int64_t lengthBufferCommitted;
        u_int64_t lengthBufferConstant;
        u_int64_t lengthBufferShPlonk;

        std::map<std::string, u_int64_t> mapBufferCommitted;
        std::map<std::string, u_int64_t> mapBufferConstant;
        std::map<std::string, u_int64_t> mapBufferShPlonk;

        FflonkInfo::FflonkInfo *fflonkInfo;

        ShPlonk::ShPlonkProver *shPlonkProver;

        PilFflonkTranscript *transcript;

        G1PointAffine *PTau;

        PilFflonkSteps pilFflonkSteps;

        FrElement *bBufferCommitted;
        FrElement *bBufferConstant;
        FrElement *bBufferShPlonk;

        std::map<std::string, AltBn128::FrElement *> ptrCommitted;
        std::map<std::string, AltBn128::FrElement *> ptrConstant;
        std::map<std::string, AltBn128::FrElement *> ptrShPlonk;

        std::vector<std::string> nonCommittedPols;

    public:
        PilFflonkProver(AltBn128::Engine &E,
                        std::string zkeyFilename, std::string fflonkInfoFilename,
                        void *reservedMemoryPtr = NULL, uint64_t reservedMemorySize = 0);

        ~PilFflonkProver();

        // Set the configuration data that is required once per prover
        void setConstantData(std::string zkeyFilename, std::string fflonkInfoFilename);

        std::tuple<json, json> prove(std::string committedPolsFilename);

        std::tuple<json, json> prove(std::string execFilename, std::string circomVerifier, std::string zkinFilename); 

        std::tuple<json, json> prove(std::string execFilename, std::string circomVerifier, nlohmann::json &zkin); 
    protected:
        std::tuple<json, json> prove();

        void stage0(PilFflonkStepsParams &params);

        void stage1(PilFflonkStepsParams &params);

        void stage2(PilFflonkStepsParams &params);

        void stage3(PilFflonkStepsParams &params);

        void stage4(PilFflonkStepsParams &params);

        void extend(u_int32_t stage, u_int32_t nPols);

        AltBn128::FrElement *getPolynomial(uint64_t polId, uint64_t offset);

        void calculateZ(u_int64_t numId, u_int64_t denId, u_int64_t zId);

        void batchInverse(u_int64_t denId);

        u_int32_t findNumberOpenings(std::string name, u_int32_t stage);

        void calculateH1H2(AltBn128::FrElement *fPol, AltBn128::FrElement *tPol, uint64_t h1Id, uint64_t h2Id);
    };
}

#endif
