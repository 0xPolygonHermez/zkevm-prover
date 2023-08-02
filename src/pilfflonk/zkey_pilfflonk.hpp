#ifndef ZKEY_PILFFLONK_H
#define ZKEY_PILFFLONK_H

#include <string>
#include <map>
#include <gmp.h>
#include <binfile_utils.hpp>
#include <binfile_writer.hpp>
#include <thread_utils.hpp>
#include <alt_bn128.hpp>

namespace PilFflonkZkey
{
    using FrElement = typename AltBn128::Engine::FrElement;
    using G1Point = typename AltBn128::Engine::G1Point;
    using G1PointAffine = typename AltBn128::Engine::G1PointAffine;

    const int PILFFLONK_PROTOCOL_ID = 12;
    const int ZKEY_HEADER_SECTION = 1;

    const int ZKEY_PF_NSECTIONS = 12;

    const int ZKEY_PF_HEADER_SECTION = 2;
    const int ZKEY_PF_F_SECTION = 3;
    const int ZKEY_PF_F_COMMITMENTS_SECTION = 4;
    const int ZKEY_PF_POLSNAMESSTAGE_SECTION = 5;
    const int ZKEY_PF_CONST_POLS_EVALS_SECTION = 6;
    const int ZKEY_PF_CONST_POLS_COEFS_SECTION = 7;
    const int ZKEY_PF_CONST_POLS_EVALS_EXT_SECTION = 8;
    const int ZKEY_PF_X_N_SECTION = 9;
    const int ZKEY_PF_X_EXT_SECTION = 10;
    const int ZKEY_PF_OMEGAS_SECTION = 11;
    const int ZKEY_PF_PTAU_SECTION = 12;

    struct ShPlonkStagePol
    {
        std::string name;
        u_int64_t degree;
    };

    struct ShPlonkStage
    {
        u_int32_t stage;
        u_int32_t nPols;
        ShPlonkStagePol *pols;
    };

    struct ShPlonkPol
    {
        uint32_t index;
        u_int64_t degree;
        uint32_t nOpeningPoints;
        uint32_t *openingPoints;
        uint32_t nPols;
        std::string *pols;
        uint32_t nStages;
        ShPlonkStage *stages;
    };

    struct ShPlonkCommitment
    {
        std::string name;
        G1PointAffine commit;
        uint64_t lenPol;
        FrElement *pol;
    };

    class PilFflonkZkey
    {
    public:
        u_int32_t n8q;
        mpz_t qPrime;
        u_int32_t n8r;
        mpz_t rPrime;

        u_int32_t power;
        u_int32_t powerW;
        u_int32_t maxQDegree;
        u_int32_t nPublics;

        void *X2;

        std::map<u_int32_t, ShPlonkPol *> f;

        std::map<u_int32_t, std::map<u_int32_t, std::string>*> polsNamesStage;

        std::map<std::string, AltBn128::FrElement> omegas;

        std::map<std::string, ShPlonkCommitment *> fCommitments;

        ~PilFflonkZkey();
    };

    void writePilFflonkZkey(PilFflonkZkey* zkey,
                            FrElement* constPols, uint64_t constPolsSize,
                            FrElement* constPolsExt, uint64_t constPolsExtSize,
                            FrElement* constPolsCoefs, uint64_t constPolsCoefsSize,
                            FrElement*  x_n, uint64_t domainSize,
                            FrElement* x_2ns, uint64_t domainSizeExt,
                            std::string zkeyFilename,
                            G1PointAffine* PTau, uint64_t pTauSize);

    void writeZkeyHeaderSection(BinFileUtils::BinFileWriter* binFile, PilFflonkZkey *pilFflonkZkey);

    void writePilFflonkHeaderSection(BinFileUtils::BinFileWriter* binFile, PilFflonkZkey *pilFflonkZkey);

    void writeFSection(BinFileUtils::BinFileWriter* binFile, PilFflonkZkey *pilFflonkZkey);

    void writeFCommitmentsSection(BinFileUtils::BinFileWriter* binFile, PilFflonkZkey *pilFflonkZkey);

    void writePolsNamesStageSection(BinFileUtils::BinFileWriter* binFile, PilFflonkZkey *pilFflonkZkey);

    void writeConstPolsEvalsSection(BinFileUtils::BinFileWriter* binFile, FrElement* constPols, uint64_t constPolsSize);

    void writeConstPolsCoefsSection(BinFileUtils::BinFileWriter* binFile, FrElement* constPolsCoefs, uint64_t constPolsCoefsSize);

    void writeConstPolsEvalsExtSection(BinFileUtils::BinFileWriter* binFile, FrElement* constPolsExt, uint64_t constPolsExtSize);

    void writeXnSection(BinFileUtils::BinFileWriter* binFile, FrElement* x_n, uint64_t domainSize);

    void writeX2nsSection(BinFileUtils::BinFileWriter* binFile, FrElement* x_2ns, uint64_t domainSizeExt);

    void writeOmegasSection(BinFileUtils::BinFileWriter* binFile, PilFflonkZkey *pilFflonkZkey);

    void writePTauSection(BinFileUtils::BinFileWriter* binFile, G1PointAffine* PTau, uint64_t pTauSize);

    PilFflonkZkey *loadPilFflonkZkey(BinFileUtils::BinFile *fdZKey);

    void readFSection(BinFileUtils::BinFile *fdZKey, PilFflonkZkey *pilFflonkZkey);

    void readFCommitmentsSection(BinFileUtils::BinFile *fdZKey, PilFflonkZkey *pilFflonkZkey);

    void readPolsNamesStageSection(BinFileUtils::BinFile *fdZKey, PilFflonkZkey *pilFflonkZkey);

    void readOmegasSection(BinFileUtils::BinFile *fdZKey, PilFflonkZkey *pilFflonkZkey);

    void readConstPols(BinFileUtils::BinFile *fdZKey, AltBn128::FrElement *evals, AltBn128::FrElement *coefs, AltBn128::FrElement *evalsExt, AltBn128::FrElement *x_n, AltBn128::FrElement *x_2ns);

    void readConstPolsEvalsSection(BinFileUtils::BinFile *fdZKey, AltBn128::FrElement *evals);

    void readConstPolsCoefsSection(BinFileUtils::BinFile *fdZKey, AltBn128::FrElement *coefs);

    void readConstPolsEvalsExtSection(BinFileUtils::BinFile *fdZKey, AltBn128::FrElement *evalsExt);

    void readXnSection(BinFileUtils::BinFile *fdZKey, AltBn128::FrElement *x_n);

    void readX2nsSection(BinFileUtils::BinFile *fdZKey, AltBn128::FrElement *x_2ns);

    void readBuffer(BinFileUtils::BinFile *fdZKey, int idSection, AltBn128::FrElement *ptrDst);
    
    int getProtocolIdFromZkeyPilFflonk(BinFileUtils::BinFile *fd);
}

#endif
