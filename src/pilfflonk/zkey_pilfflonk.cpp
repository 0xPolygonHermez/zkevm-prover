#include <stdexcept>
#include "zkey_pilfflonk.hpp"
#include "zklog.hpp"

using namespace std;
namespace PilFflonkZkey
{
    PilFflonkZkey::~PilFflonkZkey()
    {
        mpz_clear(qPrime);
        mpz_clear(rPrime);

        f.clear();

        for (auto& entry : polsNamesStage) {
            delete entry.second;
        }
        polsNamesStage.clear();

        omegas.clear();

        delete (AltBn128::Engine::G2PointAffine*)X2;
    }

    void writePilFflonkZkey(PilFflonkZkey *zkey,
                            FrElement* constPols, uint64_t constPolsSize,
                            FrElement* constPolsExt, uint64_t constPolsExtSize,
                            FrElement* constPolsCoefs, uint64_t constPolsCoefsSize,
                            FrElement*  x_n, uint64_t domainSize,
                            FrElement* x_2ns, uint64_t domainSizeExt,
                            std::string zkeyFilename, G1PointAffine *PTau, uint64_t pTauSize)
    {
        BinFileUtils::BinFileWriter* binFile = new BinFileUtils::BinFileWriter(zkeyFilename, "zkey", 1, ZKEY_PF_NSECTIONS);

        zklog.info("> Writing PILFFLONK zkey file");
        std::ostringstream ss;
        ss << "··· Writing Section " << ZKEY_HEADER_SECTION << ". Zkey Header";
        zklog.info(ss.str());
        writeZkeyHeaderSection(binFile, zkey);

        ss.str("");
        ss << "··· Writing Section " << ZKEY_PF_HEADER_SECTION << ". Zkey Pil-Fflonk Header";
        zklog.info(ss.str());
        writePilFflonkHeaderSection(binFile, zkey);

        ss.str("");
        ss << "··· Writing Section " << ZKEY_PF_F_SECTION << ". F Section";
        zklog.info(ss.str());
        writeFSection(binFile, zkey);

        ss.str("");
        ss << "··· Writing Section " << ZKEY_PF_F_COMMITMENTS_SECTION << ". F commitments Section";
        zklog.info(ss.str());
        writeFCommitmentsSection(binFile, zkey);

        ss.str("");
        ss << "··· Writing Section " << ZKEY_PF_POLSNAMESSTAGE_SECTION << ". Pols names stage Section";
        zklog.info(ss.str());
        writePolsNamesStageSection(binFile, zkey);

        ss.str("");
        ss << "··· Writing Section " << ZKEY_PF_CONST_POLS_EVALS_SECTION << ". Const Pols Evaluations";
        zklog.info(ss.str());
        writeConstPolsEvalsSection(binFile, constPols, constPolsSize);

        ss.str("");
        ss << "··· Writing Section " << ZKEY_PF_CONST_POLS_COEFS_SECTION << ". Const Pols Coefs";
        zklog.info(ss.str());
        writeConstPolsCoefsSection(binFile, constPolsCoefs, constPolsCoefsSize);

        ss.str("");
        ss << "··· Writing Section " << ZKEY_PF_CONST_POLS_EVALS_EXT_SECTION << ". Const Pols Extended Evaluations";
        zklog.info(ss.str());
        writeConstPolsEvalsExtSection(binFile, constPolsExt, constPolsExtSize);

        ss.str("");
        ss << "··· Writing Section " << ZKEY_PF_X_N_SECTION << ". X_n evaluations";
        zklog.info(ss.str());
        writeXnSection(binFile, x_n, domainSize);

        ss.str("");
        ss << "··· Writing Section " << ZKEY_PF_X_EXT_SECTION << ". X_Ext evaluations";
        zklog.info(ss.str());
        writeX2nsSection(binFile, x_2ns, domainSizeExt);

        ss.str("");
        ss << "··· Writing Section " << ZKEY_PF_OMEGAS_SECTION << ". Omegas Section";
        zklog.info(ss.str());
        writeOmegasSection(binFile, zkey);

        ss.str("");
        ss << "··· Writing Section " << ZKEY_PF_PTAU_SECTION << ". Powers of Tau Section";
        zklog.info(ss.str());
        writePTauSection(binFile, PTau, pTauSize);  

        zklog.info("> Writing PILFFLONK zkey file finished");

        binFile->close();
    }

    void writeZkeyHeaderSection(BinFileUtils::BinFileWriter* binFile, PilFflonkZkey *pilFflonkZkey)
    {
        binFile->startWriteSection(ZKEY_HEADER_SECTION);
        binFile->writeU32LE(PILFFLONK_PROTOCOL_ID);
        binFile->endWriteSection();
    }

    void writePilFflonkHeaderSection(BinFileUtils::BinFileWriter* binFile, PilFflonkZkey *pilFflonkZkey)
    {
        binFile->startWriteSection(ZKEY_PF_HEADER_SECTION);

        binFile->writeU32LE(pilFflonkZkey->n8q);
        uint8_t scalarQ[pilFflonkZkey->n8q];
        memset(scalarQ, 0, pilFflonkZkey->n8q);
        mpz_export((void *)scalarQ, NULL, -1, 8, -1, 0, pilFflonkZkey->qPrime);
        binFile->write(scalarQ, pilFflonkZkey->n8q);

        binFile->writeU32LE(pilFflonkZkey->n8r);
        uint8_t scalarR[pilFflonkZkey->n8r];
        memset(scalarR, 0, pilFflonkZkey->n8r);
        mpz_export((void *)scalarR, NULL, -1, 8, -1, 0, pilFflonkZkey->rPrime);
        binFile->write(scalarR, pilFflonkZkey->n8r);

        binFile->writeU32LE(pilFflonkZkey->power);
        binFile->writeU32LE(pilFflonkZkey->powerW);
        binFile->writeU32LE(pilFflonkZkey->nPublics);
        binFile->writeU32LE(pilFflonkZkey->maxQDegree);

        binFile->write(pilFflonkZkey->X2, pilFflonkZkey->n8q * 4);

        binFile->endWriteSection();
    }

    void writeFSection(BinFileUtils::BinFileWriter* binFile, PilFflonkZkey *pilFflonkZkey)
    {
        binFile->startWriteSection(ZKEY_PF_F_SECTION);

        const u_int32_t lenF = pilFflonkZkey->f.size();
        binFile->writeU32LE(lenF);

        for(uint32_t i = 0; i < lenF; i++)
        {
            ShPlonkPol *fi = pilFflonkZkey->f[i];

            binFile->writeU32LE(fi->index);
            binFile->writeU32LE(fi->degree);

            binFile->writeU32LE(fi->nOpeningPoints);
            for (uint32_t j = 0; j < fi->nOpeningPoints; j++)
            {
                binFile->writeU32LE(fi->openingPoints[j]);
            }

            binFile->writeU32LE(fi->nPols);

            for (uint32_t j = 0; j < fi->nPols; j++)
            {
                binFile->writeString(fi->pols[j]);
            }

            binFile->writeU32LE(fi->nStages);
            for (uint32_t j = 0; j < fi->nStages; j++)
            {
                ShPlonkStage *stage = &(fi->stages[j]);
                binFile->writeU32LE(stage->stage);
                binFile->writeU32LE(stage->nPols);
                for (uint32_t k = 0; k < stage->nPols; k++)
                {
                    ShPlonkStagePol *pol = &(stage->pols[k]);
                    binFile->writeString(pol->name);
                    binFile->writeU32LE(pol->degree);
                }
            }
        }

        binFile->endWriteSection();
    }

    void writeFCommitmentsSection(BinFileUtils::BinFileWriter* binFile, PilFflonkZkey *pilFflonkZkey)
    {
        binFile->startWriteSection(ZKEY_PF_F_COMMITMENTS_SECTION);

        const u_int32_t lenF = pilFflonkZkey->fCommitments.size();
        binFile->writeU32LE(lenF);

        for (auto it = pilFflonkZkey->fCommitments.begin(); it != pilFflonkZkey->fCommitments.end(); ++it)
        {
            auto name = it->second->name;
            auto commit = it->second->commit;
            auto lenPol = it->second->lenPol;
            auto pol = it->second->pol;

            binFile->writeString(name);
            binFile->write(&commit, 64);
            binFile->writeU32LE(lenPol * sizeof(FrElement));
            binFile->write(pol, lenPol * sizeof(FrElement));
        }

        binFile->endWriteSection();
    }

    void writePolsNamesStageSection(BinFileUtils::BinFileWriter* binFile, PilFflonkZkey *pilFflonkZkey)
    {
        binFile->startWriteSection(ZKEY_PF_POLSNAMESSTAGE_SECTION);

        auto len = pilFflonkZkey->polsNamesStage.size();
        
        binFile->writeU32LE(len);

        for (auto const &x : pilFflonkZkey->polsNamesStage)
        {
            binFile->writeU32LE(x.first);

            binFile->writeU32LE(x.second->size());

            for (auto const &y : *(x.second))
            {
                binFile->writeString(y.second);
            }
        }
        binFile->endWriteSection();
    }

    void writeConstPolsEvalsSection(BinFileUtils::BinFileWriter* binFile, FrElement* constPols, uint64_t constPolsSize)
    {
        binFile->startWriteSection(ZKEY_PF_CONST_POLS_EVALS_SECTION);
        binFile->write(constPols, constPolsSize * sizeof(FrElement));
        binFile->endWriteSection();
    }

    void writeConstPolsCoefsSection(BinFileUtils::BinFileWriter* binFile, FrElement* constPolsCoefs, uint64_t constPolsCoefsSize)
    {
        binFile->startWriteSection(ZKEY_PF_CONST_POLS_COEFS_SECTION);
        binFile->write(constPolsCoefs, constPolsCoefsSize * sizeof(FrElement));
        binFile->endWriteSection();
    }

    void writeConstPolsEvalsExtSection(BinFileUtils::BinFileWriter* binFile, FrElement* constPolsExt, uint64_t constPolsExtSize)
    {
        binFile->startWriteSection(ZKEY_PF_CONST_POLS_EVALS_EXT_SECTION);
        binFile->write(constPolsExt, constPolsExtSize * sizeof(FrElement));
        binFile->endWriteSection();
    }

    void writeXnSection(BinFileUtils::BinFileWriter* binFile, FrElement* x_n, uint64_t domainSize)
    {
        binFile->startWriteSection(ZKEY_PF_X_N_SECTION);
        binFile->write(x_n, domainSize * sizeof(FrElement));
        binFile->endWriteSection();
    }

    void writeX2nsSection(BinFileUtils::BinFileWriter* binFile, FrElement* x_2ns, uint64_t domainSizeExt)
    {
        binFile->startWriteSection(ZKEY_PF_X_EXT_SECTION);
        binFile->write(x_2ns, domainSizeExt * sizeof(FrElement));
        binFile->endWriteSection();
    }

    void writeOmegasSection(BinFileUtils::BinFileWriter* binFile, PilFflonkZkey *pilFflonkZkey)
    {
        binFile->startWriteSection(ZKEY_PF_OMEGAS_SECTION);

        auto len = pilFflonkZkey->omegas.size();

        binFile->writeU32LE(len);

        for (auto const &x : pilFflonkZkey->omegas)
        {
            binFile->writeString(x.first);
            binFile->write((void *)&x.second, sizeof(FrElement));
        }
        binFile->endWriteSection();
    }

    void writePTauSection(BinFileUtils::BinFileWriter* binFile, G1PointAffine* PTau, uint64_t pTauSize)
    {
        binFile->startWriteSection(ZKEY_PF_PTAU_SECTION);
        binFile->write(PTau, pTauSize * sizeof(G1PointAffine));
        binFile->endWriteSection();
    }

    PilFflonkZkey *loadPilFflonkZkey(BinFileUtils::BinFile *fdZKey)
    {
        auto pilFflonkZkey = new PilFflonkZkey();

        fdZKey->startReadSection(ZKEY_PF_HEADER_SECTION);  

        pilFflonkZkey->n8q = fdZKey->readU32LE();
        mpz_init(pilFflonkZkey->qPrime);
        mpz_import(pilFflonkZkey->qPrime, pilFflonkZkey->n8q, -1, 1, -1, 0, fdZKey->read(pilFflonkZkey->n8q));

        pilFflonkZkey->n8r = fdZKey->readU32LE();
        mpz_init(pilFflonkZkey->rPrime);
        mpz_import(pilFflonkZkey->rPrime, pilFflonkZkey->n8r, -1, 1, -1, 0, fdZKey->read(pilFflonkZkey->n8r));

        pilFflonkZkey->power = fdZKey->readU32LE();
        pilFflonkZkey->powerW = fdZKey->readU32LE();
        pilFflonkZkey->nPublics = fdZKey->readU32LE();
        pilFflonkZkey->maxQDegree = fdZKey->readU32LE();

        pilFflonkZkey->X2 = fdZKey->read(pilFflonkZkey->n8q * 4);

        fdZKey->endReadSection();

        readFSection(fdZKey, pilFflonkZkey);

        readPolsNamesStageSection(fdZKey, pilFflonkZkey);

        readOmegasSection(fdZKey, pilFflonkZkey);

        return pilFflonkZkey;
    }

    void readFSection(BinFileUtils::BinFile *fdZKey, PilFflonkZkey *pilFflonkZkey)
    {
        fdZKey->startReadSection(ZKEY_PF_F_SECTION);

        u_int32_t lenF = fdZKey->readU32LE();
        for (uint32_t i = 0; i < lenF; i++)
        {
            ShPlonkPol *fi = new ShPlonkPol();
            fi->index = fdZKey->readU32LE();
            fi->degree = fdZKey->readU32LE();

            fi->nOpeningPoints = fdZKey->readU32LE();
            fi->openingPoints = new u_int32_t[fi->nOpeningPoints];
            for (uint32_t j = 0; j < fi->nOpeningPoints; j++)
            {
                fi->openingPoints[j] = fdZKey->readU32LE();
            }

            fi->nPols = fdZKey->readU32LE();

            fi->pols = new string[fi->nPols];
            for (uint32_t j = 0; j < fi->nPols; j++)
            {
                fi->pols[j] = fdZKey->readString();
            }

            fi->nStages = fdZKey->readU32LE();
            fi->stages = new ShPlonkStage[fi->nStages];
            for (uint32_t j = 0; j < fi->nStages; j++)
            {
                ShPlonkStage *stage = &(fi->stages[j]);
                stage->stage = fdZKey->readU32LE();
                stage->nPols = fdZKey->readU32LE();
                stage->pols = new ShPlonkStagePol[stage->nPols];
                for (uint32_t k = 0; k < stage->nPols; k++)
                {
                    ShPlonkStagePol *pol = &(stage->pols[k]);
                    pol->name = fdZKey->readString();
                    pol->degree = fdZKey->readU32LE();
                }
            }

            pilFflonkZkey->f[fi->index] = fi;
        }

        fdZKey->endReadSection();
    }

    void readFCommitmentsSection(BinFileUtils::BinFile *fdZKey, PilFflonkZkey *pilFflonkZkey)
    {
        fdZKey->startReadSection(ZKEY_PF_F_COMMITMENTS_SECTION);

        u_int32_t lenF = fdZKey->readU32LE();
        int nThreads = omp_get_max_threads() / 2;
        for (uint32_t i = 0; i < lenF; i++)
        {
            string name = fdZKey->readString();
            G1PointAffine commit = *((G1PointAffine *)fdZKey->read(pilFflonkZkey->n8q * 2));
            uint64_t lenPolBytes = fdZKey->readU32LE();
            uint64_t lenPol = lenPolBytes * sizeof(FrElement);
            FrElement *pol = new FrElement[lenPol];

            ThreadUtils::parcpy(pol, fdZKey->read(lenPolBytes), lenPolBytes, nThreads);

            pilFflonkZkey->fCommitments[name] = new ShPlonkCommitment{name, commit, lenPol, pol};
        }
        fdZKey->endReadSection();
    }

    void readPolsNamesStageSection(BinFileUtils::BinFile *fdZKey, PilFflonkZkey *pilFflonkZkey)
    {
        fdZKey->startReadSection(ZKEY_PF_POLSNAMESSTAGE_SECTION);

        uint32_t len = fdZKey->readU32LE();

        for (u_int32_t i = 0; i < len; ++i)
        {
            u_int32_t stage = fdZKey->readU32LE();

            pilFflonkZkey->polsNamesStage[stage] = new std::map<u_int32_t, std::string>();

            u_int32_t lenPolsStage = fdZKey->readU32LE();

            for (u_int32_t j = 0; j < lenPolsStage; ++j)
            {
                (*(pilFflonkZkey->polsNamesStage[stage]))[j] = fdZKey->readString();
            }
        }

        fdZKey->endReadSection();
    }

    void readOmegasSection(BinFileUtils::BinFile *fdZKey, PilFflonkZkey *pilFflonkZkey)
    {
        fdZKey->startReadSection(ZKEY_PF_OMEGAS_SECTION);

        uint32_t len = fdZKey->readU32LE();

        for (u_int32_t i = 0; i < len; ++i)
        {
            std::string name = fdZKey->readString();
            AltBn128::FrElement omega = *(AltBn128::FrElement *)(fdZKey->read(pilFflonkZkey->n8q));
            pilFflonkZkey->omegas[name] = omega;
        }

        fdZKey->endReadSection();
    }

    void readConstPols(BinFileUtils::BinFile *fdZKey, AltBn128::FrElement *evals, AltBn128::FrElement *coefs, AltBn128::FrElement *evalsExt, AltBn128::FrElement *x_n, AltBn128::FrElement *x_2ns){
        
        readConstPolsEvalsSection(fdZKey, evals);

        readConstPolsCoefsSection(fdZKey, coefs);

        readConstPolsEvalsExtSection(fdZKey, evalsExt);

        readXnSection(fdZKey, x_n);

        readX2nsSection(fdZKey, x_2ns);
    }

    void readConstPolsEvalsSection(BinFileUtils::BinFile *fdZKey, AltBn128::FrElement *evals)
    {
        readBuffer(fdZKey, ZKEY_PF_CONST_POLS_EVALS_SECTION, evals);
    }

    void readConstPolsCoefsSection(BinFileUtils::BinFile *fdZKey, AltBn128::FrElement *coefs)
    {
        readBuffer(fdZKey, ZKEY_PF_CONST_POLS_COEFS_SECTION, coefs);
    }


    void readConstPolsEvalsExtSection(BinFileUtils::BinFile *fdZKey, AltBn128::FrElement *evalsExt)
    {
        readBuffer(fdZKey, ZKEY_PF_CONST_POLS_EVALS_EXT_SECTION, evalsExt);
    }


    void readXnSection(BinFileUtils::BinFile *fdZKey, AltBn128::FrElement *x_n)
    {
        readBuffer(fdZKey, ZKEY_PF_X_N_SECTION, x_n);
    }


    void readX2nsSection(BinFileUtils::BinFile *fdZKey, AltBn128::FrElement *x_2ns)
    {
        readBuffer(fdZKey, ZKEY_PF_X_EXT_SECTION, x_2ns);
    }

    void readBuffer(BinFileUtils::BinFile *fdZKey, int idSection, AltBn128::FrElement *ptrDst)
    {
        uint64_t size = fdZKey->getSectionSize(idSection);

        AltBn128::FrElement *buffer = ptrDst;

        ThreadUtils::parcpy(&buffer[0], (FrElement *)fdZKey->getSectionData(idSection), size, omp_get_num_threads() / 2);
    }

    int getProtocolIdFromZkeyPilFflonk(BinFileUtils::BinFile *fd) {
        fd->startReadSection(ZKEY_HEADER_SECTION);
        uint32_t protocolId = fd->readU32LE();
        fd->endReadSection();

        return protocolId;
    }
}