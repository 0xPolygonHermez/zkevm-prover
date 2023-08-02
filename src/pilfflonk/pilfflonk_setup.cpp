#include "pilfflonk_setup.hpp"
#include "timer.hpp"
#include <stdio.h>
#include <math.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "thread_utils.hpp"
#include <omp.h>
// /Improve

#include "fft.hpp"

namespace PilFflonk
{
    using FrElement = typename AltBn128::Engine::FrElement;
    using G1Point = typename AltBn128::Engine::G1Point;
    using G1PointAffine = typename AltBn128::Engine::G1PointAffine;

    PilFflonkSetup::~PilFflonkSetup()
    {
        delete zkey;
        delete fflonkInfo;
        delete fft;
        delete ntt;
        delete nttExtended;

        delete[] constPolsCoefs;
        delete[] constPolsEvals;
        delete[] constPolsEvalsExt;
        delete[] PTau;
        delete[] x_n;
        delete[] x_2ns;
    }

    void PilFflonkSetup::generateZkey(std::string shKeyFilename, std::string fflonkInfoFileName,
                                      std::string pTauFilename, std::string cnstPolsFilename,
                                      std::string zkeyFilename)
    {
        // STEP 1. Read shKey JSON file
        zkey = new PilFflonkZkey::PilFflonkZkey();

        zklog.info("PILFFLONK SETUP STARTED");

        zklog.info("> Reading shKey JSON file");
        json shKeyJson;
        file2json(shKeyFilename, shKeyJson);
        parseShKey(shKeyJson);

        // STEP 2. Read fflonkInfo JSON file
        zklog.info("> Reading fflonkInfo JSON file");
        fflonkInfo = new FflonkInfo::FflonkInfo(this->E, fflonkInfoFileName);

        zklog.info("> Opening PTau file");
        auto fdPtau = BinFileUtils::openExisting(pTauFilename, "ptau", 1);

        uint64_t maxFiDegree = 0;
        for (uint32_t i = 0; i < zkey->f.size(); i++)
        {
            maxFiDegree = max(maxFiDegree, zkey->f[i]->degree);
        }
        maxFiDegree++;

        if (fdPtau->getSectionSize(2) < maxFiDegree * sizeof(G1PointAffine))
        {
            throw new runtime_error("Powers of Tau is not big enough for this circuit size. Section 2 too small.");
        }

        PTau = new G1PointAffine[maxFiDegree];

        zklog.info("> Loading PTau data");
        int nThreads = omp_get_max_threads() / 2;
        if ((uint64_t)nThreads > maxFiDegree)
            nThreads = 1;
        ThreadUtils::parset(PTau, 0, maxFiDegree * sizeof(G1PointAffine), nThreads);
        ThreadUtils::parcpy(PTau, fdPtau->getSectionData(2), maxFiDegree * sizeof(G1PointAffine), nThreads);

        size_t sG2 = zkey->n8q * 4;
        if (fdPtau->getSectionSize(3) < sG2)
        {
            throw new runtime_error("Powers of Tau is not well prepared. Section 3 too small.");
        }
        zkey->X2 = new G2PointAffine;
        memcpy(zkey->X2, (G2PointAffine *)fdPtau->getSectionData(3) + 1, sG2);

        auto nBits = zkey->power;
        uint64_t domainSize = 1 << nBits;

        zklog.info("> Loading const polynomials file");
        u_int64_t constPolsBytes = fflonkInfo->nConstants * domainSize * sizeof(FrElement);

        constPolsEvals = new FrElement[fflonkInfo->nConstants * domainSize];

        if (constPolsBytes > 0)
        {
            auto pConstPolsAddress = copyFile(cnstPolsFilename, constPolsBytes);
            zklog.info("PilFflonk::PilFflonk() successfully copied " + to_string(constPolsBytes) + " bytes from constant file " + cnstPolsFilename);

            ThreadUtils::parcpy(constPolsEvals, (FrElement *)pConstPolsAddress, constPolsBytes, omp_get_num_threads() / 2);
        }        

        uint32_t extendBits = ceil(log2(fflonkInfo->qDeg + 1));
        auto nBitsExt = zkey->power + extendBits + fflonkInfo->nBitsZK;

        uint64_t domainSizeExt = 1 << nBitsExt;

        fft = new FFT<AltBn128::Engine::Fr>(domainSizeExt);

        constPolsCoefs = new FrElement[fflonkInfo->nConstants * domainSize];
        constPolsEvalsExt = new FrElement[fflonkInfo->nConstants * domainSizeExt];

        if (fflonkInfo->nConstants > 0)
        {
            ntt = new NTT_AltBn128(E, domainSize);
            nttExtended = new NTT_AltBn128(E, domainSizeExt);

            zklog.info("> Computing const polynomials ifft");
            ntt->INTT(constPolsCoefs, constPolsEvals, domainSize, fflonkInfo->nConstants);

            zklog.info("> Computing F commitments");
            computeFCommitments(zkey, domainSize);

            ThreadUtils::parset(constPolsEvalsExt, 0, domainSizeExt * fflonkInfo->nConstants * sizeof(AltBn128::FrElement), nThreads);
            ThreadUtils::parcpy(constPolsEvalsExt, constPolsCoefs, domainSize * fflonkInfo->nConstants * sizeof(AltBn128::FrElement), nThreads);
               
            zklog.info("> Extending const polynomials fft");
            nttExtended->NTT(constPolsEvalsExt, constPolsEvalsExt, domainSizeExt, fflonkInfo->nConstants);
        }

        // Precalculate x_n and x_2ns
        x_n = new FrElement[domainSize];
        x_2ns = new FrElement[domainSizeExt];

        zklog.info("> Computing roots");
        
#pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i++)
        {
            x_n[i] = fft->root(zkey->power, i);
        }

#pragma omp parallel for
        for (uint64_t i = 0; i < domainSizeExt; i++)
        {
            x_2ns[i] = fft->root(nBitsExt, i);
        }

        zklog.info("> Writing zkey file");

        PilFflonkZkey::writePilFflonkZkey(zkey,
                                          constPolsEvals, fflonkInfo->nConstants * domainSize,
                                          constPolsEvalsExt, fflonkInfo->nConstants * domainSizeExt,
                                          constPolsCoefs, fflonkInfo->nConstants * domainSize,
                                          x_n, domainSize,
                                          x_2ns, domainSizeExt,
                                          zkeyFilename, PTau, maxFiDegree);

        zklog.info("PILFFLONK SETUP FINISHED");
    }

    void PilFflonkSetup::parseShKey(json shKeyJson)
    {
        zkey->power = shKeyJson["power"];
        zkey->powerW = shKeyJson["powerW"];
        zkey->maxQDegree = shKeyJson["maxQDegree"];
        zkey->nPublics = shKeyJson["nPublics"];

        // These values are not in the file but we must initialize them
        zkey->n8q = shKeyJson["n8q"];
        mpz_init(zkey->qPrime);
        std::string primeQStr = shKeyJson["primeQ"];
        mpz_set_str(zkey->qPrime, primeQStr.c_str(), 10);

        zkey->n8r = shKeyJson["n8r"];
        mpz_init(zkey->rPrime);
        std::string primeRStr = shKeyJson["primeR"];
        mpz_set_str(zkey->rPrime, primeRStr.c_str(), 10);

        parseFShKey(shKeyJson);

        parsePolsNamesStageShKey(shKeyJson);

        parseOmegasShKey(shKeyJson);

    }

    void PilFflonkSetup::parseFShKey(json shKeyJson)
    {
        auto f = shKeyJson["f"];

        for (uint32_t i = 0; i < f.size(); i++)
        {
            PilFflonkZkey::ShPlonkPol *shPlonkPol = new PilFflonkZkey::ShPlonkPol();

            auto index = f[i]["index"];
            shPlonkPol->index = index;
            shPlonkPol->degree = f[i]["degree"];

            shPlonkPol->nOpeningPoints = f[i]["openingPoints"].size();
            shPlonkPol->openingPoints = new uint32_t[shPlonkPol->nOpeningPoints];
            for (uint32_t j = 0; j < shPlonkPol->nOpeningPoints; j++)
            {
                shPlonkPol->openingPoints[j] = f[i]["openingPoints"][j];
            }

            shPlonkPol->nPols = f[i]["pols"].size();
            shPlonkPol->pols = new std::string[shPlonkPol->nPols];
            for (uint32_t j = 0; j < shPlonkPol->nPols; j++)
            {
                shPlonkPol->pols[j] = f[i]["pols"][j];
            }
            shPlonkPol->nStages = f[i]["stages"].size();
            shPlonkPol->stages = new PilFflonkZkey::ShPlonkStage[shPlonkPol->nStages];
            for (uint32_t j = 0; j < shPlonkPol->nStages; j++)
            {
                shPlonkPol->stages[j].stage = f[i]["stages"][j]["stage"];
                shPlonkPol->stages[j].nPols = f[i]["stages"][j]["pols"].size();
                shPlonkPol->stages[j].pols = new PilFflonkZkey::ShPlonkStagePol[shPlonkPol->stages[j].nPols];
                for (uint32_t k = 0; k < shPlonkPol->stages[j].nPols; k++)
                {
                    shPlonkPol->stages[j].pols[k].name = f[i]["stages"][j]["pols"][k]["name"];
                    shPlonkPol->stages[j].pols[k].degree = f[i]["stages"][j]["pols"][k]["degree"];
                }
            }

            zkey->f[index] = shPlonkPol;
        }
    }

    void PilFflonkSetup::parsePolsNamesStageShKey(json shKeyJson)
    {
        auto pns = shKeyJson["polsNamesStage"];

        for (auto &el : pns.items())
        {
            std::map<u_int32_t, std::string> *polsNamesStage = new std::map<u_int32_t, std::string>();

            auto value = el.value();

            u_int32_t lenPolsStage = value.size();

            for (u_int32_t j = 0; j < lenPolsStage; ++j)
            {
                (*polsNamesStage)[j] = value[j];
            }

            zkey->polsNamesStage[stoi(el.key())] = polsNamesStage;
        }
    }

    void PilFflonkSetup::parseOmegasShKey(json shKeyJson)
    {
        auto omegas = shKeyJson.items();

        for (auto &el : omegas)
        {
            auto key = el.key();
            if (key.find("w") == 0)
            {
                FrElement omega;
                E.fr.fromString(omega, el.value());
                zkey->omegas[key] = omega;
            }
        }
    }

    void PilFflonkSetup::computeFCommitments(PilFflonkZkey::PilFflonkZkey *zkey, uint64_t domainSize)
    {
        uint32_t stage = 0;
        std::map<std::string, CommitmentAndPolynomial *> polynomialCommitments;
        for (const auto& f : zkey->f) {
            auto fIndex = f.first;
            auto pol = f.second;

            int stagePos = -1;
            for (u_int32_t i = 0; i < pol->nStages; ++i)
            {
                if (pol->stages[i].stage == stage) {
                    stagePos = i;
                    break;
                }
            }
            if (stagePos == -1) continue;

            PilFflonkZkey::ShPlonkStage *stagePol = &pol->stages[stagePos];

            u_int64_t *lengths = new u_int64_t[pol->nPols];
            u_int64_t *polsIds = new u_int64_t[pol->nPols];

            for (u_int32_t j = 0; j < stagePol->nPols; ++j)
            {

                std::string name = stagePol->pols[j].name;
                int index = find(pol->pols, pol->nPols, name);
                if (index == -1)
                {
                    throw std::runtime_error("Polynomial " + std::string(name) + " missing");
                }

                polsIds[j] = findPolId(zkey, stage, name);
                lengths[index] = findDegree(zkey, fIndex, name);
            }

            std::string index = "f" + std::to_string(pol->index);
        
            u_int32_t nPols = pol->nPols;
            u_int32_t polDegree = pol->degree;

            auto polynomial = new Polynomial<AltBn128::Engine>(E, polDegree + 1);

            u_int32_t nPolsStage = fflonkInfo->nConstants;
            
            #pragma omp parallel for
            for (u_int64_t i = 0; i < polDegree; i++) {
                for (u_int32_t j = 0; j < nPols; j++) {
                    if (lengths[j] >= 0 && i < lengths[j]) 
                    {
                            polynomial->coef[i * nPols + j] = constPolsCoefs[polsIds[j] + nPolsStage * i];
                    }
                }
            }

            polynomial->fixDegree();
    
            G1PointAffine commitAffine;
            G1Point commit = multiExponentiation(polynomial, pol->nPols, lengths);
            E.g1.copy(commitAffine, commit);

            polynomialCommitments[index] = new CommitmentAndPolynomial{commitAffine, polynomial};

            delete[] lengths;
            delete[] polsIds;
        }

        for (auto it = polynomialCommitments.begin(); it != polynomialCommitments.end(); ++it)
        {
            auto index = it->first;
            auto commit = it->second->commitment;
            auto pol = it->second->polynomial;

            auto pos = index.find("_");
            if (pos != std::string::npos)
            {
                index = index.substr(0, pos);
            }

            auto shPlonkCommitment = new PilFflonkZkey::ShPlonkCommitment{
                index,
                commit,
                pol->getDegree() + 1,
                pol->coef};
            zkey->fCommitments[index] = shPlonkCommitment;
        }
    }

    FrElement *PilFflonkSetup::polynomialFromMontgomery(Polynomial<AltBn128::Engine> *polynomial)
    {
        const u_int64_t length = polynomial->getLength();

        FrElement *result = new FrElement[length];
        int nThreads = omp_get_max_threads() / 2;
        ThreadUtils::parset(result, 0, length * sizeof(FrElement), nThreads);

#pragma omp parallel for
        for (u_int32_t index = 0; index < length; ++index)
        {
            E.fr.fromMontgomery(result[index], polynomial->coef[index]);
        }

        return result;
    }

    G1Point PilFflonkSetup::multiExponentiation(Polynomial<AltBn128::Engine> *polynomial, u_int32_t nx, u_int64_t x[])
    {
        TimerStart(PILFFLONK_SETUP_CALCULATE_MSM);
        G1Point value;
        FrElement *pol = this->polynomialFromMontgomery(polynomial);
        E.g1.multiMulByScalar(value, PTau, (uint8_t *)pol, sizeof(pol[0]), polynomial->getDegree() + 1, nx, x);
        TimerStopAndLog(PILFFLONK_SETUP_CALCULATE_MSM);
        return value;
    }

    u_int32_t PilFflonkSetup::findPolId(PilFflonkZkey::PilFflonkZkey *zkey, u_int32_t stage, std::string polName)
    {
        for (const auto &[index, name] : *zkey->polsNamesStage[stage])
        {
            if (name == polName)
                return index;
        }
        throw std::runtime_error("Polynomial name not found");
    }

    u_int32_t PilFflonkSetup::findDegree(PilFflonkZkey::PilFflonkZkey *zkey, u_int32_t fIndex, std::string name)
    {
        for (u_int32_t i = 0; i < zkey->f[fIndex]->stages[0].nPols; i++)
        {
            if (zkey->f[fIndex]->stages[0].pols[i].name == name)
            {
                return zkey->f[fIndex]->stages[0].pols[i].degree;
            }
        }
        throw std::runtime_error("Polynomial name not found");
    }

    int PilFflonkSetup::find(std::string *arr, u_int32_t n, std::string x)
    {
        for (u_int32_t i = 0; i < n; ++i)
        {
            if (arr[i] == x)
            {
                return int(i);
            }
        }

        return -1;
    }        
}
