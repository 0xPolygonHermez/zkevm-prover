#include "fflonk_setup.hpp"
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

#include "r1cs_binfile.hpp"
#include "r1cs_constraint_processor.hpp"
#include "binfile_writer.hpp"

namespace Fflonk
{
    using FrElement = typename AltBn128::Engine::FrElement;
    using G1Point = typename AltBn128::Engine::G1Point;
    using G1PointAffine = typename AltBn128::Engine::G1PointAffine;
    using G2PointAffine = typename AltBn128::Engine::G2PointAffine;

    FflonkSetup::~FflonkSetup()
    {
        // TODO! Check
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

    void FflonkSetup::generateZkey(std::string r1csFilename, std::string pTauFilename, std::string zkeyFilename)
    {
        zklog.info("FFLONK SETUP STARTED");

        // STEP 1. Read PTau file
        zklog.info("> Opening PTau file");
        auto fdPtau = BinFileUtils::openExisting(pTauFilename, "ptau", 1);
        if (!BinFileUtils::sectionExists(12))
        {
            throw new runtime_error("Powers of Tau file is not well prepared. Section 12 missing.");
        }

        // STEP 2. Read r1cs file
        zklog.info("> Opening r1cs file");
        auto fdR1cs = BinFileUtils::openExisting(r1csFilename, "r1cs", 1);

        // Read r1cs header file
        const auto r1csHeader = R1csBinFile::readHeader(fdR1cs);

        const auto sFr = curve.Fr.n8;
        const auto sG1 = curve.G1.F.n8 * 2;
        const auto sG2 = curve.G2.F.n8 * 2;

        FflonkSetupSettings settings:
        settings.nVars = r1csHeader.nVars;
        settings.nPublics = r1csHeader.nOutputs + r1csHeader.nPubInputs;

        //TODO!!!!!!!!
        // let polynomials = {};
        // let evaluations = {};
        // let PTau;

        std::vector<FrElement> plonkConstraints;
        std::vector<FrElement> plonkAdditions;

        // Process constraints inside r1cs
        zklog.info("> Processing FFlonk constraints");
        this->computeFFConstraints(FflonkSetupSettings, fdR1cs, r1csHeader, plonkConstraints, plonkAdditions);

        // As the t polynomial is n+5 whe need at least a power of 4
        //TODO check!!!!
        // NOTE : plonkConstraints + 2 = #constraints + blinding coefficients for each wire polynomial
        const FF_T_POL_DEG_MIN = 3;

        settings.cirPower = std::max(FF_T_POL_DEG_MIN, log2((plonkConstraints.length + 2) - 1) + 1);
        settings.domainSize = 1 << settings.cirPower;

        // TODO!!!!!
        if (fdPtau->getSectionSize(2) < (settings.domainSize * 9 + 18) * sG1)
        {
            throw new runtime_error("Powers of Tau is not big enough for this circuit size. Section 2 too small.");
        }
        if (fdPtau->getSectionSize(3) < sG2)
        {
            throw new runtime_error("Powers of Tau is not well prepared. Section 3 too small.");
        }

        zklog.info("----------------------------");
        zklog.info("  FFLONK SETUP SETTINGS");
        zklog.info("  Curve:         BN128");
        zklog.info("  Circuit power: " + string(settings.cirPower));
        zklog.info("  Domain size:   " + string(settings.domainSize)));
        zklog.info("  Vars:          " + string(settings.nVars));
        zklog.info("  Public vars:   " + string(settings.nPublic));
        zklog.info("  Constraints:   " + string(plonkConstraints.length));
        zklog.info("  Additions:     " + string(plonkAdditions.length));
        zklog.info("----------------------------");

        // Compute k1 and k2 to be used in the permutation checks
        zklog.info("> computing k1 and k2");
        FrElement k1, k2;
        this->computeK1K2(settings, k1, k2);

        // Compute omega 3 (w3) and omega 4 (w4) to be used in the prover and the verifier
        // w3^3 = 1 and  w4^4 = 1
        zklog.info("> computing w3");
        const auto w3 = computeW3();
        zklog.info("> computing w4");
        const auto w4 = computeW4();
        zklog.info("> computing w8");
        const auto w8 = computeW8();
        zklog.info("> computing wr");
        const auto wr = getOmegaCubicRoot(settings.cirPower, curve.Fr);

        // Write output zkey file
        this->writeZkeyFile(zkeyFilename);

        fdR1cs.close();
        fdPTau.close();

        zklog.info("FFLONK SETUP FINISHED");

        return 0;


//////////////////////////////////////////////
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

        FflonkZkeyWriter::writeFflonkZkeyWriter(zkey,
                                          constPolsEvals, fflonkInfo->nConstants * domainSize,
                                          constPolsEvalsExt, fflonkInfo->nConstants * domainSizeExt,
                                          constPolsCoefs, fflonkInfo->nConstants * domainSize,
                                          x_n, domainSize,
                                          x_2ns, domainSizeExt,
                                          zkeyFilename, PTau, maxFiDegree);

        zklog.info("PILFFLONK SETUP FINISHED");
    }

    FrElement FflonkSetup::getFFlonkConstantConstraint(uint64_t index) {
        
    }

    void FflonkSetup::computeFFConstraints(R1csSettings &r1csSettings, Binfile &r1cs, R1csHeader &r1csHeader, std::vector<FrElement> &plonkConstraints, std::vector<FrElement> &plonkAdditions)
    {
        // Add public inputs and outputs
        for (let i = 0; i < r1csSettings.nPublics; i++) {
            plonkConstraints.push_back(R1csConstraintProcessor::getFFlonkConstantConstraint(i + 1, Fr));
        }

        // Create r1cs processor
        const r1csProcessor = new r1csConstraintProcessor();

        if(!r1cs.sectionExists(R1CS_CONSTRAINTS_SECTION)) {
            throw new Error("R1CS file is not well prepared. Section 2 missing.");
        }

        // Start reading r1cs constraints section
        r1cs.startReadSection(R1CS_CONSTRAINTS_SECTION);

        for (let i=0; i< r1csHeader.nConstraints; i++) {
            auto lc = this->readR1csConstraint(r1csHeader, r1cs);
            r1csProcessor.processR1csConstraints(r1csHeader, r1csSettings, lc[0], lc[1], lc[2], plonkConstraints, plonkAdditions);
        }

        r1cs.endReadSection();
    }

    void FflonkSetup::readR1csConstraint(BinFile &r1cs) {
        vector<R1csConstraint> lc[3];
        lc[0] = readConstraintLC(r1csHeader, r1cs);
        lc[1] = readConstraintLC(r1csHeader, r1cs);
        lc[2] = readConstraintLC(r1csHeader, r1cs);

        return lc;
    }

    void FflonkSetup::readR1csConstraintLC(R1csHeader &r1csHeader, BinFile &r1cs) {
        uint32_t n = binfile.readU32LE();
        vector<R1csConstraint> lc(n);

        for (uint32_t i = 0; i < n; i++) {
            R1csContraint rc;
            rc.signal_id = r1cs.readU32LE();
            E.fr.copy(rc.value, r1cs.read(r1csHeader.n8));

            lc.push_back(rc);
        }

        return lc;
    }

    void FflonkSetup::computeK1K2(FflonkSetupSettings &settings, FrElement &k1, FrElement &k2) {
        k1 = FrElement::fromUI(2);
        vector<FrElement> kArr;
        while (isIncluded(k1, kArr, settings)) {
            k1 = E.fr.add(k1, FrElement::one());
        }

        kArr.push_back(k1);
        k2 = E.fr.add(k1, FrElement::one());
        while (isIncluded(k2, kArr, settings)) {
            k2 = E.fr.add(k2, FrElement::one());
        }
    }

    bool FflonkSetup::isIncluded(FrElement k, vector<FrElement> &kArr, FflonkSetupSettings &settings) {
        auto w = FrElement::one();
        for (uint64_t i = 0; i < settings.domainSize; i++) {
            if (E.fr.eq(k, w)) return true;

            for (let j = 0; j < kArr.length; j++) {
                if (E.fr.eq(k, E.fr.mul(kArr[j], w))) return true;
            }
            w = E.fr.mul(w, E.fr.w(settings.cirPower));
        }

        return false;
    }

    FrElement FflonkSetup::computeW3() {
        let generator = E.fr.fromUI(31624);

        // Exponent is order(r - 1) / 3
        mpz_t orderRsub1, exponent;
        mpz_init_set_str(orderRsub1, "3648040478639879203707734290876212514758060733402672390616367364429301415936", 10);
        mpz_divexact_ui(exponent, orderRsub1, 3);

        return E.fr.exp(generator, E.fr.fromMpz(exponent));
    }

    FrElement FflonkSetup::computeW4() {
        return E.fr.w(2);
    }

    FrElement FflonkSetup::computeW8() {
        return E.fr.w(3);
    }

    FrElement FflonkSetup::getOmegaCubicRoot(power, Fr) {
        // Hardcorded 3th-root of Fr.w[28]
        mpz_t firstRoot;
        mpz_init_set_str(firstRoot, "467799165886069610036046866799264026481344299079011762026774533774345988080", 10);

        return E.fr.exp(E.fr.fromMpz(firstRoot), 1 << (28 - power));
    }

    void FflonkSetup::writeZkeyFile(zkeyFilename) {
        zklog.info("> Writing the zkey file");

        BinFileUtils::BinFileWriter* binFile = new BinFileUtils::BinFileWriter(zkeyFilename, "zkey", 1, ZKEY_FF_NSECTIONS);

        zklog.info("··· Writing Section " + string(HEADER_ZKEY_SECTION) + ". Zkey Header");
        this->writeZkeyHeader(fdZKey);

        zklog.info("··· Writing Section " + string(ZKEY_FF_ADDITIONS_SECTION) + ". Additions");
        this->writeAdditions(fdZKey);

        zklog.info("··· Writing Section " + string(ZKEY_FF_A_MAP_SECTION) + ". A Map");
        // await writeWitnessMap(fdZKey, ZKEY_FF_A_MAP_SECTION, 0, "A map");

        zklog.info("··· Writing Section " + string(ZKEY_FF_B_MAP_SECTION) + ". B Map");
        // await writeWitnessMap(fdZKey, ZKEY_FF_B_MAP_SECTION, 1, "B map");

        zklog.info("··· Writing Section " + string(ZKEY_FF_C_MAP_SECTION) + ". C Map");
        // await writeWitnessMap(fdZKey, ZKEY_FF_C_MAP_SECTION, 2, "C map");

        zklog.info("··· Writing Section " + string(ZKEY_FF_QL_SECTION) + ". QL");
        // await writeQMap(fdZKey, ZKEY_FF_QL_SECTION, 3, "QL");

        zklog.info("··· Writing Section " + string(ZKEY_FF_QR_SECTION) + ". QR");
        // await writeQMap(fdZKey, ZKEY_FF_QR_SECTION, 4, "QR");

        zklog.info("··· Writing Section " + string(ZKEY_FF_QM_SECTION) + ". QM");
        // await writeQMap(fdZKey, ZKEY_FF_QM_SECTION, 5, "QM");

        zklog.info("··· Writing Section " + string(ZKEY_FF_QO_SECTION) + ". QO");
        // await writeQMap(fdZKey, ZKEY_FF_QO_SECTION, 6, "QO");

        zklog.info("··· Writing Section " + string(ZKEY_FF_QC_SECTION) + ". QC");
        // await writeQMap(fdZKey, ZKEY_FF_QC_SECTION, 7, "QC");

        zklog.info("··· Writing Sections " + string(ZKEY_FF_SIGMA1_SECTION) + ", " + string(ZKEY_FF_SIGMA2_SECTION) + ", " + string(ZKEY_FF_SIGMA3_SECTION) + ". Sigma1, Sigma2 & Sigma 3");
        // await writeSigma(fdZKey);

        zklog.info("··· Writing Section " + string(ZKEY_FF_LAGRANGE_SECTION) + ". Lagrange Polynomials");
        // await writeLagrangePolynomials(fdZKey);

        zklog.info("··· Writing Section " + string(ZKEY_FF_PTAU_SECTION) + ". Powers of Tau");
        // await writePtau(fdZKey);

        zklog.info("··· Writing Section " + string(ZKEY_FF_C0_SECTION) + ". C0");
        // await writeC0(fdZKey);

        zklog.info("··· Writing Section " + string(ZKEY_FF_HEADER_SECTION) + ". FFlonk Header");
        // await writeFFlonkHeader(fdZKey);

        zklog.info("> Writing the zkey file finished");

        await fdZKey.close();
    }

    void FflonkSetup::writeZkeyHeader(BinFileWriter &zkeyFile) {
        zkeyFile.startWriteSection(HEADER_ZKEY_SECTION);
        zkeyFile.writeU32LE(FFLONK_PROTOCOL_ID);
        zkeyFile.endWriteSection();
    }

    void FflonkSetup::writeAdditions(BinFileWriter &zkeyFile, std::vector<FrElement> &plonkAdditions) {
        zkeyFile.startWriteSection(ZKEY_FF_ADDITIONS_SECTION);

        // Written values are 2 * 32 bit integers (2 * 4 bytes) + 2 field size values ( 2 * sFr bytes)
        // const buffOut = new Uint8Array(8 + 2 * sFr);
        // const buffOutV = new DataView(buffOut.buffer);

        // for (let i = 0; i < plonkAdditions.length; i++) {
        //     const addition = plonkAdditions[i];

        //     buffOutV.setUint32(0, addition[0], true);
        //     buffOutV.setUint32(4, addition[1], true);
        //     buffOut.set(addition[2], 8);
        //     buffOut.set(addition[3], 8 + sFr);

        //     await fdZKey.write(buffOut);
        // }

        await endWriteSection(fdZKey);
    }

    // .................................................................
    // .................................................................
    // .................................................................


    void FflonkSetup::parseShKey(json shKeyJson)
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

    void FflonkSetup::parseFShKey(json shKeyJson)
    {
        auto f = shKeyJson["f"];

        for (uint32_t i = 0; i < f.size(); i++)
        {
            FflonkZkeyWriter::ShPlonkPol *shPlonkPol = new FflonkZkeyWriter::ShPlonkPol();

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
            shPlonkPol->stages = new FflonkZkeyWriter::ShPlonkStage[shPlonkPol->nStages];
            for (uint32_t j = 0; j < shPlonkPol->nStages; j++)
            {
                shPlonkPol->stages[j].stage = f[i]["stages"][j]["stage"];
                shPlonkPol->stages[j].nPols = f[i]["stages"][j]["pols"].size();
                shPlonkPol->stages[j].pols = new FflonkZkeyWriter::ShPlonkStagePol[shPlonkPol->stages[j].nPols];
                for (uint32_t k = 0; k < shPlonkPol->stages[j].nPols; k++)
                {
                    shPlonkPol->stages[j].pols[k].name = f[i]["stages"][j]["pols"][k]["name"];
                    shPlonkPol->stages[j].pols[k].degree = f[i]["stages"][j]["pols"][k]["degree"];
                }
            }

            zkey->f[index] = shPlonkPol;
        }
    }

    void FflonkSetup::parsePolsNamesStageShKey(json shKeyJson)
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

    void FflonkSetup::parseOmegasShKey(json shKeyJson)
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

    void FflonkSetup::computeFCommitments(FflonkZkeyWriter::FflonkZkeyWriter *zkey, uint64_t domainSize)
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

            FflonkZkeyWriter::ShPlonkStage *stagePol = &pol->stages[stagePos];

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

            auto shPlonkCommitment = new FflonkZkeyWriter::ShPlonkCommitment{
                index,
                commit,
                pol->getDegree() + 1,
                pol->coef};
            zkey->fCommitments[index] = shPlonkCommitment;
        }
    }

    FrElement *FflonkSetup::polynomialFromMontgomery(Polynomial<AltBn128::Engine> *polynomial)
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

    G1Point FflonkSetup::multiExponentiation(Polynomial<AltBn128::Engine> *polynomial, u_int32_t nx, u_int64_t x[])
    {
        TimerStart(PILFFLONK_SETUP_CALCULATE_MSM);
        G1Point value;
        FrElement *pol = this->polynomialFromMontgomery(polynomial);
        E.g1.multiMulByScalar(value, PTau, (uint8_t *)pol, sizeof(pol[0]), polynomial->getDegree() + 1, nx, x);
        TimerStopAndLog(PILFFLONK_SETUP_CALCULATE_MSM);
        return value;
    }

    u_int32_t FflonkSetup::findPolId(FflonkZkeyWriter::FflonkZkeyWriter *zkey, u_int32_t stage, std::string polName)
    {
        for (const auto &[index, name] : *zkey->polsNamesStage[stage])
        {
            if (name == polName)
                return index;
        }
        throw std::runtime_error("Polynomial name not found");
    }

    u_int32_t FflonkSetup::findDegree(FflonkZkeyWriter::FflonkZkeyWriter *zkey, u_int32_t fIndex, std::string name)
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

    int FflonkSetup::find(std::string *arr, u_int32_t n, std::string x)
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
