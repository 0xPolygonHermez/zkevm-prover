#include "pilfflonk_prover.hpp"
#include "timer.hpp"
#include <stdio.h>
#include <math.h>

namespace PilFflonk
{
    PilFflonkProver::PilFflonkProver(AltBn128::Engine &_E,
                                     std::string zkeyFilename, std::string fflonkInfoFilename,
                                     void *reservedMemoryPtr, uint64_t reservedMemorySize) : E(_E)
    {
        this->reservedMemoryPtr = (FrElement *)reservedMemoryPtr;
        this->reservedMemorySize = reservedMemorySize;

        curveName = "bn128";

        setConstantData(zkeyFilename, fflonkInfoFilename);
    }

    PilFflonkProver::~PilFflonkProver()
    {
        delete[] bBufferConstant;
        ptrConstant.clear();

        if (NULL == reservedMemoryPtr) delete[] bBufferCommitted;
        ptrCommitted.clear();

        delete[] bBufferShPlonk;
        ptrShPlonk.clear();

        mapBufferCommitted.clear();
        mapBufferConstant.clear();
        mapBufferShPlonk.clear();

        nonCommittedPols.clear();

        delete ntt;
        delete nttExtended;
        delete zkey;
        delete transcript;
        delete shPlonkProver;
        delete fflonkInfo;
    }

    void PilFflonkProver::setConstantData(std::string zkeyFilename, std::string fflonkInfoFile)
    {
        try
        {
            TimerStart(LOAD_ZKEY_TO_MEMORY);

            zklog.info("> Opening zkey data file");
            zkeyBinFile = BinFileUtils::openExisting(zkeyFilename, "zkey", 1);
            auto fdZkey = zkeyBinFile.get();

            if (PilFflonkZkey::getProtocolIdFromZkeyPilFflonk(fdZkey) != PilFflonkZkey::PILFFLONK_PROTOCOL_ID)
            {
                throw std::invalid_argument("zkey file is not pilfflonk");
            }

            zkey = PilFflonkZkey::loadPilFflonkZkey(fdZkey);

            fflonkInfo = new FflonkInfo::FflonkInfo(E, fflonkInfoFile);

            shPlonkProver = new ShPlonk::ShPlonkProver(AltBn128::Engine::engine, zkey);

            extendBits = ceil(log2(fflonkInfo->qDeg + 1));

            nBits = zkey->power;
            nBitsCoefs = nBits + fflonkInfo->nBitsZK;
            nBitsExt = nBits + fflonkInfo->nBitsZK + extendBits;

            extendBitsTotal = nBitsExt - nBits;

            N = 1 << nBits;
            NCoefs = 1 << nBitsCoefs;
            NExt = 1 << nBitsExt;

            ntt = new NTT_AltBn128(E, N);
            nttExtended = new NTT_AltBn128(E, NExt);

            transcript = new PilFflonkTranscript(E);

            mpz_t altBbn128r;
            mpz_init(altBbn128r);
            mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);

            if (mpz_cmp(zkey->rPrime, altBbn128r) != 0)
            {
                throw std::invalid_argument("zkey curve not supported");
            }

            // //////////////////////////////////////////////////
            // CONSTANT BIG BUFFER
            // //////////////////////////////////////////////////
            u_int64_t maxFiDegree = 0;
            for (auto const &[key, f] : zkey->f) {
                maxFiDegree = max(maxFiDegree, f->degree);
            }
            maxFiDegree += 1;

            // Polynomial evaluations
            mapBufferConstant["const_n"] = N * fflonkInfo->nConstants;
            mapBufferConstant["const_2ns"] = NExt * fflonkInfo->nConstants;
            mapBufferConstant["const_coefs"] = N * fflonkInfo->nConstants;
            mapBufferConstant["x_n"] = N;
            mapBufferConstant["x_2ns"] = NExt;
            mapBufferConstant["PTau"] = maxFiDegree * sizeof(G1PointAffine) / sizeof(FrElement);

            lengthBufferConstant = 0;
            for (auto const &[key, value] : mapBufferConstant) {
                lengthBufferConstant += value;
            }

            zklog.info("lengthBufferConstant: " + std::to_string(lengthBufferConstant));

            bBufferConstant = new FrElement[lengthBufferConstant];

            u_int64_t offset = 0;
            for(auto const &[key, value] : mapBufferConstant) {
                ptrConstant[key] = bBufferConstant + offset;
                offset += value;
            }

            // //////////////////////////////////////////////////
            // BIG BUFFER
            // //////////////////////////////////////////////////
            mapBufferCommitted["cm1_n"] = N * fflonkInfo->mapSectionsN.section[FflonkInfo::cm1_n];
            mapBufferCommitted["cm2_n"] = N * fflonkInfo->mapSectionsN.section[FflonkInfo::cm2_n];
            mapBufferCommitted["cm3_n"] = N * fflonkInfo->mapSectionsN.section[FflonkInfo::cm3_n];
            mapBufferCommitted["tmpExp_n"] = N * fflonkInfo->mapSectionsN.section[FflonkInfo::tmpExp_n];

            mapBufferCommitted["cm1_2ns"] = NExt * fflonkInfo->mapSectionsN.section[FflonkInfo::cm1_2ns];
            mapBufferCommitted["cm2_2ns"] = NExt * fflonkInfo->mapSectionsN.section[FflonkInfo::cm2_2ns];
            mapBufferCommitted["cm3_2ns"] = NExt * fflonkInfo->mapSectionsN.section[FflonkInfo::cm3_2ns];
            mapBufferCommitted["q_2ns"] = NExt * fflonkInfo->qDim;
            mapBufferCommitted["publics"] = fflonkInfo->nPublics;
            mapBufferCommitted["cm1_coefs"] = NCoefs * fflonkInfo->mapSectionsN.section[FflonkInfo::cm1_n];
            mapBufferCommitted["cm2_coefs"] = NCoefs * fflonkInfo->mapSectionsN.section[FflonkInfo::cm2_n];
            mapBufferCommitted["cm3_coefs"] = NCoefs * fflonkInfo->mapSectionsN.section[FflonkInfo::cm3_n];

            lengthBufferCommitted = 0;
            for (auto const &[key, value] : mapBufferCommitted) {
                lengthBufferCommitted += value;
            }

            zklog.info("lengthBufferCommitted: " + std::to_string(lengthBufferCommitted));

            if(reservedMemoryPtr == NULL) {
                bBufferCommitted = new FrElement[lengthBufferCommitted];
            } else {
                uint64_t requiredMemorySize = lengthBufferCommitted * sizeof(AltBn128::FrElement);
                if(reservedMemorySize < requiredMemorySize) {
                    uint64_t additionalBytes = requiredMemorySize - reservedMemorySize;
                    string errorMsg = "Insufficient reserved memory size. Additional " + std::to_string(additionalBytes) + " bytes required. Total required " + std::to_string(requiredMemorySize) + " bytes.";
                    zklog.error(errorMsg);

                    throw std::runtime_error(errorMsg);
                }
                bBufferCommitted = reservedMemoryPtr;
            }

            offset = 0;
            for(auto const &[key, value] : mapBufferCommitted) {
                ptrCommitted[key] = bBufferCommitted + offset;
                offset += value;
            }

            // //////////////////////////////////////////////////
            // SHPLONK BIG BUFFER
            // //////////////////////////////////////////////////
            u_int64_t maxDegree = 0;  
            for(u_int32_t i = 0; i < zkey->f.size(); ++i) {
                mapBufferShPlonk["f" + std::to_string(zkey->f[i]->index)] = zkey->f[i]->degree + 1;

                maxDegree = std::max(maxDegree, zkey->f[i]->degree);
            }

            u_int64_t lengthW = maxDegree + 1;
            mapBufferShPlonk["W"] = lengthW;
            mapBufferShPlonk["Wp"] = lengthW;
            
            // Add tmp buffer
            u_int64_t maxNPols = max(fflonkInfo->mapSectionsN.section[FflonkInfo::cm1_n], max(fflonkInfo->mapSectionsN.section[FflonkInfo::cm2_n], fflonkInfo->mapSectionsN.section[FflonkInfo::cm3_n]));
            u_int64_t tmpLength = max(NExt * maxNPols, lengthW);
            mapBufferShPlonk["tmp"] = tmpLength;

            lengthBufferShPlonk = 0;
            for (auto const &[key, value] : mapBufferShPlonk) {
                lengthBufferShPlonk += value;
            }

            zklog.info("lengthBufferShPlonk: " + std::to_string(lengthBufferShPlonk));

            bBufferShPlonk = new FrElement[lengthBufferShPlonk];

            offset = 0;
            for(auto const &[key, value] : mapBufferShPlonk) {
                ptrShPlonk[key] = bBufferShPlonk + offset;
                offset += value;
            }

            TimerStart(LOAD_PTAU_TO_MEMORY);
            int nThreads = omp_get_max_threads() / 2;

            u_int32_t PTauBytes = mapBufferConstant["PTau"] * sizeof(FrElement);

            ThreadUtils::parset(ptrConstant["PTau"], 0, PTauBytes, nThreads);

            ThreadUtils::parcpy(ptrConstant["PTau"],
                                (G1PointAffine *)(fdZkey->getSectionData(PilFflonkZkey::ZKEY_PF_PTAU_SECTION)),
                                PTauBytes, nThreads);

            TimerStopAndLog(LOAD_PTAU_TO_MEMORY);
        
            TimerStart(LOAD_CONST_POLS_ZKEY_TO_MEMORY);

            PilFflonkZkey::readConstPols(fdZkey, ptrConstant["const_n"], ptrConstant["const_coefs"], ptrConstant["const_2ns"], ptrConstant["x_n"], ptrConstant["x_2ns"]);

            TimerStopAndLog(LOAD_CONST_POLS_ZKEY_TO_MEMORY);

            TimerStart(LOAD_F_COMMITMENTS_ZKEY_TO_MEMORY);
            
            fdZkey->startReadSection(PilFflonkZkey::ZKEY_PF_F_COMMITMENTS_SECTION);
            u_int32_t len = fdZkey->readU32LE();

            for (u_int32_t i = 0; i < len; i++)
            {
                std::string name = fdZkey->readString();
                void *C = fdZkey->read(zkey->n8q * 2);
                
                G1Point commit;
                E.g1.copy(commit, *((G1PointAffine *)C));

                shPlonkProver->addPolynomialCommitment(name, commit);
                u_int32_t lenPol = fdZkey->readU32LE();

                ThreadUtils::parcpy(ptrShPlonk[name], (FrElement *)fdZkey->read(lenPol), lenPol, omp_get_num_threads() / 2);

                Polynomial<AltBn128::Engine> *polFi = new Polynomial<AltBn128::Engine>(E, ptrShPlonk[name], lenPol / zkey->n8q, 0, false);

                shPlonkProver->addPolynomialShPlonk(name, polFi);
            }

            fdZkey->endReadSection();

            TimerStopAndLog(LOAD_F_COMMITMENTS_ZKEY_TO_MEMORY);

            TimerStopAndLog(LOAD_ZKEY_TO_MEMORY);
        }

        catch (const std::exception &e)
        {
            std::cerr << "EXCEPTION: " << e.what() << "\n";
            exit(EXIT_FAILURE);
        }
    }

    std::tuple<json, json> PilFflonkProver::prove(std::string committedPolsFilename) {
        TimerStart(LOAD_COMMITTED_POLS_TO_MEMORY);

        u_int64_t cmtdPolsSize = fflonkInfo->mapSectionsN.section[FflonkInfo::cm1_n] * sizeof(FrElement) * N;

        auto pCommittedPolsAddress = copyFile(committedPolsFilename, cmtdPolsSize);
        zklog.info("PilFflonk::PilFflonk() successfully copied " + to_string(cmtdPolsSize) + " bytes from constant file " + committedPolsFilename);

        ThreadUtils::parcpy(ptrCommitted["cm1_n"], (FrElement *)pCommittedPolsAddress, cmtdPolsSize, omp_get_num_threads() / 2);

        TimerStopAndLog(LOAD_COMMITTED_POLS_TO_MEMORY);

        return this->prove();
    }

    std::tuple<json, json> PilFflonkProver::prove(std::string execFilename, std::string circomVerifier, std::string zkinFilename) {
        
        CircomPilFflonk::getCommittedPols(E, ptrCommitted["cm1_n"], circomVerifier, execFilename, zkinFilename, fflonkInfo->mapSectionsN.section[FflonkInfo::cm1_n], N);

        return this->prove();
    }

    std::tuple<json, json> PilFflonkProver::prove(std::string execFilename, std::string circomVerifier, nlohmann::json &zkin) {
       
        CircomPilFflonk::getCommittedPols(E, ptrCommitted["cm1_n"], circomVerifier, execFilename, zkin, fflonkInfo->mapSectionsN.section[FflonkInfo::cm1_n], N);

        return this->prove();
    }

    std::tuple<json, json> PilFflonkProver::prove()
    {
        if (NULL == zkey)
            throw std::runtime_error("Zkey data not set");

        try
        {
            zklog.info("");
            zklog.info("PIL-FFLONK PROVER STARTED");

            TimerStart(PIL_FFLONK_PROVE);

            AltBn128::FrElement* constValues = new AltBn128::FrElement[pilFflonkSteps.getNumConstValues()];

            // Initialize vars
            PilFflonkStepsParams params = {
                cm1_n : ptrCommitted["cm1_n"],
                cm2_n : ptrCommitted["cm2_n"],
                cm3_n : ptrCommitted["cm3_n"],
                tmpExp_n : ptrCommitted["tmpExp_n"],
                cm1_2ns : ptrCommitted["cm1_2ns"],
                cm2_2ns : ptrCommitted["cm2_2ns"],
                cm3_2ns : ptrCommitted["cm3_2ns"],
                const_n : ptrConstant["const_n"],
                const_2ns : ptrConstant["const_2ns"],
                challenges : challenges,
                x_n : ptrConstant["x_n"],
                x_2ns : ptrConstant["x_2ns"],
                constValues: constValues,
                publicInputs : ptrCommitted["publics"],
                q_2ns : ptrCommitted["q_2ns"]
            };

            pilFflonkSteps.setConstValues(E, params);

            if(zkey->polsNamesStage[4]->size() == 1) nonCommittedPols.push_back("Q");

            std::ostringstream ss;
            zklog.info("-----------------------------");
            zklog.info("  PIL-FFLONK PROVE SETTINGS");
            zklog.info("  Curve:           " + curveName);
            ss << "  Circuit power:   " << nBits;
            zklog.info(ss.str());
            
            ss.str("");
            ss << "  Domain size:     " << N;
            zklog.info(ss.str());
            ss.str("");
            ss << "  Extended Bits:   " << extendBits;
            zklog.info(ss.str());
            ss.str("");
            ss << "  Domain size ext: " << NExt << "(2^" << nBits + extendBits << ")";
            zklog.info(ss.str());
            ss.str("");
            ss << "  Const  pols:     " << fflonkInfo->nConstants;
            zklog.info(ss.str());
            ss.str("");
            ss << "  Stage 1 pols:    " << fflonkInfo->mapSectionsN.section[FflonkInfo::cm1_n];
            zklog.info(ss.str());
            ss.str("");
            ss << "  Stage 2 pols:    " << fflonkInfo->mapSectionsN.section[FflonkInfo::cm2_n];
            zklog.info(ss.str());
            ss.str("");
            ss << "  Stage 3 pols:    " << fflonkInfo->mapSectionsN.section[FflonkInfo::cm3_n];
            zklog.info(ss.str());
            ss.str("");
            ss << "  Stage 4 pols:    " << zkey->polsNamesStage[4]->size();
            zklog.info(ss.str());
            ss.str("");
            ss << "  Temp exp pols:   " << fflonkInfo->mapSectionsN.section[FflonkInfo::tmpExp_n];
            zklog.info("-----------------------------");

            transcript->reset();
                        
            // STAGE 0. Compute Publics and Store Constants
            zklog.info("STAGE 0. Compute Publics and Store Constants");
            stage0(params);

            // STAGE 1. Compute Trace Column Polynomials
            zklog.info("STAGE 1. Compute Trace Column Polynomials");
            stage1(params);

            // STAGE 2. Compute Inclusion Polynomials
            zklog.info("STAGE 2. Compute Inclusion Polynomials");
            stage2(params);

            // STAGE 3. Compute Grand Product and Intermediate Polynomials
            zklog.info("STAGE 3. Compute Grand Product and Intermediate Polynomials");
            stage3(params);

            // STAGE 4. Trace Quotient Polynomials
            zklog.info("STAGE 4. Compute Trace Quotient Polynomials");
            stage4(params);

            delete[] constValues;

            // Compute challenge xi seed
            transcript->reset();
            transcript->addScalar(challenges[4]);

            for(u_int32_t i = 0; i < zkey->f.size(); ++i) {
                if(zkey->f[i]->stages[0].stage == 4) {
                    G1Point commit = shPlonkProver->getPolynomialCommitment("f" + std::to_string(zkey->f[i]->index)); 
                    transcript->addPolCommitment(commit);
                }
            }

            AltBn128::FrElement challengeXiSeed = transcript->getChallenge();

            zklog.info("Challenge xi seed: " + E.fr.toString(challengeXiSeed));

            json pilFflonkProof = shPlonkProver->open((G1PointAffine *)ptrConstant["PTau"], ptrConstant["const_coefs"], ptrCommitted, ptrShPlonk, challengeXiSeed, nonCommittedPols);

            json publicSignals(nullptr);
            
            for (u_int32_t i = 0; i < fflonkInfo->nPublics; i++)
            {
                publicSignals.push_back(E.fr.toString(ptrCommitted["publics"][i]).c_str());
            }
            
            FrElement challengeXi = shPlonkProver->getChallengeXi();

            FrElement xN = E.fr.one();
            for (u_int64_t i = 0; i < N; i++)
            {
                xN = E.fr.mul(xN, challengeXi);
            }

            FrElement Z = E.fr.sub(xN, E.fr.one());

            E.fr.inv(Z, Z);
            pilFflonkProof["evaluations"]["invZh"] = E.fr.toString(Z);

            // Q should not be committed
            pilFflonkProof["evaluations"].erase("Q");

            TimerStopAndLog(PIL_FFLONK_PROVE);

            return {pilFflonkProof, publicSignals};
        }
        catch (const std::exception &e)
        {
            std::cerr << "EXCEPTION: " << e.what() << "\n";
            exit(EXIT_FAILURE);
        }
    }

    void PilFflonkProver::stage0(PilFflonkStepsParams &params) 
    {
        if (fflonkInfo->nConstants > 0)
        {
            TimerStart(PIL_FFLONK_ADD_CONSTANT_POLS_FI_COMMITMENTS);

            for(u_int32_t i = 0; i < zkey->f.size(); ++i) {
                if(zkey->f[i]->stages[0].stage == 0) {
                    G1Point commit = shPlonkProver->getPolynomialCommitment("f" + std::to_string(zkey->f[i]->index)); 
                    transcript->addPolCommitment(commit);
                }
            }

            TimerStopAndLog(PIL_FFLONK_ADD_CONSTANT_POLS_FI_COMMITMENTS);
        }

        TimerStart(PIL_FFLONK_CALCULATE_EXPS_PUBLICS);
        for (u_int32_t i = 0; i < fflonkInfo->nPublics; i++)
        {
            FflonkInfo::Publics publicPol = fflonkInfo->publics[i];
            if ("cmP" == publicPol.polType)
            {
                u_int64_t offset = (fflonkInfo->publics[i].idx * fflonkInfo->mapSectionsN.section[FflonkInfo::cm1_n] + fflonkInfo->publics[i].polId);
                ptrCommitted["publics"][i] = ptrCommitted["cm1_n"][offset];
            }
            else if ("imP" == publicPol.polType)
            {
                pilFflonkSteps.publics_first(E, params, fflonkInfo->publics[i].polId, i);
            }
            else
            {
                throw std::runtime_error("Invalid public input type");
            }
        }

        // Add all the publics to the transcript
        for (u_int32_t i = 0; i < fflonkInfo->nPublics; i++)
        {
            transcript->addScalar(ptrCommitted["publics"][i]);
        }

        TimerStopAndLog(PIL_FFLONK_CALCULATE_EXPS_PUBLICS);
    }

    void PilFflonkProver::stage1(PilFflonkStepsParams &params)
    {
        TimerStart(PIL_FFLONK_STAGE_1);
    
        // STEP 1.3 - Compute commit polynomials (coefficients + evaluations) and commit them
        if (fflonkInfo->mapSectionsN.section[FflonkInfo::cm1_n])
        {
            TimerStart(PIL_FFLONK_STAGE_1_EXTEND);
            extend(1, fflonkInfo->mapSectionsN.section[FflonkInfo::cm1_n]);
            TimerStopAndLog(PIL_FFLONK_STAGE_1_EXTEND);

            // STEP 1.4 - Commit stage 1 polynomials
            TimerStart(PIL_FFLONK_STAGE_1_COMMIT);
            shPlonkProver->commit(1, ptrCommitted["cm1_coefs"], (G1PointAffine *)ptrConstant["PTau"], ptrShPlonk);
            TimerStopAndLog(PIL_FFLONK_STAGE_1_COMMIT);
        }

        for(u_int32_t i = 0; i < zkey->f.size(); ++i) {
            if(zkey->f[i]->stages[0].stage == 1) {
                G1Point commit = shPlonkProver->getPolynomialCommitment("f" + std::to_string(zkey->f[i]->index)); 
                transcript->addPolCommitment(commit);
            }
        }

        TimerStopAndLog(PIL_FFLONK_STAGE_1);
    }

    void PilFflonkProver::stage2(PilFflonkStepsParams &params)
    {

        if(fflonkInfo->mapSectionsN.section[FflonkInfo::cm2_n] == 0 && fflonkInfo->peCtx.size() == 0) return;

        TimerStart(PIL_FFLONK_STAGE_2);
            
        zklog.info("Computing challenges alpha and beta");

        // Compute challenge alpha
        challenges[0] = transcript->getChallenge();
        zklog.info("Challenge alpha: " + E.fr.toString(challenges[0]));

        // Compute challenge beta
        transcript->reset();
        transcript->addScalar(challenges[0]);
        challenges[1] = transcript->getChallenge();
        zklog.info("Challenge beta: " + E.fr.toString(challenges[1]));
        transcript->reset();
        transcript->addScalar(challenges[1]);

        if (fflonkInfo->mapSectionsN.section[FflonkInfo::cm2_n])
        {
            // STEP 2.2 - Compute stage 2 polynomials --> h1, h2
            TimerStart(PIL_FFLONK_STAGE_2_CALCULATE_EXPS);
#pragma omp parallel for
            for (uint64_t i = 0; i < N; i++)
            {
                pilFflonkSteps.step2prev_first(E, params, i);
            }
            TimerStopAndLog(PIL_FFLONK_STAGE_2_CALCULATE_EXPS);

            auto nCm2 = fflonkInfo->mapSectionsN.section[FflonkInfo::cm1_n];

            TimerStart(PIL_FFLONK_STAGE_2_CALCULATE_H1H2);
            for (uint64_t i = 0; i < fflonkInfo->puCtx.size(); i++)
            {
                AltBn128::FrElement *fPol = getPolynomial(fflonkInfo->exp2pol[to_string(fflonkInfo->puCtx[i].fExpId)], 0);
                AltBn128::FrElement *tPol = getPolynomial(fflonkInfo->exp2pol[to_string(fflonkInfo->puCtx[i].tExpId)], N);

                uint64_t h1Id = fflonkInfo->varPolMap[fflonkInfo->cm_n[nCm2 + 2 * i]].sectionPos;
                uint64_t h2Id = fflonkInfo->varPolMap[fflonkInfo->cm_n[nCm2 + 2 * i + 1]].sectionPos;
                calculateH1H2(fPol, tPol, h1Id, h2Id);
            }

            TimerStopAndLog(PIL_FFLONK_STAGE_2_CALCULATE_H1H2);

            TimerStart(PIL_FFLONK_STAGE_2_EXTEND);
            extend(2, fflonkInfo->mapSectionsN.section[FflonkInfo::cm2_n]);
            TimerStopAndLog(PIL_FFLONK_STAGE_2_EXTEND);

            // STEP 2.3 - Commit stage 2 polynomials
            TimerStart(PIL_FFLONK_STAGE_2_COMMIT);
            shPlonkProver->commit(2, ptrCommitted["cm2_coefs"], (G1PointAffine *)ptrConstant["PTau"], ptrShPlonk);
            TimerStopAndLog(PIL_FFLONK_STAGE_2_COMMIT);

            for(u_int32_t i = 0; i < zkey->f.size(); ++i) {
                if(zkey->f[i]->stages[0].stage == 2) {
                    G1Point commit = shPlonkProver->getPolynomialCommitment("f" + std::to_string(zkey->f[i]->index)); 
                    transcript->addPolCommitment(commit);
                }
            }
        }

        TimerStopAndLog(PIL_FFLONK_STAGE_2);
    }

    void PilFflonkProver::stage3(PilFflonkStepsParams &params)
    {
        if(fflonkInfo->mapSectionsN.section[FflonkInfo::cm3_n] == 0) return;

        TimerStart(PIL_FFLONK_STAGE_3);

        zklog.info("Computing challenges gamma and delta");

        // Compute challenge gamma
        challenges[2] = transcript->getChallenge();
        zklog.info("Challenge gamma: " + E.fr.toString(challenges[2]));

        // Compute challenge delta
        transcript->reset();
        transcript->addScalar(challenges[2]);
        challenges[3] = transcript->getChallenge();
        zklog.info("Challenge delta: " + E.fr.toString(challenges[3]));
        transcript->reset();
        transcript->addScalar(challenges[3]);

    
        // STEP 3.2 - Compute stage 3 polynomials --> Plookup Z, Permutations Z & ConnectionZ polynomials
        auto nPlookups = fflonkInfo->puCtx.size();
        auto nPermutations = fflonkInfo->peCtx.size();
        auto nConnections = fflonkInfo->ciCtx.size();

        TimerStart(PIL_FFLONK_STAGE_3_PREV_CALCULATE_EXPS);
#pragma omp parallel for
        for (uint64_t i = 0; i < N; i++)
        {
            pilFflonkSteps.step3prev_first(E, params, i);
        }
        TimerStopAndLog(PIL_FFLONK_STAGE_3_PREV_CALCULATE_EXPS);

        auto nCm3 = fflonkInfo->mapSectionsN.section[FflonkInfo::cm1_n] + fflonkInfo->mapSectionsN.section[FflonkInfo::cm2_n];

        TimerStart(PIL_FFLONK_STAGE_3_CALCULATE_Z);
        for (uint64_t i = 0; i < nPlookups; i++)
        {
            zklog.info("Calculating z for plookup " + to_string(i) + " / " + to_string(nPlookups));
            u_int64_t numId = fflonkInfo->exp2pol[to_string(fflonkInfo->puCtx[i].numId)];
            u_int64_t denId = fflonkInfo->exp2pol[to_string(fflonkInfo->puCtx[i].denId)];
            u_int64_t zId = fflonkInfo->cm_n[nCm3 + i];

            calculateZ(numId, denId, zId);
        }

        for (uint64_t i = 0; i < nPermutations; i++)
        {
            zklog.info("Calculating z for permutation " + to_string(i) + " / " + to_string(nPermutations));
            u_int64_t numId = fflonkInfo->exp2pol[to_string(fflonkInfo->peCtx[i].numId)];
            u_int64_t denId = fflonkInfo->exp2pol[to_string(fflonkInfo->peCtx[i].denId)];
            u_int64_t zId = fflonkInfo->cm_n[nCm3 + nPlookups + i];

            calculateZ(numId, denId, zId);
        }

        for (uint64_t i = 0; i < nConnections; i++)
        {
            zklog.info("Calculating z for connection " + to_string(i) + " / " + to_string(nConnections));
            u_int64_t numId = fflonkInfo->exp2pol[to_string(fflonkInfo->ciCtx[i].numId)];
            u_int64_t denId = fflonkInfo->exp2pol[to_string(fflonkInfo->ciCtx[i].denId)];
            u_int64_t zId = fflonkInfo->cm_n[nCm3 + nPlookups + nPermutations + i];

            calculateZ(numId, denId, zId);
        }
        TimerStopAndLog(PIL_FFLONK_STAGE_3_CALCULATE_Z);

        TimerStart(PIL_FFLONK_STAGE_3_CALCULATE_EXPS);
#pragma omp parallel for
        for (uint64_t i = 0; i < N; i++)
        {
            pilFflonkSteps.step3_first(E, params, i);
        }
        TimerStopAndLog(PIL_FFLONK_STAGE_3_CALCULATE_EXPS);

        TimerStart(PIL_FFLONK_STAGE_3_EXTEND);
        extend(3, fflonkInfo->mapSectionsN.section[FflonkInfo::cm3_n]);
        TimerStopAndLog(PIL_FFLONK_STAGE_3_EXTEND);

        TimerStart(PIL_FFLONK_STAGE_3_COMMIT);
        shPlonkProver->commit(3, ptrCommitted["cm3_coefs"], (G1PointAffine *)ptrConstant["PTau"], ptrShPlonk);
        TimerStopAndLog(PIL_FFLONK_STAGE_3_COMMIT);
    

        for(u_int32_t i = 0; i < zkey->f.size(); ++i) {
            if(zkey->f[i]->stages[0].stage == 3) {
                G1Point commit = shPlonkProver->getPolynomialCommitment("f" + std::to_string(zkey->f[i]->index)); 
                transcript->addPolCommitment(commit);
            }
        }
        TimerStopAndLog(PIL_FFLONK_STAGE_3);
    }

    void PilFflonkProver::stage4(PilFflonkStepsParams &params)
    {
        TimerStart(PIL_FFLONK_STAGE_4);

        zklog.info("Computing challenges a");
        challenges[4] = transcript->getChallenge();
        zklog.info("Challenge a: " + E.fr.toString(challenges[4]));

        // STEP 4.2 - Compute stage 4 polynomial --> Q polynomial

        TimerStart(PIL_FFLONK_STAGE_4_CALCULATE_EXPS);
#pragma omp parallel for
        for (uint64_t i = 0; i < NExt; i++)
        {
            pilFflonkSteps.step42ns_first(E, params, i);
        }
        TimerStopAndLog(PIL_FFLONK_STAGE_4_CALCULATE_EXPS);
        
        TimerStart(PIL_FFLONK_STAGE_4_CALCULATE_Q);

        nttExtended->INTT(ptrCommitted["q_2ns"], ptrCommitted["q_2ns"], NExt, 1, ptrShPlonk["tmp"]);
        Polynomial<AltBn128::Engine> *polQ = new Polynomial<AltBn128::Engine>(E, ptrCommitted["q_2ns"], NExt, 0, false);
        polQ->divZh(N, 1 << extendBitsTotal);

        u_int64_t domainSizeQ = fflonkInfo->qDeg * N + fflonkInfo->maxPolsOpenings * (fflonkInfo->qDeg + 1);

        if (polQ->getDegree() > domainSizeQ)
        {
            throw std::runtime_error("Q Polynomial is not well calculated");
        }

        if(zkey->maxQDegree) {
            FrElement rand1 = E.fr.set(2);
            FrElement rand2 = E.fr.set(3);
            // FrElement rand1;
            // FrElement rand2;
            // randombytes_buf((void *)&(rand1), sizeof(FrElement)-1);
            // randombytes_buf((void *)&(rand2), sizeof(FrElement)-1);

            u_int64_t nQ = std::ceil(domainSizeQ - (fflonkInfo->maxPolsOpenings * (fflonkInfo->qDeg + 1)) / (zkey->maxQDegree * N));

            for(u_int32_t i = 0; i < nQ; ++i) {
                u_int64_t pos = i * zkey->maxQDegree * N;

            if(i > 0) {
                    polQ->coef[pos] = E.fr.sub(polQ->coef[pos], rand1);
                    polQ->coef[pos + 1] = E.fr.sub(polQ->coef[pos + 1], rand2);
                }

                if(i < nQ - 1) {
                    u_int64_t len = zkey->maxQDegree * N;

                    rand1 = E.fr.set(2);
                    rand2 = E.fr.set(3);

                    // randombytes_buf((void *)&(rand1), sizeof(FrElement)-1);
                    // randombytes_buf((void *)&(rand2), sizeof(FrElement)-1);

                    shPlonkProver->addRandomCoef("Q" + std::to_string(i), len, rand1);
                    shPlonkProver->addRandomCoef("Q" + std::to_string(i), len + 1, rand2);
                }
            }
        }


        TimerStopAndLog(PIL_FFLONK_STAGE_4_CALCULATE_Q);

        TimerStart(PIL_FFLONK_STAGE_4_COMMIT);
        shPlonkProver->commit(4, ptrCommitted["q_2ns"], (G1PointAffine *)ptrConstant["PTau"], ptrShPlonk);
        TimerStopAndLog(PIL_FFLONK_STAGE_4_COMMIT);

        TimerStopAndLog(PIL_FFLONK_STAGE_4);
    }

    void PilFflonkProver::extend(u_int32_t stage, u_int32_t nPols)
    {

        AltBn128::FrElement *buffSrc = ptrCommitted["cm" + std::to_string(stage) + "_n"];       // N
        AltBn128::FrElement *buffDst = ptrCommitted["cm" + std::to_string(stage) + "_2ns"];     // NExt
        AltBn128::FrElement *buffCoefs = ptrCommitted["cm" + std::to_string(stage) + "_coefs"]; // NCoefs


        int nThreads = omp_get_max_threads() / 2;

        ThreadUtils::parset(buffCoefs, 0, NCoefs * nPols * sizeof(AltBn128::FrElement), nThreads);

        TimerStart(EXTEND_INTT);
        ntt->INTT(buffCoefs, buffSrc, N, nPols, ptrShPlonk["tmp"]);
        TimerStopAndLog(EXTEND_INTT);

        for (u_int32_t i = 0; i < nPols; i++)
        {
            std::string name = (*zkey->polsNamesStage[stage])[i];
            u_int32_t openings = findNumberOpenings(name, stage);
            for (u_int32_t j = 0; j < openings; ++j)
            {
                FrElement b;
                // randombytes_buf((void *)&(b), sizeof(FrElement)-1);
                b = E.fr.set(2);

                buffCoefs[j * nPols + i] = E.fr.add(buffCoefs[j * nPols + i], E.fr.neg(b));
                buffCoefs[(j + N) * nPols + i] = E.fr.add(buffCoefs[(j + N) * nPols + i], b);
            }
        }
                
        ThreadUtils::parset(buffDst, 0, NExt * nPols * sizeof(AltBn128::FrElement), nThreads);
        ThreadUtils::parcpy(buffDst, buffCoefs, NCoefs * nPols * sizeof(AltBn128::FrElement), nThreads);
               
        TimerStart(EXTEND_NTT);
        nttExtended->NTT(buffDst, buffDst, 1 << nBitsExt, nPols, ptrShPlonk["tmp"]);
        TimerStopAndLog(EXTEND_NTT);
    }

    AltBn128::FrElement *PilFflonkProver::getPolynomial(uint64_t polId, uint64_t offset)
    {
        FflonkInfo::eSection section = fflonkInfo->varPolMap[polId].section;
        std::string sectionName = fflonkInfo->getSectionName(section);
        u_int64_t pos = fflonkInfo->varPolMap[polId].sectionPos;
        u_int64_t nPols = fflonkInfo->mapSections.section[section].size();

        AltBn128::FrElement *pol = ptrShPlonk["tmp"] + offset;

        for (uint64_t i = 0; i < N; i++)
        {
            pol[i] = ptrCommitted[sectionName][i * nPols + pos];
        }
        return pol;
    }

    void PilFflonkProver::calculateZ(u_int64_t numId, u_int64_t denId, u_int64_t zId)
    {
        
        FflonkInfo::PolInfo num = fflonkInfo->getPolInfo(numId);
        FflonkInfo::PolInfo z = fflonkInfo->getPolInfo(zId);

        TimerStart(BATCH_INVERSE);
        batchInverse(denId);
        TimerStopAndLog(BATCH_INVERSE);

        const AltBn128::FrElement* polDenI = ptrShPlonk["tmp"];
        const AltBn128::FrElement* polNum = ptrCommitted[num.sectionName];
        AltBn128::FrElement* polZ = ptrCommitted[z.sectionName];


        polZ[z.id] = E.fr.one();

        for (uint64_t i = 1; i < N; i++)
        {
            AltBn128::FrElement z_i = polZ[(i - 1) * z.nPols + z.id];
            AltBn128::FrElement num_i = polNum[(i - 1) * num.nPols + num.id];
            AltBn128::FrElement denI_i = polDenI[i - 1];

            polZ[i * z.nPols + z.id] = E.fr.mul(z_i, E.fr.mul(num_i, denI_i));
        }

        // Check that z * num * denI = 1
        zkassert(E.fr.eq(
            E.fr.one(), 
            E.fr.mul(
                polZ[(N - 1) * z.nPols + z.id], 
                E.fr.mul(
                    polNum[(N - 1) * num.nPols + num.id], 
                    polDenI[N - 1]
                )
            )
        ));
    }

    void PilFflonkProver::batchInverse(const u_int64_t denId)
    {
        const FflonkInfo::PolInfo& den = fflonkInfo->getPolInfo(denId);

        AltBn128::FrElement* tmp = ptrShPlonk["tmp"];
        
        const AltBn128::FrElement* polDen = ptrCommitted[den.sectionName];

        tmp[N] = polDen[den.id];

        for (uint64_t i = 1; i < N; i++)
        {
            tmp[N + i] = E.fr.mul(tmp[N + i - 1], polDen[i * den.nPols + den.id]);
        }

        AltBn128::FrElement z0;
        AltBn128::FrElement z1;
        E.fr.inv(z0, tmp[2*N - 1]);

        for (uint64_t i = N - 1; i > 0; i--)
        {
            z1 = E.fr.mul(z0, polDen[i * den.nPols + den.id]);
            tmp[i] = E.fr.mul(z0, tmp[N + i - 1]);
            z0 = z1;
        }

        tmp[0] = z0;
    }

    u_int32_t PilFflonkProver::findNumberOpenings(std::string name, u_int32_t stage) {
        for(u_int32_t i = 0; i < zkey->f.size(); ++i) {
            if(zkey->f[i]->stages[0].stage != stage) continue;
            for(u_int32_t j = 0; j < zkey->f[i]->nPols; ++j) {
                if(zkey->f[i]->pols[j] == name) return zkey->f[i]->nOpeningPoints + 1;
            }
        }
        return 0;
    }

    void PilFflonkProver::calculateH1H2(AltBn128::FrElement *fPol, AltBn128::FrElement *tPol, uint64_t h1Id, uint64_t h2Id)
    {
        map<AltBn128::FrElement, uint64_t, CompareFeFr> idx_t(E);
        multimap<AltBn128::FrElement, uint64_t, CompareFeFr> s(E);
        multimap<AltBn128::FrElement, uint64_t>::iterator it;
        uint64_t i = 0;

        for (uint64_t i = 0; i < N; i++)
        {
            AltBn128::FrElement key = tPol[i];
            std::pair<AltBn128::FrElement, uint64_t> pr(key, i);

            auto const result = idx_t.insert(pr);
            if (not result.second)
            {
                result.first->second = i;
            }

            s.insert(pr);
        }

        for (uint64_t i = 0; i < N; i++)
        {
            AltBn128::FrElement key = fPol[i];

            if (idx_t.find(key) == idx_t.end())
            {
                zklog.error("Polinomial::calculateH1H2() Number not included: " + E.fr.toString(fPol[i]));
                exitProcess();
            }
            uint64_t idx = idx_t[key];
            s.insert(pair<AltBn128::FrElement, uint64_t>(key, idx));
        }

        multimap<uint64_t, AltBn128::FrElement> s_sorted;
        multimap<uint64_t, AltBn128::FrElement>::iterator it_sorted;

        for (it = s.begin(); it != s.end(); it++)
        {
            s_sorted.insert(make_pair(it->second, it->first));
        }

        for (it_sorted = s_sorted.begin(); it_sorted != s_sorted.end(); it_sorted++, i++)
        {
            int ind = i/2;
            if ((i & 1) == 0)
            {   
                ptrCommitted["cm2_n"][h1Id + fflonkInfo->nCm2 * ind] = it_sorted->second;
            }
            else
            {
                ptrCommitted["cm2_n"][h2Id + fflonkInfo->nCm2 * ind] = it_sorted->second;
            }
        }
    };

}
