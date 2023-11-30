#include <fstream>
#include <iomanip>
#include <unistd.h>
#include "prover_aggregation.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "main.hpp"
#include "main.multichainPrep.hpp"
#include "main.multichainAgg.hpp"
#include "main.multichainAggF.hpp"
#include "main.multichainFinal.hpp"
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "wtns_utils.hpp"
#include "groth16.hpp"
#include "timer.hpp"
#include "execFile.hpp"
#include <math.h> /* log2 */
#include "calculateSha256Publics.hpp"
#include "proof2zkinStark.hpp"
#include "joinzkinMultichainStark.hpp"
#include "friProofC12.hpp"
#include <algorithm> // std::min
#include <openssl/sha.h>

#include "commit_pols_starks.hpp"
#include "multichainPrepSteps.hpp"
#include "multichainAggSteps.hpp"
#include "multichainAggFSteps.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

ProverAggregation::ProverAggregation(Goldilocks &fr,
               PoseidonGoldilocks &poseidon,
               const Config &config) : fr(fr),
                                       poseidon(poseidon),
                                       pCurrentRequest(NULL),
                                       config(config),
                                       lastComputedRequestEndTime(0)
{

    try
    {
        if (config.generateProofAggregation())
        {   
            mpz_init(altBbn128r);
            mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);

            StarkInfo _starkInfo(config, config.multichainPrepStarkInfo);
            uint64_t polsSize = _starkInfo.mapTotalN * sizeof(Goldilocks::Element) + _starkInfo.mapSectionsN.section[eSection::cm3_2ns] * (1 << _starkInfo.starkStruct.nBitsExt) * sizeof(Goldilocks::Element);
            
            if(config.generateFinalMultichainProof()) {
                uint64_t polsFflonkSize = 63350767616;
                polsSize = std::max(polsSize, polsFflonkSize);
                StarkInfo _starksMultichainAggF(config, config.multichainAggFStarkInfo);

                uint64_t polsSizeMultichainAggF = _starksMultichainAggF.mapTotalN * sizeof(Goldilocks::Element);

                pAddressStarksMultichainAggF = (void *)malloc(polsSizeMultichainAggF);
                if (pAddressStarksMultichainAggF == NULL)
                {
                    zklog.error("ProverAggregation failed calling malloc() for the Aggregation Final of size " + to_string(polsSizeMultichainAggF));
                    exitProcess();
                }
                zklog.info("ProverAggregation successfully allocated " + to_string(polsSizeMultichainAggF) + " bytes");

                starksMultichainAggF = new StarkRecursiveF(config, pAddressStarksMultichainAggF, true);

                zkey = BinFileUtils::openExisting(config.multichainFinalStarkZkey, "zkey", 1);
                protocolId = Zkey::getProtocolIdFromZkey(zkey.get());
                if (Zkey::GROTH16_PROTOCOL_ID == protocolId)
                {
                    zkeyHeader = ZKeyUtils::loadHeader(zkey.get());

                    if (mpz_cmp(zkeyHeader->rPrime, altBbn128r) != 0)
                    {
                        throw std::invalid_argument("zkey curve not supported");
                    }

                    groth16Prover = Groth16::makeProver<AltBn128::Engine>(
                        zkeyHeader->nVars,
                        zkeyHeader->nPublic,
                        zkeyHeader->domainSize,
                        zkeyHeader->nCoefs,
                        zkeyHeader->vk_alpha1,
                        zkeyHeader->vk_beta1,
                        zkeyHeader->vk_beta2,
                        zkeyHeader->vk_delta1,
                        zkeyHeader->vk_delta2,
                        zkey->getSectionData(4), // Coefs
                        zkey->getSectionData(5), // pointsA
                        zkey->getSectionData(6), // pointsB1
                        zkey->getSectionData(7), // pointsB2
                        zkey->getSectionData(8), // pointsC
                        zkey->getSectionData(9)  // pointsH1
                    );
                }

                prover = new Fflonk::FflonkProver<AltBn128::Engine>(AltBn128::Engine::engine, pAddress, polsSize);
                prover->setZkey(zkey.get());
            }

            if(config.generateMultichainProof()) {
                pAddress = calloc(polsSize, 1);
                if (pAddress == NULL)
                {
                    zklog.error("ProverAggregation failed calling malloc() of size " + to_string(polsSize));
                    exitProcess();
                }
                zklog.info("ProverAggregation successfully allocated " + to_string(polsSize) + " bytes");
                
                if(config.generatePrepareMultichainProof()) {
                    starksMultichainPrep = new Starks(config, {config.multichainPrepConstPols, config.mapConstPolsFile, config.multichainPrepConstantsTree, config.multichainPrepStarkInfo}, pAddress);
                }
                if(config.generateAggregatedMultichainProof()) {
                    starksMultichainAgg = new Starks(config, {config.multichainAggConstPols, config.mapConstPolsFile, config.multichainAggConstantsTree, config.multichainAggStarkInfo}, pAddress);
                }
            }
            
            lastComputedRequestEndTime = 0;

            sem_init(&pendingRequestSem, 0, 0);
            pthread_mutex_init(&mutex, NULL);
            pCurrentRequest = NULL;
            pthread_create(&proverPthread, NULL, proverAggregationThread, this);
            pthread_create(&cleanerPthread, NULL, cleanerAggregationThread, this);
        }
    }
    catch (std::exception &e)
    {
        zklog.error("ProverAggregation::Prover() got an exception: " + string(e.what()));
        exitProcess();
    }
}

ProverAggregation::~ProverAggregation()
{
    if (config.generateProofAggregation())
    {
        if(config.generateFinalMultichainProof()) {
            mpz_clear(altBbn128r);

            Groth16::Prover<AltBn128::Engine> *pGroth16 = groth16Prover.release();
            BinFileUtils::BinFile *pZkey = zkey.release();
            ZKeyUtils::Header *pZkeyHeader = zkeyHeader.release();
            assert(groth16Prover.get() == nullptr);
            assert(groth16Prover == nullptr);
            assert(zkey.get() == nullptr);
            assert(zkey == nullptr);
            assert(zkeyHeader.get() == nullptr);
            assert(zkeyHeader == nullptr);

            delete pGroth16;
            delete pZkey;
            delete pZkeyHeader;

            free(pAddressStarksMultichainAggF);
            delete starksMultichainAggF;

            delete prover;
        }

        if(config.generateMultichainProof()) {
            free(pAddress);

            if(config.generatePrepareMultichainProof()) {
                delete starksMultichainPrep;
            }
            if(config.generateAggregatedMultichainProof()) {
                delete starksMultichainAgg;
            }
        }
    }
}

void *proverAggregationThread(void *arg)
{
    ProverAggregation *pProverAggregation = (ProverAggregation *)arg;
    zklog.info("proverAggregationThread() started");

    zkassert(pProverAggregation->config.generateProofAggregation());

    while (true)
    {
        pProverAggregation->lock();

        // Wait for the pending request queue semaphore to be released, if there are no more pending requests
        if (pProverAggregation->pendingRequests.size() == 0)
        {
            pProverAggregation->unlock();
            sem_wait(&pProverAggregation->pendingRequestSem);
        }

        // Check that the pending requests queue is not empty
        if (pProverAggregation->pendingRequests.size() == 0)
        {
            pProverAggregation->unlock();
            zklog.info("proverAggregationThread() found pending requests queue empty, so ignoring");
            continue;
        }

        // Extract the first pending request (first in, first out)
        pProverAggregation->pCurrentRequest = pProverAggregation->pendingRequests[0];
        pProverAggregation->pCurrentRequest->startTime = time(NULL);
        pProverAggregation->pendingRequests.erase(pProverAggregation->pendingRequests.begin());

        zklog.info("proverAggregationThread() starting to process request with UUID: " + pProverAggregation->pCurrentRequest->uuid);

        pProverAggregation->unlock();

        // Process the request
        switch (pProverAggregation->pCurrentRequest->type)
        {
        case prt_genPrepareMultichainProof:
            pProverAggregation->genPrepareMultichainProof(pProverAggregation->pCurrentRequest);
            break;
        case prt_genAggregatedMultichainProof:
            pProverAggregation->genAggregatedMultichainProof(pProverAggregation->pCurrentRequest);
            break;
        case prt_genFinalMultichainProof:
            pProverAggregation->genFinalMultichainProof(pProverAggregation->pCurrentRequest);
            break;
        default:
            zklog.error("proverAggregationThread() got an invalid prover request type=" + to_string(pProverAggregation->pCurrentRequest->type));
            exitProcess();
        }

        // Move to completed requests
        pProverAggregation->lock();
        ProverAggregationRequest *pProverAggregationRequest = pProverAggregation->pCurrentRequest;
        pProverAggregationRequest->endTime = time(NULL);
        pProverAggregation->lastComputedRequestId = pProverAggregationRequest->uuid;
        pProverAggregation->lastComputedRequestEndTime = pProverAggregationRequest->endTime;

        pProverAggregation->completedRequests.push_back(pProverAggregation->pCurrentRequest);
        pProverAggregation->pCurrentRequest = NULL;
        pProverAggregation->unlock();

        zklog.info("proverAggregationThread() done processing request with UUID: " + pProverAggregationRequest->uuid);

        // Release the prove request semaphore to notify any blocked waiting call
        pProverAggregationRequest->notifyCompleted();
    }
    zklog.info("proverAggregationThread() done");
    return NULL;
}

void *cleanerAggregationThread(void *arg)
{
    ProverAggregation *pProverAggregation = (ProverAggregation *)arg;
    zklog.info("cleanerAggregationThread() started");

    zkassert(pProverAggregation->config.generateProofAggregation());

    while (true)
    {
        // Sleep for 10 minutes
        sleep(pProverAggregation->config.cleanerPollingPeriod);

        // Lock the prover
        pProverAggregation->lock();

        // Delete all requests older than requests persistence configuration setting
        time_t now = time(NULL);
        bool bRequestDeleted = false;
        do
        {
            bRequestDeleted = false;
            for (uint64_t i = 0; i < pProverAggregation->completedRequests.size(); i++)
            {
                if (now - pProverAggregation->completedRequests[i]->endTime > (int64_t)pProverAggregation->config.requestsPersistence)
                {
                    zklog.info("cleanerAggregationThread() deleting request with uuid: " + pProverAggregation->completedRequests[i]->uuid);
                    ProverAggregationRequest *pProverAggregationRequest = pProverAggregation->completedRequests[i];
                    pProverAggregation->completedRequests.erase(pProverAggregation->completedRequests.begin() + i);
                    pProverAggregation->requestsMap.erase(pProverAggregationRequest->uuid);
                    delete (pProverAggregationRequest);
                    bRequestDeleted = true;
                    break;
                }
            }
        } while (bRequestDeleted);

        // Unlock the prover
        pProverAggregation->unlock();
    }
    zklog.info("cleanerAggregationThread() done");
    return NULL;
}

string ProverAggregation::submitRequest(ProverAggregationRequest *pProverAggregationRequest) // returns UUID for this request
{
    zkassert(config.generateProofAggregation());
    zkassert(pProverAggregationRequest != NULL);

    zklog.info("ProverAggregation::submitRequest() started type=" + to_string(pProverAggregationRequest->type));

    // Get the prover request UUID
    string uuid = pProverAggregationRequest->uuid;

    // Add the request to the pending requests queue, and release the semaphore to notify the prover thread
    lock();
    requestsMap[uuid] = pProverAggregationRequest;
    pendingRequests.push_back(pProverAggregationRequest);
    sem_post(&pendingRequestSem);
    unlock();

    zklog.info("ProverAggregation::submitRequest() returns UUID: " + uuid);
    return uuid;
}

ProverAggregationRequest *ProverAggregation::waitForRequestToComplete(const string &uuid, const uint64_t timeoutInSeconds) // wait for the request with this UUID to complete; returns NULL if UUID is invalid
{
    zkassert(config.generateProofAggregation());
    zkassert(uuid.size() > 0);
    zklog.info("ProverAggregation::waitForRequestToComplete() waiting for request with UUID: " + uuid);

    // We will store here the address of the prove request corresponding to this UUID
    ProverAggregationRequest *pProverAggregationRequest = NULL;

    lock();

    // Map uuid to the corresponding prover request
    std::unordered_map<std::string, ProverAggregationRequest *>::iterator it = requestsMap.find(uuid);
    if (it == requestsMap.end())
    {
        zklog.error("ProverAggregation::waitForRequestToComplete() unknown uuid: " + uuid);
        unlock();
        return NULL;
    }

    // Wait for the request to complete
    pProverAggregationRequest = it->second;
    unlock();
    pProverAggregationRequest->waitForCompleted(timeoutInSeconds);
    zklog.info("ProverAggregation::waitForRequestToComplete() done waiting for request with UUID: " + uuid);

    // Return the request pointer
    return pProverAggregationRequest;
}

void ProverAggregation::calculateHash(ProverAggregationRequest *pProverAggregationRequest) {
    zkassert(config.generateProofAggregation());
    zkassert(pProverAggregationRequest != NULL);
    zkassert(pProverAggregationRequest->type == prt_calculateHash);

    TimerStart(PROVER_CALCULATE_SHA256);

    // Save input to file
    if (config.saveInputToFile)
    {
        json2file(pProverAggregationRequest->chainPublicsInput, pProverAggregationRequest->filePrefix + "chain_publics.input.json");
        if(pProverAggregationRequest->prevHashInput != nullptr) {
            json2file(pProverAggregationRequest->prevHashInput, pProverAggregationRequest->filePrefix + "prev_hash.input.json");
        }
    }
  
    pProverAggregationRequest->hashOutput = calculateSha256(pProverAggregationRequest->chainPublicsInput, pProverAggregationRequest->prevHashInput);

    // Save output to file
    if (config.saveOutputToFile)
    {
        json2file(pProverAggregationRequest->hashOutput, pProverAggregationRequest->filePrefix + "out_hash.output.json");
    }

    pProverAggregationRequest->result = ZKR_SUCCESS;

    TimerStopAndLog(PROVER_CALCULATE_SHA256);
}

void ProverAggregation::genPrepareMultichainProof(ProverAggregationRequest *pProverAggregationRequest)
{
    zkassert(config.generateProofAggregation());
    zkassert(pProverAggregationRequest != NULL);
    zkassert(pProverAggregationRequest->type == prt_genPrepareMultichainProof);

    TimerStart(PROVER_PREPARE_MULTICHAIN_PROOF);
    
    printMemoryInfo(true);
    printProcessInfo(true);

    // Save input to file
    if (config.saveInputToFile)
    {
        json2file(pProverAggregationRequest->multichainPrepProofInput, pProverAggregationRequest->filePrefix + "prep_multichain_proof.input.json");
        if(pProverAggregationRequest->multichainPrepPrevHashInput != nullptr) json2file(pProverAggregationRequest->multichainPrepPrevHashInput, pProverAggregationRequest->filePrefix + "prep_multichain_prevHash.input.json");
    }

    // Input is pProverAggregationRequest->multichainPrepProofInput and pProverAggregationRequest->multichainPrepPrevHashInput (of type json) 
    
    json verKeyMultichainPrepJson;
    file2json(config.multichainPrepVerkey, verKeyMultichainPrepJson);

    Goldilocks::Element multichainPrepVerkey[4];
    multichainPrepVerkey[0] = Goldilocks::fromU64(verKeyMultichainPrepJson["constRoot"][0]);
    multichainPrepVerkey[1] = Goldilocks::fromU64(verKeyMultichainPrepJson["constRoot"][1]);
    multichainPrepVerkey[2] = Goldilocks::fromU64(verKeyMultichainPrepJson["constRoot"][2]);
    multichainPrepVerkey[3] = Goldilocks::fromU64(verKeyMultichainPrepJson["constRoot"][3]);

    ordered_json verKeyMultichainAgg;
    file2json(config.multichainAggVerkey, verKeyMultichainAgg);

    json zkinInputMultichainPrep = pProverAggregationRequest->multichainPrepProofInput;

    zkinInputMultichainPrep["rootC"] = ordered_json::array();
    for (int i = 0; i < 4; i++)
    {
        zkinInputMultichainPrep["rootC"][i] = to_string(verKeyMultichainAgg["constRoot"][i]);
    }

    std::string nPrevBlocksSha256;
    std::string prevHash[8]; 

    if(pProverAggregationRequest->multichainPrepPrevHashInput != nullptr) {
        json prevHashInfo = pProverAggregationRequest->multichainPrepPrevHashInput;

        nPrevBlocksSha256 = to_string(prevHashInfo["nPrevBlocks"]);
        for (int i = 0; i < 8; i++)
        {
            prevHash[i] = to_string(prevHashInfo["prevHash"][i]);
        }
    } else {
        nPrevBlocksSha256 = "0";
        prevHash[0] = "1779033703";
        prevHash[1] = "3144134277";
        prevHash[2] = "1013904242";
        prevHash[3] = "2773480762";
        prevHash[4] = "1359893119";
        prevHash[5] = "2600822924";
        prevHash[6] = "528734635";
        prevHash[7] = "1541459225";
    }

    // Add Previous Hash information
    zkinInputMultichainPrep["nPrevBlocksSha256"] = nPrevBlocksSha256;
    zkinInputMultichainPrep["prevHash"] = json::array();
    for(int i = 0; i < 8; ++i) {
        zkinInputMultichainPrep["prevHash"][i] = prevHash[i];
    }

    json outHashInfo = json::object();

    CommitPolsStarks cmPolsPrep(pAddress, (1 << starksMultichainPrep->starkInfo.starkStruct.nBits), starksMultichainPrep->starkInfo.nCm1);
    CircomMultichainPrep::getCommitedPols(&cmPolsPrep, config.multichainPrepVerifier, config.multichainPrepExec, zkinInputMultichainPrep, (1 << starksMultichainPrep->starkInfo.starkStruct.nBits), starksMultichainPrep->starkInfo.nCm1, outHashInfo);

    Goldilocks::Element multichainAggVerkeyValues[4];
    multichainAggVerkeyValues[0] = Goldilocks::fromU64(verKeyMultichainAgg["constRoot"][0]);
    multichainAggVerkeyValues[1] = Goldilocks::fromU64(verKeyMultichainAgg["constRoot"][1]);
    multichainAggVerkeyValues[2] = Goldilocks::fromU64(verKeyMultichainAgg["constRoot"][2]);
    multichainAggVerkeyValues[3] = Goldilocks::fromU64(verKeyMultichainAgg["constRoot"][3]);

    Goldilocks::Element publics[starksMultichainPrep->starkInfo.nPublics];

    uint64_t nBlocksSha256 = outHashInfo["nBlocks"];

    publics[0] = Goldilocks::fromU64(nBlocksSha256);

    uint64_t outHash[8];
    for(uint64_t i = 0; i < 8; i++) {
        outHash[i] = outHashInfo["outHash"][i];
    }
  
    for (uint64_t i = 0; i < 8; i++)
    {
        publics[i + 1] = Goldilocks::fromU64(outHash[i]);
    }

    publics[9] = Goldilocks::fromString(nPrevBlocksSha256);

    for (uint64_t i = 0; i < 8; i++)
    {
        publics[i + 10] = Goldilocks::fromString(prevHash[i]);
    }

    for (uint64_t i = 0; i < verKeyMultichainAgg["constRoot"].size(); i++)
    {
        publics[i + 18] = Goldilocks::fromU64(verKeyMultichainAgg["constRoot"][i]);
    }

    // void *pointerCmMultichainPrepPols = mapFile("config/multichainPrep/multichainPrep.commit", cmPolsPrep.size(), true);
    // memcpy(pointerCmMultichainPrepPols, cmPolsPrep.address(), cmPolsPrep.size());
    // unmapFile(pointerCmMultichainPrepPols, cmPolsPrep.size());

    //-------------------------------------------
    // Generate Multichain Prep proof
    //-------------------------------------------

    TimerStart(STARK_MULTICHAIN_PREPARE_PROOF);
    uint64_t polBitsMultichainPrep = starksMultichainPrep->starkInfo.starkStruct.steps[starksMultichainPrep->starkInfo.starkStruct.steps.size() - 1].nBits;
    FRIProof fproofMultichainPrep((1 << polBitsMultichainPrep), FIELD_EXTENSION, starksMultichainPrep->starkInfo.starkStruct.steps.size(), starksMultichainPrep->starkInfo.evMap.size(), starksMultichainPrep->starkInfo.nPublics);
    MultichainPrepSteps multichainPrepSteps;
    starksMultichainPrep->genProof(fproofMultichainPrep, publics, multichainPrepVerkey, &multichainPrepSteps);
    TimerStopAndLog(STARK_MULTICHAIN_PREPARE_PROOF);

    // Save the proof & zkinproof
    TimerStart(SAVE_PROOF);

    nlohmann::ordered_json jProofMultichainPrep = fproofMultichainPrep.proofs.proof2json();
    nlohmann::ordered_json zkinMultichainPrep = proof2zkinStark(jProofMultichainPrep);
    zkinMultichainPrep["nPrevBlocksSha256"] = zkinInputMultichainPrep["nPrevBlocksSha256"];
    zkinMultichainPrep["prevHash"] = zkinInputMultichainPrep["prevHash"];

    zkinMultichainPrep["nBlocksSha256"] = to_string(nBlocksSha256);
    for(uint64_t i = 0; i < 8; ++i) {
            zkinMultichainPrep["outHash"][i] = to_string(outHash[i]);
    }
    
    // Output is pProverAggregationRequest->multichainPrepProofOutput (of type json)
    pProverAggregationRequest->multichainPrepProofOutput = zkinMultichainPrep;

    json hashInfo = json::object();
    hashInfo["nPrevBlocks"] = outHashInfo["nBlocks"];
    hashInfo["prevHash"] = outHashInfo["outHash"];

    pProverAggregationRequest->multichainPrepHashOutput = hashInfo;

    // Save output to file
    if (config.saveOutputToFile)
    {
        json2file(pProverAggregationRequest->multichainPrepProofOutput, pProverAggregationRequest->filePrefix + "prepare_multichain_proof.output.json");
        json2file(pProverAggregationRequest->multichainPrepHashOutput, pProverAggregationRequest->filePrefix + "prepare_multichain_out_hash.output.json");

    }

    // Save proof to file
    if (config.saveProofToFile)
    {
        jProofMultichainPrep["nPrevBlocksSha256"] = zkinInputMultichainPrep["nPrevBlocksSha256"];
        jProofMultichainPrep["prevHash"] = zkinInputMultichainPrep["prevHash"];
        json2file(jProofMultichainPrep, pProverAggregationRequest->filePrefix + "prepare_multichain_proof.proof.json");
    }

    json publicsJson = json::array();

    publicsJson[0] = to_string(nBlocksSha256);
    for (uint64_t i = 0; i < 8; i++)
    {
        publicsJson[i + 1] = to_string(outHashInfo["outHash"][i]);
    }
    publicsJson[9] = zkinInputMultichainPrep["nPrevBlocksSha256"];
    for (uint64_t i = 0; i < 8; i++)
    {
        publicsJson[i + 10] = zkinInputMultichainPrep["prevHash"][i];
    }

    // Add the multichainAgg verification key
    publicsJson[18] = to_string(verKeyMultichainAgg["constRoot"][0]);
    publicsJson[19] = to_string(verKeyMultichainAgg["constRoot"][1]);
    publicsJson[20] = to_string(verKeyMultichainAgg["constRoot"][2]);
    publicsJson[21] = to_string(verKeyMultichainAgg["constRoot"][3]);
   
    json2file(publicsJson, pProverAggregationRequest->publicsOutputFile());

    TimerStopAndLog(SAVE_PROOF);

    pProverAggregationRequest->result = ZKR_SUCCESS;

    TimerStopAndLog(PROVER_PREPARE_MULTICHAIN_PROOF);
}

void ProverAggregation::genAggregatedMultichainProof(ProverAggregationRequest *pProverAggregationRequest)
{

    zkassert(config.generateProofAggregation());
    zkassert(pProverAggregationRequest != NULL);
    zkassert(pProverAggregationRequest->type == prt_genAggregatedMultichainProof);

    TimerStart(PROVER_AGGREGATED_MULTICHAIN_PROOF);

    printMemoryInfo(true);
    printProcessInfo(true);

    // Save input to file
    if (config.saveInputToFile)
    {
        json2file(pProverAggregationRequest->aggregatedMultichainProofInput1, pProverAggregationRequest->filePrefix + "aggregated_multichain_proof.input_1.json");
        json2file(pProverAggregationRequest->aggregatedMultichainProofInput2, pProverAggregationRequest->filePrefix + "aggregated_multichain_proof.input_2.json");
    }

    // Input is pProverAggregationRequest->aggregatedMultichainProofInput1 and pProverAggregationRequest->aggregatedMultichainProofInput2 (of type json)

    ordered_json verKeyMultichainAgg;
    file2json(config.multichainAggVerkey, verKeyMultichainAgg);

    // ----------------------------------------------
    // CHECKS
    // ----------------------------------------------
    // Check prev number of blocks sha256

    if (pProverAggregationRequest->aggregatedMultichainProofInput1["nBlocksSha256"] != pProverAggregationRequest->aggregatedMultichainProofInput2["nPrevBlocksSha256"])
    {
        zklog.error("ProverAggregation::genAggregatedMultiChainProof() nBlocksSha256 and nPrevBlocksSha256 are not consistent " + pProverAggregationRequest->aggregatedMultichainProofInput1["nBlocksSha256"].dump() + "!=" + pProverAggregationRequest->aggregatedMultichainProofInput2["nPrevBlocksSha256"].dump());
        pProverAggregationRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
        return;
    }

    // Check intermediate sha256 value
    for (int i = 0; i < 8; i++)
    {
        if (pProverAggregationRequest->aggregatedMultichainProofInput1["outHash"][i] != pProverAggregationRequest->aggregatedMultichainProofInput2["prevHash"][i])
        {
            zklog.error("ProverAggregation::genAggregatedProof() The outHash and the prevHash are not consistent " + pProverAggregationRequest->aggregatedMultichainProofInput1["outHash"][i].dump() + "!=" + pProverAggregationRequest->aggregatedMultichainProofInput1["prevHash"][i].dump());
            pProverAggregationRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
            return;
        }
    }
    
    json zkinInputAggregated = joinzkinmultichain(pProverAggregationRequest->aggregatedMultichainProofInput1, pProverAggregationRequest->aggregatedMultichainProofInput2, verKeyMultichainAgg, starksMultichainAgg->starkInfo.starkStruct.steps.size());

    Goldilocks::Element multichainAggVerkeyValues[4];
    multichainAggVerkeyValues[0] = Goldilocks::fromU64(verKeyMultichainAgg["constRoot"][0]);
    multichainAggVerkeyValues[1] = Goldilocks::fromU64(verKeyMultichainAgg["constRoot"][1]);
    multichainAggVerkeyValues[2] = Goldilocks::fromU64(verKeyMultichainAgg["constRoot"][2]);
    multichainAggVerkeyValues[3] = Goldilocks::fromU64(verKeyMultichainAgg["constRoot"][3]);

    Goldilocks::Element publics[22];

    publics[0] = Goldilocks::fromString(pProverAggregationRequest->aggregatedMultichainProofInput2["nBlocksSha256"]);

    for (uint64_t i = 0; i < 8; i++)
    {
        publics[i + 1] = Goldilocks::fromString(pProverAggregationRequest->aggregatedMultichainProofInput2["outHash"][i]);
    }

    publics[9] = Goldilocks::fromString(pProverAggregationRequest->aggregatedMultichainProofInput1["nPrevBlocksSha256"]);

    for (uint64_t i = 0; i < 8; i++)
    {
        publics[i + 10] = Goldilocks::fromString(pProverAggregationRequest->aggregatedMultichainProofInput1["prevHash"][i]);
    }

    for (uint64_t i = 0; i < verKeyMultichainAgg["constRoot"].size(); i++)
    {
        publics[i + 18] = Goldilocks::fromU64(verKeyMultichainAgg["constRoot"][i]);
    }

    CommitPolsStarks cmPolsAgg(pAddress, (1 << starksMultichainAgg->starkInfo.starkStruct.nBits), starksMultichainAgg->starkInfo.nCm1);
    CircomMultichainAgg::getCommitedPols(&cmPolsAgg, config.multichainAggVerifier, config.multichainAggExec, zkinInputAggregated, (1 << starksMultichainAgg->starkInfo.starkStruct.nBits), starksMultichainAgg->starkInfo.nCm1);

    // void *pointerCmMultichainAggPols = mapFile("config/multichainAgg/multichainAgg.commit", cmPolsAgg.size(), true);
    // memcpy(pointerCmMultichainAggPols, cmPolsAgg.address(), cmPolsAgg.size());
    // unmapFile(pointerCmMultichainAggPols, cmPolsAgg.size());

    //-------------------------------------------
    // Generate Multichain Agg proof
    //-------------------------------------------

    TimerStart(STARK_MULTICHAIN_AGGREGATED_PROOF);
    uint64_t polBitsMultichainAgg = starksMultichainAgg->starkInfo.starkStruct.steps[starksMultichainAgg->starkInfo.starkStruct.steps.size() - 1].nBits;
    FRIProof fproofMultichainAgg((1 << polBitsMultichainAgg), FIELD_EXTENSION, starksMultichainAgg->starkInfo.starkStruct.steps.size(), starksMultichainAgg->starkInfo.evMap.size(), starksMultichainAgg->starkInfo.nPublics);
    MultichainAggSteps multichainAggSteps;
    starksMultichainAgg->genProof(fproofMultichainAgg, publics, multichainAggVerkeyValues, &multichainAggSteps);
    TimerStopAndLog(STARK_MULTICHAIN_AGGREGATED_PROOF);

    // Save the proof & zkinproof
    TimerStart(SAVE_PROOF);

    nlohmann::ordered_json jProofMultichainAgg = fproofMultichainAgg.proofs.proof2json();
    nlohmann::ordered_json zkinMultichainAgg = proof2zkinStark(jProofMultichainAgg);
    zkinMultichainAgg["nBlocksSha256"] = zkinInputAggregated["b_nBlocksSha256"];
    zkinMultichainAgg["outHash"] = zkinInputAggregated["b_outHash"];
    zkinMultichainAgg["nPrevBlocksSha256"] = zkinInputAggregated["a_nPrevBlocksSha256"];
    zkinMultichainAgg["prevHash"] = zkinInputAggregated["a_prevHash"];

    // Output is pProverAggregationRequest->aggregatedMultichainProofOutput (of type json)
    pProverAggregationRequest->aggregatedMultichainProofOutput = zkinMultichainAgg;

    // Save output to file
    if (config.saveOutputToFile)
    {
        json2file(pProverAggregationRequest->aggregatedMultichainProofOutput, pProverAggregationRequest->filePrefix + "aggregated_multichain_proof.output.json");
    }
    // Save proof to file
    if (config.saveProofToFile)
    {
        jProofMultichainAgg["nBlocksSha256"] = zkinInputAggregated["nBlocksSha256"];
        jProofMultichainAgg["outHash"] = zkinInputAggregated["outHash"];
        jProofMultichainAgg["nPrevBlocksSha256"] = zkinInputAggregated["nPrevBlocksSha256"];
        jProofMultichainAgg["prevHash"] = zkinInputAggregated["prevHash"];
        json2file(jProofMultichainAgg, pProverAggregationRequest->filePrefix + "aggregated_multichain_proof.proof.json");
    }

    json publicsJson = json::array();

    publicsJson[0] = zkinInputAggregated["b_nBlocksSha256"];
    for (uint64_t i = 0; i < 8; i++)
    {
        publicsJson[i + 1] = zkinInputAggregated["b_outHash"][i];
    }
    publicsJson[9] = zkinInputAggregated["a_nPrevBlocksSha256"];
    for (uint64_t i = 0; i < 8; i++)
    {
        publicsJson[i + 10] = zkinInputAggregated["a_prevHash"][i];
    }

    // Add the multichainAgg verification key
    publicsJson[18] = to_string(verKeyMultichainAgg["constRoot"][0]);
    publicsJson[19] = to_string(verKeyMultichainAgg["constRoot"][1]);
    publicsJson[20] = to_string(verKeyMultichainAgg["constRoot"][2]);
    publicsJson[21] = to_string(verKeyMultichainAgg["constRoot"][3]);
   
    json2file(publicsJson, pProverAggregationRequest->publicsOutputFile());

    TimerStopAndLog(SAVE_PROOF);

    pProverAggregationRequest->result = ZKR_SUCCESS;

    TimerStopAndLog(PROVER_AGGREGATED_MULTICHAIN_PROOF);
}


void ProverAggregation::genFinalMultichainProof(ProverAggregationRequest *pProverAggregationRequest)
{
    zkassert(config.generateProofAggregation());
    zkassert(pProverAggregationRequest != NULL);
    zkassert(pProverAggregationRequest->type == prt_genFinalMultichainProof);

    TimerStart(PROVER_MULTICHAIN_FINAL_PROOF);

    printMemoryInfo(true);
    printProcessInfo(true);

    // Save input to file
    if (config.saveInputToFile)
    {
        json2file(pProverAggregationRequest->finalMultichainProofInput, pProverAggregationRequest->filePrefix + "final_proof.input.json");
    }

    json zkinMultichainFinal = pProverAggregationRequest->finalMultichainProofInput;

    Goldilocks::Element publics[starksMultichainAggF->starkInfo.nPublics];

    publics[0] = Goldilocks::fromString(zkinMultichainFinal["nBlocksSha256"]);
    for (uint64_t i = 0; i < 8; i++)
    {
        publics[i + 1] = Goldilocks::fromString(zkinMultichainFinal["outHash"][i]);
    }

    CommitPolsStarks cmPolsMultichainAggF(pAddressStarksMultichainAggF, (1 << starksMultichainAggF->starkInfo.starkStruct.nBits), starksMultichainAggF->starkInfo.nCm1);
    CircomMultichainAggF::getCommitedPols(&cmPolsMultichainAggF, config.multichainAggFVerifier, config.multichainAggFExec, zkinMultichainFinal, (1 << starksMultichainAggF->starkInfo.starkStruct.nBits), starksMultichainAggF->starkInfo.nCm1);

    // void *pointercmPolsMultichainAggF = mapFile("config/multichainAggF/multichainAggF.commit", cmPolsMultichainAggF.size(), true);
    // memcpy(pointercmPolsMultichainAggF, cmPolsMultichainAggF.address(), cmPolsMultichainAggF.size());
    // unmapFile(pointercmPolsMultichainAggF, cmPolsMultichainAggF.size());

    //  ----------------------------------------------
    //  Generate Multichain Aggregation Final proof
    //  ----------------------------------------------

    TimerStart(STARK_MULTICHAIN_AGG_F_PROOF);
    MultichainAggFSteps multichainAggFsteps;
    uint64_t polBitsMultichainAggF = starksMultichainAggF->starkInfo.starkStruct.steps[starksMultichainAggF->starkInfo.starkStruct.steps.size() - 1].nBits;
    FRIProofC12 fproofMultichainAggF((1 << polBitsMultichainAggF), FIELD_EXTENSION, starksMultichainAggF->starkInfo.starkStruct.steps.size(), starksMultichainAggF->starkInfo.evMap.size(), starksMultichainAggF->starkInfo.nPublics);
    starksMultichainAggF->genProof(fproofMultichainAggF, publics, &multichainAggFsteps);
    TimerStopAndLog(STARK_MULTICHAIN_AGG_F_PROOF);

    // Save the proof & zkinproof
    nlohmann::ordered_json jProofMultichainAggF = fproofMultichainAggF.proofs.proof2json();
    json zkinMultichainAggF = proof2zkinStark(jProofMultichainAggF);
 
    zkinMultichainAggF["nBlocksSha256"] = zkinMultichainFinal["nBlocksSha256"];
    zkinMultichainAggF["outHash"] = zkinMultichainFinal["outHash"];
    
    zkinMultichainAggF["aggregatorAddr"] = pProverAggregationRequest->aggregatorAddress;
    
    // Save proof to file
    if (config.saveProofToFile)
    {
        jProofMultichainAggF["nBlocksSha256"] = zkinMultichainAggF["nBlocksSha256"];
        jProofMultichainAggF["outHash"] = zkinMultichainAggF["outHash"];
        json2file(jProofMultichainAggF, pProverAggregationRequest->filePrefix + "multichainAggF.proof.json");
        json2file(zkinMultichainAggF, pProverAggregationRequest->filePrefix + "multichainAggF.proof.output.json");
    }

    //  ----------------------------------------------
    //  Verifier Multichain final
    //  ----------------------------------------------
    
    TimerStart(CIRCOM_LOAD_CIRCUIT_MULTICHAIN_FINAL);
    CircomMultichainFinal::Circom_Circuit *circuitMultichainFinal = CircomMultichainFinal::loadCircuit(config.multichainFinalVerifier);
    TimerStopAndLog(CIRCOM_LOAD_CIRCUIT_MULTICHAIN_FINAL);

    TimerStart(CIRCOM_MULTICHAIN_FINAL_LOAD_JSON);
    CircomMultichainFinal::Circom_CalcWit *ctxMultichainFinal = new CircomMultichainFinal::Circom_CalcWit(circuitMultichainFinal);

    CircomMultichainFinal::loadJsonImpl(ctxMultichainFinal, zkinMultichainAggF);
    if (ctxMultichainFinal->getRemaingInputsToBeSet() != 0)
    {
        zklog.error("ProverAggregation::genProof() Not all inputs have been set. Only " + to_string(CircomMultichainFinal::get_main_input_signal_no() - ctxMultichainFinal->getRemaingInputsToBeSet()) + " out of " + to_string(CircomMultichainFinal::get_main_input_signal_no()));
        exitProcess();
    }
    TimerStopAndLog(CIRCOM_MULTICHAIN_FINAL_LOAD_JSON);

    TimerStart(CIRCOM_GET_BIN_WITNESS_MULTICHAIN_FINAL);
    AltBn128::FrElement *pWitnessMultichainFinal = NULL;
    uint64_t witnessSizeMultichainFinal = 0;
    CircomMultichainFinal::getBinWitness(ctxMultichainFinal, pWitnessMultichainFinal, witnessSizeMultichainFinal);
    CircomMultichainFinal::freeCircuit(circuitMultichainFinal);
    delete ctxMultichainFinal;

    TimerStopAndLog(CIRCOM_GET_BIN_WITNESS_MULTICHAIN_FINAL);

    TimerStart(SAVE_PUBLICS_JSON);
    // Save public file
    json publicJson;
    AltBn128::FrElement aux;
    AltBn128::Fr.toMontgomery(aux, pWitnessMultichainFinal[1]);
    
    publicJson[0] = AltBn128::Fr.toString(aux);
    json2file(publicJson, pProverAggregationRequest->publicsOutputFile());
    TimerStopAndLog(SAVE_PUBLICS_JSON);

    if (Zkey::GROTH16_PROTOCOL_ID != protocolId)
    {
        TimerStart(RAPID_SNARK);
        try
        {
            auto [jsonProof, publicSignalsJson] = prover->prove(pWitnessMultichainFinal);
            // Save proof to file
            if (config.saveProofToFile)
            {
                json2file(jsonProof, pProverAggregationRequest->filePrefix + "multichain_final_proof.proof.json");
            }
            TimerStopAndLog(RAPID_SNARK);

            pProverAggregationRequest->proof.load(jsonProof, publicSignalsJson);
            pProverAggregationRequest->result = ZKR_SUCCESS;
        }
        catch (std::exception &e)
        {
            zklog.error("ProverAggregation::genProof() got exception in rapid SNARK:" + string(e.what()));
            exitProcess();
        }
    }
    else
    {
        // Generate Groth16 via rapid SNARK
        TimerStart(GROTH_16);
        json jsonProof;
        try
        {
            auto proof = groth16Prover->prove(pWitnessMultichainFinal);
            jsonProof = proof->toJson();
        }
        catch (std::exception &e)
        {
            zklog.error("ProverAggregation::genProof() got exception in rapid SNARK:" + string(e.what()));
            exitProcess();
        }
        TimerStopAndLog(GROTH_16);

        // Save proof to file
        if (config.saveProofToFile)
        {
            json2file(jsonProof, pProverAggregationRequest->filePrefix + "multichain_final_proof.proof.json");
        }
        // Populate Proof with the correct data
        pProverAggregationRequest->proof.load(jsonProof, publicJson);
        pProverAggregationRequest->result = ZKR_SUCCESS;
    }

    /***********/
    /* Cleanup */
    /***********/
    free(pWitnessMultichainFinal);

    TimerStopAndLog(PROVER_MULTICHAIN_FINAL_PROOF);
}
