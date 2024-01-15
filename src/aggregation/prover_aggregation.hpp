#ifndef PROVER_AGGREGATION_HPP
#define PROVER_AGGREGATION_HPP

#include <map>
#include <pthread.h>
#include <semaphore.h>
#include "goldilocks_base_field.hpp"
#include "proof_fflonk.hpp"
#include "alt_bn128.hpp"
#include "groth16.hpp"
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "prover_aggregation_request.hpp"
#include "poseidon_goldilocks.hpp"
#include "sm/pols_generated/constant_pols.hpp"


#include "starkRecursiveF.hpp"
#include "starkpil/stark_info.hpp"
#include "starks.hpp"
#include "constant_pols_starks.hpp"
#include "fflonk_prover.hpp"

class ProverAggregation
{
    Goldilocks &fr;
    PoseidonGoldilocks &poseidon;

    Starks *starksMultichainPrep;
    Starks *starksMultichainAgg;

    StarkRecursiveF *starksMultichainAggF;

    Fflonk::FflonkProver<AltBn128::Engine> *prover;
    std::unique_ptr<Groth16::Prover<AltBn128::Engine>> groth16Prover;
    std::unique_ptr<BinFileUtils::BinFile> zkey;
    std::unique_ptr<ZKeyUtils::Header> zkeyHeader;
    mpz_t altBbn128r;

public:
    unordered_map<string, ProverAggregationRequest *> requestsMap; // Map uuid -> ProveRequest pointer

    vector<ProverAggregationRequest *> pendingRequests;   // Queue of pending requests
    ProverAggregationRequest *pCurrentRequest;            // Request currently being processed by the prover thread in server mode
    vector<ProverAggregationRequest *> completedRequests; // Map uuid -> ProveRequest pointer

private:
    pthread_t proverPthread;  // Prover thread
    pthread_t cleanerPthread; // Garbage collector
    pthread_mutex_t mutex;    // Mutex to protect the requests queues
    void *pAddress = NULL;
    void *pAddressStarksMultichainAggF = NULL;
    int protocolId;
public:
    const Config &config;
    sem_t pendingRequestSem; // Semaphore to wakeup prover thread when a new request is available
    string lastComputedRequestId;
    uint64_t lastComputedRequestEndTime;

    uint64_t polsSize;
    uint64_t polsSizeMultichainAggF;

    ProverAggregation(Goldilocks &fr,
           PoseidonGoldilocks &poseidon,
           const Config &config);

    ~ProverAggregation();

    void calculateHash(ProverAggregationRequest *pProverAggregationRequest);
    void genPrepareMultichainProof(ProverAggregationRequest *pProverAggregationRequest);
    void genAggregatedMultichainProof(ProverAggregationRequest *pProverAggregationRequest);
    void genFinalMultichainProof(ProverAggregationRequest *pProverAggregationRequest);
   
    string submitRequest(ProverAggregationRequest *pProverAggregationRequest);                                          // returns UUID for this request
    ProverAggregationRequest *waitForRequestToComplete(const string &uuid, const uint64_t timeoutInSeconds); // wait for the request with this UUID to complete; returns NULL if UUID is invalid

    void lock(void) { pthread_mutex_lock(&mutex); };
    void unlock(void) { pthread_mutex_unlock(&mutex); };
};

void *proverAggregationThread(void *arg);
void *cleanerAggregationThread(void *arg);

#endif