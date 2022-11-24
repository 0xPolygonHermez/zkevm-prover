#ifndef PROVER_HPP
#define PROVER_HPP

#include <map>
#include <pthread.h>
#include <semaphore.h>
#include "goldilocks_base_field.hpp"
#include "input.hpp"
#include "rom.hpp"
#include "proof.hpp"
#include "alt_bn128.hpp"
#include "groth16.hpp"
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "prover_request.hpp"
#include "poseidon_goldilocks.hpp"
#include "executor/executor.hpp"
#include "sm/pols_generated/constant_pols.hpp"
#include "stark.hpp"
#include "starkC12a.hpp"
#include "starkRecursive1.hpp"
#include "starkRecursive2.hpp"
#include "starkRecursiveF.hpp"
#include "starkpil/stark_info.hpp"
#include "starks.hpp"

class Prover
{
    Goldilocks &fr;
    PoseidonGoldilocks &poseidon;
    Executor executor;
    Stark stark;
    StarkC12a starkC12a;
    StarkRecursive1 starkRecursive1;
    StarkRecursive2 starkRecursive2;
    StarkRecursiveF starkRecursiveF;

    // Starks<ConstantPolsC12a> starksC12a;
    // Starks<ConstantPolsRecursive1> starksRecursive1;
    // Starks<ConstantPolsRecursive2> starksRecursive2;

    std::unique_ptr<Groth16::Prover<AltBn128::Engine>> groth16Prover;
    std::unique_ptr<BinFileUtils::BinFile> zkey;
    std::unique_ptr<ZKeyUtils::Header> zkeyHeader;
    mpz_t altBbn128r;

public:
    unordered_map<string, ProverRequest *> requestsMap; // Map uuid -> ProveRequest pointer

    vector<ProverRequest *> pendingRequests;   // Queue of pending requests
    ProverRequest *pCurrentRequest;            // Request currently being processed by the prover thread in server mode
    vector<ProverRequest *> completedRequests; // Map uuid -> ProveRequest pointer

private:
    pthread_t proverPthread;  // Prover thread
    pthread_t cleanerPthread; // Garbage collector
    pthread_mutex_t mutex;    // Mutex to protect the requests queues

public:
    const Config &config;
    sem_t pendingRequestSem; // Semaphore to wakeup prover thread when a new request is available
    string lastComputedRequestId;
    uint64_t lastComputedRequestEndTime;

    Prover(Goldilocks &fr,
           PoseidonGoldilocks &poseidon,
           const Config &config);

    ~Prover();

    void genBatchProof(ProverRequest *pProverRequest);
    void genAggregatedProof(ProverRequest *pProverRequest);
    void genFinalProof(ProverRequest *pProverRequest);
    void processBatch(ProverRequest *pProverRequest);
    void execute(ProverRequest *pProverRequest);
    
    string submitRequest(ProverRequest *pProverRequest);                                          // returns UUID for this request
    ProverRequest *waitForRequestToComplete(const string &uuid, const uint64_t timeoutInSeconds); // wait for the request with this UUID to complete; returns NULL if UUID is invalid

    void lock(void) { pthread_mutex_lock(&mutex); };
    void unlock(void) { pthread_mutex_unlock(&mutex); };
};

void *proverThread(void *arg);
void *cleanerThread(void *arg);

#endif