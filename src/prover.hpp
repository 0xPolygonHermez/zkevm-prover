#ifndef PROVER_HPP
#define PROVER_HPP

#include <map>
#include <pthread.h>
#include <semaphore.h>
#include "goldilocks/goldilocks_base_field.hpp"
#include "input.hpp"
#include "rom.hpp"
#include "script.hpp"
#include "proof.hpp"
#include "alt_bn128.hpp"
#include "groth16.hpp"
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "prover_request.hpp"
#include "poseidon_opt/poseidon_goldilocks.hpp"
#include "sm/executor/executor.hpp"
#include "sm/pols_generated/constant_pols.hpp"
#include "pil.hpp"

class Prover
{
    Goldilocks &fr;
    Poseidon_goldilocks &poseidon;
    const Rom &romData;
    Executor executor;

    const Script &script;
    const Pil &pil;
    const ConstantPols &constPols;

    std::unique_ptr<Groth16::Prover<AltBn128::Engine>> groth16Prover;
    std::unique_ptr<BinFileUtils::BinFile> zkey;
    std::unique_ptr<ZKeyUtils::Header> zkeyHeader;
    mpz_t altBbn128r;

    Reference constRefs[NCONSTPOLS];

public:
    map< string, ProverRequest * > requestsMap; // Map uuid -> ProveRequest pointer
    
    vector< ProverRequest * > pendingRequests; // Queue of pending requests
    ProverRequest * pCurrentRequest; // Request currently being processed by the prover thread in server mode
    vector< ProverRequest * > completedRequests; // Map uuid -> ProveRequest pointer

private:
    pthread_t proverPthread; // Prover thread
    pthread_t cleanerPthread; // Garbage collector
    pthread_mutex_t mutex; // Mutex to protect the requests queues

public:
    const Config &config;
    sem_t pendingRequestSem; // Semaphore to wakeup prover thread when a new request is available
    string lastComputedRequestId;
    uint64_t lastComputedRequestEndTime;

    Prover( Goldilocks &fr,
            Poseidon_goldilocks &poseidon,
            const Rom &romData,
            const Script &script,
            const Pil &pil,
            const ConstantPols &constPols,
            const Config &config ) ;

    ~Prover();

    void prove (ProverRequest * pProverRequest);
    void execute (ProverRequest * pProverRequest);
    void processBatch (ProverRequest * pProverRequest);
    string submitRequest (ProverRequest * pProverRequest); // returns UUID for this request
    ProverRequest * waitForRequestToComplete (const string & uuid, const uint64_t timeoutInSeconds); // wait for the request with this UUID to complete; returns NULL if UUID is invalid
    
    void lock (void) { pthread_mutex_lock(&mutex); };
    void unlock (void) { pthread_mutex_unlock(&mutex); };
};

void* proverThread(void* arg);
void* cleanerThread(void* arg);

#endif