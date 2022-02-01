#ifndef PROVER_HPP
#define PROVER_HPP

#include <map>
#include <pthread.h>
#include <semaphore.h>
#include "ffiasm/fr.hpp"
#include "input.hpp"
#include "rom.hpp"
#include "executor.hpp"
#include "script.hpp"
#include "proof.hpp"
#include "alt_bn128.hpp"
#include "groth16.hpp"
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "prover_request.hpp"

class Prover
{
    RawFr &fr;
    const Rom &romData;
    Executor executor;
    const Script &script;
    const Pil &pil;
    const Pols &constPols;
    const Config &config;

    std::unique_ptr<Groth16::Prover<AltBn128::Engine>> groth16Prover;
    std::unique_ptr<BinFileUtils::BinFile> zkey;
    std::unique_ptr<ZKeyUtils::Header> zkeyHeader;
    mpz_t altBbn128r;

    Reference constRefs[NCONSTPOLS];

public:
    map< string, ProverRequest * > requestsMap; // Map uuid -> ProveRequest pointer
    vector< ProverRequest * > pendingRequests; // Queue of pending requests
    sem_t pendingRequestSem; // Semaphore to wakeup prover thread when a new request is available
    ProverRequest * pCurrentRequest;
    vector< ProverRequest * > completedRequests; // Map uuid -> ProveRequest pointer

private:
    pthread_t t;

public:
    Prover( RawFr &fr,
            const Rom &romData,
            const Script &script,
            const Pil &pil,
            const Pols &constPols,
            const Config &config ) ;

    ~Prover();

    void prove (ProverRequest * pProverRequest);
    string submitRequest (ProverRequest * pProvefRequest); // returns UUID for this request
    ProverRequest * waitForRequestToComplete (const string & uuid); // wait for the request with this UUID to complete; returns NULL if UUID is invalid
};

void* proverThread(void* arg);

#endif