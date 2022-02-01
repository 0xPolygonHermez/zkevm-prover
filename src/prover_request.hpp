#ifndef PROVER_REQUEST_HPP
#define PROVER_REQUEST_HPP

#include <semaphore.h>
#include "input.hpp"
#include "proof.hpp"
#include "counters.hpp"

class ProverRequest
{
private:
    RawFr &fr;
    sem_t completedSem; // Semaphore to wakeup waiting thread when the request is completed

public:
    /* IDs */
    string uuid;
    string timestamp;
    
    /* Files */
    string inputFile;
    string inputFileEx;
    string publicFile;
    string proofFile;

    /* Executor */
    Input input;
    Database db;
    Counters counters;

    /* Result */
    Proof proof;
    bool bCompleted;

    /* Constructor */
    ProverRequest (RawFr &fr) : fr(fr), input(fr), db(fr), bCompleted(false)
    {
        sem_init(&completedSem, 0, 0);
    }

    /* Init, to be called before Prover::prove() */
    void init (const Config &config);

    /* Block until completed */
    void waitForCompleted (void)
    {
        if (bCompleted) return;
        sem_wait(&completedSem);
    }
    
    /* Unblock waiter thread */
    void notifyCompleted (void)
    {
        bCompleted = true;
        sem_post(&completedSem);
    }
};

#endif