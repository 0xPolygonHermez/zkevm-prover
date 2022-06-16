#ifndef PROVER_REQUEST_HPP
#define PROVER_REQUEST_HPP

#include <semaphore.h>
#include "input.hpp"
#include "proof.hpp"
#include "counters.hpp"

class ProverRequest
{
private:
    Goldilocks &fr;
    sem_t completedSem; // Semaphore to wakeup waiting thread when the request is completed

public:
    /* IDs */
    string uuid;
    string timestamp; // Timestamp, when requested, used as a prefix in the output files
    time_t startTime; // Time when the request started being processed
    time_t endTime; // Time when the request ended
    
    /* Files */
    string inputFile;
    string inputFileEx;
    string publicFile;
    string proofFile;

    /* Executor */
    Input input;
    Database db;
    Counters counters;

    /* Process Batch */
    bool bProcessBatch;
    bool bUpdateMerkleTree; // only used if bProcessBatch
    bool bGenerateExecuteTrace; // only used if bProcessBatch
    bool bGenerateCallTrace; // only used if bProcessBatch

    /* Result */
    Proof proof;
    bool bCompleted;
    bool bCancelling; // set to true to request to cancel this request

    /* Executor EVM events */
    vector<string> receipts;
    vector<string> logs;

    /* Constructor */
    ProverRequest (Goldilocks &fr) :
        fr(fr),
        startTime(0),
        endTime(0),
        input(fr),
        db(fr),
        bCompleted(false),
        bCancelling(false),
        bProcessBatch(false),
        bUpdateMerkleTree(false),
        bGenerateExecuteTrace(false),
        bGenerateCallTrace(false)
    {
        sem_init(&completedSem, 0, 0);
    }

    /* Init, to be called before Prover::prove() */
    void init (const Config &config);

    /* Block until completed */
    void waitForCompleted (const uint64_t timeoutInSeconds)
    {
        if (bCompleted) return;
        timespec t;
        t.tv_sec = timeoutInSeconds;
        t.tv_nsec = 0;
        sem_timedwait(&completedSem, &t);
    }
    
    /* Unblock waiter thread */
    void notifyCompleted (void)
    {
        bCompleted = true;
        sem_post(&completedSem);
    }
};

#endif