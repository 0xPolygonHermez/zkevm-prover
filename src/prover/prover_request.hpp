#ifndef PROVER_REQUEST_HPP
#define PROVER_REQUEST_HPP

#include <semaphore.h>
#include "input.hpp"
#include "proof.hpp"
#include "counters.hpp"
#include "full_tracer.hpp"

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
    string filePrefix;
    string inputFile;
    string inputFileEx;
    string publicFile;
    string proofFile;

    /* Executor */
    Input input;
    Counters counters;

    /* Process Batch */
    bool bProcessBatch;
    bool bUpdateMerkleTree; // only used if bProcessBatch
    string txHashToGenerateExecuteTrace; // only used if bProcessBatch
    string txHashToGenerateCallTrace; // only used if bProcessBatch
    bool bNoCounters; // set to true if counters should not be used

    /* Full tracer */
    FullTracer fullTracer;

    /* State */
    Proof proof;
    bool bCompleted;
    bool bCancelling; // set to true to request to cancel this request

    /* Result */
    zkresult result;

    /* Executor EVM events */
    vector<string> receipts;
    vector<string> logs;

    /* Constructor */
    ProverRequest (Goldilocks &fr) :
        fr(fr),
        startTime(0),
        endTime(0),
        input(fr),
        bProcessBatch(false),
        bUpdateMerkleTree(true),
        bNoCounters(false),
        fullTracer(fr),
        bCompleted(false),
        bCancelling(false),
        result(ZKR_UNSPECIFIED)
    {
        sem_init(&completedSem, 0, 0);
    }

    /* Init, to be called before Prover::prove() */
    void init (const Config &config, bool bExecutor); // bExecutor must be true if this is a process batch request; false if a proof
    
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

    /* Generate FullTracer call traces if true */
    bool generateCallTraces (void)
    {
        return (txHashToGenerateExecuteTrace.size() > 0) || (txHashToGenerateCallTrace.size() > 0);
    }
};

#endif