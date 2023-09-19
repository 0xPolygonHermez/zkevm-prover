#ifndef PROVER_REQUEST_HPP
#define PROVER_REQUEST_HPP

#include <semaphore.h>
#include <unordered_set>
#include "input.hpp"
#include "proof_fflonk.hpp"
#include "counters.hpp"
#include "full_tracer_interface.hpp"
#include "database_map.hpp"
#include "prover_request_type.hpp"

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

class ProverRequest
{
public:
    Goldilocks &fr;
    const Config &config;
private:
    sem_t completedSem; // Semaphore to wakeup waiting thread when the request is completed

public:
    /* IDs */
    string uuid;
    string contextId; // Externally provided context ID
    vector<LogTag> tags; // Tags used in logs
    string timestamp; // Timestamp, when requested, used as a prefix in the output files
    time_t startTime; // Time when the request started being processed
    time_t endTime; // Time when the request ended

    /* Output files prefix */
    string filePrefix;

    /* Prrover request type */
    tProverRequestType type;

    /* Input batch L2 data for processBatch, genProof and genBatchProof; */
    Input input;

    /* Flush ID and last sent flush ID, to track when data reaches DB, i.e. batch trusted state */
    uint64_t  flushId;
    uint64_t  lastSentFlushId;

    /* genBatchProof output */
    nlohmann::ordered_json batchProofOutput;

    /* genAggregatedProof input and output */
    nlohmann::ordered_json aggregatedProofInput1;
    nlohmann::ordered_json aggregatedProofInput2;
    nlohmann::ordered_json aggregatedProofOutput;

    /* genFinalProof input */
    nlohmann::ordered_json finalProofInput;

    /* genProof and genFinalProof output */
    Proof proof;

    /* Execution generated data */
    Counters counters; // Counters of the batch execution
    DatabaseMap *dbReadLog; // Database reads logs done during the execution (if enabled)
    FullTracerInterface * pFullTracer; // Execution traces interface

    /* State */
    bool bCompleted;
    bool bCancelling; // set to true to request to cancel this request

    /* Result */
    zkresult result;

    /* Executor EVM events */
    vector<string> receipts;
    vector<string> logs;

    /* Keys */
    unordered_set<string> nodesKeys;
    unordered_set<string> programKeys;

    /* Constructor */
    ProverRequest (Goldilocks &fr, const Config &config, tProverRequestType type);
    ~ProverRequest();

    void CreateFullTracer(void);
    void DestroyFullTracer(void);

    /* Output file names */
    string proofFile (void);
    string inputFile (void);
    string inputDbFile (void);
    string publicsOutputFile (void);

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

    static void onDBReadLogChangeCallback(void *p, DatabaseMap *dbMap)
    {
        ((ProverRequest *)p) -> onDBReadLogChange(dbMap);
    }

    void onDBReadLogChange(DatabaseMap *dbMap);
};

#endif