#ifndef PROVER_AGGREGATION_REQUEST_HPP
#define PROVER_AGGREGATION_REQUEST_HPP

#include <semaphore.h>
#include <unordered_set>
#include "proof_fflonk.hpp"
#include "prover_aggregation_request_type.hpp"
#include "zkresult.hpp"
#include "config.hpp"
#include "goldilocks_base_field.hpp"

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

class ProverAggregationRequest
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
    string timestamp; // Timestamp, when requested, used as a prefix in the output files
    time_t startTime; // Time when the request started being processed
    time_t endTime; // Time when the request ended

    /* Output files prefix */
    string filePrefix;

    /* Prover request type */
    tProverAggregationRequestType type;

    /* AggregatorAddress */
    string aggregatorAddress;

    /* calculateHash input and output */
    nlohmann::ordered_json prevHashInput;
    nlohmann::ordered_json chainPublicsInput;
    nlohmann::ordered_json hashOutput;
 
    /* genPrepMultichainProof input and output */
    nlohmann::ordered_json multichainPrepProofInput;
    nlohmann::ordered_json multichainPrepPrevHashInput;
    nlohmann::ordered_json multichainPrepProofOutput;
    nlohmann::ordered_json multichainPrepHashOutput;


    /* genAggregatedMultichainProof input and output */
    nlohmann::ordered_json aggregatedMultichainProofInput1;
    nlohmann::ordered_json aggregatedMultichainProofInput2;
    nlohmann::ordered_json aggregatedMultichainProofOutput;

    /* genFinalMultichainProof input */
    nlohmann::ordered_json finalMultichainProofInput;

    /* genProof and genFinalProof output */
    Proof proof;

    /* State */
    bool bCompleted;
    bool bCancelling; // set to true to request to cancel this request

    /* Result */
    zkresult result;

    /* Keys */
    unordered_set<string> nodesKeys;
    unordered_set<string> programKeys;

    /* Constructor */
    ProverAggregationRequest (Goldilocks &fr, const Config &config, tProverAggregationRequestType type);
    ~ProverAggregationRequest();


    /* Output file names */
    string proofFile (void);
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
};

#endif