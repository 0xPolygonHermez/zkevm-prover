#ifndef EXECUTOR_SERVICE_HPP
#define EXECUTOR_SERVICE_HPP

#include "grpc/gen/executor.grpc.pb.h"
#include "proof_fflonk.hpp"
#include "goldilocks_base_field.hpp"
#include "prover.hpp"
#include "config.hpp"
#include "zkresult.hpp"

//#define PROCESS_BATCH_STREAM

class ExecutorServiceImpl final : public executor::v1::ExecutorService::Service
{
    Goldilocks &fr;
    Config &config;
    Prover &prover;

    // These attributes are used to measure the throughput of the executor
    // process batch service, in gas per second
    uint64_t counter; // Total number of calls to ProcessBatch
    uint64_t totalGas; // Total gas, i.e. the sum of gas of all calls to ProcessBatch
    uint64_t totalBytes; // Total amount of batch L2 data bytes procesed
    uint64_t totalTX; // Total amount of transactions returned
    double totalTime; // Total time, i.e. the sum of time (in seconds) of all calls to ProcessBatch
    struct timeval lastTotalTime; // Time when the last total was calculated
    struct timeval firstTotalTime; // Time when the executor service instance started
    uint64_t lastTotalGas; // Gas when the last total was calculated
    uint64_t lastTotalBytes; // Bytes when the last total was calculated
    uint64_t lastTotalTX; // TXs when the last total was calculated
    double totalTPG; // Total throughput in gas/s, calculated when time since lastTotalTime > 1s
    double totalTPB; // Total throughput in B/s, calculated when time since lastTotalTime > 1s
    double totalTPTX; // Total throughput in TX/s, calculated when time since lastTotalTime > 1s
    pthread_mutex_t mutex; // Mutex to protect the access to the throughput attributes

public:
    ExecutorServiceImpl (Goldilocks &fr, Config &config, Prover &prover) :
        fr(fr),
        config(config),
        prover(prover),
        counter(0),
        totalGas(0),
        totalBytes(0),
        totalTX(0),
        totalTime(0),
        lastTotalGas(0),
        lastTotalBytes(0),
        lastTotalTX(0),
        totalTPG(0),
        totalTPB(0),
        totalTPTX(0)
    {
        pthread_mutex_init(&mutex, NULL);
        lastTotalTime = {0,0};
        firstTotalTime = {0, 0};
    };
    void lock(void) { pthread_mutex_lock(&mutex); };
    void unlock(void) { pthread_mutex_unlock(&mutex); };
    ::grpc::Status ProcessBatch (::grpc::ServerContext* context, const ::executor::v1::ProcessBatchRequest* request, ::executor::v1::ProcessBatchResponse* response) override;
#ifdef PROCESS_BATCH_STREAM
    ::grpc::Status ProcessBatchStream (::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::executor::v1::ProcessBatchResponse, ::executor::v1::ProcessBatchRequest>* stream) override;
#endif    
    ::executor::v1::RomError string2error (string &errorString);
    ::executor::v1::ExecutorError zkresult2error (zkresult &result);
};

#endif