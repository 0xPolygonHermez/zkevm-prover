#ifndef EXECUTOR_SERVICE_HPP
#define EXECUTOR_SERVICE_HPP

#include "grpc/gen/executor.grpc.pb.h"
#include "proof.hpp"
#include "goldilocks_base_field.hpp"
#include "prover.hpp"
#include "config.hpp"
#include "zkresult.hpp"

class ExecutorServiceImpl final : public executor::v1::ExecutorService::Service
{
    Goldilocks &fr;
    Config &config;
    Prover &prover;

    // These attributes are used to measure the throughput of the executor
    // process batch service, in gas per second
    uint64_t counter; // Total number of calls to ProcessBatch
    uint64_t totalGas; // Total gas, i.e. the sum of gas of all calls to ProcessBatch
    double totalTime; // Total time, i.e. the sum of time (in seconds) of all calls to ProcessBatch
    pthread_mutex_t mutex; // Mutex to protect the access to the throughput attributes

public:
    ExecutorServiceImpl (Goldilocks &fr, Config &config, Prover &prover) : fr(fr), config(config), prover(prover), counter(0), totalGas(0), totalTime(0)
    {
        pthread_mutex_init(&mutex, NULL);
    };
    void lock(void) { pthread_mutex_lock(&mutex); };
    void unlock(void) { pthread_mutex_unlock(&mutex); };
    ::grpc::Status ProcessBatch (::grpc::ServerContext* context, const ::executor::v1::ProcessBatchRequest* request, ::executor::v1::ProcessBatchResponse* response) override;
    ::executor::v1::Error string2error (string &errorString);
    ::executor::v1::Error zkresult2error (zkresult &result);
};

#endif