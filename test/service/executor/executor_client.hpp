#ifndef EXECUTOR_CLIENT_HPP
#define EXECUTOR_CLIENT_HPP

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include "executor.grpc.pb.h"
#include "goldilocks_base_field.hpp"
#include "prover.hpp"

#define EXECUTOR_CLIENT_MULTITHREAD_N_THREADS  10
#define EXECUTOR_CLIENT_MULTITHREAD_N_FILES 10

class ExecutorClient
{
public:
    Goldilocks &fr;
    const Config &config;
    executor::v1::ExecutorService::Stub * stub;
    pthread_t t; // Client thread
    pthread_t threads[EXECUTOR_CLIENT_MULTITHREAD_N_THREADS]; // Client threads

public:
    ExecutorClient (Goldilocks &fr, const Config &config);
    ~ExecutorClient ();

    // Mono-thread
    void runThread (void);
    void waitForThread (void);

    // Multi-thread
    void runThreads (void);
    void waitForThreads (void);

    bool ProcessBatch (void);
};

void* executorClientThread  (void* arg); // One process batch
void* executorClientThreads (void* arg); // Many process batches

#endif