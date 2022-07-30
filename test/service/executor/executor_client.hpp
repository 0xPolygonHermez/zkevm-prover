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

class ExecutorClient
{
public:
    Goldilocks &fr;
    const Config &config;
    executor::v1::ExecutorService::Stub * stub;
    pthread_t t; // Client thread

public:
    ExecutorClient (Goldilocks &fr, const Config &config);

    void runThread (void);
    void waitForThread (void);
    bool ProcessBatch (void);
};

void* executorClientThread (void* arg);

#endif