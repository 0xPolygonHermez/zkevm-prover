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
public:
    ExecutorServiceImpl (Goldilocks &fr, Config &config, Prover &prover) : fr(fr), config(config), prover(prover) {};
    ::grpc::Status ProcessBatch (::grpc::ServerContext* context, const ::executor::v1::ProcessBatchRequest* request, ::executor::v1::ProcessBatchResponse* response) override;
    ::executor::v1::Error string2error (string &errorString);
    ::executor::v1::Error zkresult2error (zkresult &result);
};

#endif