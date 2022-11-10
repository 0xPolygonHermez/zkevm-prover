#ifndef ZKPROVER_SERVICE_HPP
#define ZKPROVER_SERVICE_HPP

#include "zk_prover.grpc.pb.h"
#include "proof.hpp"
#include "goldilocks_base_field.hpp"
#include "prover.hpp"

class ZKProverServiceImpl final : public zkprover::v1::ZKProverService::Service
{
    Goldilocks &fr;
    Prover &prover;
public:
    ZKProverServiceImpl(Goldilocks &fr, Prover &prover) : fr(fr), prover(prover) {};
    ::grpc::Status GetStatus(::grpc::ServerContext* context, const ::zkprover::v1::GetStatusRequest* request, ::zkprover::v1::GetStatusResponse* response) override;
    ::grpc::Status GenProof(::grpc::ServerContext* context, const ::zkprover::v1::GenProofRequest* request, ::zkprover::v1::GenProofResponse* response) override;
    ::grpc::Status Cancel(::grpc::ServerContext* context, const ::zkprover::v1::CancelRequest* request, ::zkprover::v1::CancelResponse* response) override;
    ::grpc::Status GetProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::zkprover::v1::GetProofResponse, ::zkprover::v1::GetProofRequest>* stream) override;
};

#endif