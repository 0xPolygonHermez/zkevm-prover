#ifndef ZKPROVER_SERVICE_HPP
#define ZKPROVER_SERVICE_HPP

#include "zk-prover.grpc.pb.h"
#include "proof.hpp"
#include "ff/ff.hpp"
#include "prover.hpp"

class ZKProverServiceImpl final : public zkprover::ZKProver::Service
{
    FiniteField &fr;
    Prover &prover;
public:
    ZKProverServiceImpl(FiniteField &fr, Prover &prover) : fr(fr), prover(prover) {};

    ::grpc::Status GetStatus (::grpc::ServerContext* context, const ::zkprover::NoParams* request, ::zkprover::ResGetStatus* response) override;
    ::grpc::Status GenProof  (::grpc::ServerContext* context, const ::zkprover::InputProver* request, ::zkprover::ResGenProof* response) override;
    ::grpc::Status Cancel    (::grpc::ServerContext* context, const ::zkprover::RequestId* request, ::zkprover::ResCancel* response) override;
    ::grpc::Status GetProof  (::grpc::ServerContext* context, const ::zkprover::RequestId* request, ::zkprover::ResGetProof* response) override;
    ::grpc::Status Execute   (::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::zkprover::ResExecute, ::zkprover::InputProver>* stream) override;
};

#endif