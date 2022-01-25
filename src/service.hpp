#ifndef ZKPROVER_SERVICE_HPP
#define ZKPROVER_SERVICE_HPP

#include "zk-prover.grpc.pb.h"
#include "proof.hpp"
#include "ffiasm/fr.hpp"
#include "prover.hpp"

class ZKProverServiceImpl final : public zkprover::ZKProver::Service {
    zkprover::State_Status status;
    zkprover::Proof lastProof;
    bool bCancelling;

    RawFr &fr;
    Prover &prover;
public:
    ZKProverServiceImpl(RawFr &fr, Prover &prover) : fr(fr), prover(prover) { status = zkprover::State::IDLE; bCancelling = false; };

    ::grpc::Status GetStatus(::grpc::ServerContext* context, const ::zkprover::NoParams* request, ::zkprover::State* response) override;
    ::grpc::Status GenProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::zkprover::State, ::zkprover::InputProver>* stream) override;
    ::grpc::Status Cancel(::grpc::ServerContext* context, const ::zkprover::NoParams* request, ::zkprover::State* response) override;
    ::grpc::Status GetProof(::grpc::ServerContext* context, const ::zkprover::NoParams* request, ::zkprover::Proof* response) override;

private:
    void inputProver2Input (zkprover::InputProver &inputProver, Input &input);
    void proof2ProofProver (Proof &proof, zkprover::Proof &proofProver);
};

#endif