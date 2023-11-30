#ifndef MULTICHAIN_CLIENT_HPP
#define MULTICHAIN_CLIENT_HPP

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include "multichain.grpc.pb.h"
#include "proof_fflonk.hpp"
#include "goldilocks_base_field.hpp"
#include "prover_aggregation.hpp"

class MultichainClient
{
public:
    Goldilocks &fr;
    const Config &config;
    ProverAggregation &prover;
    multichain::v1::MultichainService::Stub * stub;
    pthread_t t; // Client thread

public:
    MultichainClient (Goldilocks &fr, const Config &config, ProverAggregation &prover);

    void runThread (void);
    void waitForThread (void);
    bool GetStatus (multichain::v1::GetStatusResponse &getStatusResponse);
    bool GenPrepareMultichainProof (const multichain::v1::GenPrepareMultichainProofRequest &genPrepareMultichainProofRequest, multichain::v1::GenPrepareMultichainProofResponse &genPrepareMultichainProofResponse);
    bool GenAggregatedMultichainProof (const multichain::v1::GenAggregatedMultichainProofRequest &genAggregatedMultichainProofRequest, multichain::v1::GenAggregatedMultichainProofResponse &genAggregatedMultichainProofResponse);
    bool GenFinalMultichainProof (const multichain::v1::GenFinalMultichainProofRequest &genFinalMultichainProofRequest, multichain::v1::GenFinalMultichainProofResponse &genFinalMultichainProofResponse);
    bool CalculateSha256(const multichain::v1::CalculateSha256Request &calculateSha256Request, multichain::v1::CalculateSha256Response &calculateSha256Response);
    bool Cancel (const multichain::v1::CancelRequest &cancelRequest, multichain::v1::CancelResponse &cancelResponse);
    bool GetProof (const multichain::v1::GetProofRequest &getProofRequest, multichain::v1::GetProofResponse &getProofResponse);
};

void* multichainClientThread(void* arg);

#endif