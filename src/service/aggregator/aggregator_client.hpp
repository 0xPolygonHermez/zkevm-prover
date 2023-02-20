#ifndef AGGREGATOR_CLIENT_HPP
#define AGGREGATOR_CLIENT_HPP

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include "aggregator.grpc.pb.h"
#include "proof_fflonk.hpp"
#include "goldilocks_base_field.hpp"
#include "prover.hpp"

class AggregatorClient
{
public:
    Goldilocks &fr;
    const Config &config;
    Prover &prover;
    aggregator::v1::AggregatorService::Stub * stub;
    pthread_t t; // Client thread

public:
    AggregatorClient (Goldilocks &fr, const Config &config, Prover &prover);

    void runThread (void);
    void waitForThread (void);
    bool GetStatus (aggregator::v1::GetStatusResponse &getStatusResponse);
    bool GenBatchProof (const aggregator::v1::GenBatchProofRequest &genBatchProofRequest, aggregator::v1::GenBatchProofResponse &genBatchProofResponse);
    bool GenAggregatedProof (const aggregator::v1::GenAggregatedProofRequest &genAggregatedProofRequest, aggregator::v1::GenAggregatedProofResponse &genAggregatedProofResponse);
    bool GenFinalProof (const aggregator::v1::GenFinalProofRequest &genFinalProofRequest, aggregator::v1::GenFinalProofResponse &genFinalProofResponse);
    bool Cancel (const aggregator::v1::CancelRequest &cancelRequest, aggregator::v1::CancelResponse &cancelResponse);
    bool GetProof (const aggregator::v1::GetProofRequest &getProofRequest, aggregator::v1::GetProofResponse &getProofResponse);
};

void* aggregatorClientThread(void* arg);

#endif