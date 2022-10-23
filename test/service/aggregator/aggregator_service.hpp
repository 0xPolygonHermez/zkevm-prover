#ifndef AGGREGATOR_SERVICE_HPP
#define AGGREGATOR_SERVICE_HPP

#include "aggregator.grpc.pb.h"
#include "proof.hpp"
#include "goldilocks_base_field.hpp"
#include "prover.hpp"

class AggregatorServiceImpl final : public aggregator::v1::AggregatorService::Service
{
    Goldilocks &fr;
    Config &config;
public:
    AggregatorServiceImpl(Goldilocks &fr, Config &config) : fr(fr), config(config) {};
    ::grpc::Status Channel(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream) override;
};

#endif