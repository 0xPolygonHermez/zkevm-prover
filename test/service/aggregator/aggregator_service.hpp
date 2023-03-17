#ifndef AGGREGATOR_SERVICE_HPP
#define AGGREGATOR_SERVICE_HPP

#include "aggregator.grpc.pb.h"
#include "proof_fflonk.hpp"
#include "goldilocks_base_field.hpp"
#include "prover.hpp"

class AggregatorServiceImpl final : public aggregator::v1::AggregatorService::Service
{
    Goldilocks &fr;
    Config &config;
    uint64_t messageId;
public:
    AggregatorServiceImpl(Goldilocks &fr, Config &config) : fr(fr), config(config), messageId(0) {};
    ::grpc::Status Channel(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream) override;
    ::grpc::Status ChannelOld(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream);
    ::grpc::Status GetStatus(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream);
    ::grpc::Status Cancel(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & requestID, aggregator::v1::Result &result);
    ::grpc::Status GenBatchProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFile, string &requestID);
    ::grpc::Status GenAggregatedProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFileA, const string & inputFileB, string &requestID);
    ::grpc::Status GenFinalProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFile, string &requestID);
    ::grpc::Status GetProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string &requestID, aggregator::v1::GetProofResponse_Result &result, string &proof);
    ::grpc::Status GenAndGetBatchProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFile, const string &outputFile);
    ::grpc::Status GenAndGetAggregatedProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFileA, const string & inputFileB, const string &outputFile);
    ::grpc::Status GenAndGetFinalProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFile, const string &outputFile);
    
};

#endif