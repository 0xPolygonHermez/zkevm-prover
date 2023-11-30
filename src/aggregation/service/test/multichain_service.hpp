#ifndef MULTICHAIN_SERVICE_HPP
#define MULTICHAIN_SERVICE_HPP

#include "multichain.grpc.pb.h"
#include "proof_fflonk.hpp"
#include "goldilocks_base_field.hpp"
#include "prover_aggregation.hpp"

class MultichainServiceImpl final : public multichain::v1::MultichainService::Service
{
    Goldilocks &fr;
    Config &config;
    uint64_t messageId;
public:
    MultichainServiceImpl(Goldilocks &fr, Config &config) : fr(fr), config(config), messageId(0) {};
    ::grpc::Status Channel(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream) override;
    ::grpc::Status GetStatus(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream);
    ::grpc::Status Cancel(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string & requestID, multichain::v1::Result &result);
    ::grpc::Status GenPrepareMultichainProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string & inputFile, const string & previousHashFile, string &requestID);
    ::grpc::Status GenAggregatedMultichainProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string & inputFileA, const string & inputFileB, string &requestID);
    ::grpc::Status GenFinalMultichainProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string & inputFile, string &requestID);
    ::grpc::Status GetProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string &requestID, multichain::v1::GetProofResponse_Result &result, string &proof);
    ::grpc::Status GetProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string &requestID, multichain::v1::GetProofResponse_Result &result, string &proof, string &hashInfo);
    ::grpc::Status GenAndGetPrepareMultichainProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string & inputFile, const string & previousHashFile, const string &outputFile, const string &hashInfoFile);
    ::grpc::Status GenAndGetAggregatedMultichainProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string & inputFileA, const string & inputFileB, const string &outputFile);
    ::grpc::Status GenAndGetFinalMultichainProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string & inputFile, const string &outputFile);
    ::grpc::Status CalculateSha256Publics(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string & publicsFile, const string & previousHashFile, const string &outputHashFile);
};

#endif