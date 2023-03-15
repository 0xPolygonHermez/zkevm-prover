#ifndef STATEDB_SERVICE_HPP
#define STATEDB_SERVICE_HPP

#include "statedb.grpc.pb.h"
#include "goldilocks_base_field.hpp"
#include "statedb.hpp"
#include "statedb_singleton.hpp"
#include <mutex>

class StateDBServiceImpl final : public statedb::v1::StateDBService::Service
{
    Goldilocks &fr;
    const Config &config;
    StateDB * pStateDB;

public:
    StateDBServiceImpl (Goldilocks &fr, const Config& config, const bool autoCommit, const bool asyncWrite) : fr(fr), config(config)
    {
        pStateDB = stateDBSingleton.get(fr, config);
    };
    ::grpc::Status Set (::grpc::ServerContext* context, const ::statedb::v1::SetRequest* request, ::statedb::v1::SetResponse* response) override;
    ::grpc::Status Get (::grpc::ServerContext* context, const ::statedb::v1::GetRequest* request, ::statedb::v1::GetResponse* response) override;
    ::grpc::Status SetProgram (::grpc::ServerContext* context, const ::statedb::v1::SetProgramRequest* request, ::statedb::v1::SetProgramResponse* response) override;
    ::grpc::Status GetProgram (::grpc::ServerContext* context, const ::statedb::v1::GetProgramRequest* request, ::statedb::v1::GetProgramResponse* response) override;
    ::grpc::Status LoadDB(::grpc::ServerContext* context, const ::statedb::v1::LoadDBRequest* request, ::google::protobuf::Empty* response);
    ::grpc::Status LoadProgramDB(::grpc::ServerContext* context, const ::statedb::v1::LoadProgramDBRequest* request, ::google::protobuf::Empty* response);
    ::grpc::Status Flush (::grpc::ServerContext* context, const ::google::protobuf::Empty* request, ::statedb::v1::FlushResponse* response) override;
};

#endif