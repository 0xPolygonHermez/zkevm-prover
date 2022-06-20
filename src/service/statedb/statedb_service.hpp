#ifndef STATEDB_SERVICE_HPP
#define STATEDB_SERVICE_HPP

#include "statedb.grpc.pb.h"
#include "goldilocks/goldilocks_base_field.hpp"
#include "database.hpp"
#include "statedb.hpp"

class StateDBServiceImpl final : public statedb::v1::StateDBService::Service
{
    Goldilocks &fr;
    StateDB &stateDB;
public:
    StateDBServiceImpl(Goldilocks &fr, StateDB &stateDB) : fr(fr), stateDB(stateDB) {};
    ::grpc::Status Set(::grpc::ServerContext* context, const ::statedb::v1::SetRequest* request, ::statedb::v1::SetResponse* response) override;
    ::grpc::Status Get(::grpc::ServerContext* context, const ::statedb::v1::GetRequest* request, ::statedb::v1::GetResponse* response) override;
    ::grpc::Status SetProgram(::grpc::ServerContext* context, const ::statedb::v1::SetProgramRequest* request, ::statedb::v1::SetProgramResponse* response) override;
    ::grpc::Status GetProgram(::grpc::ServerContext* context, const ::statedb::v1::GetProgramRequest* request, ::statedb::v1::GetProgramResponse* response) override;
    ::grpc::Status Flush(::grpc::ServerContext* context, const ::google::protobuf::Empty* request, ::google::protobuf::Empty* response) override;
};

#endif