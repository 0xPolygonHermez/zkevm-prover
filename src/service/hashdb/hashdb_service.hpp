#ifndef HASHDB_SERVICE_HPP
#define HASHDB_SERVICE_HPP

#include "hashdb.grpc.pb.h"
#include "goldilocks_base_field.hpp"
#include "hashdb.hpp"
#include "hashdb_singleton.hpp"
#include <mutex>

class HashDBServiceImpl final : public hashdb::v1::HashDBService::Service
{
    Goldilocks &fr;
    const Config &config;
    HashDB * pHashDB;

public:
    HashDBServiceImpl (Goldilocks &fr, const Config& config, const bool autoCommit, const bool asyncWrite) : fr(fr), config(config)
    {
        pHashDB = hashDBSingleton.get();
    };
    ::grpc::Status Set (::grpc::ServerContext* context, const ::hashdb::v1::SetRequest* request, ::hashdb::v1::SetResponse* response) override;
    ::grpc::Status Get (::grpc::ServerContext* context, const ::hashdb::v1::GetRequest* request, ::hashdb::v1::GetResponse* response) override;
    ::grpc::Status SetProgram (::grpc::ServerContext* context, const ::hashdb::v1::SetProgramRequest* request, ::hashdb::v1::SetProgramResponse* response) override;
    ::grpc::Status GetProgram (::grpc::ServerContext* context, const ::hashdb::v1::GetProgramRequest* request, ::hashdb::v1::GetProgramResponse* response) override;
    ::grpc::Status LoadDB(::grpc::ServerContext* context, const ::hashdb::v1::LoadDBRequest* request, ::google::protobuf::Empty* response);
    ::grpc::Status LoadProgramDB(::grpc::ServerContext* context, const ::hashdb::v1::LoadProgramDBRequest* request, ::google::protobuf::Empty* response);
    ::grpc::Status Flush (::grpc::ServerContext* context, const ::hashdb::v1::FlushRequest* request, ::hashdb::v1::FlushResponse* response) override;
    ::grpc::Status GetFlushStatus (::grpc::ServerContext* context, const ::google::protobuf::Empty* request, ::hashdb::v1::GetFlushStatusResponse* response) override;
    ::grpc::Status GetFlushData (::grpc::ServerContext* context, const ::hashdb::v1::GetFlushDataRequest* request, ::hashdb::v1::GetFlushDataResponse* response) override;
    ::grpc::Status ConsolidateState (::grpc::ServerContext* context, const ::hashdb::v1::ConsolidateStateRequest* request, ::hashdb::v1::ConsolidateStateResponse* response) override;
    ::grpc::Status ReadTree (::grpc::ServerContext* context, const ::hashdb::v1::ReadTreeRequest* request, ::hashdb::v1::ReadTreeResponse* response);
};

#endif