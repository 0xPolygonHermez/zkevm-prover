#ifndef STATEDB_UTILS_HPP
#define STATEDB_UTILS_HPP

#include "statedb.grpc.pb.h"
#include "goldilocks_base_field.hpp"
#include <google/protobuf/port_def.inc>
#include "database.hpp"

using namespace std;

void fea2grpc(Goldilocks &fr, const Goldilocks::Element (&fea)[4], ::statedb::v1::Fea *grpcFea);
void grpc2fea(Goldilocks &fr, const ::statedb::v1::Fea& grpcFea, Goldilocks::Element (&fea)[4]);
void mtMap2grpc(Goldilocks &fr, const DatabaseMap::MTMap &map, ::PROTOBUF_NAMESPACE_ID::Map<string, ::statedb::v1::FeList> *grpcMap);
void programMap2grpc(Goldilocks &fr, const DatabaseMap::ProgramMap &map, ::PROTOBUF_NAMESPACE_ID::Map<string, string> *grpcMap);
void grpc2mtMap(Goldilocks &fr, const ::PROTOBUF_NAMESPACE_ID::Map<string, ::statedb::v1::FeList> &grpcMap, DatabaseMap::MTMap &map);
void grpc2programMap(Goldilocks &fr, const ::PROTOBUF_NAMESPACE_ID::Map<string, string> &grpcMap, DatabaseMap::ProgramMap &map);

#endif
