#ifndef HASHDB_UTILS_HPP
#define HASHDB_UTILS_HPP

#include "hashdb.grpc.pb.h"
#include "goldilocks_base_field.hpp"
#include <google/protobuf/port_def.inc>
#include "database.hpp"

using namespace std;

void fea2grpc(Goldilocks &fr, const Goldilocks::Element (&fea)[4], ::hashdb::v1::Fea *grpcFea);
void grpc2fea(Goldilocks &fr, const ::hashdb::v1::Fea& grpcFea, Goldilocks::Element (&fea)[4]);
void mtMap2grpc(Goldilocks &fr, const DatabaseMap::MTMap &map, ::PROTOBUF_NAMESPACE_ID::Map<string, ::hashdb::v1::FeList> *grpcMap);
void programMap2grpc(Goldilocks &fr, const DatabaseMap::ProgramMap &map, ::PROTOBUF_NAMESPACE_ID::Map<string, string> *grpcMap);
bool grpc2mtMap(Goldilocks &fr, const ::PROTOBUF_NAMESPACE_ID::Map<string, ::hashdb::v1::FeList> &grpcMap, DatabaseMap::MTMap &map);
bool grpc2programMap(Goldilocks &fr, const ::PROTOBUF_NAMESPACE_ID::Map<string, string> &grpcMap, DatabaseMap::ProgramMap &map);

#endif
