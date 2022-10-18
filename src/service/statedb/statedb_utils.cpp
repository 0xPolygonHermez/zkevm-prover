#include "statedb_utils.hpp"
#include "statedb.grpc.pb.h"
#include "goldilocks_base_field.hpp"
#include "database.hpp"

void fea2grpc (Goldilocks &fr, const Goldilocks::Element (&fea)[4], ::statedb::v1::Fea *grpcFea)
{
    grpcFea->set_fe0(fr.toU64(fea[0]));
    grpcFea->set_fe1(fr.toU64(fea[1]));
    grpcFea->set_fe2(fr.toU64(fea[2]));
    grpcFea->set_fe3(fr.toU64(fea[3]));
}

void grpc2fea (Goldilocks &fr, const ::statedb::v1::Fea& grpcFea, Goldilocks::Element (&fea)[4])
{
    fea[0] = fr.fromU64(grpcFea.fe0());
    fea[1] = fr.fromU64(grpcFea.fe1());
    fea[2] = fr.fromU64(grpcFea.fe2());
    fea[3] = fr.fromU64(grpcFea.fe3());
}

void mtMap2grpc(Goldilocks &fr, const DatabaseMap::MTMap &map, ::PROTOBUF_NAMESPACE_ID::Map<string, ::statedb::v1::FeList> *grpcMap)
{
    DatabaseMap::MTMap::const_iterator it;
    for (it = map.begin(); it != map.end(); it++)
    {
        ::statedb::v1::FeList list;
        for (uint64_t i=0; i<it->second.size(); i++)
        {
            list.add_fe(fr.toU64(it->second[i]));
        }
        (*grpcMap)[it->first] = list;
    }
}

void programMap2grpc(Goldilocks &fr, const DatabaseMap::ProgramMap &map, ::PROTOBUF_NAMESPACE_ID::Map<string, string> *grpcMap)
{
    DatabaseMap::ProgramMap::const_iterator it;
    for (it = map.begin(); it != map.end(); it++)
    {
        string sData;
        for (uint64_t i = 0; i < it->second.size(); i++) {
            sData.push_back((char)it->second.at(i));
        }
        (*grpcMap)[it->first] = sData;
    }
}

void grpc2mtMap(Goldilocks &fr, const ::PROTOBUF_NAMESPACE_ID::Map<string, ::statedb::v1::FeList> &grpcMap, DatabaseMap::MTMap &map)
{
    ::PROTOBUF_NAMESPACE_ID::Map<string, ::statedb::v1::FeList>::const_iterator it;
    for (it = grpcMap.begin(); it != grpcMap.end(); it++)
    {
        vector<Goldilocks::Element> list;
        for (int i = 0; i < it->second.fe_size(); i++)
        {
            list.push_back(fr.fromU64(it->second.fe(i)));
        }
        map[it->first] = list;
    }
}

void grpc2programMap(Goldilocks &fr, const ::PROTOBUF_NAMESPACE_ID::Map<string, string> &grpcMap, DatabaseMap::ProgramMap &map)
{
    ::PROTOBUF_NAMESPACE_ID::Map<string, string>::const_iterator it;
    for (it = grpcMap.begin(); it != grpcMap.end(); it++)
    {
        vector<uint8_t> list;
        for (size_t i=0; i < it->second.size(); i++)
        {
            list.push_back(it->second.at(i));
        }
        map[it->first] = list;
    }
}


