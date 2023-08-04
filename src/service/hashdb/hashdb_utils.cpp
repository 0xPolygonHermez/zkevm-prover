#include "hashdb_utils.hpp"
#include "hashdb.grpc.pb.h"
#include "goldilocks_base_field.hpp"
#include "database.hpp"
#include "scalar.hpp"
#include "zklog.hpp"

void fea2grpc (Goldilocks &fr, const Goldilocks::Element (&fea)[4], ::hashdb::v1::Fea *grpcFea)
{
    grpcFea->set_fe0(fr.toU64(fea[0]));
    grpcFea->set_fe1(fr.toU64(fea[1]));
    grpcFea->set_fe2(fr.toU64(fea[2]));
    grpcFea->set_fe3(fr.toU64(fea[3]));
}

void grpc2fea (Goldilocks &fr, const ::hashdb::v1::Fea& grpcFea, Goldilocks::Element (&fea)[4])
{
    fea[0] = fr.fromU64(grpcFea.fe0());
    fea[1] = fr.fromU64(grpcFea.fe1());
    fea[2] = fr.fromU64(grpcFea.fe2());
    fea[3] = fr.fromU64(grpcFea.fe3());
}

void mtMap2grpc (Goldilocks &fr, const DatabaseMap::MTMap &map, ::PROTOBUF_NAMESPACE_ID::Map<string, ::hashdb::v1::FeList> *grpcMap)
{
    DatabaseMap::MTMap::const_iterator it;
    for (it = map.begin(); it != map.end(); it++)
    {
        ::hashdb::v1::FeList list;
        for (uint64_t i=0; i<it->second.size(); i++)
        {
            list.add_fe(fr.toU64(it->second[i]));
        }
        (*grpcMap)[it->first] = list;
    }
}

void programMap2grpc (Goldilocks &fr, const DatabaseMap::ProgramMap &map, ::PROTOBUF_NAMESPACE_ID::Map<string, string> *grpcMap)
{
    DatabaseMap::ProgramMap::const_iterator it;
    for (it = map.begin(); it != map.end(); it++)
    {
        string sData;
        for (uint64_t i = 0; i < it->second.size(); i++)
        {
            sData.push_back((char)it->second.at(i));
        }
        (*grpcMap)[it->first] = sData;
    }
}

bool grpc2mtMap (Goldilocks &fr, const ::PROTOBUF_NAMESPACE_ID::Map<string, ::hashdb::v1::FeList> &grpcMap, DatabaseMap::MTMap &map)
{
    string key;
    ::PROTOBUF_NAMESPACE_ID::Map<string, ::hashdb::v1::FeList>::const_iterator it;
    for (it = grpcMap.begin(); it != grpcMap.end(); it++)
    {
        // Get key
        key = it->first;
        Remove0xIfPresentNoCopy(key);
        if (key.size() > 64)
        {
            zklog.error("grpc2mtMap() got db key too long, size=" + to_string(key.size()));
            return false;
        }
        if (!stringIsHex(key))
        {
            zklog.error("grpc2mtMap() got db key not hex, key=" + key);
            return false;
        }
        PrependZerosNoCopy(key, 64);

        // Get value
        vector<Goldilocks::Element> value;
        for (int i = 0; i < it->second.fe_size(); i++)
        {
            value.push_back(fr.fromU64(it->second.fe(i)));
        }

        // Set key-value
        map[key] = value;
    }

    return true;
}

bool grpc2programMap (Goldilocks &fr, const ::PROTOBUF_NAMESPACE_ID::Map<string, string> &grpcMap, DatabaseMap::ProgramMap &map)
{
    string key;
    ::PROTOBUF_NAMESPACE_ID::Map<string, string>::const_iterator it;
    for (it = grpcMap.begin(); it != grpcMap.end(); it++)
    {
        // Get key
        key = it->first;
        Remove0xIfPresentNoCopy(key);
        if (key.size() > 64)
        {
            zklog.error("grpc2programMap() got db key too long, size=" + to_string(key.size()));
            return false;
        }
        if (!stringIsHex(key))
        {
            zklog.error("grpc2programMap() got db key not hex, key=" + key);
            return false;
        }
        PrependZerosNoCopy(key, 64);

        // Get value
        vector<uint8_t> list;
        for (size_t i=0; i < it->second.size(); i++)
        {
            list.push_back(it->second.at(i));
        }

        // Set key-value
        map[key] = list;
    }

    return true;
}


