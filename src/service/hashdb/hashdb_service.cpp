#include "hashdb_service.hpp"
#include <grpcpp/grpcpp.h>
#include "smt.hpp"
#include "goldilocks_base_field.hpp"
#include "hashdb_utils.hpp"
#include "definitions.hpp"
#include "scalar.hpp"
#include "zkresult.hpp"
#include <iomanip>
#include "zklog.hpp"
#include "exit_process.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

::grpc::Status HashDBServiceImpl::Set(::grpc::ServerContext* context, const ::hashdb::v1::SetRequest* request, ::hashdb::v1::SetResponse* response)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

    SmtSetResult r;
    try
    {
        // Get old state root
        Goldilocks::Element oldRoot[4];
        grpc2fea(fr, request->old_root(), oldRoot);

        // Get key
        Goldilocks::Element key[4];
        grpc2fea(fr, request->key(), key);

        // Get value
        if (request->value().size() > 64)
        {
            zklog.error("HashDBServiceImpl::Set() got a too big value: " + request->value());
            return Status::CANCELLED;
        }
        if (!stringIsHex(request->value()))
        {
            zklog.error("HashDBServiceImpl::Set() got a non-hex value: " + request->value());
            return Status::CANCELLED;
        }
        mpz_class value(request->value(), 16);

        // Get persistence flag
        Persistence persistence = (Persistence)request->persistence();

#ifdef LOG_HASHDB_SERVICE
        zklog.info("HashDBServiceImpl::Set() called. odlRoot=" + fea2string(fr, oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]) +
            " key=" + fea2string(fr, key[0], key[1], key[2], key[3]) +
            " value=" +  value.get_str(16) +
            " persistent=" + to_string(persistent));
#endif
        // Get database read log flag
        DatabaseMap *dbReadLog = NULL;
        if (request->get_db_read_log())
        {
            dbReadLog = new DatabaseMap();
            if (dbReadLog == NULL)
            {
                zklog.error("HashDBServiceImpl::Set() failed allocating a new DatabaseMap");
                return Status::CANCELLED;
            }
        }

        // Get batch UUID
        string batchUUID = request->batch_uuid();

        // Get TX number
        uint64_t tx = request->tx();

        // Call SMT set
        Goldilocks::Element newRoot[4];
        zkresult zkr = pHashDB->set(batchUUID, tx, oldRoot, key, value, persistence, newRoot, &r, dbReadLog);

        // Return database read log
        if (request->get_db_read_log())
        {
            mtMap2grpc(fr, dbReadLog->getMTDB(), response->mutable_db_read_log());
            delete dbReadLog;
        }

        // Return new state root
        ::hashdb::v1::Fea* resNewRoot = new ::hashdb::v1::Fea();
        fea2grpc(fr, r.newRoot, resNewRoot);
        response->set_allocated_new_root(resNewRoot);

        // If requested, return details
        if (request->details())
        {
            // Return old state root
            ::hashdb::v1::Fea* resOldRoot = new ::hashdb::v1::Fea();
            fea2grpc(fr, r.oldRoot, resOldRoot);
            response->set_allocated_old_root(resOldRoot);

            // Return key
            ::hashdb::v1::Fea* resKey = new ::hashdb::v1::Fea();
            fea2grpc(fr, r.key, resKey);
            response->set_allocated_key(resKey);

            // Return siblings
            std::map<uint64_t, std::vector<Goldilocks::Element>>::iterator it;
            for (it=r.siblings.begin(); it!=r.siblings.end(); it++)
            {
                ::hashdb::v1::SiblingList list;
                for (uint64_t i=0; i<it->second.size(); i++)
                {
                    list.add_sibling(fr.toU64(it->second[i]));
                }
                (*response->mutable_siblings())[it->first] = list;
            }

            // Return ins key
            ::hashdb::v1::Fea* resInsKey = new ::hashdb::v1::Fea();
            fea2grpc(fr, r.insKey, resInsKey);
            response->set_allocated_ins_key(resInsKey);

            // Return ins value
            response->set_ins_value(r.insValue.get_str(16));

            // Return is old0
            response->set_is_old0(r.isOld0);

            // Return old value
            response->set_old_value(r.oldValue.get_str(16));

            // Return new value
            response->set_new_value(r.newValue.get_str(16));

            // Return mode
            response->set_mode(r.mode);

            // Return hash counter
            response->set_proof_hash_counter(r.proofHashCounter);
        }

        // Return result
        ::hashdb::v1::ResultCode* rc = new ::hashdb::v1::ResultCode();
        rc->set_code(static_cast<::hashdb::v1::ResultCode_Code>(zkr));
        response->set_allocated_result(rc);
    }
    catch (const std::exception &e)
    {
        zklog.error("HashDBServiceImpl::Set() exception: " + string(e.what()));
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::Set() completed. newRoot= " + fea2string(fr, r.newRoot[0], r.newRoot[1], r.newRoot[2], r.newRoot[3]));
#endif

    return Status::OK;
}

::grpc::Status HashDBServiceImpl::Get(::grpc::ServerContext* context, const ::hashdb::v1::GetRequest* request, ::hashdb::v1::GetResponse* response)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

    SmtGetResult r;
    try
    {
        // Get state root
        Goldilocks::Element root[4];
        grpc2fea (fr, request->root(), root);

        // Get key
        Goldilocks::Element key[4];
        grpc2fea (fr, request->key(), key);

#ifdef LOG_HASHDB_SERVICE
        zklog.info("HashDBServiceImpl::Get() called. root=" + fea2string(fr, root[0], root[1], root[2], root[3]) +
            " key=" + fea2string(fr, key[0], key[1], key[2], key[3]));
#endif

        // Get database read log flag
        DatabaseMap *dbReadLog = NULL;
        if (request->get_db_read_log())
        {
            dbReadLog = new DatabaseMap();
            if (dbReadLog == NULL)
            {
                zklog.error("HashDBServiceImpl::Get() failed allocating a new DatabaseMap");
                return Status::CANCELLED;
            }
        }

        // Get batch uuid
        string batchUUID = request->batch_uuid();

        // Call SMT get
        mpz_class value;
        zkresult zkr = pHashDB->get(batchUUID, root, key, value, &r, dbReadLog);

        // Return database read log
        if (request->get_db_read_log())
        {
            mtMap2grpc(fr, dbReadLog->getMTDB(), response->mutable_db_read_log());
            delete dbReadLog;
        }

        // Return value
        response->set_value(PrependZeros(r.value.get_str(16), 64));

        // If requested, return details
        if (request->details())
        {
            // Return state root
            ::hashdb::v1::Fea* resRoot = new ::hashdb::v1::Fea();
            fea2grpc(fr, r.root, resRoot);
            response->set_allocated_root(resRoot);

            // Return key
            ::hashdb::v1::Fea* resKey = new ::hashdb::v1::Fea();
            fea2grpc(fr, r.key, resKey);
            response->set_allocated_key(resKey);

            // Return siblings
            std::map<uint64_t, std::vector<Goldilocks::Element>>::iterator it;
            for (it=r.siblings.begin(); it!=r.siblings.end(); it++)
            {
                ::hashdb::v1::SiblingList list;
                for (uint64_t i=0; i<it->second.size(); i++)
                {
                    list.add_sibling(fr.toU64(it->second[i]));
                }
                (*response->mutable_siblings())[it->first] = list;
            }

            // Return ins key
            ::hashdb::v1::Fea* resInsKey = new ::hashdb::v1::Fea();
            fea2grpc (fr, r.insKey, resInsKey);
            response->set_allocated_ins_key(resInsKey);

            // Return ins value
            response->set_ins_value(r.insValue.get_str(16));

            // Return is old0
            response->set_is_old0(r.isOld0);

            // Return hash counter
            response->set_proof_hash_counter(r.proofHashCounter);
        }

        // Return result
        ::hashdb::v1::ResultCode* rc = new ::hashdb::v1::ResultCode();
        rc->set_code(static_cast<::hashdb::v1::ResultCode_Code>(zkr));
        response->set_allocated_result(rc);
    }
    catch (const std::exception &e)
    {
        zklog.error("HashDBServiceImpl::Get() exception: " + string(e.what()));
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::Get() completed. value=" +  r.value.get_str(16));
#endif

    return Status::OK;
}

::grpc::Status HashDBServiceImpl::SetProgram(::grpc::ServerContext* context, const ::hashdb::v1::SetProgramRequest* request, ::hashdb::v1::SetProgramResponse* response)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

    try
    {
        // Get key
        Goldilocks::Element key[4];
        grpc2fea(fr, request->key(), key);

        // Get data
        vector<uint8_t> data;
        for (uint64_t i=0; i<request->data().size(); i++)
        {
            data.push_back(request->data().at(i));
        }

#ifdef LOG_HASHDB_SERVICE
        {
            string s = "HashDBServiceImpl::SetProgram() called. key=" + fea2string(fr, key[0], key[1], key[2], key[3]) + " data=";
            for (uint64_t i=0; i<data.size(); i++)
                s += byte2string(data[i]);
            s += " persistent=" + to_string(request->persistent());
            zklog.info(s);
        }
#endif
        // Call set program
        zkresult r = pHashDB->setProgram(key, data, request->persistent());

        // Return result
        ::hashdb::v1::ResultCode* result = new ::hashdb::v1::ResultCode();
        result->set_code(static_cast<::hashdb::v1::ResultCode_Code>(r));
        response->set_allocated_result(result);
    }
    catch (const std::exception &e)
    {
        zklog.error("HashDBServiceImpl::SetProgram() exception: " + string(e.what()));
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::SetProgram() completed.");
#endif

    return Status::OK;
}

::grpc::Status HashDBServiceImpl::GetProgram(::grpc::ServerContext* context, const ::hashdb::v1::GetProgramRequest* request, ::hashdb::v1::GetProgramResponse* response)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

    try
    {
        // Get key
        Goldilocks::Element key[4];
        grpc2fea (fr, request->key(), key);

#ifdef LOG_HASHDB_SERVICE
        zklog.info("HashDBServiceImpl::GetProgram() called. key=" + fea2string(fr, key[0], key[1], key[2], key[3]));
#endif

        // Call get program
        vector<uint8_t> value;
        zkresult r = pHashDB->getProgram(key, value, NULL);

        // Return data
        string sData;
        for (uint64_t i=0; i<value.size(); i++)
        {
            sData.push_back((char)value.at(i));
        }
        response->set_data(sData);

        // Return result
        ::hashdb::v1::ResultCode* result = new ::hashdb::v1::ResultCode();
        result->set_code(static_cast<::hashdb::v1::ResultCode_Code>(r));
        response->set_allocated_result(result);
    }
    catch (const std::exception &e)
    {
        zklog.error("HashDBServiceImpl::GetProgram() exception: " + string(e.what()));
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    {
        string s = "HashDBServiceImpl::GetProgram() completed. data=";
        for (uint64_t i=0; i<sData.size(); i++)
            s += byte2string(sData.at(i));
        zklog.info(s);
    }
#endif

    return Status::OK;
}

::grpc::Status HashDBServiceImpl::LoadDB(::grpc::ServerContext* context, const ::hashdb::v1::LoadDBRequest* request, ::google::protobuf::Empty* response)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::LoadDB called.");
#endif

    // Get Merkle tree map
    DatabaseMap::MTMap map;
    bool bResult = grpc2mtMap(fr, request->input_db(), map);
    if (!bResult)
    {
        zklog.error("HashDBServiceImpl::LoadDB() failed calling grpc2mtMap()");
        return Status::CANCELLED;
    }

    // Call load database
    pHashDB->loadDB(map, request->persistent());

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::LoadDB() completed.");
#endif

    return Status::OK;
}

::grpc::Status HashDBServiceImpl::LoadProgramDB(::grpc::ServerContext* context, const ::hashdb::v1::LoadProgramDBRequest* request, ::google::protobuf::Empty* response)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::LoadProgramDB called.");
#endif

    // Get program map
    DatabaseMap::ProgramMap mapProgram;
    bool bResult = grpc2programMap(fr, request->input_program_db(), mapProgram);
    if (!bResult)
    {
        zklog.error("HashDBServiceImpl::LoadProgramDB() failed calling grpc2programMap()");
        return Status::CANCELLED;
    }

    // Call load program database
    pHashDB->loadProgramDB(mapProgram, request->persistent());

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::LoadProgramDB() completed.");
#endif

    return Status::OK;
}

::grpc::Status HashDBServiceImpl::Flush(::grpc::ServerContext* context, const ::hashdb::v1::FlushRequest* request, ::hashdb::v1::FlushResponse* response)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::Flush called.");
#endif

    try
    {
        // Call the HashDB flush method
        uint64_t flushId = 0, storedFlushId = 0;
        zkresult zkres = pHashDB->flush(request->batch_uuid(), request->new_state_root(), (Persistence)(uint64_t)request->persistence(), flushId, storedFlushId);

        // return the result in the response
        ::hashdb::v1::ResultCode* result = new ::hashdb::v1::ResultCode();
        result->set_code(static_cast<::hashdb::v1::ResultCode_Code>(zkres));
        response->set_allocated_result(result);
        response->set_flush_id(flushId);
        response->set_stored_flush_id(storedFlushId);
    }
    catch (const std::exception &e)
    {
        zklog.error("HashDBServiceImpl::Flush() exception: " + string(e.what()));
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::Flush() completed.");
#endif

    return Status::OK;
}

::grpc::Status HashDBServiceImpl::GetFlushStatus (::grpc::ServerContext* context, const ::google::protobuf::Empty* request, ::hashdb::v1::GetFlushStatusResponse* response)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::GetFlushStatus called.");
#endif

    try
    {
        uint64_t storedFlushId;
        uint64_t storingFlushId;
        uint64_t lastFlushId;
        uint64_t pendingToFlushNodes;
        uint64_t pendingToFlushProgram;
        uint64_t storingNodes;
        uint64_t storingProgram;
        string proverId;

        pHashDB->getFlushStatus(storedFlushId, storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram, proverId);

        response->set_stored_flush_id(storedFlushId);
        response->set_storing_flush_id(storingFlushId);
        response->set_last_flush_id(lastFlushId);
        response->set_pending_to_flush_nodes(pendingToFlushNodes);
        response->set_pending_to_flush_program(pendingToFlushProgram);
        response->set_storing_nodes(storingNodes);
        response->set_storing_program(storingProgram);
        response->set_prover_id(proverId);
    }
    catch (const std::exception &e)
    {
        zklog.error("HashDBServiceImpl::GetFlushStatus() exception: " + string(e.what()));
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::GetFlushStatus() completed.");
#endif
    
    return Status::OK;
}

::grpc::Status HashDBServiceImpl::GetFlushData (::grpc::ServerContext* context, const ::hashdb::v1::GetFlushDataRequest* request, ::hashdb::v1::GetFlushDataResponse* response)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::GetFlushData called.");
#endif

    try
    {
        // Declare local variables to store the result
        uint64_t storedFlushId;
        unordered_map<string, string> nodes;
        unordered_map<string, string> program;
        string nodesStateRoot;

        // Call the local getFlushData method
        pHashDB->getFlushData(request->flush_id(), storedFlushId, nodes, program, nodesStateRoot);

        // Set the last sent flush ID
        response->set_stored_flush_id(storedFlushId);

        // Set the nodes
        unordered_map<string, string>::const_iterator it;
        for (it = nodes.begin(); it != nodes.end(); it++)
        {
            (*response->mutable_nodes())[it->first] = it->second;
        }

        // Set the program
        for (it = program.begin(); it != program.end(); it++)
        {
            (*response->mutable_program())[it->first] = it->second;
        }

        // Set the nodes state root
        response->set_nodes_state_root(nodesStateRoot);
    }
    catch (const std::exception &e)
    {
        zklog.error("HashDBServiceImpl::GetFlushData() exception: " + string(e.what()));
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::GetFlushData() completed.");
#endif
    
    return Status::OK;
}

::grpc::Status HashDBServiceImpl::ConsolidateState (::grpc::ServerContext* context, const ::hashdb::v1::ConsolidateStateRequest* request, ::hashdb::v1::ConsolidateStateResponse* response)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::ConsolidateState called.");
#endif

    try
    {
        // Call the HashDB flush method
        uint64_t flushId = 0, storedFlushId = 0;
        Goldilocks::Element virtualStateRoot[4];
        Goldilocks::Element consolidatedStateRoot[4];
        grpc2fea(fr, request->virtual_state_root(), virtualStateRoot);
        zkresult zkres = pHashDB->consolidateState(virtualStateRoot, (Persistence)(uint64_t)request->persistence(), consolidatedStateRoot, flushId, storedFlushId);

        // return the result in the response
        ::hashdb::v1::ResultCode* result = new ::hashdb::v1::ResultCode();
        if (result == NULL)
        {
            zklog.error("HashDBServiceImpl::ConsolidateState() failed allocating hashdb::v1::ResultCode");
            exitProcess();
        }
        result->set_code(static_cast<::hashdb::v1::ResultCode_Code>(zkres));
        response->set_allocated_result(result);
        response->set_flush_id(flushId);
        response->set_stored_flush_id(storedFlushId);
        hashdb::v1::Fea *consolidated_state_root = new hashdb::v1::Fea();
        if (consolidated_state_root == NULL)
        {
            zklog.error("HashDBServiceImpl::ConsolidateState() failed allocating hashdb::v1::Fea");
            exitProcess();
        }
        fea2grpc(fr, consolidatedStateRoot, consolidated_state_root);
        response->set_allocated_consolidated_state_root(consolidated_state_root);
    }
    catch (const std::exception &e)
    {
        zklog.error("HashDBServiceImpl::ConsolidateState() exception: " + string(e.what()));
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::ConsolidateState() completed.");
#endif

    return Status::OK;
}

::grpc::Status HashDBServiceImpl::ReadTree(::grpc::ServerContext* context, const ::hashdb::v1::ReadTreeRequest* request, ::hashdb::v1::ReadTreeResponse* response)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::ReadTree called.");
#endif

    try
    {
        // Get the keys from the request
        vector<KeyValue> keyValues;
        for (int64_t i=0; i<request->keys_size(); i++)
        {
            KeyValue keyValue;
            grpc2fea(fr, request->keys(i), keyValue.key);
            keyValues.emplace_back(keyValue);
        }

        // Get the state root from the request
        Goldilocks::Element stateRoot[4];
        grpc2fea(fr, request->state_root(), stateRoot);

        // Call readTree()
        vector<HashValueGL> hashValues;
        zkresult zkr = pHashDB->readTree(stateRoot, keyValues, hashValues);

        // Return the result in the response
        ::hashdb::v1::ResultCode* result = new ::hashdb::v1::ResultCode();
        if (result == NULL)
        {
            zklog.error("HashDBServiceImpl::ReadTree() failed allocating hashdb::v1::ResultCode");
            exitProcess();
        }
        result->set_code(static_cast<::hashdb::v1::ResultCode_Code>(zkr));
        response->set_allocated_result(result);

        // Return the key-value pairs in the response
        for (uint64_t i=0; i<keyValues.size(); i++)
        {
            hashdb::v1::Fea *pKey = new hashdb::v1::Fea();
            zkassertpermanent(pKey != NULL);
            pKey->set_fe0(fr.toU64(keyValues[i].key[0]));
            pKey->set_fe1(fr.toU64(keyValues[i].key[1]));
            pKey->set_fe2(fr.toU64(keyValues[i].key[2]));
            pKey->set_fe3(fr.toU64(keyValues[i].key[3]));
            hashdb::v1::KeyValue *pKeyValue = response->add_key_value();
            zkassertpermanent(pKeyValue != NULL);
            pKeyValue->set_allocated_key(pKey);
            pKeyValue->set_value(keyValues[i].value.get_str(16));
        }

        // Return the hash-value pairs in the response
        for (uint64_t i=0; i<hashValues.size(); i++)
        {
            hashdb::v1::Fea *pHash = new hashdb::v1::Fea();
            zkassertpermanent(pHash != NULL);
            pHash->set_fe0(fr.toU64(hashValues[i].hash[0]));
            pHash->set_fe1(fr.toU64(hashValues[i].hash[1]));
            pHash->set_fe2(fr.toU64(hashValues[i].hash[2]));
            pHash->set_fe3(fr.toU64(hashValues[i].hash[3]));
            hashdb::v1::Fea12 *pValue = new hashdb::v1::Fea12();
            zkassertpermanent(pValue != NULL);
            pValue->set_fe0(fr.toU64(hashValues[i].value[0]));
            hashdb::v1::HashValueGL *pHashValue = response->add_hash_value();
            zkassertpermanent(pHashValue != NULL);
            pHashValue->set_allocated_hash(pHash);
            pHashValue->set_allocated_value(pValue);
        }        
    }
    catch (const std::exception &e)
    {
        zklog.error("HashDBServiceImpl::ReadTree() exception: " + string(e.what()));
        return Status::CANCELLED;
    }

#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::ReadTree() completed.");
#endif

    return Status::OK;
}



