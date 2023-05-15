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

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

::grpc::Status HashDBServiceImpl::Set(::grpc::ServerContext* context, const ::hashdb::v1::SetRequest* request, ::hashdb::v1::SetResponse* response)
{
    SmtSetResult r;
    try {
        Goldilocks::Element oldRoot[4];
        grpc2fea (fr, request->old_root(), oldRoot);

        Goldilocks::Element key[4];
        grpc2fea (fr, request->key(), key);

        mpz_class value(request->value(),16);
        bool persistent = request->persistent();
#ifdef LOG_HASHDB_SERVICE
        zklog.info("HashDBServiceImpl::Set() called. odlRoot=" + fea2string(fr, oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]) +
            " key=" + fea2string(fr, key[0], key[1], key[2], key[3]) +
            " value=" +  value.get_str(16) +
            " persistent=" + to_string(persistent));
#endif
        DatabaseMap *dbReadLog = NULL;
        if (request->get_db_read_log())
            dbReadLog = new DatabaseMap();

        Goldilocks::Element newRoot[4];
        zkresult zkr = pHashDB->set(oldRoot, key, value, persistent, newRoot, &r, dbReadLog);

        if (request->get_db_read_log())
        {
            mtMap2grpc(fr, dbReadLog->getMTDB(), response->mutable_db_read_log());
            delete dbReadLog;
        }

        ::hashdb::v1::Fea* resNewRoot = new ::hashdb::v1::Fea();
        fea2grpc (fr, r.newRoot, resNewRoot);
        response->set_allocated_new_root(resNewRoot);

        if (request->details()) {
            ::hashdb::v1::Fea* resOldRoot = new ::hashdb::v1::Fea();
            fea2grpc (fr, r.oldRoot, resOldRoot);
            response->set_allocated_old_root(resOldRoot);

            ::hashdb::v1::Fea* resKey = new ::hashdb::v1::Fea();
            fea2grpc (fr, r.key, resKey);
            response->set_allocated_key(resKey);

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

            ::hashdb::v1::Fea* resInsKey = new ::hashdb::v1::Fea();
            fea2grpc (fr, r.insKey, resInsKey);
            response->set_allocated_ins_key(resInsKey);

            response->set_ins_value(r.insValue.get_str(16));
            response->set_is_old0(r.isOld0);
            response->set_old_value(r.oldValue.get_str(16));
            response->set_new_value(r.newValue.get_str(16));
            response->set_mode(r.mode);
            response->set_proof_hash_counter(r.proofHashCounter);
        }

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
    SmtGetResult r;
    try
    {
        Goldilocks::Element root[4];
        grpc2fea (fr, request->root(), root);

        Goldilocks::Element key[4];
        grpc2fea (fr, request->key(), key);
#ifdef LOG_HASHDB_SERVICE
        zklog.info("HashDBServiceImpl::Get() called. root=" + fea2string(fr, root[0], root[1], root[2], root[3]) +
            " key=" + fea2string(fr, key[0], key[1], key[2], key[3]));
#endif

        DatabaseMap *dbReadLog = NULL;
        if (request->get_db_read_log())
            dbReadLog = new DatabaseMap();

        mpz_class value;
        zkresult zkr = pHashDB->get(root, key, value, &r, dbReadLog);

        if (request->get_db_read_log())
        {
            mtMap2grpc(fr, dbReadLog->getMTDB(), response->mutable_db_read_log());
            delete dbReadLog;
        }

        response->set_value(PrependZeros(r.value.get_str(16), 64));

        if (request->details()) {
            ::hashdb::v1::Fea* resRoot = new ::hashdb::v1::Fea();
            fea2grpc (fr, r.root, resRoot);
            response->set_allocated_root(resRoot);

            ::hashdb::v1::Fea* resKey = new ::hashdb::v1::Fea();
            fea2grpc (fr, r.key, resKey);
            response->set_allocated_key(resKey);

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

            ::hashdb::v1::Fea* resInsKey = new ::hashdb::v1::Fea();
            fea2grpc (fr, r.insKey, resInsKey);
            response->set_allocated_ins_key(resInsKey);

            response->set_ins_value(r.insValue.get_str(16));
            response->set_is_old0(r.isOld0);
            response->set_proof_hash_counter(r.proofHashCounter);
        }

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
    try
    {
        Goldilocks::Element key[4];
        grpc2fea (fr, request->key(), key);

        vector<uint8_t> data;
        std:string sData;

        sData = request->data();

        for (uint64_t i=0; i<sData.size(); i++) {
            data.push_back(sData.at(i));
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
        zkresult r = pHashDB->setProgram(key, data, request->persistent());

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
    string sData;
    try
    {
        Goldilocks::Element key[4];
        grpc2fea (fr, request->key(), key);
#ifdef LOG_HASHDB_SERVICE
        zklog.info("HashDBServiceImpl::GetProgram() called. key=" + fea2string(fr, key[0], key[1], key[2], key[3]));
#endif
        vector<uint8_t> value;
        zkresult r = pHashDB->getProgram(key, value, NULL);

        for (uint64_t i=0; i<value.size(); i++) {
            sData.push_back((char)value.at(i));
        }
        response->set_data(sData);

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
#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::LoadDB called.");
#endif
    try
    {
        DatabaseMap::MTMap map;
        grpc2mtMap(fr, request->input_db(), map);
        pHashDB->loadDB(map, request->persistent());
    }
    catch (const std::exception &e)
    {
        zklog.error("HashDBServiceImpl::LoadDB() exception: " + string(e.what()));
        return Status::CANCELLED;
    }
#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::LoadDB() completed.");
#endif
    return Status::OK;
}

::grpc::Status HashDBServiceImpl::LoadProgramDB(::grpc::ServerContext* context, const ::hashdb::v1::LoadProgramDBRequest* request, ::google::protobuf::Empty* response)
{
#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::LoadProgramDB called.");
#endif
        DatabaseMap::ProgramMap mapProgram;
        grpc2programMap(fr, request->input_program_db(), mapProgram);
        pHashDB->loadProgramDB(mapProgram, request->persistent());
#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::LoadProgramDB() completed.");
#endif
    return Status::OK;
}

::grpc::Status HashDBServiceImpl::Flush(::grpc::ServerContext* context, const ::google::protobuf::Empty* request, ::hashdb::v1::FlushResponse* response)
{
#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::Flush called.");
#endif
    try
    {
        // Call the HashDB flush method
        uint64_t flushId, storedFlushId;
        zkresult zkres = pHashDB->flush(flushId, storedFlushId);

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
#ifdef LOG_HASHDB_SERVICE
    zklog.info("HashDBServiceImpl::GetFlushData called.");
#endif
    try
    {
        // Declare local variables to store the result
        uint64_t storedFlushId;
        vector<FlushData> nodes;
        vector<FlushData> nodesUpdate;
        vector<FlushData> program;
        vector<FlushData> programUpdate;
        string nodesStateRoot;

        // Call the local getFlushData method
        pHashDB->getFlushData(request->flush_id(), storedFlushId, nodes, nodesUpdate, program, programUpdate, nodesStateRoot);

        // Set the last sent flush ID
        response->set_stored_flush_id(storedFlushId);

        // Set the nodes
        for (uint64_t i=0; i<nodes.size(); i++)
        {
            hashdb::v1::FlushData * pFlushData = response->add_nodes();
            if (pFlushData == NULL)
            {
                zklog.error("HashDBServiceImpl::GetFlushData() failed calling response->add_nodes()");
                exitProcess();
            }
            pFlushData->set_key(nodes[i].key);
            pFlushData->set_value(nodes[i].value);
        }

        // Set the nodes update
        for (uint64_t i=0; i<nodesUpdate.size(); i++)
        {
            hashdb::v1::FlushData * pFlushData = response->add_nodes_update();
            if (pFlushData == NULL)
            {
                zklog.error("HashDBServiceImpl::GetFlushData() failed calling response->add_nodes_update()");
                exitProcess();
            }
            pFlushData->set_key(nodes[i].key);
            pFlushData->set_value(nodesUpdate[i].value);
        }

        // Set the program
        for (uint64_t i=0; i<program.size(); i++)
        {
            hashdb::v1::FlushData * pFlushData = response->add_program();
            if (pFlushData == NULL)
            {
                zklog.error("HashDBServiceImpl::GetFlushData() failed calling response->add_program()");
                exitProcess();
            }
            pFlushData->set_key(nodes[i].key);
            pFlushData->set_value(program[i].value);
        }

        // Set the program update
        for (uint64_t i=0; i<programUpdate.size(); i++)
        {
            hashdb::v1::FlushData * pFlushData = response->add_program_update();
            if (pFlushData == NULL)
            {
                zklog.error("HashDBServiceImpl::GetFlushData() failed calling response->add_program_update()");
                exitProcess();
            }
            pFlushData->set_key(nodes[i].key);
            pFlushData->set_value(programUpdate[i].value);
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






