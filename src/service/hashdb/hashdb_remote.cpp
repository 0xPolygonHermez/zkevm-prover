#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include "hashdb_interface.hpp"
#include "scalar.hpp"
#include "hashdb_utils.hpp"
#include "hashdb_remote.hpp"
#include "zkresult.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

using namespace std;
using json = nlohmann::json;

HashDBRemote::HashDBRemote (Goldilocks &fr, const Config &config) : fr(fr), config(config)
{
	//options = [('grpc.max_message_length', 100 * 1024 * 1024)]

    grpc::ChannelArguments channelArguments;
    channelArguments.SetMaxReceiveMessageSize(100*1024*1024);

    // Create channel
    std::shared_ptr<grpc::Channel> channel = ::grpc::CreateCustomChannel(config.hashDBURL, grpc::InsecureChannelCredentials(), channelArguments);

    // Create stub (i.e. client)
    stub = new hashdb::v1::HashDBService::Stub(channel);
}

HashDBRemote::~HashDBRemote()
{
    delete stub;
    
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.print("HashDBRemote");
#endif    
}

zkresult HashDBRemote::set (const string &batchUUID, uint64_t tx, const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const Persistence persistence, Goldilocks::Element (&newRoot)[4], SmtSetResult *result, DatabaseMap *dbReadLog)
{
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    gettimeofday(&t, NULL);
#endif

    ::grpc::ClientContext context;
    ::hashdb::v1::SetRequest request;
    ::hashdb::v1::SetResponse response;

    ::hashdb::v1::Fea* reqOldRoot = new ::hashdb::v1::Fea();
    fea2grpc(fr, oldRoot, reqOldRoot);
    request.set_allocated_old_root(reqOldRoot);

    ::hashdb::v1::Fea* reqKey = new ::hashdb::v1::Fea();
    fea2grpc(fr, key, reqKey);
    request.set_allocated_key(reqKey);

    request.set_value(value.get_str(16));
    request.set_persistence((hashdb::v1::Persistence)persistence);
    request.set_details(result != NULL);
    request.set_get_db_read_log((dbReadLog != NULL));
    request.set_batch_uuid(batchUUID);
    request.set_tx(tx);

    grpc::Status s = stub->Set(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK)
    {
        zklog.error("HashDBRemote::set() GRPC error(" + to_string(s.error_code()) + "): " + s.error_message());
        return ZKR_HASHDB_GRPC_ERROR;
    }

    grpc2fea(fr, response.new_root(), newRoot);

    if (result != NULL)
    {
        grpc2fea(fr, response.old_root(), result->oldRoot);
        grpc2fea(fr, response.key(), result->key);
        grpc2fea(fr, response.new_root(), result->newRoot);

        google::protobuf::Map<google::protobuf::uint64, hashdb::v1::SiblingList>::iterator it;
        google::protobuf::Map<google::protobuf::uint64, hashdb::v1::SiblingList> siblings;
        siblings = *response.mutable_siblings();
        result->siblings.clear();
        for (it=siblings.begin(); it!=siblings.end(); it++)
        {
            vector<Goldilocks::Element> list;
            for (int i=0; i<it->second.sibling_size(); i++)
            {
                list.push_back(fr.fromU64(it->second.sibling(i)));
            }
            result->siblings[it->first]=list;
        }

        grpc2fea(fr, response.ins_key(), result->insKey);
        result->insValue.set_str(response.ins_value(),16);
        result->isOld0 = response.is_old0();
        result->oldValue.set_str(response.old_value(),16);
        result->newValue.set_str(response.new_value(),16);
        result->mode = response.mode();
        result->proofHashCounter = response.proof_hash_counter();
    }

    if (dbReadLog != NULL)
    {
        DatabaseMap::MTMap mtMap;
        grpc2mtMap(fr, *response.mutable_db_read_log(), mtMap);
        dbReadLog->add(mtMap);
    }

#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.add("set", TimeDiff(t));
#endif

    return static_cast<zkresult>(response.result().code());
}

zkresult HashDBRemote::get (const string &batchUUID, const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result, DatabaseMap *dbReadLog)
{
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    gettimeofday(&t, NULL);
#endif

    ::grpc::ClientContext context;
    ::hashdb::v1::GetRequest request;
    ::hashdb::v1::GetResponse response;

    ::hashdb::v1::Fea* reqRoot = new ::hashdb::v1::Fea();
    fea2grpc(fr, root, reqRoot);
    request.set_allocated_root(reqRoot);

    ::hashdb::v1::Fea* reqKey = new ::hashdb::v1::Fea();
    fea2grpc(fr, key, reqKey);
    request.set_allocated_key(reqKey);
    request.set_details(result != NULL);
    request.set_get_db_read_log((dbReadLog != NULL));
    request.set_batch_uuid(batchUUID);

    grpc::Status s = stub->Get(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK)
    {
        zklog.error("HashDBRemote::get() GRPC error(" + to_string(s.error_code()) + "): " + s.error_message());
        return ZKR_HASHDB_GRPC_ERROR;
    }

    value.set_str(response.value(),16);

    if (result != NULL)
    {
        grpc2fea(fr, response.root(), result->root);
        grpc2fea(fr, response.key(), result->key);
        result->value.set_str(response.value(),16);

        google::protobuf::Map<google::protobuf::uint64, hashdb::v1::SiblingList>::iterator it;
        google::protobuf::Map<google::protobuf::uint64, hashdb::v1::SiblingList> siblings;
        siblings = *response.mutable_siblings();
        result->siblings.clear();
        for (it=siblings.begin(); it!=siblings.end(); it++)
        {
            vector<Goldilocks::Element> list;
            for (int i=0; i<it->second.sibling_size(); i++)
            {
                list.push_back(fr.fromU64(it->second.sibling(i)));
            }
            result->siblings[it->first]=list;
        }

        grpc2fea(fr, response.ins_key(), result->insKey);
        result->insValue.set_str(response.ins_value(),16);
        result->isOld0 = response.is_old0();
        result->proofHashCounter = response.proof_hash_counter();
    }

    if (dbReadLog != NULL)
    {
        DatabaseMap::MTMap mtMap;
        grpc2mtMap(fr, *response.mutable_db_read_log(), mtMap);
        dbReadLog->add(mtMap);
    }

#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.add("get", TimeDiff(t));
#endif

    return static_cast<zkresult>(response.result().code());
}

zkresult HashDBRemote::setProgram (const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const bool persistent)
{
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    gettimeofday(&t, NULL);
#endif

    ::grpc::ClientContext context;
    ::hashdb::v1::SetProgramRequest request;
    ::hashdb::v1::SetProgramResponse response;

    ::hashdb::v1::Fea* reqKey = new ::hashdb::v1::Fea();
    fea2grpc(fr, key, reqKey);
    request.set_allocated_key(reqKey);

    std::string sData;
    for (uint64_t i=0; i<data.size(); i++)
    {
        sData.push_back((char)data.at(i));
    }
    request.set_data(sData);

    request.set_persistent(persistent);

    grpc::Status s = stub->SetProgram(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK)
    {
        zklog.error("HashDBRemote::setProgram() GRPC error(" + to_string(s.error_code()) + "): " + s.error_message());
        return ZKR_HASHDB_GRPC_ERROR;
    }

#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.add("setProgram", TimeDiff(t));
#endif

    return static_cast<zkresult>(response.result().code());
}

zkresult HashDBRemote::getProgram (const Goldilocks::Element (&key)[4], vector<uint8_t> &data, DatabaseMap *dbReadLog)
{
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    gettimeofday(&t, NULL);
#endif

    ::grpc::ClientContext context;
    ::hashdb::v1::GetProgramRequest request;
    ::hashdb::v1::GetProgramResponse response;

    ::hashdb::v1::Fea* reqKey = new ::hashdb::v1::Fea();
    fea2grpc(fr, key, reqKey);
    request.set_allocated_key(reqKey);

    grpc::Status s = stub->GetProgram(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK)
    {
        zklog.error("HashDBRemote::getProgram() GRPC error(" + to_string(s.error_code()) + "): " + s.error_message());
        return ZKR_HASHDB_GRPC_ERROR;
    }

    std:string sData;

    sData = response.data();
    data.clear();
    for (uint64_t i=0; i<sData.size(); i++)
    {
        data.push_back(sData.at(i));
    }

    if (dbReadLog != NULL)
    {
        dbReadLog->add(fea2string(fr, key), data, false, 0);
    }

#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.add("getProgram", TimeDiff(t));
#endif

    return static_cast<zkresult>(response.result().code());
}

void HashDBRemote::loadDB(const DatabaseMap::MTMap &input, const bool persistent)
{
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    gettimeofday(&t, NULL);
#endif
    ::grpc::ClientContext context;
    ::hashdb::v1::LoadDBRequest request;
    ::google::protobuf::Empty response;

    mtMap2grpc(fr, input, request.mutable_input_db());

    request.set_persistent(persistent);

    grpc::Status s = stub->LoadDB(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK)
    {
        zklog.error("HashDBRemote:loadDB() GRPC error(" + to_string(s.error_code()) + "): " + s.error_message());
    }

#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.add("loadDB", TimeDiff(t));
#endif
}

void HashDBRemote::loadProgramDB(const DatabaseMap::ProgramMap &input, const bool persistent)
{
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    gettimeofday(&t, NULL);
#endif
    ::grpc::ClientContext context;
    ::hashdb::v1::LoadProgramDBRequest request;
    ::google::protobuf::Empty response;

    programMap2grpc(fr, input, request.mutable_input_program_db());

    request.set_persistent(persistent);

    grpc::Status s = stub->LoadProgramDB(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK)
    {
        zklog.error("HashDBRemote:loadProgramDB() GRPC error(" + to_string(s.error_code()) + "): " + s.error_message());
    }

#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.add("loadProgramDB", TimeDiff(t));
#endif
}

zkresult HashDBRemote::flush(const string &batchUUID, const string &newStateRoot, const Persistence persistence, uint64_t &flushId, uint64_t &storedFlushId)
{
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    gettimeofday(&t, NULL);
#endif
    ::grpc::ClientContext context;
    ::hashdb::v1::FlushRequest request;
    request.set_batch_uuid(batchUUID);
    request.set_new_state_root(newStateRoot);
    request.set_persistence((hashdb::v1::Persistence)persistence);
    ::hashdb::v1::FlushResponse response;
    grpc::Status s = stub->Flush(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK)
    {
        zklog.error("HashDBRemote:flush() GRPC error(" + to_string(s.error_code()) + "): " + s.error_message());
    }
    flushId = response.flush_id();
    storedFlushId = response.stored_flush_id();

#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.add("flush", TimeDiff(t));
#endif
    return static_cast<zkresult>(response.result().code());
}

void HashDBRemote::semiFlush (const string &batchUUID, const string &newStateRoot, const Persistence persistence)
{
    ::grpc::ClientContext context;
    ::hashdb::v1::SemiFlushRequest request;
    request.set_batch_uuid(batchUUID);
    request.set_new_state_root(newStateRoot);
    request.set_persistence((hashdb::v1::Persistence)persistence);
    ::google::protobuf::Empty response;
    grpc::Status s = stub->SemiFlush(&context, request, &response);
}

zkresult HashDBRemote::purge (const string &batchUUID, const Goldilocks::Element (&newStateRoot)[4], const Persistence persistence)
{
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    gettimeofday(&t, NULL);
#endif
    ::grpc::ClientContext context;
    ::hashdb::v1::PurgeRequest request;
    request.set_batch_uuid(batchUUID);
    hashdb::v1::Fea *new_state_root = new hashdb::v1::Fea();
    if (new_state_root == NULL)
    {
        zklog.error("HashDBRemote::purge() failed allocating hashdb::v1::Fea for new_state_root");
        exitProcess();
    }
    new_state_root->set_fe0(fr.toU64(newStateRoot[0]));
    new_state_root->set_fe1(fr.toU64(newStateRoot[1]));
    new_state_root->set_fe2(fr.toU64(newStateRoot[2]));
    new_state_root->set_fe3(fr.toU64(newStateRoot[3]));
    request.set_allocated_new_state_root(new_state_root);
    request.set_persistence((hashdb::v1::Persistence)persistence);
    ::hashdb::v1::PurgeResponse response;
    grpc::Status s = stub->Purge(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK)
    {
        zklog.error("HashDBRemote:purge() GRPC error(" + to_string(s.error_code()) + "): " + s.error_message());
    }

#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.add("purge", TimeDiff(t));
#endif
    return static_cast<zkresult>(response.result().code());
}

zkresult HashDBRemote::consolidateState (const Goldilocks::Element (&virtualStateRoot)[4], const Persistence persistence, Goldilocks::Element (&consolidatedStateRoot)[4], uint64_t &flushId, uint64_t &storedFlushId)
{
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    gettimeofday(&t, NULL);
#endif
    ::grpc::ClientContext context;

    // Prepare the ConsolidateStateRequest
    ::hashdb::v1::ConsolidateStateRequest request;
    hashdb::v1::Fea *virtual_state_root = new hashdb::v1::Fea();
    if (virtual_state_root == NULL)
    {
        zklog.error("HashDBRemote::consolidateState() failed allocating hashdb::v1::Fea");
        exitProcess();
    }
    fea2grpc(fr, virtualStateRoot, virtual_state_root);
    request.set_allocated_virtual_state_root(virtual_state_root);
    request.set_persistence((hashdb::v1::Persistence)persistence);

    // Call ConsolidateState()
    ::hashdb::v1::ConsolidateStateResponse response;
    grpc::Status s = stub->ConsolidateState(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK)
    {
        zklog.error("HashDBRemote:consolidateState() GRPC error(" + to_string(s.error_code()) + "): " + s.error_message());
    }

    // Get return data from the response
    grpc2fea(fr, response.consolidated_state_root(), consolidatedStateRoot);
    flushId = response.flush_id();
    storedFlushId = response.stored_flush_id();

#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.add("consolidateState", TimeDiff(t));
#endif
    return static_cast<zkresult>(response.result().code());
}

zkresult HashDBRemote::getFlushStatus(uint64_t &storedFlushId, uint64_t &storingFlushId, uint64_t &lastFlushId, uint64_t &pendingToFlushNodes, uint64_t &pendingToFlushProgram, uint64_t &storingNodes, uint64_t &storingProgram, string &proverId)
{
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    gettimeofday(&t, NULL);
#endif
    ::grpc::ClientContext context;
    ::google::protobuf::Empty request;
    ::hashdb::v1::GetFlushStatusResponse response;

    // Call GetFlushStatus()
    grpc::Status s = stub->GetFlushStatus(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK)
    {
        zklog.error("HashDBRemote:getFlushStatus() GRPC error(" + to_string(s.error_code()) + "): " + s.error_message());
        return ZKR_HASHDB_GRPC_ERROR;
    }

    // Get response data
    storedFlushId = response.stored_flush_id();
    storingFlushId = response.storing_flush_id();
    lastFlushId = response.last_flush_id();
    pendingToFlushNodes = response.pending_to_flush_nodes();
    pendingToFlushProgram = response.pending_to_flush_program();
    storingNodes = response.storing_nodes();
    storingProgram = response.storing_program();
    proverId = response.prover_id();

#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.add("getFlushStatus", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

zkresult HashDBRemote::getFlushData(uint64_t flushId, uint64_t &storedFlushId, unordered_map<string, string> (&nodes), unordered_map<string, string> (&program), string &nodesStateRoot)
{
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    gettimeofday(&t, NULL);
#endif
    ::grpc::ClientContext context;

    // Prepare the request
    ::hashdb::v1::GetFlushDataRequest request;
    request.set_flush_id(flushId);

    // Declare the response
    ::hashdb::v1::GetFlushDataResponse response;

    // Call the gRPC GetFlushData method
    grpc::Status s = stub->GetFlushData(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK)
    {
        zklog.error("HashDBRemote:getFlushData() GRPC error(" + to_string(s.error_code()) + "): " + s.error_message());
        return ZKR_HASHDB_GRPC_ERROR;
    }

    // Copy the last sent flush ID
    storedFlushId = response.stored_flush_id();

    // Copy the nodes vector
    nodes.clear();
    ::PROTOBUF_NAMESPACE_ID::Map<string, string>::const_iterator it;
    for (it = response.nodes().begin(); it != response.nodes().end(); it++)
    {
        nodes[it->first] = it->second;
    }

    // Copy the program vector
    program.clear();
    for (it = response.program().begin(); it != response.program().end(); it++)
    {
        program[it->first] = it->second;
    }

    // Copy the nodes state root
    nodesStateRoot = response.nodes_state_root();    

#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.add("getFlushData", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

zkresult HashDBRemote::readTree (const Goldilocks::Element (&root)[4], vector<KeyValue> &keyValues, vector<HashValueGL> &hashValues)
{
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    gettimeofday(&t, NULL);
#endif
    ::grpc::ClientContext context;

    // Prepare the ReadTreeRequest
    ::hashdb::v1::ReadTreeRequest request;
    hashdb::v1::Fea *state_root = new hashdb::v1::Fea();
    if (state_root == NULL)
    {
        zklog.error("HashDBRemote::readTree() failed allocating hashdb::v1::Fea");
        exitProcess();
    }
    fea2grpc(fr, root, state_root);
    request.set_allocated_state_root(state_root);
    
    // Prepare the keys
    for (uint64_t i=0; i<keyValues.size(); i++)
    {
        hashdb::v1::Fea *keyFea = request.add_keys();
        fea2grpc(fr, keyValues[i].key, keyFea);
    }

    // Call ReadTree()
    ::hashdb::v1::ReadTreeResponse response;
    grpc::Status s = stub->ReadTree(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK)
    {
        zklog.error("HashDBRemote:readTree() GRPC error(" + to_string(s.error_code()) + "): " + s.error_message());
    }
    else
    {
        // Get return data from the response
        keyValues.clear();
        for (int64_t i=0; i<response.key_value_size(); i++)
        {
            KeyValue kv;
            kv.key[0] = fr.fromU64(response.key_value(i).key().fe0());
            kv.key[1] = fr.fromU64(response.key_value(i).key().fe1());
            kv.key[2] = fr.fromU64(response.key_value(i).key().fe2());
            kv.key[3] = fr.fromU64(response.key_value(i).key().fe3());
            string valueString = response.key_value(i).value();
            if (!stringIsHex(valueString))
            {
                zklog.error("HashDBRemote:readTree() found invalid valueString=" + valueString);
                exitProcess();
            }
            if (valueString.size() > 64)
            {
                zklog.error("HashDBRemote:readTree() found invalid valueString.size=" + to_string(valueString.size()));
                exitProcess();
            }
            kv.value.set_str(response.key_value(i).value(), 16);
            keyValues.emplace_back(kv);
        }
    }

#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.add("consolidateState", TimeDiff(t));
#endif

    return static_cast<zkresult>(response.result().code());
}

zkresult HashDBRemote::writeTree (const Goldilocks::Element (&oldRoot)[4], const vector<KeyValue> &keyValues, Goldilocks::Element (&newRoot)[4], const bool persistent)
{
    zklog.error("HashDBRemote::writeTree() called, but this method is only allowed locally");
    return ZKR_UNSPECIFIED;
}

zkresult HashDBRemote::cancelBatch (const string &batchUUID)
{
#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    gettimeofday(&t, NULL);
#endif
    ::grpc::ClientContext context;
    ::hashdb::v1::CancelBatchRequest request;
    request.set_batch_uuid(batchUUID);
    ::hashdb::v1::CancelBatchResponse response;
    grpc::Status s = stub->CancelBatch(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK)
    {
        zklog.error("HashDBRemote:cancelBatch() GRPC error(" + to_string(s.error_code()) + "): " + s.error_message());
    }

#ifdef LOG_TIME_STATISTICS_HASHDB_REMOTE
    tms.add("cancelBatch", TimeDiff(t));
#endif
    return static_cast<zkresult>(response.result().code());
}
