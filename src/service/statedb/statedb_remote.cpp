#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include "statedb_interface.hpp"
#include "scalar.hpp"
#include "statedb_utils.hpp"
#include "statedb_remote.hpp"
#include "zkresult.hpp"

using namespace std;
using json = nlohmann::json;

StateDBRemote::StateDBRemote (Goldilocks &fr, const Config &config) : fr(fr), config(config)
{
    // Create channel
    std::shared_ptr<grpc_impl::Channel> channel = ::grpc::CreateChannel(config.stateDBURL, grpc::InsecureChannelCredentials());

    // Create stub (i.e. client)
    stub = new statedb::v1::StateDBService::Stub(channel);
}

StateDBRemote::~StateDBRemote()
{
    delete stub;
    
#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    tms.print("StateDBRemote");
#endif    
}

zkresult StateDBRemote::set (const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, Goldilocks::Element (&newRoot)[4], SmtSetResult *result, DatabaseMap *dbReadLog)
{
#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    gettimeofday(&t, NULL);
#endif

    ::grpc::ClientContext context;
    ::statedb::v1::SetRequest request;
    ::statedb::v1::SetResponse response;

    ::statedb::v1::Fea* reqOldRoot = new ::statedb::v1::Fea();
    fea2grpc(fr, oldRoot, reqOldRoot);
    request.set_allocated_old_root(reqOldRoot);

    ::statedb::v1::Fea* reqKey = new ::statedb::v1::Fea();
    fea2grpc(fr, key, reqKey);
    request.set_allocated_key(reqKey);

    request.set_value(value.get_str(16));
    request.set_persistent(persistent);
    request.set_details(result != NULL);
    request.set_get_db_read_log((dbReadLog != NULL));

    grpc::Status s = stub->Set(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK) {
        cerr << "Error: StateDBRemote::set() GRPC error(" << s.error_code() << "): " << s.error_message() << endl;
        return ZKR_STATEDB_GRPC_ERROR;
    }

    grpc2fea(fr, response.new_root(), newRoot);

    if (result != NULL) {
        grpc2fea(fr, response.old_root(), result->oldRoot);
        grpc2fea(fr, response.key(), result->key);
        grpc2fea(fr, response.new_root(), result->newRoot);

        google::protobuf::Map<google::protobuf::uint64, statedb::v1::SiblingList>::iterator it;
        google::protobuf::Map<google::protobuf::uint64, statedb::v1::SiblingList> siblings;
        siblings = *response.mutable_siblings();
        result->siblings.clear();
        for (it=siblings.begin(); it!=siblings.end(); it++)
        {
            vector<Goldilocks::Element> list;
            for (int i=0; i<it->second.sibling_size(); i++) {
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

    if (dbReadLog != NULL) {
        DatabaseMap::MTMap mtMap;
        grpc2mtMap(fr, *response.mutable_db_read_log(), mtMap);
        dbReadLog->add(mtMap);
    }

#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    tms.add("set", TimeDiff(t));
#endif

    return static_cast<zkresult>(response.result().code());
}

zkresult StateDBRemote::get (const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result, DatabaseMap *dbReadLog)
{
#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    gettimeofday(&t, NULL);
#endif

    ::grpc::ClientContext context;
    ::statedb::v1::GetRequest request;
    ::statedb::v1::GetResponse response;

    ::statedb::v1::Fea* reqRoot = new ::statedb::v1::Fea();
    fea2grpc(fr, root, reqRoot);
    request.set_allocated_root(reqRoot);

    ::statedb::v1::Fea* reqKey = new ::statedb::v1::Fea();
    fea2grpc(fr, key, reqKey);
    request.set_allocated_key(reqKey);
    request.set_details(result != NULL);
    request.set_get_db_read_log((dbReadLog != NULL));

    grpc::Status s = stub->Get(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK) {
        cerr << "Error: StateDBRemote::get() GRPC error(" << s.error_code() << "): " << s.error_message() << endl;
        return ZKR_STATEDB_GRPC_ERROR;
    }

    value.set_str(response.value(),16);

    if (result != NULL) {
        grpc2fea(fr, response.root(), result->root);
        grpc2fea(fr, response.key(), result->key);
        result->value.set_str(response.value(),16);

        google::protobuf::Map<google::protobuf::uint64, statedb::v1::SiblingList>::iterator it;
        google::protobuf::Map<google::protobuf::uint64, statedb::v1::SiblingList> siblings;
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

    if (dbReadLog != NULL) {
        DatabaseMap::MTMap mtMap;
        grpc2mtMap(fr, *response.mutable_db_read_log(), mtMap);
        dbReadLog->add(mtMap);
    }

#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    tms.add("get", TimeDiff(t));
#endif

    return static_cast<zkresult>(response.result().code());
}

zkresult StateDBRemote::setProgram (const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const bool persistent)
{
#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    gettimeofday(&t, NULL);
#endif

    ::grpc::ClientContext context;
    ::statedb::v1::SetProgramRequest request;
    ::statedb::v1::SetProgramResponse response;

    ::statedb::v1::Fea* reqKey = new ::statedb::v1::Fea();
    fea2grpc(fr, key, reqKey);
    request.set_allocated_key(reqKey);

    std::string sData;
    for (uint64_t i=0; i<data.size(); i++) {
        sData.push_back((char)data.at(i));
    }
    request.set_data(sData);

    request.set_persistent(persistent);

    grpc::Status s = stub->SetProgram(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK) {
        cerr << "Error: StateDBRemote::setProgram() GRPC error(" << s.error_code() << "): " << s.error_message() << endl;
        return ZKR_STATEDB_GRPC_ERROR;
    }

#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    tms.add("setProgram", TimeDiff(t));
#endif

    return static_cast<zkresult>(response.result().code());
}

zkresult StateDBRemote::getProgram (const Goldilocks::Element (&key)[4], vector<uint8_t> &data, DatabaseMap *dbReadLog)
{
#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    gettimeofday(&t, NULL);
#endif

    ::grpc::ClientContext context;
    ::statedb::v1::GetProgramRequest request;
    ::statedb::v1::GetProgramResponse response;

    ::statedb::v1::Fea* reqKey = new ::statedb::v1::Fea();
    fea2grpc(fr, key, reqKey);
    request.set_allocated_key(reqKey);

    grpc::Status s = stub->GetProgram(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK) {
        cerr << "Error: StateDBRemote::getProgram() GRPC error(" << s.error_code() << "): " << s.error_message() << endl;
        return ZKR_STATEDB_GRPC_ERROR;
    }

    std:string sData;

    sData = response.data();
    data.clear();
    for (uint64_t i=0; i<sData.size(); i++) {
        data.push_back(sData.at(i));
    }

    if (dbReadLog != NULL) {
        dbReadLog->add(fea2string(fr, key), data);
    }

#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    tms.add("getProgram", TimeDiff(t));
#endif

    return static_cast<zkresult>(response.result().code());
}

void StateDBRemote::loadDB(const DatabaseMap::MTMap &input, const bool persistent)
{
#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    gettimeofday(&t, NULL);
#endif
    ::grpc::ClientContext context;
    ::statedb::v1::LoadDBRequest request;
    ::google::protobuf::Empty response;

    mtMap2grpc(fr, input, request.mutable_input_db());

    request.set_persistent(persistent);

    grpc::Status s = stub->LoadDB(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK) {
        cerr << "Error: StateDBRemote:loadDB() GRPC error(" << s.error_code() << "): " << s.error_message() << endl;
    }

#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    tms.add("loadDB", TimeDiff(t));
#endif
}

void StateDBRemote::loadProgramDB(const DatabaseMap::ProgramMap &input, const bool persistent)
{
#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    gettimeofday(&t, NULL);
#endif
    ::grpc::ClientContext context;
    ::statedb::v1::LoadProgramDBRequest request;
    ::google::protobuf::Empty response;

    programMap2grpc(fr, input, request.mutable_input_program_db());

    request.set_persistent(persistent);

    grpc::Status s = stub->LoadProgramDB(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK) {
        cerr << "Error: StateDBRemote:loadProgramDB() GRPC error(" << s.error_code() << "): " << s.error_message() << endl;
    }

#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    tms.add("loadProgramDB", TimeDiff(t));
#endif
}

zkresult StateDBRemote::flush()
{
#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    gettimeofday(&t, NULL);
#endif
    ::grpc::ClientContext context;
    ::google::protobuf::Empty request;
    ::statedb::v1::FlushResponse response;
    grpc::Status s = stub->Flush(&context, request, &response);
    if (s.error_code() != grpc::StatusCode::OK) {
        cerr << "Error: StateDBRemote:flush() GRPC error(" << s.error_code() << "): " << s.error_message() << endl;
    }

#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    tms.add("flush", TimeDiff(t));
#endif
    return static_cast<zkresult>(response.result().code());
}