#include <nlohmann/json.hpp>
#include "statedb_client.hpp"
#include "scalar.hpp"
#include "statedb_utils.hpp"

using namespace std;
using json = nlohmann::json;

//#define LOG_STATEDB_CLIENT

StateDBClient::StateDBClient (Goldilocks &fr, const Config &config) : fr(fr), config(config)
{
    // Create channel
    std::shared_ptr<grpc_impl::Channel> channel = ::grpc::CreateChannel("localhost:" + to_string(config.stateDBClientPort), grpc::InsecureChannelCredentials());

    // Create stub (i.e. client)
    stub = new statedb::v1::StateDBService::Stub(channel);
}

void StateDBClient::set (const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtSetResult &result)
{
    set (oldRoot, key, value, true, true, result);
}

void StateDBClient::set (const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], mpz_class &value, const bool persistent, const bool details, SmtSetResult &result)
{
    ::grpc::ClientContext context;
    ::statedb::v1::SetRequest request;
    ::statedb::v1::SetResponse response;

    ::statedb::v1::fea* reqOldRoot = new ::statedb::v1::fea();
    fea2grpc(fr, oldRoot, reqOldRoot);
    request.set_allocated_old_root(reqOldRoot);

    ::statedb::v1::fea* reqKey = new ::statedb::v1::fea();
    fea2grpc(fr, key, reqKey);
    request.set_allocated_key(reqKey);

    request.set_value(value.get_str(16));
    request.set_persistent(persistent);
    request.set_details(details);

    stub->Set(&context, request, &response);

    grpc2fea(fr, response.new_root(), result.newRoot);

    if (details) {
        grpc2fea(fr, response.old_root(), result.oldRoot);
        grpc2fea(fr, response.key(), result.key);

        for (auto & [level, siblingList] : *response.mutable_siblings())
        {
            vector<Goldilocks::Element> list;
            for (int i=0; i<siblingList.sibling_size(); i++) {
                list.push_back(fr.fromU64(siblingList.sibling(i)));
            }
            result.siblings[level]=list;
        }

        grpc2fea(fr, response.ins_key(), result.insKey);
        result.insValue.set_str(response.ins_value(),16);
        result.isOld0 = response.is_old0();
        result.oldValue.set_str(response.old_value(),16);
        result.newValue.set_str(response.new_value(),16);
        result.mode = response.mode();
    }

#ifdef LOG_STATEDB_CLIENT
    cout << "StateDBClient::set() response: " << response.DebugString() << endl;
#endif    
}

void StateDBClient::get (const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], const bool details, SmtGetResult &result)
{
    ::grpc::ClientContext context;
    ::statedb::v1::GetRequest request;
    ::statedb::v1::GetResponse response;
    
    ::statedb::v1::fea* reqRoot = new ::statedb::v1::fea();    
    fea2grpc(fr, root, reqRoot);
    request.set_allocated_root(reqRoot);

    ::statedb::v1::fea* reqKey = new ::statedb::v1::fea();
    fea2grpc(fr, key, reqKey);
    request.set_allocated_key(reqKey);
    request.set_details(details);

    stub->Get(&context, request, &response);

    result.value.set_str(response.value(),16);

    if (details) {
        grpc2fea(fr, response.root(), result.root);
        grpc2fea(fr, response.key(), result.key);

        for (auto & [level, siblingList] : *response.mutable_siblings())
        {
            vector<Goldilocks::Element> list;
            for (int i=0; i<siblingList.sibling_size(); i++) {
                list.push_back(fr.fromU64(siblingList.sibling(i)));
            }
            result.siblings[level]=list;
        }  

        grpc2fea(fr, response.ins_key(), result.insKey);
        result.insValue.set_str(response.ins_value(),16);
        result.isOld0 = response.is_old0();
    }

#ifdef LOG_STATEDB_CLIENT
    cout << "StateDBClient::get() response: " << response.DebugString() << endl;
#endif    
}

void StateDBClient::flush()
{
    ::grpc::ClientContext context;
    ::google::protobuf::Empty request;
    ::google::protobuf::Empty response;
    stub->Flush(&context, request, &response);
}
