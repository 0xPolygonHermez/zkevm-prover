#include "statedb_service.hpp"
#include <grpcpp/grpcpp.h>
#include "smt.hpp"
#include "goldilocks/goldilocks_base_field.hpp"
#include "statedb_utils.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

//#define LOG_STATEDB_SERVICE
StateDBServiceImpl::StateDBServiceImpl (Goldilocks &fr, const Config& config, const bool autoCommit, const bool asyncWrite) : fr(fr), config(config), db(fr), smt(fr)
{
    db.init(config);
}

::grpc::Status StateDBServiceImpl::Set(::grpc::ServerContext* context, const ::statedb::v1::SetRequest* request, ::statedb::v1::SetResponse* response)
{
#ifdef LOG_STATEDB_SERVICE
    cout << "StateDBServiceImpl::set() called with request: " << request->DebugString() << endl;
#endif
    std::lock_guard<std::mutex> lock(mutex);

    SmtSetResult r;

    Goldilocks::Element oldRoot[4];
    grpc2fea (fr, request->old_root(), oldRoot);

    Goldilocks::Element key[4];
    grpc2fea (fr, request->key(), key);

    mpz_class value(request->value(),16);
    bool persistent = request->persistent();

    smt.set (db, oldRoot, key, value, persistent, r);

    ::statedb::v1::Fea* resNewRoot = new ::statedb::v1::Fea();
    fea2grpc (fr, r.newRoot, resNewRoot);
    response->set_allocated_new_root(resNewRoot);

    if (request->details()) {
        ::statedb::v1::Fea* resOldRoot = new ::statedb::v1::Fea();
        fea2grpc (fr, r.oldRoot, resOldRoot);
        response->set_allocated_old_root(resOldRoot);

        ::statedb::v1::Fea* resKey = new ::statedb::v1::Fea();
        fea2grpc (fr, r.key, resKey);
        response->set_allocated_key(resKey);    

        for (auto & [level, siblingList] : r.siblings) {
            ::statedb::v1::SiblingList list;
            for (uint64_t i=0; i<siblingList.size(); i++) {
                list.add_sibling(fr.toU64(siblingList[i]));
            }
            (*response->mutable_siblings())[level] = list;
        }

        ::statedb::v1::Fea* resInsKey = new ::statedb::v1::Fea();
        fea2grpc (fr, r.insKey, resInsKey);
        response->set_allocated_key(resInsKey);  

        response->set_ins_value(r.insValue.get_str(16));
        response->set_is_old0(r.isOld0);
        response->set_old_value(r.oldValue.get_str(16));
        response->set_new_value(r.newValue.get_str(16));
        response->set_mode(r.mode);
    }

#ifdef LOG_STATEDB_SERVICE
    cout << "StateDBServiceImpl::set() returns: " << response->DebugString() << endl;
#endif
    return Status::OK;
}

::grpc::Status StateDBServiceImpl::Get(::grpc::ServerContext* context, const ::statedb::v1::GetRequest* request, ::statedb::v1::GetResponse* response)
{
#ifdef LOG_STATEDB_SERVICE
    cout << "StateDBServiceImpl::Get() called with request: " << request->DebugString() << endl;
#endif
    std::lock_guard<std::mutex> lock(mutex);

    SmtGetResult r;
    
    ::statedb::v1::Fea reqRoot;
    reqRoot = request->root();
    Goldilocks::Element root[4] = {reqRoot.fe0(), reqRoot.fe1(), reqRoot.fe2(), reqRoot.fe3()};

    ::statedb::v1::Fea reqKey;
    reqKey = request->key();
    Goldilocks::Element key[4] = {reqKey.fe0(), reqKey.fe1(), reqKey.fe2(), reqKey.fe3()};

    smt.get (db, root, key, r);      

    response->set_value(r.value.get_str(16));

    if (request->details()) {
        ::statedb::v1::Fea* resRoot = new ::statedb::v1::Fea();
        fea2grpc (fr, r.root, resRoot);
        response->set_allocated_root(resRoot);

        ::statedb::v1::Fea* resKey = new ::statedb::v1::Fea();
        fea2grpc (fr, r.key, resKey);
        response->set_allocated_key(resKey);

        for (auto & [level, siblingList] : r.siblings) {
            ::statedb::v1::SiblingList list;
            for (uint64_t i=0; i<siblingList.size(); i++) {
                list.add_sibling(fr.toU64(siblingList[i]));
            }
            (*response->mutable_siblings())[level] = list;
        }

        ::statedb::v1::Fea* resInsKey = new ::statedb::v1::Fea();
        fea2grpc (fr, r.insKey, resInsKey);
        response->set_allocated_key(resInsKey);

        response->set_ins_value(r.insValue.get_str(16));
        response->set_is_old0(r.isOld0);
    }
#ifdef LOG_STATEDB_SERVICE
    cout << "StateDBServiceImpl::Get() returns: " << response->DebugString() << endl;
#endif
    return Status::OK;
}

::grpc::Status StateDBServiceImpl::SetProgram(::grpc::ServerContext* context, const ::statedb::v1::SetProgramRequest* request, ::statedb::v1::SetProgramResponse* response)
{
#ifdef LOG_STATEDB_SERVICE
    cout << "StateDBServiceImpl::SetProgram() called with request: " << request->DebugString() << endl;
#endif

    vector<uint8_t> value;
    std:string sValue;

    sValue = request->data();

    for (uint64_t i=0; sValue.size(); i++) {
        value.push_back(sValue.at(i));
    }
    
    db.setProgram (request->hash(), value, request->persistent());

    ::statedb::v1::ResultCode* result = new ::statedb::v1::ResultCode();
    //· Devolver codigo resultado correcto
    result->set_code(::statedb::v1::ResultCode_Code_CODE_SUCCESS);
    response->set_allocated_result(result);

#ifdef LOG_STATEDB_SERVICE
    cout << "StateDBServiceImpl::Get() returns: " << response->DebugString() << endl;
#endif
    return Status::OK;
}

::grpc::Status StateDBServiceImpl::GetProgram(::grpc::ServerContext* context, const ::statedb::v1::GetProgramRequest* request, ::statedb::v1::GetProgramResponse* response)
{
#ifdef LOG_STATEDB_SERVICE
    cout << "StateDBServiceImpl::GetProgram() called with request: " << request->DebugString() << endl;
#endif
    vector<uint8_t> value;

    db.getProgram(request->hash(), value);

    std::string sValue;
    for (uint64_t i=0; i<value.size(); i++) {
        sValue.push_back((char)value.at(i));
    }
    response->set_data(sValue);

    ::statedb::v1::ResultCode* result = new ::statedb::v1::ResultCode();
    //· Devolver codigo resultado correcto
    result->set_code(::statedb::v1::ResultCode_Code_CODE_SUCCESS);
    response->set_allocated_result(result);

#ifdef LOG_STATEDB_SERVICE
    cout << "StateDBServiceImpl::Get() returns: " << response->DebugString() << endl;
#endif
    return Status::OK;
}

::grpc::Status StateDBServiceImpl::Flush(::grpc::ServerContext* context, const ::google::protobuf::Empty* request, ::google::protobuf::Empty* response)
{
    db.flush();
    return Status::OK;
}






