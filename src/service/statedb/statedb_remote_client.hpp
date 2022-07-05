#ifndef STATEDB_REMOTE_CLIENT_HPP
#define STATEDB_REMOTE_CLIENT_HPP

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include "statedb.grpc.pb.h"
#include "goldilocks/goldilocks_base_field.hpp"
#include "smt.hpp"
#include "statedb_client.hpp"

class StateDBRemoteClient : public StateDBClient
{
private:
    Goldilocks &fr;
    const Config &config;
    ::statedb::v1::StateDBService::Stub *stub;

public:
    StateDBRemoteClient (Goldilocks &fr, const Config &config);

    int set (const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, Goldilocks::Element (&newRoot)[4], SmtSetResult *result);
    int get (const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result);
    int setProgram (const string &hash, const vector<uint8_t> &data, const bool persistent);
    int getProgram (const string &hash, vector<uint8_t> &data);
    void flush();
};

#endif