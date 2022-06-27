#ifndef STATEDB_CLIENT_HPP
#define STATEDB_CLIENT_HPP

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include "statedb.grpc.pb.h"
#include "goldilocks/goldilocks_base_field.hpp"
#include "smt.hpp"

class StateDBClient
{
public:
    Goldilocks &fr;
    const Config &config;
    ::statedb::v1::StateDBService::Stub * stub;

public:
    StateDBClient (Goldilocks &fr, const Config &config);

    void set (const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], mpz_class &value, const bool persistent, const bool details, SmtSetResult &result);
    void set (const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtSetResult &result);
    void get (const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], const bool details, SmtGetResult &result);
    void flush();
};

#endif