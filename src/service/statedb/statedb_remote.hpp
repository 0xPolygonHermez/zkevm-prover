#ifndef STATEDB_REMOTE_HPP
#define STATEDB_REMOTE_HPP

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <sys/time.h>
#include "statedb.grpc.pb.h"
#include "goldilocks_base_field.hpp"
#include "smt.hpp"
#include "statedb_interface.hpp"
#include "zkresult.hpp"
#include "utils/time_metric.hpp"
#include "timer.hpp"

class StateDBRemote : public StateDBInterface
{
private:
    Goldilocks &fr;
    const Config &config;
    ::statedb::v1::StateDBService::Stub *stub;
#ifdef LOG_TIME_STATISTICS_STATEDB_REMOTE
    TimeMetricStorage tms;
    struct timeval t;
#endif
public:
    StateDBRemote(Goldilocks &fr, const Config &config);
    ~StateDBRemote();
    zkresult set(const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, Goldilocks::Element (&newRoot)[4], SmtSetResult *result, DatabaseMap *dbReadLog);
    zkresult get(const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result, DatabaseMap *dbReadLog);
    zkresult setProgram(const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const bool persistent);
    zkresult getProgram(const Goldilocks::Element (&key)[4], vector<uint8_t> &data, DatabaseMap *dbReadLog);
    void loadDB(const DatabaseMap::MTMap &input, const bool persistent);
    void loadProgramDB(const DatabaseMap::ProgramMap &input, const bool persistent);
    zkresult flush();
};

#endif