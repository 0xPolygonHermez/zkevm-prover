#ifndef STATEDB_LOCAL_CLIENT_HPP
#define STATEDB_LOCAL_CLIENT_HPP

#include "goldilocks_base_field.hpp"
#include "database.hpp"
#include "config.hpp"
#include "smt.hpp"
#include "statedb_client.hpp"
#include "zkresult.hpp"

class StateDBLocalClient : public StateDBClient
{
private:
    Goldilocks &fr;
    const Config &config;
    Database db;
    Smt smt;

public:
    StateDBLocalClient (Goldilocks &fr, const Config &config);
    zkresult set (const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, Goldilocks::Element (&newRoot)[4], SmtSetResult *result);
    zkresult get (const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result);
    zkresult setProgram (const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const bool persistent);
    zkresult getProgram (const Goldilocks::Element (&key)[4], vector<uint8_t> &data);
    void flush ();
    Database * getDatabase (void);

    // Methods added for testing purposes
    void setAutoCommit (const bool autoCommit);
    void commit ();
    void hashSave (const Goldilocks::Element (&a)[8], const Goldilocks::Element (&c)[4], const bool persistent, Goldilocks::Element (&hash)[4]);
};

#endif