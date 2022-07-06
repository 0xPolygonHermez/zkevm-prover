#ifndef STATEDB_CLIENT_HPP
#define STATEDB_CLIENT_HPP

#include "goldilocks/goldilocks_base_field.hpp"
#include "database.hpp"
#include "config.hpp"
#include "smt.hpp"
#include <mutex>

class StateDBClient 
{
public:
    virtual ~StateDBClient() {};
    virtual int set (const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, Goldilocks::Element (&newRoot)[4], SmtSetResult *result) = 0;
    virtual int get (const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result) = 0;
    virtual int setProgram (const string &hash, const vector<uint8_t> &data, const bool persistent) = 0;
    virtual int getProgram (const string &hash, vector<uint8_t> &data) = 0;
    virtual void flush() = 0;
};

#endif