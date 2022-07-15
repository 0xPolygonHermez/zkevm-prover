#ifndef STATEDB_CLIENT_HPP
#define STATEDB_CLIENT_HPP

#include "goldilocks/goldilocks_base_field.hpp"
#include "database.hpp"
#include "config.hpp"
#include "smt.hpp"
#include <mutex>
#include "zkresult.hpp"

class StateDBClient 
{
public:
    virtual ~StateDBClient() {};
    virtual zkresult set (const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, Goldilocks::Element (&newRoot)[4], SmtSetResult *result) = 0;
    virtual zkresult get (const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result) = 0;
    virtual zkresult setProgram (const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const bool persistent) = 0;
    virtual zkresult getProgram (const Goldilocks::Element (&key)[4], vector<uint8_t> &data) = 0;
    virtual void flush() = 0;
};

#endif