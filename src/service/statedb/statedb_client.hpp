#ifndef STATEDB_CLIENT_HPP
#define STATEDB_CLIENT_HPP

#include "goldilocks/goldilocks_base_field.hpp"
#include "database.hpp"
#include "config.hpp"
#include "smt.hpp"
#include <mutex>
#include "result.hpp"

class StateDBClient 
{
public:
    virtual ~StateDBClient() {};
    virtual result_t set (const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, Goldilocks::Element (&newRoot)[4], SmtSetResult *result) = 0;
    virtual result_t get (const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result) = 0;
    virtual result_t setProgram (const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const bool persistent) = 0;
    virtual result_t getProgram (const Goldilocks::Element (&key)[4], vector<uint8_t> &data) = 0;
    virtual void flush() = 0;
};

#endif