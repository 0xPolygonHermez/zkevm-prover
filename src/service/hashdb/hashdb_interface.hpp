#ifndef HASHDB_INTERFACE_HPP
#define HASHDB_INTERFACE_HPP

#include "goldilocks_base_field.hpp"
#include "database.hpp"
#include "database_map.hpp"
#include "config.hpp"
#include "smt.hpp"
#include <mutex>
#include "zkresult.hpp"

class HashDBInterface
{
public:
    virtual ~HashDBInterface(){};
    virtual zkresult set(const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, Goldilocks::Element (&newRoot)[4], SmtSetResult *result, DatabaseMap *dbReadLog) = 0;
    virtual zkresult get(const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result, DatabaseMap *dbReadLog) = 0;
    virtual zkresult setProgram(const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const bool persistent) = 0;
    virtual zkresult getProgram(const Goldilocks::Element (&key)[4], vector<uint8_t> &data, DatabaseMap *dbReadLog) = 0;
    virtual void loadDB(const DatabaseMap::MTMap &input, const bool persistent) = 0;
    virtual void loadProgramDB(const DatabaseMap::ProgramMap &input, const bool persistent) = 0;
    virtual zkresult flush(uint64_t &flushId, uint64_t &storedFlushId) = 0;
    virtual zkresult getFlushStatus(uint64_t &storedFlushId, uint64_t &storingFlushId, uint64_t &lastFlushId, uint64_t &pendingToFlushNodes, uint64_t &pendingToFlushProgram, uint64_t &storingNodes, uint64_t &storingProgram, string &proverId) = 0;
    virtual zkresult getFlushData(uint64_t flushId, uint64_t &storedFlushId, unordered_map<string, string> (&nodes), unordered_map<string, string> (&nodesUpdate), unordered_map<string, string> (&program), unordered_map<string, string> (&programUpdate), string &nodesStateRoot) = 0;
    virtual void clearCache(void) = 0;
};

#endif