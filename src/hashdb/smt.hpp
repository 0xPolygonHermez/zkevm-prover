#ifndef SMT_HPP
#define SMT_HPP

#include <vector>
#include <map>
#include <gmpxx.h>

#include "poseidon_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#include "compare_fe.hpp"
#include "database.hpp"
#include "database_map.hpp"
#include "zkresult.hpp"

using namespace std;

// SMT set method result data
class SmtSetResult
{
public:
    Goldilocks::Element oldRoot[4];
    Goldilocks::Element newRoot[4];
    Goldilocks::Element key[4];
    map< uint64_t, vector<Goldilocks::Element> > siblings;
    Goldilocks::Element insKey[4];
    mpz_class insValue;
    bool isOld0;
    mpz_class oldValue;
    mpz_class newValue;
    string mode;
    uint64_t proofHashCounter;
    string toString (Goldilocks &fr);
};

// SMT get method result data
class SmtGetResult
{
public:
    Goldilocks::Element root[4]; // merkle-tree root
    Goldilocks::Element key[4]; // key to look for
    map< uint64_t, vector<Goldilocks::Element> > siblings; // array of siblings // array of fields??
    Goldilocks::Element insKey[4]; // key found
    mpz_class insValue; // value found
    bool isOld0; // is new insert or delete
    mpz_class value; // value retrieved
    uint64_t proofHashCounter;
    string toString (Goldilocks &fr);
};

// SMT class
class Smt
{
private:
    Goldilocks  &fr;
    PoseidonGoldilocks poseidon;
public:
    Smt(Goldilocks &fr) : fr(fr) {}
    zkresult set(Database &db, const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, SmtSetResult &result, DatabaseMap *dbReadLog = NULL);
    zkresult get(Database &db, const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], SmtGetResult &result, DatabaseMap *dbReadLog = NULL);
    void splitKey(const Goldilocks::Element (&key)[4], vector<uint64_t> &result);
    void joinKey(const vector<uint64_t> &bits, const Goldilocks::Element (&rkey)[4], Goldilocks::Element (&key)[4]);
    void removeKeyBits(const Goldilocks::Element (&key)[4], uint64_t nBits, Goldilocks::Element (&rkey)[4]);
    zkresult hashSave(Database &db, const Goldilocks::Element (&a)[8], const Goldilocks::Element (&c)[4], const bool persistent, Goldilocks::Element (&hash)[4]);
    zkresult saveStateRoot(Database &db, const Goldilocks::Element (&stateRoot)[4]);
    int64_t getUniqueSibling(vector<Goldilocks::Element> &a);
};

#endif