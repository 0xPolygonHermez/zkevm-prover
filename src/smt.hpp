#ifndef SMT_HPP
#define SMT_HPP

#include <vector>
#include <map>
#include <gmpxx.h>

#include "poseidon_opt/poseidon_opt.hpp"
#include "ffiasm/fr.hpp"
#include "compare_fe.hpp"
#include "database.hpp"

using namespace std;

// SMT set method result data
class SmtSetResult
{
public:
    RawFr::Element oldRoot;
    RawFr::Element newRoot;
    RawFr::Element key;
    map< uint64_t, vector<RawFr::Element> > siblings;
    RawFr::Element insKey;
    mpz_class insValue;
    bool isOld0;
    mpz_class oldValue;
    mpz_class newValue;
    string mode;
};

// SMT get method result data
class SmtGetResult
{
public:
    RawFr::Element root;
    RawFr::Element key;
    map< uint64_t, vector<RawFr::Element> > siblings;
    RawFr::Element insKey;
    mpz_class insValue;
    bool isOld0;
    mpz_class value;
};

// SMT class
class Smt
{
    mpz_class    mask; // 0x0F
    uint64_t     maxLevels; // 40 (160 bits)
    Poseidon_opt poseidon;
    uint64_t     arity; // 4
public:
    Smt(uint64_t arity) : arity(arity) {
        mask = (1<<arity)-1; //15, 0x0F
        maxLevels = 160/arity; // 40
    }
    void set (RawFr &fr, Database &db, RawFr::Element &oldRoot, RawFr::Element &key, mpz_class &value, SmtSetResult &result);
    void get (RawFr &fr, Database &db, RawFr::Element &oldRoot, RawFr::Element &key, SmtGetResult &result);
    void splitKey (RawFr &fr, RawFr::Element &key, vector<uint64_t> &result);
    void hashSave (RawFr &fr, Database &db, vector<RawFr::Element> &a, RawFr::Element &hash);
    int64_t getUniqueSibling(RawFr &fr, vector<RawFr::Element> &a);
};

#endif