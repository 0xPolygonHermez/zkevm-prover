#ifndef SMT_HPP
#define SMT_HPP

#include "context.hpp"
#include "poseidon_opt/poseidon_opt.hpp"
#include "ffiasm/fr.hpp"

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

class Smt
{
    mpz_class mask;
    uint64_t maxLevels;
    Poseidon_opt &poseidon;
    RawFr &fr;
    uint64_t arity;
    map< RawFr::Element, vector<RawFr::Element>, CompareFe > &db;
public:
    Smt(RawFr &fr, uint64_t arity, Poseidon_opt &poseidon, map< RawFr::Element, vector<RawFr::Element>, CompareFe > &db) : fr(fr), arity(arity), poseidon(poseidon), db(db) {
        mask = (1<<arity)-1;
        maxLevels = 160/arity;
    }
    void set (RawFr::Element &oldRoot, RawFr::Element &key, mpz_class &value, SmtSetResult &result);
    void get (RawFr::Element &oldRoot, RawFr::Element &key, SmtGetResult &result);
    void splitKey (RawFr::Element &key, vector<uint64_t> &result);
    void hashSave (vector<RawFr::Element> &a, RawFr::Element &hash);
    int64_t getUniqueSibling(vector<RawFr::Element> &a);
};

#endif