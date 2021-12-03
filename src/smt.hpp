#ifndef SMT_HPP
#define SMT_HPP

#include <vector>
#include <map>
#include <gmpxx.h>

#include "poseidon_opt/poseidon_opt.hpp"
#include "ffiasm/fr.hpp"

using namespace std;

bool CompareFeImpl(const RawFr::Element &a, const RawFr::Element &b);

class CompareFe {
public:
    bool operator()(const RawFr::Element &a, const RawFr::Element &b) const
    {
        return CompareFeImpl(a, b);
    }
};

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
    Poseidon_opt poseidon;
    uint64_t arity;
public:
    Smt(uint64_t arity) : arity(arity) {
        mask = (1<<arity)-1;
        maxLevels = 160/arity;
    }
    void set (RawFr &fr, map< RawFr::Element, vector<RawFr::Element>, CompareFe > &db, RawFr::Element &oldRoot, RawFr::Element &key, mpz_class &value, SmtSetResult &result);
    void get (RawFr &fr, map< RawFr::Element, vector<RawFr::Element>, CompareFe > &db, RawFr::Element &oldRoot, RawFr::Element &key, SmtGetResult &result);
    void splitKey (RawFr &fr, RawFr::Element &key, vector<uint64_t> &result);
    void hashSave (RawFr &fr, map< RawFr::Element, vector<RawFr::Element>, CompareFe > &db, vector<RawFr::Element> &a, RawFr::Element &hash);
    int64_t getUniqueSibling(RawFr &fr, vector<RawFr::Element> &a);
};

#endif