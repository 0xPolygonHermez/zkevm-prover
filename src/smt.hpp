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
    Poseidon_opt poseidon;
public:
    Smt() {
        mask = (1<<ARITY)-1;
        maxLevels = 160/ARITY;
    }
    void set (Context &ctx, RawFr::Element &oldRoot, RawFr::Element &key, mpz_class &value, SmtSetResult &result);
    void get (Context &ctx, RawFr::Element &oldRoot, RawFr::Element &key, SmtGetResult &result);
    void splitKey (Context &ctx, RawFr::Element &key, vector<uint64_t> &result);
    void hashSave (Context &ctx, vector<RawFr::Element> &a, RawFr::Element &hash);
    int64_t getUniqueSibling(Context &ctx, vector<RawFr::Element> &a);
};

/* TODO: Remove dependency with Context.  Pass all data in constructor, like in the JavaScript class:

    constructor(db, arity, hash, F) {
        this.db = db;
        this.arity = arity;
        this.hash = hash;
        this.F = F;
        this.mask = Scalar.e((1<<this.arity)-1);

        this.maxLevels = 160/this.arity;
    }
    */


#endif