#ifndef SMT_HPP
#define SMT_HPP

#include <vector>
#include <map>
#include <gmpxx.h>

#include "poseidon_opt/poseidon_goldilocks.hpp"
#include "ff/ff.hpp"
#include "compare_fe.hpp"
#include "database.hpp"

using namespace std;

// SMT set method result data
class SmtSetResult
{
public:
    FieldElement oldRoot0;
    FieldElement oldRoot1;
    FieldElement oldRoot2;
    FieldElement oldRoot3;
    FieldElement newRoot0;
    FieldElement newRoot1;
    FieldElement newRoot2;
    FieldElement newRoot3;
    FieldElement key;
    map< uint64_t, vector<FieldElement> > siblings;
    FieldElement insKey;
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
    FieldElement root;
    FieldElement key;
    map< uint64_t, vector<FieldElement> > siblings;
    FieldElement insKey;
    mpz_class insValue;
    bool isOld0;
    mpz_class value;
};

// SMT class
class Smt
{
    mpz_class    mask; // 0x0F
    uint64_t     maxLevels; // 40 (160 bits)
    Poseidon_goldilocks poseidon;
    uint64_t     arity; // 4
public:
    Smt(uint64_t arity) : arity(arity) {
        mask = (1<<arity)-1; //15, 0x0F
        maxLevels = 160/arity; // 40
    }
    void set ( FiniteField &fr, Database &db,
               FieldElement &oldRoot0, FieldElement &oldRoot1, FieldElement &oldRoot2, FieldElement &oldRoot3, 
               FieldElement &key0, FieldElement &key1, FieldElement &key2, FieldElement &key3,
               mpz_class &value, SmtSetResult &result );
    void get ( FiniteField &fr, Database &db,
               FieldElement &oldRoot0, FieldElement &oldRoot1, FieldElement &oldRoot2, FieldElement &oldRoot3,
               FieldElement &key0, FieldElement &key1, FieldElement &key2, FieldElement &key3,
               SmtGetResult &result );
    void splitKey (FiniteField &fr, FieldElement &key, vector<uint64_t> &result);
    void hashSave (FiniteField &fr, Database &db, vector<FieldElement> &a, FieldElement &hash);
    int64_t getUniqueSibling(FiniteField &fr, vector<FieldElement> &a);
};

#endif