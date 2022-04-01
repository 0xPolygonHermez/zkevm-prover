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
    FieldElement oldRoot[4];
    FieldElement newRoot[4];
    FieldElement key[4];
    map< uint64_t, vector<FieldElement> > siblings;
    FieldElement insKey[4];
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
    FieldElement root[4]; // merkle-tree root
    mpz_class key[4]; // key to look for
    map< uint64_t, vector<FieldElement> > siblings; // array of siblings // array of fields??
    FieldElement insKey[4]; // key found
    mpz_class insValue; // value found
    bool isOld0; // is new insert or delete
    mpz_class value; // value retrieved
};

// SMT class
class Smt
{
private:
    FiniteField  &fr;
    Poseidon_goldilocks poseidon;
public:
    Smt(FiniteField &fr) : fr(fr) {;}
    void set ( Database &db, FieldElement (&oldRoot)[4], FieldElement (&key)[4], mpz_class &value, SmtSetResult &result );
    void get ( Database &db, const FieldElement (&root)[4], const FieldElement (&key)[4], SmtGetResult &result );
    void splitKey ( const FieldElement (&key)[4], vector<uint64_t> &result);
    void joinKey ( const vector<uint64_t> &bits, const FieldElement (&rkey)[4], FieldElement (&key)[4] );
    void removeKeyBits ( const FieldElement (&key)[4], uint64_t nBits, FieldElement (&rkey)[4]);
    void hashSave ( Database &db, const FieldElement (&a)[8], const FieldElement (&c)[4], FieldElement (&hash)[4]);
    int64_t getUniqueSibling(vector<FieldElement> &a);
};

#endif