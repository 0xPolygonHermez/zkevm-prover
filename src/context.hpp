#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <vector>
#include <gmpxx.h>
#include "config.hpp"
#include "rom.hpp"
#include "rom_command.hpp"
#include "ff/ff.hpp"
#include "smt.hpp"
#include "pols.hpp"
#include "database.hpp"
#include "input.hpp"

using namespace std;
using json = nlohmann::json;

class HashValue
{
public:
    vector<uint8_t> data;
    string hash;
};

class LastSWrite
{
public:
    uint64_t step;
    FieldElement key0;
    FieldElement key1;
    FieldElement key2;
    FieldElement key3;
    FieldElement newRoot0;
    FieldElement newRoot1;
    FieldElement newRoot2;
    FieldElement newRoot3;
};

class Fea
{
public:
    FieldElement fe0;
    FieldElement fe1;
    FieldElement fe2;
    FieldElement fe3;
    FieldElement fe4;
    FieldElement fe5;
    FieldElement fe6;
    FieldElement fe7;
};

class Context {
public:

    FiniteField &fr; // Finite field reference
    mpz_class prime; // Prime number used to generate the finite field fr
    Pols &pols; // PIL JSON file polynomials data
    const Input &input; // Input JSON file data
    Database &db; // Database reference
    Context(FiniteField &fr, Pols &pols, const Input &input, Database &db) : fr(fr), pols(pols), input(input), db(db) { ; }; // Constructor, setting references

    // Evaluations data
    uint64_t zkPC; // Zero-knowledge program counter
    uint64_t step; // Iteration, instruction execution loop counter, polynomial evaluation counter
#ifdef LOG_FILENAME
    string   fileName; // From ROM JSON file instruction
    uint64_t line; // From ROM JSON file instruction
#endif

    // Storage
#ifdef USE_LOCAL_STORAGE
    map< FieldElement, mpz_class, CompareFe> sto; // Input JSON will include the initial values of the rellevant storage positions
#endif
    LastSWrite lastSWrite; // Keep track of the last storage write

    // Hash database, used in hashRD and hashWR
    map< uint64_t, HashValue > hash;

    // Variables database, used in evalCommand() declareVar/setVar/getVar
    map< string, FieldElement > vars; 
    
    // Memory map, using absolute address as key, and field element array as value
    map< uint64_t, Fea > mem;

    // Used to write byte4_freeIN and byte4_out polynomials after all evaluations have been done
    map< uint32_t, bool > byte4;

};

/* Declare Context ctx to use pols(A0)[i] */
#define pol(name) ctx.pols.name.pData

#endif