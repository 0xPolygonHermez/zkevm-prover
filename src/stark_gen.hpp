#ifndef STARK_GEN_HPP
#define STARK_GEN_HPP

#include "pols.hpp"
#include "pil.hpp"

class StarkCm
{
public:
    vector<RawFr::Element> v_n;
};

class StarkExpression
{
public:
    vector<RawFr::Element> v_n;
    vector<RawFr::Element> v_2ns;
};

class StarkQ
{
};

class StarkConstants
{
public:
    vector<RawFr::Element> v_n;
    vector<RawFr::Element> v_2ns;
};

class StarkPols
{
public:
    vector<StarkCm> cm;
    vector<StarkExpression> exps;
    vector<StarkQ> q;
    vector<StarkConstants> constants;
};

class StarkGen
{
public:
    RawFr &fr;
    Pil &pil;
    StarkGen(RawFr &fr, Pil &pil) : fr(fr), pil(pil) {};

    /* Polynomials generation */
    StarkPols pols;
    void generate (Pols &cmPols, Pols &constPols, Pols &constTree);

    /* From polutils.js -> should we move it to polutils.cpp? */
    vector<RawFr::Element> ZHInv;
    void buildZhInv (RawFr &fr, uint64_t Nbits, uint64_t extendBits);
    void getZhInv (RawFr::Element &fe, uint64_t i);

    void eval (Expression &exp, const string &subPol, vector<RawFr::Element> &r);
    void calculateExpression (uint64_t expId, const string &subPol);
    void calculateDependencies (const Expression &exp, const string &subPol);
};



#endif