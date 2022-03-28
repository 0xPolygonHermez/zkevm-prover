#ifndef BATCHMACHINE_EXECUTOR_HPP
#define BATCHMACHINE_EXECUTOR_HPP

#include "ffiasm/fr.hpp"
#include "mem.hpp"
#include "script.hpp"

class BatchMachineExecutor
{
    RawFr &fr;
    const Script &script;
public:
    BatchMachineExecutor (RawFr &fr, const Script &script) : fr(fr), script(script) {};
    void execute (Mem &mem, json &proof);
    json dereference (const Mem &mem, const Output &output);
    json refToObject (const Mem &mem, const Reference &ref);
    void calculateH1H2(Reference &f, Reference &t, Reference &h1, Reference &h2);
    static void batchInverse (RawFr &fr, Reference &source, Reference &result);
    static void batchInverseTest (RawFr &fr);
    void evalPol (RawFr::Element *pPol, uint64_t polSize, RawFr::Element &x, RawFr::Element &result);
    void polMulAxi (RawFr::Element *pPol, uint64_t polSize, RawFr::Element &init, RawFr::Element &acc);
};

#endif