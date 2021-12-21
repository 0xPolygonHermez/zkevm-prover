#ifndef BATCHMACHINE_EXECUTOR_HPP
#define BATCHMACHINE_EXECUTOR_HPP

#include "mem.hpp"
#include "script.hpp"

void batchMachineExecutor (RawFr &fr, Mem &mem, Script &script);
json dereference(RawFr &fr, Mem &mem, Output &output);
json refToObject(RawFr &fr, Mem &mem, Reference &ref);
void calculateH1H2(RawFr &fr, Reference &f, Reference &t, Reference &h1, Reference &h2);
void batchInverse (RawFr &fr, Reference &source, Reference &result);

#endif