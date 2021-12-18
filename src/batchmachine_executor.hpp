#ifndef BATCHMACHINE_EXECUTOR_HPP
#define BATCHMACHINE_EXECUTOR_HPP

#include "mem.hpp"
#include "script.hpp"

void batchMachineExecutor (RawFr &fr, Mem &mem, Script &script);
void calculateH1H2(RawFr &fr, Reference &f, Reference &t, Reference &h1, Reference &h2);

#endif