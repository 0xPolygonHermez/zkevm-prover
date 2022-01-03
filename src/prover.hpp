#ifndef PROVER_HPP
#define PROVER_HPP

#include "ffiasm/fr.hpp"
#include "input.hpp"
#include "rom.hpp"
#include "executor.hpp"
#include "script.hpp"

class Prover
{
    RawFr &fr;
    Rom &romData;
    Executor executor;
    Script &script;
public:
    Prover(RawFr &fr, Rom &romData, Script &script) : fr(fr), romData(romData), executor(fr, romData), script(script) {};
    void prove (Input &input, Pols &cmPols, Pols &constPols);
};

#endif