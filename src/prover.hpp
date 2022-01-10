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
    const Rom &romData;
    Executor executor;
    const Script &script;
    const Pil &pil;
    const Pols &constPols;
    const string &cmPolsOutputFile;
public:
    Prover(RawFr &fr, const Rom &romData, const Script &script, const Pil &pil, const Pols &constPols, const string &cmPolsOutputFile) :
        fr(fr), romData(romData), executor(fr, romData), script(script), pil(pil), constPols(constPols), cmPolsOutputFile(cmPolsOutputFile) {};

    void prove (const Input &input);
};

#endif