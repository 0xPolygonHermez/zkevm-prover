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
    Pil &pil;
    Pols &constPols;
    string &cmPolsOutputFile;
public:
    Prover(RawFr &fr, Rom &romData, Script &script, Pil &pil, Pols &constPols, string &cmPolsOutputFile) :
        fr(fr), romData(romData), executor(fr, romData), script(script), pil(pil), constPols(constPols), cmPolsOutputFile(cmPolsOutputFile) {};

    void prove (Input &input);
};

#endif