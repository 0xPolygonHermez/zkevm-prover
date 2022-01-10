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
    string &constTreePolsInputFile;

public:
    Prover(RawFr &fr, Rom &romData, Script &script, Pil &pil, Pols &constPols, string &cmPolsOutputFile,string &constTreePolsInputFile) :
        fr(fr), romData(romData), executor(fr, romData), script(script), pil(pil), constPols(constPols), cmPolsOutputFile(cmPolsOutputFile), constTreePolsInputFile(constTreePolsInputFile) {};

    void prove (Input &input);
};

#endif