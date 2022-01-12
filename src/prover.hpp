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
    const string &constTreePolsInputFile;
    const string &inputFile;
    const string &starkFile;
    const string &verifierFile;
    const string &witnessFile;
public:
    Prover( RawFr &fr,
            const Rom &romData,
            const Script &script,
            const Pil &pil,
            const Pols &constPols,
            const string &cmPolsOutputFile,
            const string &constTreePolsInputFile,
            const string &inputFile,
            const string &starkFile,
            const string &verifierFile,
            const string &witnessFile ) :
        fr(fr),
        romData(romData),
        executor(fr, romData),
        script(script),
        pil(pil),
        constPols(constPols),
        cmPolsOutputFile(cmPolsOutputFile),
        constTreePolsInputFile(constTreePolsInputFile),
        inputFile(inputFile),
        starkFile(starkFile),
        verifierFile(verifierFile),
        witnessFile(witnessFile) {};

    void prove (const Input &input);
};

#endif