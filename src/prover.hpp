#ifndef PROVER_HPP
#define PROVER_HPP

#include "ffiasm/fr.hpp"
#include "input.hpp"
#include "rom.hpp"
#include "executor.hpp"
#include "script.hpp"
#include "proof.hpp"
#include "alt_bn128.hpp"
#include "groth16.hpp"
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"

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
    const string &starkVerifierFile;
    const string &proofFile;

    std::unique_ptr<Groth16::Prover<AltBn128::Engine>> groth16Prover;
    std::unique_ptr<BinFileUtils::BinFile> zkey;
    std::unique_ptr<ZKeyUtils::Header> zkeyHeader;
    mpz_t altBbn128r;

    Reference constRefs[NCONSTPOLS];

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
            const string &witnessFile,
            const string &starkVerifierFile,
            const string &proofFile,
            const DatabaseConfig &databaseConfig ) ;

    ~Prover();

    void prove (const Input &input, Proof &proof);
};

#endif