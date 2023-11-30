#ifndef ZKEVM_VERIFIER_MAIN_MULTICHAIN_PREP_HPP
#define ZKEVM_VERIFIER_MAIN_MULTICHAIN_PREP_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "calcwit.multichainPrep.hpp"
#include "circom.multichainPrep.hpp"
#include "fr_goldilocks.hpp"

#include "timer.hpp"
#include <iostream>
#include <unistd.h>
#include "commit_pols_starks.hpp"
using namespace std;

namespace CircomMultichainPrep
{
    using json = nlohmann::json;
    Circom_Circuit *loadCircuit(std::string const &datFileName);
    void freeCircuit(Circom_Circuit *circuit);
    void loadJson(Circom_CalcWit *ctx, std::string filename);
    void loadJsonImpl(Circom_CalcWit *ctx, json &j);
    void writeBinWitness(Circom_CalcWit *ctx, std::string wtnsFileName);
    void getBinWitness(Circom_CalcWit *ctx, FrGElement *&pWitness, uint64_t &witnessSize);
    void getCommitedPols(CommitPolsStarks *commitPols, const std::string multichainPrepVerifier, const std::string execFile, nlohmann::json &zkin, uint64_t N, uint64_t nCols, nlohmann::json &outHashInfo);
    bool check_valid_number(std::string &s, uint base);
}
#endif
