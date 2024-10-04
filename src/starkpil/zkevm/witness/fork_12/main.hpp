#ifndef ZKEVM_VERIFIER_MAIN_2_HPP_fork_12
#define ZKEVM_VERIFIER_MAIN_2_HPP_fork_12

#include <string>
#include <nlohmann/json.hpp>
#include "fork_12/calcwit.hpp"
#include "fork_12/circom.hpp"
#include "fr_goldilocks.hpp"

#include "timer.hpp"
#include <iostream>
#include <unistd.h>
#include "commit_pols_starks.hpp"
using namespace std;

namespace CircomFork12
{
    Circom_Circuit *loadCircuit(std::string const &datFileName);
    void freeCircuit(Circom_Circuit *circuit);
    void loadJson(Circom_CalcWit *ctx, std::string filename);
    void loadJsonImpl(Circom_CalcWit *ctx, json &j);
    void writeBinWitness(Circom_CalcWit *ctx, std::string wtnsFileName);
    void getBinWitness(Circom_CalcWit *ctx, FrGElement *&pWitness, uint64_t &witnessSize);
    void getCommitedPols(CommitPolsStarks *commitPols, const std::string zkevmVerifier, const std::string execFile, nlohmann::json &zkin, uint64_t N, uint64_t nCols);
    bool check_valid_number(std::string &s, uint base);

}
#endif
