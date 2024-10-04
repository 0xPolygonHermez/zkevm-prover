#ifndef ZKEVM_VERIFIER_MAIN_FINAL_2_HPP
#define ZKEVM_VERIFIER_MAIN_FINAL_2_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "fork_13/calcwit.final.hpp"
#include "fork_13/circom.final.hpp"
#include "fr.hpp"

namespace CircomFinalFork13
{
    Circom_Circuit *loadCircuit(std::string const &datFileName);
    void freeCircuit(Circom_Circuit *circuit);
    void loadJson(Circom_CalcWit *ctx, std::string filename);
    void loadJsonImpl(Circom_CalcWit *ctx, json &j);
    void writeBinWitness(Circom_CalcWit *ctx, std::string wtnsFileName);
    void getBinWitness(Circom_CalcWit *ctx, RawFr::Element *&pWitness, uint64_t &witnessSize);
    bool check_valid_number(std::string &s, uint base);
}
#endif