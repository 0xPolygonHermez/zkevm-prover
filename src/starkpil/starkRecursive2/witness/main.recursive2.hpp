#ifndef ZKEVM_VERIFIER_MAIN_RECURSIVE_2_HPP
#define ZKEVM_VERIFIER_MAIN_RECURSIVE_2_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "calcwit.recursive2.hpp"
#include "circom.recursive2.hpp"
#include "fr_goldilocks.hpp"

namespace CircomRecursive2
{
    using json = nlohmann::json;
    Circom_Circuit *loadCircuit(std::string const &datFileName);
    void freeCircuit(Circom_Circuit *circuit);
    void loadJson(Circom_CalcWit *ctx, std::string filename);
    void loadJsonImpl(Circom_CalcWit *ctx, json &j);
    void writeBinWitness(Circom_CalcWit *ctx, std::string wtnsFileName);
    void getBinWitness(Circom_CalcWit *ctx, FrGElement *&pWitness, uint64_t &witnessSize);
}
#endif