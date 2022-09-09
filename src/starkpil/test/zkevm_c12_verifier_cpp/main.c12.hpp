#ifndef MAIN_ZKEVM__VERIFIER_MOCK_HPP
#define MAIN_ZKEVM__VERIFIER_MOCK_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "starkpil/test/zkevm_c12_verifier_cpp/calcwit.c12.hpp"
#include "starkpil/test/zkevm_c12_verifier_cpp/circom.c12.hpp"
#include "fr.hpp"

namespace MockCircomC12
{
    using json = nlohmann::json;

    Circom_Circuit *loadCircuit(std::string const &datFileName);
    void freeCircuit(Circom_Circuit *circuit);
    void loadJson(Circom_CalcWit *ctx, std::string filename);
    void loadJsonImpl(Circom_CalcWit *ctx, json &j);
    void writeBinWitness(Circom_CalcWit *ctx, std::string wtnsFileName);
    void getBinWitness(Circom_CalcWit *ctx, RawFr::Element *&pWitness, uint64_t &witnessSize);
}
#endif