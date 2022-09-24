#ifndef MAIN_ZKEVM__VERIFIER_C12A_HPP
#define MAIN_ZKEVM__VERIFIER_C12A_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "zkevm_c12a_verifier_cpp/calcwit.c12a.hpp"
#include "zkevm_c12a_verifier_cpp/circom.c12a.hpp"
#include "fr.hpp"

namespace CircomC12a
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