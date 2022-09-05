#ifndef MAIN_ZKEVM_C12_VERIFIER_HPP
#define MAIN_ZKEVM_C12_VERIFIER_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "calcwit.c12.hpp"
#include "circom.c12.hpp"
#include "fr.hpp"

using json = nlohmann::json;

Circom_CircuitC12 *loadCircuitC12(std::string const &datFileName);
void freeCircuitC12 (Circom_CircuitC12* circuitC12);
void loadJsonC12(Circom_CalcWitC12 *ctx, std::string filename);
void loadJsonImplC12(Circom_CalcWitC12 *ctx, json &j);
void writeBinWitnessC12(Circom_CalcWitC12 *ctx, std::string wtnsFileName);
void getBinWitnessC12(Circom_CalcWitC12 *ctx, RawFr::Element *&pWitness, uint64_t &witnessSize);

#endif