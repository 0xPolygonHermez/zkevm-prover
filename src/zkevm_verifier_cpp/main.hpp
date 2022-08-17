#ifndef MAIN_HPP_OLD
#define MAIN_HPP_OLD

#include <string>
#include <nlohmann/json.hpp>
#include "calcwit.hpp"
#include "circom.hpp"
#include "fr_goldilocks.hpp"

Circom_Circuit *loadCircuit(std::string const &datFileName);
void loadJson(Circom_CalcWit *ctx, std::string filename);
void loadJsonImpl(Circom_CalcWit *ctx, json &j);
void writeBinWitness(Circom_CalcWit *ctx, std::string wtnsFileName);
void getBinWitness(Circom_CalcWit *ctx, FrGElement *&pWitness, uint64_t &witnessSize);

#endif