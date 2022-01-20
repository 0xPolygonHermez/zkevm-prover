#ifndef MAIN_HPP
#define MAIN_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "calcwit.hpp"
#include "circom.hpp"
#include "alt_bn128.hpp"

Circom_Circuit* loadCircuit(std::string const &datFileName);
void loadJson(Circom_CalcWit *ctx, std::string filename);
void loadJsonImpl(Circom_CalcWit *ctx, json &j);
void writeBinWitness(Circom_CalcWit *ctx, std::string wtnsFileName);
void getBinWitness(Circom_CalcWit *ctx, AltBn128::FrElement * &pWitness, uint64_t &witnessSize);


#endif