#ifndef MAIN_HPP
#define MAIN_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "calcwit.hpp"
#include "circom.hpp"

Circom_Circuit* loadCircuit(std::string const &datFileName);
void loadJson(Circom_CalcWit *ctx, std::string filename);
void writeBinWitness(Circom_CalcWit *ctx, std::string wtnsFileName);

#endif