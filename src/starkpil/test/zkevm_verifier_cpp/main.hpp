#ifndef ZKEVM_VERIFIER_MAIN_TEST_HPP
#define ZKEVM_VERIFIER_MAIN_TEST_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "starkpil/test/zkevm_verifier_cpp/calcwit.hpp"
#include "circom.hpp"
#include "fr_goldilocks.hpp"

namespace testCircom
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