#ifndef SNARK_PROOF_HPP
#define SNARK_PROOF_HPP

#include <map>
#include <string>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

template<typename Engine>
class SnarkProof {
    using FrElement = typename Engine::FrElement;
    using G1Point = typename Engine::G1Point;
    using G1PointAffine = typename Engine::G1PointAffine;

    Engine &E;
    std::string protocol;
    std::string curve;

    std::map <std::string, G1Point> polynomialCommitments;
    std::map <std::string, FrElement> evaluationCommitments;
public :
    SnarkProof(Engine &E, const std::string &protocol);

    void reset();

    void addPolynomialCommitment(const std::string &key, G1Point &polynomialCommitment);

    G1Point getPolynomialCommitment(const std::string &key);

    void addEvaluationCommitment(const std::string &key, FrElement evaluationCommitment);

    FrElement getEvaluationCommitment(const std::string &key);

    json toJson();
};

#include "snark_proof.c.hpp"

#endif
