#ifndef PROOF_FFLONK_HPP
#define PROOF_FFLONK_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "public_inputs_extended.hpp"

using namespace std;
using json = nlohmann::json;

class ProofEvalutations
{
public:
    string a;
    string b;
    string c;
    string inv;
    string qc;
    string ql;
    string qm;
    string qo;
    string qr;
    string s1;
    string s2;
    string s3;
    string t1w;
    string t2w;
    string z;
    string zw;
};

class ProofPolynomials
{
public:
    vector<string> C1;
    vector<string> C2;
    vector<string> W1;
    vector<string> W2;
};

class Proof
{
public:
    ProofEvalutations evaluations;
    ProofPolynomials polynomials;

    PublicInputsExtended publicInputsExtended;

    vector<string> publics;

    void load(json &proof, json &publicSignalsJson);
    std::string getStringProof();
};

#endif