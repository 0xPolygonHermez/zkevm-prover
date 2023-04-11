#include <iostream>
#include "config.hpp"
#include "zkassert.hpp"
#include "proof_fflonk.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

using namespace std;

void Proof::load(json &proof, json &publicSignalsJson)
{
    string aux;

    if (!proof.contains("evaluations") || !proof["evaluations"].is_object())
    {
        zklog.error("Proof::load() proof does not contain evaluations");
        exitProcess();
    }
    if (!proof["evaluations"].contains("a") ||
        !proof["evaluations"].contains("b") ||
        !proof["evaluations"].contains("c") ||
        !proof["evaluations"].contains("inv") ||
        !proof["evaluations"].contains("qc") ||
        !proof["evaluations"].contains("ql") ||
        !proof["evaluations"].contains("qm") ||
        !proof["evaluations"].contains("qo") ||
        !proof["evaluations"].contains("qr") ||
        !proof["evaluations"].contains("s1") ||
        !proof["evaluations"].contains("s2") ||
        !proof["evaluations"].contains("s3") ||
        !proof["evaluations"].contains("t1w") ||
        !proof["evaluations"].contains("t2w") ||
        !proof["evaluations"].contains("z") ||
        !proof["evaluations"].contains("zw"))
    {
        zklog.error("Proof::load() proof does not contain one evaluation");
        exitProcess();
    }

    evaluations.a = proof["evaluations"]["a"];
    evaluations.b = proof["evaluations"]["b"];
    evaluations.c = proof["evaluations"]["c"];
    evaluations.inv = proof["evaluations"]["inv"];
    evaluations.ql = proof["evaluations"]["ql"];
    evaluations.qc = proof["evaluations"]["qc"];
    evaluations.qm = proof["evaluations"]["qm"];
    evaluations.qo = proof["evaluations"]["qo"];
    evaluations.qr = proof["evaluations"]["qr"];
    evaluations.s1 = proof["evaluations"]["s1"];
    evaluations.s2 = proof["evaluations"]["s2"];
    evaluations.s3 = proof["evaluations"]["s3"];
    evaluations.t1w = proof["evaluations"]["t1w"];
    evaluations.t2w = proof["evaluations"]["t2w"];
    evaluations.z = proof["evaluations"]["z"];
    evaluations.zw = proof["evaluations"]["zw"];

    if (!proof.contains("polynomials") || !proof["evaluations"].is_object())
    {
        zklog.error("Proof::load() proof does not contain polynomials");
        exitProcess();
    }

    if (!proof["polynomials"].contains("C1") || !proof["polynomials"]["C1"].is_array())
    {
        zklog.error("Proof::load() proof does not contain C1 polynomial");
        exitProcess();
    }

    if (!proof["polynomials"].contains("C2") || !proof["polynomials"]["C2"].is_array())
    {
        zklog.error("Proof::load() proof does not contain C1 polynomial");
        exitProcess();
    }

    if (!proof["polynomials"].contains("W1") || !proof["polynomials"]["W1"].is_array())
    {
        zklog.error("Proof::load() proof does not contain C1 polynomial");
        exitProcess();
    }

    if (!proof["polynomials"].contains("W2") || !proof["polynomials"]["W2"].is_array())
    {
        zklog.error("Proof::load() proof does not contain C1 polynomial");
        exitProcess();
    }

    for (uint64_t i = 0; i < proof["polynomials"]["C1"].size(); i++)
    {
        zkassert(proof["polynomials"]["C1"][i].is_string());
        aux = proof["polynomials"]["C1"][i];
        polynomials.C1.push_back(aux);
    }

    for (uint64_t i = 0; i < proof["polynomials"]["C2"].size(); i++)
    {
        zkassert(proof["polynomials"]["C2"][i].is_string());
        aux = proof["polynomials"]["C2"][i];
        polynomials.C2.push_back(aux);
    }

    for (uint64_t i = 0; i < proof["polynomials"]["W1"].size(); i++)
    {
        zkassert(proof["polynomials"]["W1"][i].is_string());
        aux = proof["polynomials"]["W1"][i];
        polynomials.W1.push_back(aux);
    }

    for (uint64_t i = 0; i < proof["polynomials"]["W2"].size(); i++)
    {
        zkassert(proof["polynomials"]["W2"][i].is_string());
        aux = proof["polynomials"]["W2"][i];
        polynomials.W2.push_back(aux);
    }

    // Load public inputs
    if (!publicSignalsJson.is_array())
    {
        zklog.error("Proof::load() publicSignalsJson is not an array");
        exitProcess();
    }

    for (uint64_t i = 0; i < publicSignalsJson.size(); i++)
    {
        publics.push_back(publicSignalsJson[i]);
    }
}

std::string Proof::getStringProof()
{
    mpz_class aux;
    string result = "0x";

    aux.set_str(polynomials.C1[0], 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);
    aux.set_str(polynomials.C1[1], 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(polynomials.C2[0], 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);
    aux.set_str(polynomials.C2[1], 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(polynomials.W1[0], 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);
    aux.set_str(polynomials.W1[1], 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(polynomials.W2[0], 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);
    aux.set_str(polynomials.W2[1], 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.ql, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.qr, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.qm, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.qo, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.qc, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.s1, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.s2, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.s3, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.a, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.b, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.c, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.z, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.zw, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.t1w, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.t2w, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    aux.set_str(evaluations.inv, 10);
    result += NormalizeToNFormat(aux.get_str(16), 64);

    return result;
}