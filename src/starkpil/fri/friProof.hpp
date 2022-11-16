#ifndef FRI_PROOF
#define FRI_PROOF

#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"

#include <vector>

using ordered_json = nlohmann::ordered_json;

class MerkleProof
{
public:
    std::vector<std::vector<Goldilocks::Element>> v;
    std::vector<std::vector<Goldilocks::Element>> mp;

    MerkleProof(uint64_t nLinears, uint64_t elementsTree, Goldilocks::Element *pointer) : v(nLinears, std::vector<Goldilocks::Element>(1, Goldilocks::zero())), mp(elementsTree, std::vector<Goldilocks::Element>(HASH_SIZE, Goldilocks::zero()))
    {
        for (uint64_t i = 0; i < nLinears; i++)
        {
            std::memcpy(&v[i][0], &pointer[i], sizeof(Goldilocks::Element));
        }
        for (uint64_t j = 0; j < elementsTree; j++)
        {
            std::memcpy(&mp[j][0], &pointer[nLinears + j * HASH_SIZE], HASH_SIZE * sizeof(Goldilocks::Element));
        }
    };
    ordered_json merkleProof2json()
    {
        ordered_json j = ordered_json::array();
        ordered_json json_v = ordered_json::array();
        for (uint i = 0; i < v.size(); i++)
        {
            if (v[i].size() > 1)
            {
                ordered_json element = ordered_json::array();
                for (uint j = 0; j < v[i].size(); j++)
                {
                    element.push_back(Goldilocks::toString(v[i][j]));
                }
                json_v.push_back(element);
            }
            else
            {
                json_v.push_back(Goldilocks::toString(v[i][0]));
            }
        }
        j.push_back(json_v);

        ordered_json json_mp = ordered_json::array();
        for (uint i = 0; i < mp.size(); i++)
        {
            ordered_json element = ordered_json::array();
            for (uint j = 0; j < mp[i].size(); j++)
            {
                element.push_back(Goldilocks::toString(mp[i][j]));
            }
            json_mp.push_back(element);
        }
        j.push_back(json_mp);
        return j;
    }
};

class ProofTree
{
public:
    std::vector<Goldilocks::Element> root;
    std::vector<std::vector<MerkleProof>> polQueries;

    ProofTree() : root(HASH_SIZE){};
    void setRoot(Goldilocks::Element *_root)
    {
        std::memcpy(&root[0], &_root[0], HASH_SIZE * sizeof(Goldilocks::Element));
    };

    ordered_json ProofTree2json()
    {
        ordered_json j_ProofTree2json = ordered_json::object();

        ordered_json json_root = ordered_json::array();
        for (uint i = 0; i < root.size(); i++)
        {
            json_root.push_back(Goldilocks::toString(root[i]));
        }
        if (Goldilocks::toU64(root[0]) != 0 && Goldilocks::toU64(root[1]) != 0 && Goldilocks::toU64(root[2]) != 0 && Goldilocks::toU64(root[3]) != 0)
            j_ProofTree2json.erase("root");

        ordered_json json_polQueries = ordered_json::array();
        for (uint i = 0; i < polQueries.size(); i++)
        {
            ordered_json element = ordered_json::array();
            if (polQueries[i].size() != 1)
            {
                for (uint j = 0; j < polQueries[i].size(); j++)
                {
                    element.push_back(polQueries[i][j].merkleProof2json());
                }
                json_polQueries.push_back(element);
            }
            else
            {
                json_polQueries.push_back(polQueries[i][0].merkleProof2json());
            }
        }
        j_ProofTree2json["root"] = json_root;
        j_ProofTree2json["polQueries"] = json_polQueries;

        return j_ProofTree2json;
    }
};

class Fri
{
public:
    std::vector<std::vector<Goldilocks::Element>> pol;
    std::vector<ProofTree> trees;

    Fri(uint64_t polN, uint64_t dim, uint64_t numSteps) : pol(polN, std::vector<Goldilocks::Element>(dim, Goldilocks::zero())),
                                                          trees(numSteps){};

    void setPol(Goldilocks::Element *pPol)
    {
        for (uint64_t i = 0; i < pol.size(); i++)
        {
            std::memcpy(&pol[i][0], &pPol[i * pol[i].size()], pol[i].size() * sizeof(Goldilocks::Element));
        }
    }

    ordered_json FriP2json()
    {
        ordered_json j = ordered_json::array();

        for (uint i = 0; i < trees.size(); i++)
        {
            j.push_back((trees[i].ProofTree2json()));
        }

        ordered_json json_pol = ordered_json::array();
        for (uint i = 0; i < pol.size(); i++)
        {
            ordered_json element = ordered_json::array();
            for (uint j = 0; j < pol[i].size(); j++)
            {
                element.push_back(Goldilocks::toString(pol[i][j]));
            }
            json_pol.push_back(element);
        }
        j.push_back(json_pol);
        return j;
    }
};

class Proofs
{
public:
    std::vector<Goldilocks::Element> root1;
    std::vector<Goldilocks::Element> root2;
    std::vector<Goldilocks::Element> root3;
    std::vector<Goldilocks::Element> root4;
    Fri fri;
    std::vector<std::vector<Goldilocks::Element>> evals;
    Proofs(uint64_t polN, uint64_t dim, uint64_t numSteps, uint64_t evalSize) : root1(HASH_SIZE, Goldilocks::zero()),
                                                                                root2(HASH_SIZE, Goldilocks::zero()),
                                                                                root3(HASH_SIZE, Goldilocks::zero()),
                                                                                root4(HASH_SIZE, Goldilocks::zero()),
                                                                                fri(polN, dim, numSteps),
                                                                                evals(evalSize, std::vector<Goldilocks::Element>(dim, Goldilocks::zero())){};

    void setEvals(Goldilocks::Element *_evals)
    {
        for (uint64_t i = 0; i < evals.size(); i++)
        {
            std::memcpy(&evals[i][0], &_evals[i * evals[i].size()], evals[i].size() * sizeof(Goldilocks::Element));
        }
    }
    ordered_json proof2json()
    {
        ordered_json j = ordered_json::object();

        ordered_json json_root1 = ordered_json::array();
        for (uint i = 0; i < root1.size(); i++)
        {
            json_root1.push_back(Goldilocks::toString(root1[i]));
        }
        ordered_json json_root2 = ordered_json::array();
        for (uint i = 0; i < root2.size(); i++)
        {
            json_root2.push_back(Goldilocks::toString(root2[i]));
        }
        ordered_json json_root3 = ordered_json::array();
        for (uint i = 0; i < root3.size(); i++)
        {
            json_root3.push_back(Goldilocks::toString(root3[i]));
        }
        ordered_json json_root4 = ordered_json::array();
        for (uint i = 0; i < root4.size(); i++)
        {
            json_root4.push_back(Goldilocks::toString(root4[i]));
        }
        j["root1"] = json_root1;
        j["root2"] = json_root2;
        j["root3"] = json_root3;
        j["root4"] = json_root4;

        ordered_json json_evals = ordered_json::array();
        for (uint i = 0; i < evals.size(); i++)
        {
            ordered_json element = ordered_json::array();
            for (uint j = 0; j < evals[i].size(); j++)
            {
                element.push_back(Goldilocks::toString(evals[i][j]));
            }
            json_evals.push_back(element);
        }
        j["evals"] = json_evals;
        j["fri"] = fri.FriP2json();
        return j;
    }
};

class FRIProof
{
public:
    Proofs proofs;
    std::vector<Goldilocks::Element> publics;

    FRIProof(
        uint64_t polN,
        uint64_t dim,
        uint64_t numTrees,
        uint64_t evalSize,
        uint64_t nPublics) : proofs(polN, dim, numTrees, evalSize),
                             publics(nPublics){};
};

#endif