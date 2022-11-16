#ifndef FRI_PROOF_C12
#define FRI_PROOF_C12

#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "fr.hpp"
#include <vector>

using ordered_json = nlohmann::ordered_json;

class MerkleProofC12
{
public:
    std::vector<std::vector<Goldilocks::Element>> v;
    std::vector<std::vector<RawFr::Element>> mp;

    MerkleProofC12(uint64_t nLinears, uint64_t elementsTree, void *pointer) : v(nLinears, std::vector<Goldilocks::Element>(1, Goldilocks::zero())), mp(elementsTree, std::vector<RawFr::Element>(16, RawFr::field.zero()))
    {
        for (uint64_t i = 0; i < nLinears; i++)
        {
            std::memcpy(&v[i][0], &((Goldilocks::Element *)pointer)[i], sizeof(Goldilocks::Element));
        }
        RawFr::Element *mpCursor = (RawFr::Element *)&((Goldilocks::Element *)pointer)[nLinears];
        for (uint64_t j = 0; j < elementsTree; j++)
        {
            std::memcpy(&mp[j][0], &mpCursor[j * 16], 16 * sizeof(RawFr::Element));
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
                element.push_back(RawFr::field.toString(mp[i][j]));
            }
            json_mp.push_back(element);
        }
        j.push_back(json_mp);
        return j;
    }
};

class ProofTreeC12
{
public:
    std::vector<RawFr::Element> root;
    std::vector<std::vector<MerkleProofC12>> polQueries;

    ProofTreeC12() : root(1){};
    void setRoot(RawFr::Element *_root)
    {
        std::memcpy(&root[0], &_root[0], sizeof(RawFr::Element));
    };

    ordered_json ProofTree2json()
    {
        ordered_json j_ProofTree2json = ordered_json::object();
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
        j_ProofTree2json["root"] = RawFr::field.toString(root[0]);
        j_ProofTree2json["polQueries"] = json_polQueries;

        return j_ProofTree2json;
    }
};

class FriC12
{
public:
    std::vector<std::vector<Goldilocks::Element>> pol;
    std::vector<ProofTreeC12> trees;

    FriC12(uint64_t polN, uint64_t dim, uint64_t numSteps) : pol(polN, std::vector<Goldilocks::Element>(dim, Goldilocks::zero())),
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

class ProofsC12
{
public:
    std::vector<RawFr::Element> root1;
    std::vector<RawFr::Element> root2;
    std::vector<RawFr::Element> root3;
    std::vector<RawFr::Element> root4;
    FriC12 fri;
    std::vector<std::vector<Goldilocks::Element>> evals;
    ProofsC12(uint64_t polN, uint64_t dim, uint64_t numSteps, uint64_t evalSize) : root1(1, RawFr::field.zero()),
                                                                                   root2(1, RawFr::field.zero()),
                                                                                   root3(1, RawFr::field.zero()),
                                                                                   root4(1, RawFr::field.zero()),
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
        ordered_json json_root2 = ordered_json::array();
        ordered_json json_root3 = ordered_json::array();
        ordered_json json_root4 = ordered_json::array();

        j["root1"] = RawFr::field.toString(root1[0]);
        j["root2"] = RawFr::field.toString(root2[0]);
        j["root3"] = RawFr::field.toString(root3[0]);
        j["root4"] = RawFr::field.toString(root4[0]);

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

class FRIProofC12
{
public:
    ProofsC12 proofs;
    std::vector<RawFr::Element> publics;

    FRIProofC12(
        uint64_t polN,
        uint64_t dim,
        uint64_t numTrees,
        uint64_t evalSize,
        uint64_t nPublics) : proofs(polN, dim, numTrees, evalSize),
                             publics(nPublics){};
};

#endif