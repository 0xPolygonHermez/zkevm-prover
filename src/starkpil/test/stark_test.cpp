#include "stark_test.hpp"

#include "starkMock.hpp"

#define NUM_CHALLENGES 8

void StarkTest(void)
{
#include "public_inputs_all.hpp"
#define starkInfo_File "all.starkinfo.json"
#define commited_file "all.commit"
#define constant_file "all.const"
#define constant_tree_file "all.consttree"
    // Load config & test vectors
    Config cfg;
    cfg.starkInfoFile = starkInfo_File;
    cfg.constPolsFile = constant_file;
    cfg.mapConstPolsFile = false;
    cfg.runProverServer = true;
    cfg.constantsTreeFile = "all.consttree";
    StarkInfo starkInfo(cfg);
    StarkMock stark(cfg);

    void *pAddress = malloc(starkInfo.mapTotalN * sizeof(Goldilocks::Element));
    void *pCommitedAddress = mapFile(commited_file, starkInfo.nCm1 * starkInfo.mapDeg.section[eSection::cm1_n] * sizeof(Goldilocks::Element), false);
    std::memcpy(pAddress, pCommitedAddress, starkInfo.nCm1 * starkInfo.mapDeg.section[eSection::cm1_n] * sizeof(Goldilocks::Element));

    CommitPolsAll cmP(pAddress, starkInfo.mapDeg.section[eSection::cm1_n]);
    PublicInputsAll publics;
    publics.inputs[0] = Goldilocks::fromU64(1);
    publics.inputs[1] = Goldilocks::fromU64(2);
    publics.inputs[2] = Goldilocks::fromU64(74469561660084004);

    void *pConstantAddress = NULL;
    pConstantAddress = mapFile(constant_file, starkInfo.nConstants * (1 << starkInfo.starkStruct.nBits) * sizeof(Goldilocks::Element), false);
    ConstantPolsAll const_n(pConstantAddress, (1 << starkInfo.starkStruct.nBits));

    Proof proof;

    stark.genProof(pAddress, cmP, const_n, publics, proof);
}

void StarkMock::calculateH1H2(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol)
{
    map<std::vector<Goldilocks::Element>, uint64_t, CompareGL3> idx_t;
    multimap<std::vector<Goldilocks::Element>, uint64_t, CompareGL3> s;
    multimap<std::vector<Goldilocks::Element>, uint64_t>::iterator it;
    uint64_t i = 0;

    for (uint64_t i = 0; i < tPol.degree(); i++)
    {
        vector<Goldilocks::Element> key = Goldilocks3::toVector((Goldilocks3::Element *)tPol[i]);
        std::pair<vector<Goldilocks::Element>, uint64_t> pr(key, i);
        idx_t.insert(pr);
        s.insert(pr);
    }

    for (uint64_t i = 0; i < fPol.degree(); i++)
    {
        vector<Goldilocks::Element> key = Goldilocks3::toVector((Goldilocks3::Element *)fPol[i]);

        if (idx_t.find(key) == idx_t.end())
        {
            cerr << "Error: calculateH1H2() Number not included: " << Goldilocks::toString(fPol[i], 16) << endl;
            exit(-1);
        }
        uint64_t idx = idx_t[key];
        s.insert(pair<vector<Goldilocks::Element>, uint64_t>(key, idx));
    }

    multimap<uint64_t, vector<Goldilocks::Element>> s_sorted;
    multimap<uint64_t, vector<Goldilocks::Element>>::iterator it_sorted;

    for (it = s.begin(); it != s.end(); it++)
    {
        s_sorted.insert(make_pair(it->second, it->first));
    }
    for (it_sorted = s_sorted.begin(); it_sorted != s_sorted.end(); it_sorted++, i++)
    {
        Goldilocks::Element *h = it_sorted->second.data();

        if ((i & 1) == 0)
        {
            Goldilocks3::copy((Goldilocks3::Element *)h1[i / 2], (Goldilocks3::Element *)h);
        }
        else
        {
            Goldilocks3::copy((Goldilocks3::Element *)h2[i / 2], (Goldilocks3::Element *)h);
        }
    }
}
