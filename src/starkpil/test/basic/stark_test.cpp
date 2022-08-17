#include "stark_test.hpp"
#include "starkMock.hpp"

#define NUM_CHALLENGES 8

#define starkInfo_File "basic.starkinfo.json"
#define commited_file "basic.commit"
#define constant_file "basic.const"
#define constant_tree_file "basic.consttree"

void StarkTest(void)
{
    // Load config & test vectors
    Config cfg;
    cfg.starkInfoFile = starkInfo_File;
    cfg.constPolsFile = constant_file;
    cfg.mapConstPolsFile = false;
    cfg.runProverServer = true;
    cfg.constantsTreeFile = constant_tree_file;
    StarkInfo starkInfo(cfg);
    StarkMock stark(cfg);

    void *pAddress = malloc(starkInfo.mapTotalN * sizeof(Goldilocks::Element));
    void *pCommitedAddress = mapFile(commited_file, starkInfo.nCm1 * starkInfo.mapDeg.section[eSection::cm1_n] * sizeof(Goldilocks::Element), false);
    std::memcpy(pAddress, pCommitedAddress, starkInfo.nCm1 * starkInfo.mapDeg.section[eSection::cm1_n] * sizeof(Goldilocks::Element));

    CommitPolsBasic cmP(pAddress, starkInfo.mapDeg.section[eSection::cm1_n]);

    void *pConstantAddress = NULL;
    pConstantAddress = mapFile(constant_file, starkInfo.nConstants * (1 << starkInfo.starkStruct.nBits) * sizeof(Goldilocks::Element), false);
    ConstantPolsBasic const_n(pConstantAddress, (1 << starkInfo.starkStruct.nBits));

    Proof proof;

    stark.genProof(pAddress, proof);
}

void StarkMock::calculateH1H2(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol)
{
    map<std::vector<Goldilocks::Element>, uint64_t, CompareGL3> idx_t;
    multimap<std::vector<Goldilocks::Element>, uint64_t, CompareGL3> s;
    multimap<std::vector<Goldilocks::Element>, uint64_t>::iterator it;
    uint64_t i = 0;

    for (uint64_t i = 0; i < tPol.degree(); i++)
    {
        vector<Goldilocks::Element> key = tPol.toVector(i);
        std::pair<vector<Goldilocks::Element>, uint64_t> pr(key, i);

        auto const result = idx_t.insert(pr);
        if (not result.second)
        {
            result.first->second = i;
        }

        s.insert(pr);
    }

    for (uint64_t i = 0; i < fPol.degree(); i++)
    {
        vector<Goldilocks::Element> key = fPol.toVector(i);

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
        if ((i & 1) == 0)
        {
            Polinomial::copyElement(h1, i / 2, it_sorted->second);
        }
        else
        {
            Polinomial::copyElement(h2, i / 2, it_sorted->second);
        }
    }
}
