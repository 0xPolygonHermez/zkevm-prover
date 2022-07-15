#include <iostream>
#include "stark_test.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "utils.hpp"
#include "stark_info.hpp"
#include "transcript.hpp"
#include "zhInv.hpp"
#include "commit_pols_all.hpp"
#include "constant_pols_all.hpp"
#include "merklehash_goldilocks.hpp"
#include "timer.hpp"
#include "utils.hpp"
#include "polinomial.hpp"
#include "ntt_goldilocks.hpp"
#include "calculateExps_all.hpp"
#include <vector>

// Test vectors files
#define starkInfo_File "all.starkinfo.json"
#define commited_file "all.commit"
#define constant_file "all.const"
#define constant_tree_file "all.consttree"

#define NUM_CHALLENGES_TEST 8

using namespace std;

class CompareGL3
{
public:
    bool operator()(const vector<Goldilocks::Element> &a, const vector<Goldilocks::Element> &b) const
    {
        return Goldilocks::toU64(a[1]) < Goldilocks::toU64(b[1]);
    }
};

void calculateH1H2(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol)
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
};

void StarkTest(void)
{

    // Load config & test vectors
    Config cfg;
    cfg.starkInfoFile = starkInfo_File;
    cfg.runProverServer = true;
    StarkInfo starkInfo(cfg);

    // Computed vars
    uint64_t N = 1 << starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;
    // uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
    // uint64_t numCommited = starkInfo.nCm1;

    // Load test vector data
    uint64_t constTreeSize = starkInfo.nConstants * NExtended + NExtended * HASH_SIZE + (NExtended - 1) * HASH_SIZE + MERKLEHASHGOLDILOCKS_HEADER_SIZE;
    uint64_t constTreeSizeBytes = constTreeSize * sizeof(Goldilocks::Element);

    void *pAddress = malloc(starkInfo.mapTotalN * sizeof(Goldilocks::Element));
    void *pCommitedAddress = NULL;
    void *pConstantAddress = NULL;
    void *pConstant2nsAddress = NULL;
    void *pConstTreeAddress = NULL;

    pCommitedAddress = mapFile(commited_file, starkInfo.nCm1 * starkInfo.mapDeg.section[eSection::cm1_n] * sizeof(Goldilocks::Element), false);
    pConstantAddress = mapFile(constant_file, starkInfo.nConstants * (1 << starkInfo.starkStruct.nBits) * sizeof(Goldilocks::Element), false);
    pConstTreeAddress = mapFile(constant_tree_file, constTreeSizeBytes, false);
    pConstant2nsAddress = (void *)malloc(starkInfo.nConstants * (1 << starkInfo.starkStruct.nBitsExt) * sizeof(Goldilocks::Element));

    std::memcpy(pAddress, pCommitedAddress, starkInfo.nCm1 * starkInfo.mapDeg.section[eSection::cm1_n] * sizeof(Goldilocks::Element));
    CommitPolsAll cmP(pAddress, starkInfo.mapDeg.section[eSection::cm1_n]);
    ConstantPolsAll const_n(pConstantAddress, (1 << starkInfo.starkStruct.nBits));
    ConstantPolsAll const_2ns(pConstant2nsAddress, (1 << starkInfo.starkStruct.nBitsExt));

    Goldilocks::Element *mem = (Goldilocks::Element *)pAddress;

    ////////////////
    /// CONSTRUCTOR
    ////////////////
    Transcript transcript;
    ZhInv zi(starkInfo.starkStruct.nBits, starkInfo.starkStruct.nBitsExt);

    TimerStart(LOAD_CONST_2NS_POLS_TO_MEMORY);
    for (uint64_t i = 0; i < starkInfo.nConstants; i++)
    {
        for (uint64_t j = 0; j < NExtended; j++)
        {
            MerklehashGoldilocks::getElement(const_2ns.getElement(i, j), (Goldilocks::Element *)pConstTreeAddress, j, i);
        }
    }
    TimerStopAndLog(LOAD_CONST_2NS_POLS_TO_MEMORY);

    // Compute x_n and x_2ns this could be pre-computed (TODO)
    Polinomial x_n(N, 1);
    Polinomial x_2ns(NExtended, 1);
    Polinomial challenges(NUM_CHALLENGES_TEST, 3);
    NTT_Goldilocks ntt(N);
    Goldilocks::Element xx = Goldilocks::one();

    for (uint i = 0; i < N; i++)
    {
        *x_n[i] = xx;
        Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBits));
    }
    xx = Goldilocks::shift();
    for (uint i = 0; i < NExtended; i++)
    {
        *x_2ns[i] = xx;
        Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBitsExt));
    }
    // TODO: Implement publics computation
    Goldilocks::Element publics[3];
    publics[0] = Goldilocks::fromU64(1);
    publics[1] = Goldilocks::fromU64(2);
    publics[2] = Goldilocks::fromU64(74469561660084004);

    std::cout << "Merkelizing 1...." << std::endl;

    uint64_t numElementsTree1 = MerklehashGoldilocks::getTreeNumElements(starkInfo.mapSectionsN1.section[eSection::cm1_n] + starkInfo.mapSectionsN3.section[eSection::cm1_n] * FIELD_EXTENSION, NExtended);

    Polinomial tree1(numElementsTree1, 1);
    Polinomial root1(HASH_SIZE, 1);

    Goldilocks::Element *p_cm2_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm2_2ns]];
    Goldilocks::Element *p_cm1_n = &mem[starkInfo.mapOffsets.section[eSection::cm1_n]];

    ntt.extendPol(p_cm2_2ns, p_cm1_n, NExtended, N, starkInfo.nCm1);

    PoseidonGoldilocks::merkletree(tree1.address(), p_cm2_2ns, starkInfo.nCm1, NExtended);

    MerklehashGoldilocks::root(root1.address(), tree1.address(), tree1.length());
    std::cout << "MerkleTree root 1: [ " << root1.toString(4) << " ]" << std::endl;
    transcript.put(root1.address(), HASH_SIZE);

    ///////////
    // 2.- Caluculate plookups h1 and h2
    ///////////
    transcript.getField(challenges[0]); // u
    transcript.getField(challenges[1]); // defVal
    CalculateExpsAll::step2prev_first(mem, const_n, (Goldilocks3::Element *)challenges.address(), 0);

#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        CalculateExpsAll::step2prev_i(mem, const_n, (Goldilocks3::Element *)challenges.address(), i);
    }
    CalculateExpsAll::step2prev_last(mem, const_n, (Goldilocks3::Element *)challenges.address(), N - 1);

    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        Polinomial fPol = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].fExpId]);
        Polinomial tPol = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].tExpId]);

        std::cout << fPol.toString(3) << std::endl;
        std::cout << tPol.toString(3) << std::endl;

        Polinomial h1(fPol.degree(), 3, "h1");
        Polinomial h2(tPol.degree(), 3, "h2");

        calculateH1H2(h1, h2, fPol, tPol);
        std::cout << h1.toString(3) << std::endl;
        std::cout << h2.toString(3) << std::endl;
        /*
        Goldilocks3::Element h1[getPolN(starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].fExpId])];
        Goldilocks3::Element h2[getPolN(starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].tExpId])];

        calculateH1H2(h1, h2, (Goldilocks3::Element *)fPol, (Goldilocks3::Element *)tPol, getPolN(starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].fExpId]), getPolN(starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].tExpId]));

        setPol((Goldilocks::Element *)pAddress, h1, starkInfo, starkInfo.cm_n[numCommited++]);
        setPol((Goldilocks::Element *)pAddress, h2, starkInfo, starkInfo.cm_n[numCommited++]);
        free(fPol);
        free(tPol);
        */
    }
    /*
    std::cout << "Merkelizing 2...." << std::endl;
    Goldilocks::Element *pols = (Goldilocks::Element *)pAddress;
    Goldilocks::Element *dst = &pols[starkInfo.mapOffsets.section[eSection::cm2_2ns]];
    Goldilocks::Element *src = &pols[starkInfo.mapOffsets.section[eSection::cm2_n]];

    ntt.extendPol(dst, src, NExtended, N, starkInfo.mapSectionsN1.section[eSection::cm2_n] + starkInfo.mapSectionsN3.section[eSection::cm2_n] * FIELD_EXTENSION);
    uint64_t numElementsTree2 = MerklehashGoldilocks::getTreeNumElements(starkInfo.mapSectionsN1.section[eSection::cm2_n] + starkInfo.mapSectionsN3.section[eSection::cm2_n] * FIELD_EXTENSION, NExtended);
    Goldilocks::Element *tree2 = (Goldilocks::Element *)calloc(numElementsTree2, sizeof(Goldilocks::Element));

    Goldilocks::Element root2[HASH_SIZE];
    PoseidonGoldilocks::merkletree(tree2, dst, starkInfo.mapSectionsN1.section[eSection::cm2_n] + starkInfo.mapSectionsN3.section[eSection::cm2_n] * FIELD_EXTENSION, NExtended);
    MerklehashGoldilocks::root(root2, tree2, numElementsTree2);
    std::cout << Goldilocks::toString(&(root2[0]), HASH_SIZE, 10) << std::endl;
    transcript.put(&(root2[0]), HASH_SIZE);
*/
    return;
}
