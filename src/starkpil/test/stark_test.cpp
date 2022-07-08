#include <iostream>
#include "stark_test.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <utility>
#include <tuple>

#include "utils.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "transcript.hpp"
#include "poseidon_goldilocks.hpp"
#include "merklehash_goldilocks.hpp"
#include "ntt_goldilocks.hpp"
#include "stark_info.hpp"
#include "commit_pols_all.hpp"
#include "constant_pols_all.hpp"
#include "zhInv.hpp"
#include "calculateExps.hpp"

#define commited_file "all.commit"
#define constant_file "all.const"
#define constant_tree_file "all.consttree"

#define starkInfo_File "all.starkinfo.json"

#define NUM_CHALLENGES 8

using namespace std;

void calculateZ(Goldilocks3::Element *z, Goldilocks3::Element *num, Goldilocks3::Element *den, uint64_t size)
{
    Goldilocks3::Element denI[size];
    Goldilocks3::Element checkVal;
    Goldilocks3::copy(z[0], Goldilocks3::one());

    Goldilocks3::batchInverse(denI, den, size);
    for (uint64_t i = 1; i < size; i++)
    {
        Goldilocks3::Element tmp;
        Goldilocks3::mul(tmp, num[i - 1], denI[i - 1]);
        Goldilocks3::mul(z[i], z[i - 1], tmp);
    }
    Goldilocks3::Element tmp;
    Goldilocks3::mul(tmp, num[size - 1], denI[size - 1]);
    Goldilocks3::mul(checkVal, z[size - 1], tmp);
    zkassert(Goldilocks3::isOne(checkVal));
}

uint64_t getPolSize(StarkInfo starkInfo, Expression exp)
{
    zkassert(!exp.isNull);
    VarPolMap p = starkInfo.varPolMap[exp.value];
    uint64_t N = starkInfo.mapDeg.getSection(p.section);
    return N * p.dim;
}

uint64_t getPolN(StarkInfo starkInfo, Expression exp)
{
    zkassert(!exp.isNull);
    VarPolMap p = starkInfo.varPolMap[exp.value];
    uint64_t N = starkInfo.mapDeg.getSection(p.section);
    return N;
}

uint64_t getPolDim(StarkInfo starkInfo, Expression exp)
{
    zkassert(!exp.isNull);
    VarPolMap p = starkInfo.varPolMap[exp.value];
    return p.dim;
}

uint64_t getTreeNumElements(uint64_t numCols, uint64_t degree)
{
    return numCols * degree + degree * HASH_SIZE + (degree - 1) * HASH_SIZE + MERKLEHASHGOLDILOCKS_HEADER_SIZE;
};

void getPol(Goldilocks::Element *res, Goldilocks::Element *pols, StarkInfo starkInfo, Expression exp)
{
    zkassert(!exp.isNull);
    VarPolMap p = starkInfo.varPolMap[exp.value];
    uint64_t N = starkInfo.mapDeg.getSection(p.section);
    uint64_t offset = starkInfo.mapOffsets.getSection(p.section);
    offset += p.sectionPos;
    uint64_t size = starkInfo.mapSectionsN.getSection(p.section);

    //#pragma omp parallel for
    for (uint64_t i = 0; i < N; i++)
    {
        std::memcpy(&res[i * p.dim], &pols[offset + i * size], p.dim * sizeof(Goldilocks::Element));
    }
};

void setPol(Goldilocks::Element *pols, Goldilocks::Element *pol, StarkInfo starkInfo, uint64_t i)
{
    VarPolMap p = starkInfo.varPolMap[i];
    uint64_t N = starkInfo.mapDeg.getSection(p.section);
    uint64_t offset = starkInfo.mapOffsets.getSection(p.section);
    offset += p.sectionPos;
    uint64_t size = starkInfo.mapSectionsN.getSection(p.section);

#pragma omp parallel for
    for (uint64_t i = 0; i < N; i++)
    {
        std::memcpy(&pols[offset + i * size], &pol[i * p.dim], p.dim * sizeof(Goldilocks::Element));
    }
};

void setPol(Goldilocks::Element *pols, Goldilocks3::Element *pol, StarkInfo starkInfo, uint64_t i)
{
    VarPolMap p = starkInfo.varPolMap[i];
    uint64_t N = starkInfo.mapDeg.getSection(p.section);
    uint64_t offset = starkInfo.mapOffsets.getSection(p.section);
    offset += p.sectionPos;
    uint64_t size = starkInfo.mapSectionsN.getSection(p.section);

    //#pragma omp parallel for
    for (uint64_t i = 0; i < N; i++)
    {
        std::memcpy(&pols[offset + i * size], &pol[i], p.dim * sizeof(Goldilocks::Element));
    }
};

class CompareGL3
{
public:
    bool operator()(const vector<Goldilocks::Element> &a, const vector<Goldilocks::Element> &b) const
    {
        return Goldilocks::toU64(a[1]) < Goldilocks::toU64(b[1]);
    }
};

void calculateH1H2(Goldilocks3::Element *h1, Goldilocks3::Element *h2, Goldilocks3::Element *fPol, Goldilocks3::Element *tPol, uint64_t fN, uint64_t tN)
{
    map<vector<Goldilocks::Element>, uint64_t, CompareGL3> idx_t;
    multimap<vector<Goldilocks::Element>, uint64_t, CompareGL3> s;
    multimap<vector<Goldilocks::Element>, uint64_t>::iterator it;
    uint64_t i = 0;

    for (uint64_t i = 0; i < tN; i++)
    {
        vector<Goldilocks::Element> key = Goldilocks3::toVector(tPol[i]);
        std::pair<vector<Goldilocks::Element>, uint64_t> pr(key, i);
        idx_t.insert(pr);
        s.insert(pr);
    }

    for (uint64_t i = 0; i < fN; i++)
    {
        vector<Goldilocks::Element> key = Goldilocks3::toVector(fPol[i]);

        if (idx_t.find(key) == idx_t.end())
        {
            cerr << "Error: calculateH1H2() Number not included: " << Goldilocks::toString(fPol[i], 16) << endl;
            exit(-1);
        }
        uint64_t idx = idx_t[key];
        s.insert(pair<vector<Goldilocks::Element>, uint64_t>(key, idx));
        // std::cout << Goldilocks3::toString(fPol[i]) << std::endl;
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
            Goldilocks3::copy(h1[i / 2], (Goldilocks3::Element &)*h);
        }
        else
        {
            Goldilocks3::copy(h2[i / 2], (Goldilocks3::Element &)*h);
        }
    }
};

void StarkTest(void)
{
    Config cfg;
    cfg.starkInfoFile = starkInfo_File;
    StarkInfo starkInfo(cfg);

    uint64_t numCommited = 0;

    uint64_t N = 1 << starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;
    uint64_t constTreeSize = starkInfo.nConstants * NExtended + NExtended * HASH_SIZE + (NExtended - 1) * HASH_SIZE + MERKLEHASHGOLDILOCKS_HEADER_SIZE;
    uint64_t constTreeSizeBytes = constTreeSize * sizeof(Goldilocks::Element);

    void *pAddress = malloc(starkInfo.mapTotalN * sizeof(Goldilocks::Element));
    void *pCommitedAddress = NULL;
    void *pConstantAddress = NULL;
    void *pConstant2nsAddress = NULL;
    void *pConstTreeAddress = NULL;

    Transcript transcript;
    ZhInv zi(starkInfo.starkStruct.nBits, starkInfo.starkStruct.nBitsExt);

    pCommitedAddress = mapFile(commited_file, starkInfo.nCm1 * starkInfo.mapDeg.cm1_n * sizeof(Goldilocks::Element), false);
    numCommited = starkInfo.nCm1;
    pConstantAddress = mapFile(constant_file, starkInfo.nConstants * (1 << starkInfo.starkStruct.nBits) * sizeof(Goldilocks::Element), false);
    pConstTreeAddress = mapFile(constant_tree_file, constTreeSizeBytes, false);

    pConstant2nsAddress = (void *)malloc(starkInfo.nConstants * (1 << starkInfo.starkStruct.nBitsExt) * sizeof(Goldilocks::Element));

    std::memcpy(pAddress, pCommitedAddress, starkInfo.nCm1 * starkInfo.mapDeg.cm1_n * sizeof(Goldilocks::Element));

    CommitPolsAll cmP(pAddress, starkInfo.mapDeg.cm1_n);
    ConstantPolsAll const_n(pConstantAddress, (1 << starkInfo.starkStruct.nBits));
    ConstantPolsAll const_2ns(pConstant2nsAddress, (1 << starkInfo.starkStruct.nBitsExt));

    for (uint64_t i = 0; i < starkInfo.nConstants; i++)
    {
        for (uint64_t j = 0; j < NExtended; j++)
        {
            MerklehashGoldilocks::getElement(const_2ns.getElement(i, j), (Goldilocks::Element *)pConstTreeAddress, j, i);
        }
    }

    // Compute x_n and x_2ns this could be pre-computed (TODO)
    Goldilocks::Element x_n[N];
    Goldilocks::Element x_2ns[NExtended];
    Goldilocks3::Element challenges[NUM_CHALLENGES];

    Goldilocks::Element xx = Goldilocks::one();
    for (uint i = 0; i < N; i++)
    {
        x_n[i] = xx;
        Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBits));
    }
    xx = Goldilocks::shift();
    for (uint i = 0; i < NExtended; i++)
    {
        x_2ns[i] = xx;
        Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBitsExt));
    }

    // TODO: Implement publics computation
    Goldilocks::Element publics[starkInfo.nPublics];
    publics[0] = Goldilocks::fromU64(1);
    publics[1] = Goldilocks::fromU64(2);
    publics[2] = Goldilocks::fromU64(74469561660084004);

    std::cout << "Merkelizing 1...." << std::endl;

    NTT_Goldilocks ntt(N);
    Goldilocks::Element *output = (Goldilocks::Element *)((uint8_t *)pAddress + (starkInfo.mapOffsets.cm1_2ns * sizeof(Goldilocks::Element)));
    ntt.extendPol(output, (Goldilocks::Element *)pAddress, NExtended, N, starkInfo.nCm1);

    uint64_t numElementsTree = getTreeNumElements(starkInfo.mapSectionsN1.cm1_n + starkInfo.mapSectionsN3.cm1_n * FIELD_EXTENSION, NExtended);

    // std::cout << numElementsTree << std::endl;
    Goldilocks::Element *tree1 = (Goldilocks::Element *)calloc(numElementsTree * sizeof(Goldilocks::Element), 1);

    Goldilocks::Element root[HASH_SIZE];
    PoseidonGoldilocks::merkletree(tree1, output, starkInfo.nCm1, NExtended);
    MerklehashGoldilocks::root(root, tree1, numElementsTree);

    transcript.put(root, HASH_SIZE);
    ///////////
    // 2.- Caluculate plookups h1 and h2
    ///////////
    transcript.getField(challenges[0]); // u
    transcript.getField(challenges[1]); // defVal

    CalculateExps::step2prev_first((Goldilocks::Element *)pAddress, const_n, challenges, 0);
#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        CalculateExps::step2prev_i((Goldilocks::Element *)pAddress, const_n, challenges, i);
    }
    CalculateExps::step2prev_last((Goldilocks::Element *)pAddress, const_n, challenges, N - 1);

    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        Goldilocks::Element *fPol = (Goldilocks::Element *)malloc(getPolSize(starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].fExpId]) * sizeof(Goldilocks::Element));
        Goldilocks::Element *tPol = (Goldilocks::Element *)malloc(getPolSize(starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].tExpId]) * sizeof(Goldilocks::Element));
        getPol(&fPol[0], (Goldilocks::Element *)pAddress, starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].fExpId]);
        getPol(&tPol[0], (Goldilocks::Element *)pAddress, starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].tExpId]);

        Goldilocks3::Element h1[getPolN(starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].fExpId])];
        Goldilocks3::Element h2[getPolN(starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].tExpId])];

        calculateH1H2(h1, h2, (Goldilocks3::Element *)fPol, (Goldilocks3::Element *)tPol, getPolN(starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].fExpId]), getPolN(starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].tExpId]));

        setPol((Goldilocks::Element *)pAddress, h1, starkInfo, starkInfo.cm_n[numCommited++]);
        setPol((Goldilocks::Element *)pAddress, h2, starkInfo, starkInfo.cm_n[numCommited++]);
        free(fPol);
        free(tPol);
    }
    std::cout << "Merkelizing 2...." << std::endl;
    Goldilocks::Element *pols = (Goldilocks::Element *)pAddress;
    Goldilocks::Element *dst = &pols[starkInfo.mapOffsets.cm2_2ns];
    Goldilocks::Element *src = &pols[starkInfo.mapOffsets.cm2_n];

    ntt.extendPol(dst, src, NExtended, N, starkInfo.mapSectionsN1.cm2_n + starkInfo.mapSectionsN3.cm2_n * FIELD_EXTENSION);

    uint64_t numElementsTree2 = getTreeNumElements(starkInfo.mapSectionsN1.cm2_n + starkInfo.mapSectionsN3.cm2_n * FIELD_EXTENSION, NExtended);
    // std::cout << numElementsTree2 << std::endl;
    Goldilocks::Element *tree2 = (Goldilocks::Element *)calloc(numElementsTree2, sizeof(Goldilocks::Element));

    PoseidonGoldilocks::merkletree(tree2, dst, starkInfo.mapSectionsN1.cm2_n + starkInfo.mapSectionsN3.cm2_n * FIELD_EXTENSION, NExtended);
    MerklehashGoldilocks::root(root, tree2, numElementsTree2);
    Goldilocks::Element *r = &(root[0]);

    // std::cout << Goldilocks::toString(r, HASH_SIZE, 10) << std::endl;
    transcript.put(root, HASH_SIZE);

    ///////////
    // 3.- Compute Z polynomials
    ///////////
    transcript.getField(challenges[2]); // gamma
    transcript.getField(challenges[3]); // betta

    CalculateExps::step3prev_first((Goldilocks::Element *)pAddress, const_n, challenges, x_n, 0);
#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        CalculateExps::step3prev_i((Goldilocks::Element *)pAddress, const_n, challenges, x_n, i);
    }
    CalculateExps::step3prev_last((Goldilocks::Element *)pAddress, const_n, challenges, x_n, N - 1);

    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        uint64_t N = getPolN(starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].numId]);
        zkassert(N == getPolN(starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].denId]));

        Goldilocks3::Element *pNum = (Goldilocks3::Element *)malloc(N * sizeof(Goldilocks3::Element));
        Goldilocks3::Element *pDen = (Goldilocks3::Element *)malloc(N * sizeof(Goldilocks3::Element));
        Goldilocks3::Element *z = (Goldilocks3::Element *)malloc(N * sizeof(Goldilocks3::Element));

        getPol((Goldilocks::Element *)&pNum[0], (Goldilocks::Element *)pAddress, starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].numId]);
        getPol((Goldilocks::Element *)&pDen[0], (Goldilocks::Element *)pAddress, starkInfo, starkInfo.exps_n[starkInfo.puCtx[i].denId]);

        calculateZ(z, pNum, pDen, N);

        setPol((Goldilocks::Element *)pAddress, z, starkInfo, starkInfo.cm_n[numCommited++]);

        free(z);
        free(pNum);
        free(pDen);
    }
    for (uint64_t i = 0; i < starkInfo.peCtx.size(); i++)
    {
        uint64_t N = getPolN(starkInfo, starkInfo.exps_n[starkInfo.peCtx[i].numId]);
        zkassert(N == getPolN(starkInfo, starkInfo.exps_n[starkInfo.peCtx[i].denId]));

        Goldilocks3::Element *pNum = (Goldilocks3::Element *)malloc(N * sizeof(Goldilocks3::Element));
        Goldilocks3::Element *pDen = (Goldilocks3::Element *)malloc(N * sizeof(Goldilocks3::Element));
        Goldilocks3::Element *z = (Goldilocks3::Element *)malloc(N * sizeof(Goldilocks3::Element));

        getPol((Goldilocks::Element *)&pNum[0], (Goldilocks::Element *)pAddress, starkInfo, starkInfo.exps_n[starkInfo.peCtx[i].numId]);
        getPol((Goldilocks::Element *)&pDen[0], (Goldilocks::Element *)pAddress, starkInfo, starkInfo.exps_n[starkInfo.peCtx[i].denId]);

        calculateZ(z, pNum, pDen, N);

        setPol((Goldilocks::Element *)pAddress, z, starkInfo, starkInfo.cm_n[numCommited++]);
        free(z);
        free(pNum);
        free(pDen);
    }
    for (uint64_t i = 0; i < starkInfo.ciCtx.size(); i++)
    {
        uint64_t N = getPolN(starkInfo, starkInfo.exps_n[starkInfo.ciCtx[i].numId]);
        zkassert(N == getPolN(starkInfo, starkInfo.exps_n[starkInfo.ciCtx[i].denId]));

        Goldilocks3::Element *pNum = (Goldilocks3::Element *)malloc(N * sizeof(Goldilocks3::Element));
        Goldilocks3::Element *pDen = (Goldilocks3::Element *)malloc(N * sizeof(Goldilocks3::Element));
        Goldilocks3::Element *z = (Goldilocks3::Element *)malloc(N * sizeof(Goldilocks3::Element));

        getPol((Goldilocks::Element *)&pNum[0], (Goldilocks::Element *)pAddress, starkInfo, starkInfo.exps_n[starkInfo.ciCtx[i].numId]);
        getPol((Goldilocks::Element *)&pDen[0], (Goldilocks::Element *)pAddress, starkInfo, starkInfo.exps_n[starkInfo.ciCtx[i].denId]);

        calculateZ(z, pNum, pDen, N);

        setPol((Goldilocks::Element *)pAddress, z, starkInfo, starkInfo.cm_n[numCommited++]);
        free(z);
        free(pNum);
        free(pDen);
    }

    std::cout << "Merkelizing 3...." << std::endl;
    dst = &pols[starkInfo.mapOffsets.cm3_2ns];
    src = &pols[starkInfo.mapOffsets.cm3_n];

    ntt.extendPol(dst, src, NExtended, N, starkInfo.mapSectionsN1.cm3_n + starkInfo.mapSectionsN3.cm3_n * FIELD_EXTENSION);

    uint64_t numElementsTree3 = getTreeNumElements(starkInfo.mapSectionsN1.cm3_n + starkInfo.mapSectionsN3.cm3_n * FIELD_EXTENSION, NExtended);
    Goldilocks::Element *tree3 = (Goldilocks::Element *)calloc(numElementsTree3, sizeof(Goldilocks::Element));

    PoseidonGoldilocks::merkletree(tree3, dst, starkInfo.mapSectionsN1.cm3_n + starkInfo.mapSectionsN3.cm3_n * FIELD_EXTENSION, NExtended);
    MerklehashGoldilocks::root(root, tree3, numElementsTree3);

    std::cout << Goldilocks::toString(&(root[0]), HASH_SIZE, 10) << std::endl;
    transcript.put(root, HASH_SIZE);

    ///////////
    // 4. Compute C Polynomial
    ///////////
    transcript.getField(challenges[4]); // gamma

    CalculateExps::step4_first((Goldilocks::Element *)pAddress, const_n, const_2ns, challenges, x_n, x_2ns, publics, zi, 0);
#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        CalculateExps::step4_i((Goldilocks::Element *)pAddress, const_n, const_2ns, challenges, x_n, x_2ns, publics, zi, i);
    }
    CalculateExps::step4_last((Goldilocks::Element *)pAddress, const_n, const_2ns, challenges, x_n, x_2ns, publics, zi, N - 1);
    dst = &pols[starkInfo.mapOffsets.exps_withq_2ns];
    src = &pols[starkInfo.mapOffsets.exps_withq_n];

    ntt.extendPol(dst, src, NExtended, N, starkInfo.mapSectionsN1.exps_withq_n + starkInfo.mapSectionsN3.exps_withq_n * FIELD_EXTENSION);

    uint64_t next = 1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits);

    for (uint64_t i = 0; i < next; i++)
    {
        CalculateExps::step42ns_first((Goldilocks::Element *)pAddress, const_n, const_2ns, challenges, x_n, x_2ns, publics, zi, i);
    }
#pragma omp parallel for
    for (uint64_t i = next; i < NExtended - next; i++)
    {
        CalculateExps::step42ns_i((Goldilocks::Element *)pAddress, const_n, const_2ns, challenges, x_n, x_2ns, publics, zi, i);
    }
    for (uint64_t i = NExtended - next; i < NExtended; i++)
    {
        CalculateExps::step42ns_last((Goldilocks::Element *)pAddress, const_n, const_2ns, challenges, x_n, x_2ns, publics, zi, i);
    }

    std::cout << "Merkelizing 4...." << std::endl;
    dst = &pols[starkInfo.mapOffsets.q_2ns];

    std::cout << Goldilocks::toString((Goldilocks::Element *)&dst[0], N * starkInfo.mapSectionsN.q_2ns + 1000, 10) << std::endl;

    uint64_t numElementsTree4 = getTreeNumElements(starkInfo.mapSectionsN.q_2ns, NExtended);
    Goldilocks::Element *tree4 = (Goldilocks::Element *)calloc(numElementsTree4, sizeof(Goldilocks::Element));

    PoseidonGoldilocks::merkletree(tree4, dst, starkInfo.mapSectionsN.q_2ns, NExtended);
    MerklehashGoldilocks::root(root, tree4, numElementsTree4);

    std::cout << Goldilocks::toString(&(root[0]), HASH_SIZE, 10) << std::endl;
    transcript.put(root, HASH_SIZE);

    free(tree1);
    free(tree2);
}
