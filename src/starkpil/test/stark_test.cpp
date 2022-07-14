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

// Test vectors files
#define starkInfo_File "all.starkinfo.json"
#define commited_file "all.commit"
#define constant_file "all.const"
#define constant_tree_file "all.consttree"

#define NUM_CHALLENGES_TEST 8

using namespace std;

void StarkTest(void)
{

    // Load config & test vectors
    Config cfg;
    cfg.starkInfoFile = starkInfo_File;
    StarkInfo starkInfo(cfg);

    // Computed vars
    uint64_t N = 1 << starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;
    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
    uint64_t numCommited = starkInfo.nCm1;

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

    ////////////////
    /// CONSTRUCTOR
    ////////////////
    Transcript transcript;
    ZhInv zi(cfg, starkInfo.starkStruct.nBits, starkInfo.starkStruct.nBitsExt);

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
    Goldilocks::Element publics[3];
    publics[0] = Goldilocks::fromU64(1);
    publics[1] = Goldilocks::fromU64(2);
    publics[2] = Goldilocks::fromU64(74469561660084004);

    std::cout << "Merkelizing 1...." << std::endl;

    uint64_t numElementsTree1 = MerklehashGoldilocks::getTreeNumElements(starkInfo.mapSectionsN1.section[eSection::cm1_n] + starkInfo.mapSectionsN3.section[eSection::cm1_n] * FIELD_EXTENSION, NExtended);

    Polinomial tree1(numElementsTree1, 1);
    Polinomial root1(HASH_SIZE, 1);

    Goldilocks::Element *mem = (Goldilocks::Element *)pAddress;

    Goldilocks::Element *p_cm2_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm2_2ns]];
    Goldilocks::Element *p_cm1_n = &mem[starkInfo.mapOffsets.section[eSection::cm1_n]];

    ntt.extendPol(p_cm2_2ns, p_cm1_n, NExtended, N, starkInfo.nCm1);

    PoseidonGoldilocks::merkletree(tree1.address(), p_cm2_2ns, starkInfo.nCm1, NExtended);

    MerklehashGoldilocks::root(root1.address(), tree1.address(), tree1.length());
    std::cout << "MerkleTree root 1: [ " << root1.toString(4) << " ]" << std::endl;
    transcript.put(root1.address(), HASH_SIZE);

    return;
}
