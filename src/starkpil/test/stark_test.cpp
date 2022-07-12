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

// Test vectors files
#define starkInfo_File "all.starkinfo.json"
#define commited_file "all.commit"
#define constant_file "all.const"
#define constant_tree_file "all.consttree"

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
    return;
}
