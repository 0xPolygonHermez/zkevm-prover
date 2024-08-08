#ifndef CONST_POLS_STARKS_HPP
#define CONST_POLS_STARKS_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"
#include "stark_info.hpp"
#include "zklog.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "ntt_goldilocks.hpp"
#include "merkleTreeBN128.hpp"
#include "merkleTreeGL.hpp"


template <typename ElementType>
class ConstPols 
{
private:
    StarkInfo& starkInfo;
    uint64_t N;
    uint64_t NExtended;

    uint64_t nFieldElements;
    uint64_t merkleTreeArity;
    uint64_t merkleTreeCustom;

public:
    using MerkleTreeType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, MerkleTreeGL, MerkleTreeBN128>;

    void *pConstPolsAddress;
    void *pConstPolsAddress2ns;
    void *pConstTreeAddress;

    ConstantPolsStarks *pConstPols;
    ConstantPolsStarks *pConstPols2ns;

    ConstPols(StarkInfo& starkInfo_, std::string constPolsFile): starkInfo(starkInfo_), N(1 << starkInfo.starkStruct.nBits), NExtended(1 << starkInfo.starkStruct.nBitsExt) {
        
        if(starkInfo.starkStruct.verificationHashType == std::string("BN128")) {
            nFieldElements = 1;
            merkleTreeArity = starkInfo.starkStruct.merkleTreeArity;
            merkleTreeCustom = starkInfo.starkStruct.merkleTreeCustom;
        } else {
            nFieldElements = HASH_SIZE;
            merkleTreeArity = 2;
            merkleTreeCustom = true;
        }

        loadConstPols(starkInfo, constPolsFile);

        TimerStart(CALCULATE_CONST_TREE_TO_MEMORY);
        pConstTreeAddress = malloc(getConstTreeSize());
        if(pConstTreeAddress == NULL)
        {
            zklog.error("Starks::Starks() failed to allocate pConstTreeAddress");
            exitProcess();
        }
        pConstPolsAddress2ns = (uint8_t *)pConstTreeAddress + 2 * sizeof(uint64_t);

        TimerStart(EXTEND_CONST_POLS);
        NTT_Goldilocks ntt(N);
        ntt.extendPol((Goldilocks::Element *)pConstPolsAddress2ns, (Goldilocks::Element *)pConstPolsAddress, NExtended, N, starkInfo.nConstants);
        TimerStopAndLog(EXTEND_CONST_POLS);
        TimerStart(MERKELIZE_CONST_TREE);
        MerkleTreeGL mt(merkleTreeArity, merkleTreeCustom, NExtended, starkInfo.nConstants, (Goldilocks::Element *)pConstPolsAddress2ns);
        mt.merkelize();
        TimerStopAndLog(MERKELIZE_CONST_TREE);

        uint64_t* constTreePtr = (uint64_t*)pConstTreeAddress;
        constTreePtr[0] = starkInfo.nConstants;  
        constTreePtr[1] = NExtended;
        memcpy(constTreePtr + 2 + starkInfo.nConstants * NExtended, mt.nodes, mt.numNodes * sizeof(Goldilocks::Element));

        TimerStopAndLog(CALCULATE_CONST_TREE_TO_MEMORY);

        // Allocate ConstantPols2ns
        pConstPols2ns = new ConstantPolsStarks(pConstPolsAddress2ns, NExtended, starkInfo.nConstants);
    }

    ConstPols(StarkInfo& starkInfo_, std::string constPolsFile, std::string constTreeFile) : starkInfo(starkInfo_), N(1 << starkInfo.starkStruct.nBits), NExtended(1 << starkInfo.starkStruct.nBitsExt) {
        
        if(starkInfo.starkStruct.verificationHashType == std::string("BN128")) {
            nFieldElements = 1;
            merkleTreeArity = starkInfo.starkStruct.merkleTreeArity;
            merkleTreeCustom = starkInfo.starkStruct.merkleTreeCustom;
        } else {
            nFieldElements = HASH_SIZE;
            merkleTreeArity = 2;
            merkleTreeCustom = true;
        }

        loadConstPols(starkInfo, constPolsFile);

        TimerStart(LOAD_CONST_TREE_TO_MEMORY);
            
        uint64_t constTreeSizeBytes = getConstTreeSize();

        pConstTreeAddress = loadFileParallel(constPolsFile, constTreeSizeBytes);
        zklog.info("Starks::Starks() successfully copied " + to_string(constTreeSizeBytes) + " bytes from constant file " + constTreeFile);
        
        pConstPolsAddress2ns = (uint8_t *)pConstTreeAddress + 2 * sizeof(uint64_t);
        TimerStopAndLog(LOAD_CONST_TREE_TO_MEMORY);
                
        // Allocate ConstantPols2ns
        pConstPols2ns = new ConstantPolsStarks(pConstPolsAddress2ns, NExtended, starkInfo.nConstants);
    }

    void loadConstPols(StarkInfo& starkInfo, std::string constPolsFile) {
        // Allocate an area of memory, mapped to file, to read all the constant polynomials,
        // and create them using the allocated address
        TimerStart(LOAD_CONST_POLS_TO_MEMORY);

        uint64_t constPolsSize = starkInfo.nConstants * sizeof(Goldilocks::Element) * N;
        
        pConstPolsAddress = loadFileParallel(constPolsFile, constPolsSize);
        zklog.info("Starks::Starks() successfully copied " + to_string(constPolsSize) + " bytes from constant file " + constPolsFile);
        
        // Allocate ConstantPols
        pConstPols = new ConstantPolsStarks(pConstPolsAddress, constPolsSize, starkInfo.nConstants);
        TimerStopAndLog(LOAD_CONST_POLS_TO_MEMORY);
    }

    uint64_t getConstTreeSize()
    {   
        uint n_tmp = NExtended;
        uint64_t nextN = floor(((double)(n_tmp - 1) / merkleTreeArity) + 1);
        uint64_t acc = nextN * merkleTreeArity;
        while (n_tmp > 1)
        {
            // FIll with zeros if n nodes in the leve is not even
            n_tmp = nextN;
            nextN = floor((n_tmp - 1) / merkleTreeArity) + 1;
            if (n_tmp > 1)
            {
                acc += nextN * merkleTreeArity;
            }
            else
            {
                acc += 1;
            }
        }

        uint64_t numElements = (1 << starkInfo.starkStruct.nBitsExt) * starkInfo.nConstants * sizeof(Goldilocks::Element);
        uint64_t total = numElements + acc * nFieldElements * sizeof(ElementType);
        if(starkInfo.starkStruct.verificationHashType == std::string("BN128")) {
            total += 16; // HEADER
        } else {
            total += merkleTreeArity * sizeof(ElementType);
        }
        return total; 
        
    };

    ~ConstPols()
    {
        free(pConstPolsAddress);
        free(pConstTreeAddress);

        delete pConstPols;
        delete pConstPols2ns;
    }
};

#endif // CONST_POLS_STARKS_HPP

