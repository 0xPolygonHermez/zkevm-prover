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

    Goldilocks::Element *pConstPolsAddress;
    Goldilocks::Element *pConstPolsAddressExtended;
    Goldilocks::Element *pConstTreeAddress;

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
        pConstTreeAddress = (Goldilocks::Element *)malloc(getConstTreeSize());
        if(pConstTreeAddress == NULL)
        {
            zklog.error("Starks::Starks() failed to allocate pConstTreeAddress");
            exitProcess();
        }
        pConstPolsAddressExtended = &pConstTreeAddress[2];

        TimerStart(EXTEND_CONST_POLS);
        NTT_Goldilocks ntt(N);
        ntt.extendPol((Goldilocks::Element *)pConstPolsAddressExtended, (Goldilocks::Element *)pConstPolsAddress, NExtended, N, starkInfo.nConstants);
        TimerStopAndLog(EXTEND_CONST_POLS);
        TimerStart(MERKELIZE_CONST_TREE);
        MerkleTreeGL mt(merkleTreeArity, merkleTreeCustom, NExtended, starkInfo.nConstants, (Goldilocks::Element *)pConstPolsAddressExtended);
        mt.merkelize();
        TimerStopAndLog(MERKELIZE_CONST_TREE);

        pConstTreeAddress[0] = Goldilocks::fromU64(starkInfo.nConstants);  
        pConstTreeAddress[1] = Goldilocks::fromU64(NExtended);
        memcpy(&pConstTreeAddress[2 + starkInfo.nConstants * NExtended], mt.nodes, mt.numNodes * sizeof(Goldilocks::Element));

        TimerStopAndLog(CALCULATE_CONST_TREE_TO_MEMORY);
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

        pConstTreeAddress = (Goldilocks::Element *)loadFileParallel(constPolsFile, constTreeSizeBytes);
        zklog.info("Starks::Starks() successfully copied " + to_string(constTreeSizeBytes) + " bytes from constant file " + constTreeFile);
        
        pConstPolsAddressExtended = &pConstTreeAddress[2];
        TimerStopAndLog(LOAD_CONST_TREE_TO_MEMORY);
    }

    void loadConstPols(StarkInfo& starkInfo, std::string constPolsFile) {
        // Allocate an area of memory, mapped to file, to read all the constant polynomials,
        // and create them using the allocated address
        TimerStart(LOAD_CONST_POLS_TO_MEMORY);

        uint64_t constPolsSize = starkInfo.nConstants * sizeof(Goldilocks::Element) * N;
        
        pConstPolsAddress = (Goldilocks::Element *)loadFileParallel(constPolsFile, constPolsSize);
        zklog.info("Starks::Starks() successfully copied " + to_string(constPolsSize) + " bytes from constant file " + constPolsFile);
        
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
    }
};

#endif // CONST_POLS_STARKS_HPP

