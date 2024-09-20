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
    Goldilocks::Element *pConstPolsAddress;
    Goldilocks::Element *pConstPolsAddressExtended;
    Goldilocks::Element *pConstTreeAddress;
    Goldilocks::Element *zi;
    Goldilocks::Element *S;
    Goldilocks::Element *x;
    Goldilocks::Element *x_n; // Needed for PIL1 compatibility
    Goldilocks::Element *x_2ns; // Needed for PIL1 compatibility

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

        NTT_Goldilocks ntt(N);
        ntt.extendPol((Goldilocks::Element *)pConstPolsAddressExtended, (Goldilocks::Element *)pConstPolsAddress, NExtended, N, starkInfo.nConstants);
        MerkleTreeGL mt(merkleTreeArity, merkleTreeCustom, NExtended, starkInfo.nConstants, (Goldilocks::Element *)pConstPolsAddressExtended);
        mt.merkelize();

        pConstTreeAddress[0] = Goldilocks::fromU64(starkInfo.nConstants);  
        pConstTreeAddress[1] = Goldilocks::fromU64(NExtended);
        memcpy(&pConstTreeAddress[2 + starkInfo.nConstants * NExtended], mt.nodes, mt.numNodes * sizeof(Goldilocks::Element));

        TimerStopAndLog(CALCULATE_CONST_TREE_TO_MEMORY);

        computeZerofier();

        computeX();

        computeConnectionsX(); // Needed for PIL1 compatibility
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

        pConstTreeAddress = (Goldilocks::Element *)loadFileParallel(constTreeFile, constTreeSizeBytes);
        zklog.info("Starks::Starks() successfully copied " + to_string(constTreeSizeBytes) + " bytes from constant file " + constTreeFile);
        
        pConstPolsAddressExtended = &pConstTreeAddress[2];
        TimerStopAndLog(LOAD_CONST_TREE_TO_MEMORY);

        computeZerofier();

        computeX();

        computeConnectionsX(); // Needed for PIL1 compatibility
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

        uint64_t elementSize = starkInfo.starkStruct.verificationHashType == std::string("BN128") ? sizeof(RawFr::Element) : sizeof(Goldilocks::Element);
        uint64_t numElements = NExtended * starkInfo.nConstants * sizeof(Goldilocks::Element);
        uint64_t total = numElements + acc * nFieldElements * elementSize;
        if(starkInfo.starkStruct.verificationHashType == std::string("BN128")) {
            total += 16; // HEADER
        } else {
            total += merkleTreeArity * elementSize;
        }
        return total; 
        
    };

    void computeZerofier() {
        zi = new Goldilocks::Element[starkInfo.boundaries.size() * NExtended];

        for(uint64_t i = 0; i < starkInfo.boundaries.size(); ++i) {
            Boundary boundary = starkInfo.boundaries[i];
            if(boundary.name == "everyRow") {
                buildZHInv();
            } else if(boundary.name == "firstRow") {
                buildOneRowZerofierInv(i, 0);
            } else if(boundary.name == "lastRow") {
                buildOneRowZerofierInv(i, N);
            } else if(boundary.name == "everyRow") {
                buildFrameZerofierInv(i, boundary.offsetMin, boundary.offsetMax);
            }
        }
    }

    void computeConnectionsX() {
        uint64_t N = 1 << starkInfo.starkStruct.nBits;
        uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;
        x_n = new Goldilocks::Element[N];
        Goldilocks::Element xx = Goldilocks::one();
        for (uint64_t i = 0; i < N; i++)
        {
            x_n[i] = xx;
            Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBits));
        }
        xx = Goldilocks::shift();
        x_2ns = new Goldilocks::Element[NExtended];
        for (uint64_t i = 0; i < NExtended; i++)
        {
            x_2ns[i] = xx;
            Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBitsExt));
        }
    }

    void computeX() {
        uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
        x = new Goldilocks::Element[N << extendBits];
        x[0] = Goldilocks::shift();
        for (uint64_t k = 1; k < (N << extendBits); k++)
        {
            x[k] = x[k - 1] * Goldilocks::w(starkInfo.starkStruct.nBits + extendBits);
        }

        S = new Goldilocks::Element[starkInfo.qDeg];
        Goldilocks::Element shiftIn = Goldilocks::exp(Goldilocks::inv(Goldilocks::shift()), N);
        S[0] = Goldilocks::one();
        for(uint64_t i = 1; i < starkInfo.qDeg; i++) {
            S[i] = Goldilocks::mul(S[i - 1], shiftIn);
        }
    }

    void buildZHInv()
    {
        uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
        uint64_t extend = (1 << extendBits);
        
        Goldilocks::Element w = Goldilocks::one();
        Goldilocks::Element sn = Goldilocks::shift();

        for (uint64_t i = 0; i < starkInfo.starkStruct.nBits; i++) Goldilocks::square(sn, sn);

        for (uint64_t i=0; i<extend; i++) {
            Goldilocks::inv(zi[i], (sn * w) - Goldilocks::one());
            Goldilocks::mul(w, w, Goldilocks::w(extendBits));
        }

        #pragma omp parallel for
        for (uint64_t i=extend; i<NExtended; i++) {
            zi[i] = zi[i % extend];
        }
    };

    void buildOneRowZerofierInv(uint64_t offset, uint64_t rowIndex)
    {
        Goldilocks::Element root = Goldilocks::one();

        for(uint64_t i = 0; i < rowIndex; ++i) {
            root = root * Goldilocks::w(starkInfo.starkStruct.nBits);
        }

        Goldilocks::Element w = Goldilocks::one();
        Goldilocks::Element sn = Goldilocks::shift();

        for(uint64_t i = 0; i < NExtended; ++i) {
            Goldilocks::Element x = sn * w;
            Goldilocks::inv(zi[i + offset * NExtended], (x - root) * zi[i]);
            w = w * Goldilocks::w(starkInfo.starkStruct.nBitsExt);
        }
    }

    void buildFrameZerofierInv(uint64_t offset, uint64_t offsetMin, uint64_t offsetMax)
    {
        uint64_t nRoots = offsetMin + offsetMax;
        Goldilocks::Element roots[nRoots];

        for(uint64_t i = 0; i < offsetMin; ++i) {
            roots[i] = Goldilocks::one();
            for(uint64_t j = 0; j < i; ++j) {
                roots[i] = roots[i] * Goldilocks::w(starkInfo.starkStruct.nBits);
            }
        }

        for(uint64_t i = 0; i < offsetMax; ++i) {
            roots[i + offsetMin] = Goldilocks::one();
            for(uint64_t j = 0; j < (N - i - 1); ++j) {
                roots[i + offsetMin] = roots[i + offsetMin] * Goldilocks::w(starkInfo.starkStruct.nBits);
            }
        }

        Goldilocks::Element w = Goldilocks::one();
        Goldilocks::Element sn = Goldilocks::shift();

        for(uint64_t i = 0; i < NExtended; ++i) {
            zi[i + offset*NExtended] = Goldilocks::one();
            Goldilocks::Element x = sn * w;
            for(uint64_t j = 0; j < nRoots; ++j) {
                zi[i + offset*NExtended] = zi[i + offset*NExtended] * (x - roots[j]);
            }
            w = w * Goldilocks::w(starkInfo.starkStruct.nBitsExt);
        }
    }

    ~ConstPols()
    {
        free(pConstPolsAddress);
        free(pConstTreeAddress);
        delete zi;
        delete S;
        delete x;
        delete x_n; // Needed for PIL1 compatibility
        delete x_2ns; // Needed for PIL1 compatibility
    }
};

#endif // CONST_POLS_STARKS_HPP

