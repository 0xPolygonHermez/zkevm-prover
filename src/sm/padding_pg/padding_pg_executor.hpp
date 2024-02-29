#ifndef PADDING_PG_EXECUTOR_HPP
#define PADDING_PG_EXECUTOR_HPP

#include <vector>
#include <array>
#include <gmpxx.h>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"

USING_PROVER_FORK_NAMESPACE;

using namespace std;

class PaddingPGExecutorInput
{
public:

    typedef struct{
        char * data;
        uint8_t * dataBytes;
        uint64_t dataBytes_size;
        uint64_t * reads;
        uint64_t reads_size;
        bool digestCalled;
        bool lenCalled; 
    } DTO;
    string data;
    vector<uint8_t> dataBytes;
    uint64_t realLen;
    vector<uint64_t> reads;
    mpz_class hash;
    bool digestCalled;
    bool lenCalled;
    PaddingPGExecutorInput() : realLen(0), digestCalled(false), lenCalled(false) {};

    inline void toDTO(DTO* dto){
        dto->data = (char*)data.c_str();
        dto->dataBytes = dataBytes.data();
        dto->dataBytes_size = dataBytes.size();
        dto->reads = reads.data();
        dto->reads_size = reads.size();
        dto->digestCalled = digestCalled;
        dto->lenCalled = lenCalled;
    }
    inline void fromDTO(DTO* dto){
        data = string(dto->data);
        dataBytes.clear();
        if(dto->dataBytes_size > 0)
            dataBytes.assign(dto->dataBytes, dto->dataBytes + dto->dataBytes_size);
        reads.clear();
        if(dto->reads_size > 0)
            reads.assign(dto->reads, dto->reads + dto->reads_size);
        digestCalled = dto->digestCalled;
        lenCalled = dto->lenCalled;
    }
    static inline DTO*  toDTO(vector<PaddingPGExecutorInput> &input){
        DTO* dto = new DTO[input.size()];
        for (uint64_t i = 0; i < input.size(); i++){
            input[i].toDTO(dto + i);
        }
        return dto;
    }
    static inline void fromDTO(DTO* dto, uint64_t dto_size, vector<PaddingPGExecutorInput> &output){
        output.resize(dto_size);
        for (uint64_t i = 0; i < dto_size; i++){
            output[i].fromDTO(dto + i);
        }
    }
};

class PaddingPGExecutor
{
private:
    Goldilocks &fr;
    PoseidonGoldilocks &poseidon;
    const uint64_t bytesPerElement;
    const uint64_t nElements;
    const uint64_t bytesPerBlock;
    const uint64_t N;

uint64_t prepareInput (vector<PaddingPGExecutorInput> &input);

public:
    PaddingPGExecutor(Goldilocks &fr, PoseidonGoldilocks &poseidon) :
        fr(fr),
        poseidon(poseidon),
        bytesPerElement(7),
        nElements(8),
        bytesPerBlock(bytesPerElement*nElements),
        N(PROVER_FORK_NAMESPACE::PaddingPGCommitPols::pilDegree()) {};
    void execute (vector<PaddingPGExecutorInput> &input, PROVER_FORK_NAMESPACE::PaddingPGCommitPols &pols, vector<array<Goldilocks::Element, 17>> &required);
    inline void execute (vector<PaddingPGExecutorInput> &input, Goldilocks::Element *pAddress, void *pRequests){
        PROVER_FORK_NAMESPACE::PaddingPGCommitPols pols(pAddress, N);
        vector<array<Goldilocks::Element, 17>> required;
        execute(input, pols, required);
#ifdef __ZKEVM_SM__
        //add_padding_kk_bit_inputs(pRequests, (void *)required.data(), (uint64_t) required.size());
#endif
    }
};


#endif