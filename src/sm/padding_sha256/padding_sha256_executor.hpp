#ifndef PADDING_SHA256_EXECUTOR_HPP
#define PADDING_SHA256_EXECUTOR_HPP

#include <vector>
#include <gmpxx.h>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "goldilocks_base_field.hpp"
#include "padding_sha256bit_executor.hpp"
#include "scalar.hpp"
#include "sha256.hpp"

USING_PROVER_FORK_NAMESPACE;

using namespace std;

class PaddingSha256ExecutorInput
{
public:

    //Data Transfer Object, realLean and hast are not included since are overwritten in the prepareInput function
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
    
    PaddingSha256ExecutorInput() : realLen(0), digestCalled(false), lenCalled(false) {};
    
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
    static inline DTO*  toDTO(vector<PaddingSha256ExecutorInput> &input){
        DTO* dto = new DTO[input.size()];
        for (uint64_t i = 0; i < input.size(); i++){
            input[i].toDTO(dto + i);
        }
        return dto;
    }
    static inline void fromDTO(DTO* dto, uint64_t dto_size, vector<PaddingSha256ExecutorInput> &output){
        output.resize(dto_size);
        for (uint64_t i = 0; i < dto_size; i++){
            output[i].fromDTO(dto + i);
        }
    }

};


class PaddingSha256Executor
{
private:

    /* Goldilocks reference */
    Goldilocks &fr;

    /* Constant values */
    const uint64_t blockSize;
    const uint64_t bytesPerBlock;
    const uint64_t bitsPerElement;
    const uint64_t N;

    /* Hash of an empty/zero message */
    mpz_class hashZeroScalar;
    Goldilocks::Element hash0[8];

uint64_t prepareInput (vector<PaddingSha256ExecutorInput> &input);

public:

    /* Constructor */
    PaddingSha256Executor(Goldilocks &fr) :
        fr(fr),
        blockSize(31488),
        bytesPerBlock(64),
        bitsPerElement(7),
        N(PROVER_FORK_NAMESPACE::PaddingSha256CommitPols::pilDegree())
    {
        SHA256(NULL, 0, hashZeroScalar);
        scalar2fea(fr, hashZeroScalar, hash0);
    };

    /* Executor */
    void execute (vector<PaddingSha256ExecutorInput> &input, PROVER_FORK_NAMESPACE::PaddingSha256CommitPols &pols, vector<PaddingSha256BitExecutorInput> &required);
    
    inline void execute (vector<PaddingSha256ExecutorInput> &input, Goldilocks::Element *pAddress, void* pSMRquests){
        PROVER_FORK_NAMESPACE::PaddingSha256CommitPols pols(pAddress, N);
        vector<PaddingSha256BitExecutorInput> required;
        execute(input, pols, required);
        #ifdef __ZKEVM_SM__
            add_padding_sha256_bit_inputs(pSMRquests, (void *)required.data(), (uint64_t) required.size());
        #endif
    }
};


#endif