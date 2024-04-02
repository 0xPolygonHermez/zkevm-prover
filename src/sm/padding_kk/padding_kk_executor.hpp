#ifndef PADDING_KK_EXECUTOR_HPP
#define PADDING_KK_EXECUTOR_HPP

#include <vector>
#include <gmpxx.h>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "goldilocks_base_field.hpp"
#include "sm/padding_kkbit/padding_kkbit_executor.hpp"
#include "scalar.hpp"

USING_PROVER_FORK_NAMESPACE;

using namespace std;

class PaddingKKExecutorInput
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
    PaddingKKExecutorInput() : realLen(0), digestCalled(false), lenCalled(false) {};

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
    static inline DTO*  toDTO(vector<PaddingKKExecutorInput> &input){
        DTO* dto = new DTO[input.size()];
        for (uint64_t i = 0; i < input.size(); i++){
            input[i].toDTO(dto + i);
        }
        return dto;
    }
    static inline void fromDTO(DTO* dto, uint64_t dto_size, vector<PaddingKKExecutorInput> &output){
        output.resize(dto_size);
        for (uint64_t i = 0; i < dto_size; i++){
            output[i].fromDTO(dto + i);
        }
    }
};

class PaddingKKExecutor
{
private:

    /* Goldilocks reference */
    Goldilocks &fr;

    /* Constant values */
    const uint64_t blockSize;
    const uint64_t bytesPerBlock;
    const uint64_t N;

    /* Hash of an empty/zero message */
    mpz_class hashZeroScalar;
    Goldilocks::Element hash0[8];

uint64_t prepareInput (vector<PaddingKKExecutorInput> &input);

public:

    /* Constructor */
    PaddingKKExecutor(Goldilocks &fr) :
        fr(fr),
        blockSize(155286),
        bytesPerBlock(136),
        N(PROVER_FORK_NAMESPACE::PaddingKKCommitPols::pilDegree())
    {
        keccak256(NULL, 0, hashZeroScalar);
        scalar2fea(fr, hashZeroScalar, hash0);
    };

    /* Executor */
    void execute (vector<PaddingKKExecutorInput> &input, PROVER_FORK_NAMESPACE::PaddingKKCommitPols &pols, vector<PaddingKKBitExecutorInput> &required);
};


#endif