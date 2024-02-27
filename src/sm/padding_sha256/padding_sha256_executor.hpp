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
    string data;
    vector<uint8_t> dataBytes;
    uint64_t realLen;
    vector<uint64_t> reads;
    mpz_class hash;
    bool digestCalled;
    bool lenCalled;
    PaddingSha256ExecutorInput() : realLen(0), digestCalled(false), lenCalled(false) {};

    uint32_t serialize_size(){
        uint32_t sum =0;
        sum += sizeof(uint32_t); //data.size  
        sum += data.size();
        sum += sizeof(uint32_t); //reads.size
        sum += reads.size()*sizeof(uint64_t);
        sum += sizeof(digestCalled);
        sum += sizeof(lenCalled);
    }
    uint32_t serialize(uint8_t *buffer){
        uint32_t offset = 0;
        uint32_t size = data.size();
        memcpy(buffer+offset, &size, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(buffer+offset, data.c_str(), size);
        offset += size;
        size = reads.size();
        memcpy(buffer+offset, &size, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(buffer+offset, reads.data(), size*sizeof(uint64_t));
        offset += size*sizeof(uint64_t);
        memcpy(buffer+offset, &digestCalled, sizeof(digestCalled));
        offset += sizeof(digestCalled);
        memcpy(buffer+offset, &lenCalled, sizeof(lenCalled));
        offset += sizeof(lenCalled);
        return offset;
    }
    uint32_t deserialize(uint8_t *buffer){
        uint32_t offset = 0;
        uint32_t size;
        memcpy(&size, buffer+offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        data = string((char*)buffer+offset, size);
        offset += size;
        memcpy(&size, buffer+offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        reads.resize(size);
        memcpy(reads.data(), buffer+offset, size*sizeof(uint64_t));
        offset += size*sizeof(uint64_t);
        memcpy(&digestCalled, buffer+offset, sizeof(digestCalled));
        offset += sizeof(digestCalled);
        memcpy(&lenCalled, buffer+offset, sizeof(lenCalled));
        offset += sizeof(lenCalled);
        return offset;
    }
    
    static void serialize(vector<PaddingSha256ExecutorInput> &input, std::vector<uint8_t> &buffer){
        uint32_t sum=sizeof(uint32_t);
        for(uint32_t i=0; i<input.size(); i++){
            sum += input[i].serialize_size();
        }
        uint8_t *buffer_ = new uint8_t[sum];
        uint32_t size = input.size();
        memcpy(buffer_, &size, sizeof(uint32_t));
        uint32_t offset = sizeof(uint32_t);
        for(uint32_t i=0; i<input.size(); i++){
            offset += input[i].serialize(buffer_+offset);
        }
        buffer.resize(sum);
        memcpy(buffer.data(), buffer_, sum);
        delete[] buffer_;
    }
    static void deserialize(uint8_t *buffer, vector<PaddingSha256ExecutorInput> &input){
        uint32_t offset = 0;
        uint32_t size;
        memcpy(&size,buffer, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        input.resize(size);
        for(uint32_t i=0; i<size; i++){
            offset += input[i].deserialize(buffer+offset);
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
    
    inline void execute (vector<PaddingSha256ExecutorInput> &input, Goldilocks::Element *pAddress){
        PROVER_FORK_NAMESPACE::PaddingSha256CommitPols pols(pAddress, N);
        vector<PaddingSha256BitExecutorInput> required;
        execute(input, pols, required);
    }
};


#endif