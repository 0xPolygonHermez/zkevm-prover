#include "config.hpp"
#include "bits2field_sha256_executor.hpp"
#include "zkassert.hpp"
#include "utils.hpp"
#include "zklog.hpp"


void Bits2FieldSha256Executor::execute (vector<Bits2FieldSha256ExecutorInput> &input, Bits2FieldSha256CommitPols &pols, vector<Sha256FExecutorInput> &required)
{
    /* Check input size (the number of keccaks blocks to process) is not bigger than
       the capacity of the SM (the number of slots that fit into the evaluations multiplied by the number of bits per field element) */
    if (input.size() > nSlots*bitsPerElement)
    {
        zklog.error("Bits2FieldSha256Executor::execute() too many entries input.size()=" + to_string(input.size()) + " > nSlots*bitsPerElement=" + to_string(nSlots*bitsPerElement));
        exitProcess();
    }

    /* Evaluation counter
       The first position 0 is reserved since it will contain the Zero^One gate in Keccak-f SM, so we start at position 1 */
    uint64_t p = 1;

    /* Accumulator field element */
    Goldilocks::Element accField = fr.zero();

    /* For every slot i */
    for (uint64_t i=0; i<nSlots; i++)
    {
        Sha256FExecutorInput req;
        vector<Goldilocks::Element> stOut;

        for(uint64_t j=0; j<1024; j++){
            for(uint64_t k=0; k<bitsPerElement; k++){
                if(j<256){
                    pols.bit[p] = getBit(input, i*bitsPerElement + k, BitType::STATE_IN, j);
                } else if (j<512){
                    pols.bit[p] = getBit(input, i*bitsPerElement + k, BitType::STATE_OUT, j-256);
                } else{
                    pols.bit[p] = getBit(input, i*bitsPerElement + k, BitType::BLOCK_IN, j-512);
                }
                if (k==0){
                    accField = pols.bit[p];
                } else{
                    accField = accField + fr.fromU64(fr.toU64(pols.bit[p]) << k);
                }
                pols.packField[p] = accField;
                p+=1;
            }
            if(j<256){
                req.stIn.push_back(accField);
            } else if (j<512){
                stOut.push_back(accField);
            } else{
                req.rIn.push_back(accField);
            }
        }
        for (uint64_t j = 1024 * bitsPerElement; j < slotSize; j++) {
            p += 1;
        }

        /* Add the rquest to the Sha256-f SM*/
        required.push_back(req);
       
    }

    /* Sanity check */
    zkassert(p <= N);

    zklog.info("Bits2FieldSha256Executor successfully processed " + to_string(input.size()) + " Sha256 hashes (" + to_string((double(input.size())*slotSize*100)/(bitsPerElement*N)) + "%)");
}

inline uint64_t getStateBit ( const uint32_t (&state)[8], uint64_t i )
{
    uint64_t sh = 31 - (i%32);
    return (state[i/32] >> sh) & 1;
}

Goldilocks::Element Bits2FieldSha256Executor::getBit (vector<Bits2FieldSha256ExecutorInput> &input, uint64_t block, BitType type, uint64_t pos)
{
    /* If we run out of input, simply return zero */
    if (block >= input.size())
    {
        return fr.zero();
    }

    /* Return the bit "pos" of the input or output part of the state */
    if (type == BitType::STATE_IN)
    {
        return fr.fromU64(getStateBit(input[block].inputState, pos));
    }
    else if( type == BitType::STATE_OUT)
    {
        return fr.fromU64(getStateBit(input[block].outputState, pos));
    } else if( type == BitType::BLOCK_IN) {
        uint64_t byte = pos/8;
        uint64_t sh = 7 - (pos%8);
        return fr.fromU64((input[block].inBlock[byte] >> sh) & 1);
    } else {
        zklog.error("Bits2FieldSha256Executor::getBit() found invalid type value: " + to_string(type));
        exitProcess();
    }
    return fr.zero();
}
