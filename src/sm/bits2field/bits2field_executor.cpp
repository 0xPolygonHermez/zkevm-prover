#include "config.hpp"
#include "bits2field_executor.hpp"
#include "zkassert.hpp"
#include "utils.hpp"
#include "zklog.hpp"

/*
    This SM inserts 44 bits of 44 different Keccak-f inputs/outputs in the lower bits of a field element.
    The first evaluation is reserved to match the keccak-f gates topology.
    The evaluations usage are: 1 (reserved) + 44 (one per bit0 of 44 inputs) + 44 + ...

    As per PIL spec:
      field44' = (1-FieldLatch)*field44 + bit*Factor;

    Where FieldLatch and Factor are constant polynomials:
      FieldLatch = 0, 43 zeros, 1, 43 zeros, 1, ...
      Factor = 0, 1, 2, 4, ..., 1<<43, 1, 2, 4, ...

    For example, if bit_0 of the first 44 inputs were alterned ones and zeros, and bit_N=bit_N-1:

             Constant pols:             Committed pols:

    Pos 0:   Factor=0     FieldLatch=0  bit=0 field44=0b0 (reserved)
    Pos 1:   Factor=1     FieldLatch=0  bit=1 field44=0b0
    Pos 2:   Factor=2     FieldLatch=0  bit=0 field44=0b1
    Pos 3:   Factor=4     FieldLatch=0  bit=1 field44=0b01
    Pos 4:   Factor=8     FieldLatch=0  bit=0 field44=0b101
    Pos 5:   Factor=16    FieldLatch=0  bit=1 field44=0b0101
    Pos 6:   Factor=32    FieldLatch=0  bit=0 field44=0b10101
    ...
    Pos 44:  Factor=1<<43 FieldLatch=0  bit=0 field44=0b1010101010101010101010101010101010101010101
    Pos 45:  Factor=1     FieldLatch=1  bit=1 field44=0b01010101010101010101010101010101010101010101 (completed field element)
    Pos 46:  Factor=2     FieldLatch=0  bit=0 field44=0b1
    Pos 47:  Factor=4     FieldLatch=0  bit=1 field44=0b01
    Pos 48:  Factor=8     FieldLatch=0  bit=0 field44=0b101
    Pos 49:  Factor=16    FieldLatch=0  bit=1 field44=0b0101
    Pos 50:  Factor=32    FieldLatch=0  bit=0 field44=0b10101
    ...
    Pos 88:  Factor=1<<43 FieldLatch=0  bit=0 field44=0b1010101010101010101010101010101010101010101
    Pos 89:  Factor=1     FieldLatch=1  bit=1 field44=0b01010101010101010101010101010101010101010101 (completed field element)
    Pos 90:  Factor=2     FieldLatch=0  bit=0 field44=0b1
    etc.
*/

inline uint64_t getBitFromState ( const uint8_t (&state)[200], uint64_t i )
{
    return (state[i/8] >> (i%8)) & 1;
}

void Bits2FieldExecutor::execute (vector<Bits2FieldExecutorInput> &input, Bits2FieldCommitPols &pols, vector<vector<Goldilocks::Element>> &required)
{
    /* Check input size (the number of keccaks blocks to process) is not bigger than
       the capacity of the SM (the number of slots that fit into the evaluations multiplied by the number of bits per field element) */
    if (input.size() > nSlots*44)
    {
        zklog.error("Bits2FieldExecutor::execute() too many entries input.size()=" + to_string(input.size()) + " > nSlots*44=" + to_string(nSlots*44));
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
        vector<Goldilocks::Element> keccakFSlot;

        /* For every input bit */
        for (uint64_t j=0; j<1600; j++)
        {
            /* For every field element bit */
            for (uint64_t k=0; k<44; k++)
            {
                /* Get this input bit and store it in pols.bit[] */
                pols.bit[p] = getBit(input, i*44 + k, false, j);

                /* Store the accumulated field in pols.field44[] */
                pols.field44[p] = accField;

                /* Add this bit to accField, left-shifting the rest */
                if (k == 0)
                {
                    accField = pols.bit[p];
                }
                else
                {
                    accField = fr.add( accField, fr.fromU64(fr.toU64(pols.bit[p]) << k) );
                }

                /* Increment the pol evaluation counter */
                p++;
            }

            /* Store the accField in the input vector for the keccak-f SM */
            keccakFSlot.push_back(accField);
        }

        /* For every output bit */
        for (uint64_t j=0; j<1600; j++)
        {
            /* For every field element bit */
            for (uint64_t k=0; k<44; k++)
            {
                /* Get this output bit and store it in pols.bit[] */
                pols.bit[p] = getBit(input, i*44 + k, true, j);

                /* Store the accumulated field in pols.field44[] */
                pols.field44[p] = accField;

                /* Add this bit to accField, left-shifting the rest */
                if (k == 0)
                {
                    accField = pols.bit[p];
                }
                else
                {
                    accField = fr.add( accField, fr.fromU64((fr.toU64(pols.bit[p]) << k)) );
                }

                /* Increment the pol evaluation counter */
                p++;
            }
        }

        /* Add the accumulated field elements as a required input for the keccak-f SM */
        required.push_back(keccakFSlot);

        /* Store the accumulated field into pols.field44[] */
        pols.field44[p] = accField;
        accField = fr.zero();
        p++;

        /* Skip the rest of the gates of the keccak-f SM: slot size minus the consumed evaluations */
        p += slotSize - (3200*44+1);
    }

    /* Sanity check */
    zkassert(p <= N);

    zklog.info("Bits2FieldExecutor successfully processed " + to_string(input.size()) + " Keccak hashes (" + to_string((double(input.size())*slotSize*100)/(44*N)) + "%)");
}

Goldilocks::Element Bits2FieldExecutor::getBit (vector<Bits2FieldExecutorInput> &input, uint64_t block, bool isOutput, uint64_t pos)
{
    /* If we run out of input, simply return zero */
    if (block >= input.size())
    {
        return fr.zero();
    }

    /* Return the bit "pos" of the input or output part of the state */
    if (isOutput)
    {
        return fr.fromU64(getBitFromState(input[block].outputState, pos));
    }
    else
    {
        return fr.fromU64(getBitFromState(input[block].inputState, pos));
    }
}
