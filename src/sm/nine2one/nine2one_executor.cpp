#include "config.hpp"
#include "nine2one_executor.hpp"
#include "zkassert.hpp"
#include "utils.hpp"

/*
    This SM inserts 44 bits of 44 different Keccak-f inputs/outputs in the lower bits of a field element.
    The first evaluation is skipped to match the keccak-f gates topology.
    The evaluations usage are: 1 (skipped) + 44 (one per bit) + 1 (accumulated field element) + 44 + 1 + ...

    Pos 0: skipped

    Pos 1: bit j of input i*44 + 0
    ...
    Pos 44: bit j of input i*44 + 43
    Pos 45: field element with 44 bits

    etc.
*/

void Nine2OneExecutor::execute (vector<Nine2OneExecutorInput> &input, Nine2OneCommitPols &pols, vector<vector<Goldilocks::Element>> &required)
{
    /* Check input size (the number of keccaks blocks to process) is not bigger than
       the capacity of the SM (the number of slots that fit into the evaluations multiplied by the number of bits per field element) */
    if (input.size() > nSlots*44)
    {
        cerr << "Error: Nine2OneExecutor::execute() too many entries input.size()=" << input.size() << " > nSlots*44=" << nSlots*44 << endl;
        exitProcess();
    }

    /* Evaluation counter.  The first position is skipped since it will contain the Zero^One gate in Keccak-f SM*/
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

    cout << "Nine2OneExecutor successfully processed " << input.size() << " Keccak hashes (" << (double(input.size())*slotSize*100)/(44*N) << "%)" << endl;
}

Goldilocks::Element Nine2OneExecutor::getBit (vector<Nine2OneExecutorInput> &input, uint64_t block, bool isOut, uint64_t pos)
{
    /* If we run out of input, simply return zero */
    if (block >= input.size())
    {
        return fr.zero();
    }

    /* Return the bit "pos" of the input or output part of the state */
    return bitFromState(isOut ? input[block].st[1] : input[block].st[0], pos);
}

Goldilocks::Element Nine2OneExecutor::bitFromState (uint64_t (&st)[5][5][2], uint64_t i)
{
    uint64_t y  = i/320;
    uint64_t x  = (i%320)/64;
    uint64_t z  = i%64;
    uint64_t z1 = z/32; /* First 32 bits of z are stored in st[x][y][0], second 32 bits of z are stored in st[x][y][0] */
    uint64_t z2 = z%32; /* z2 is the number of the bit: 0...32 */
    return fr.fromU64((st[x][y][z1] >> z2) & 1);
}
