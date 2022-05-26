#include "config.hpp"
#include "nine2one_executor.hpp"

void Nine2OneExecutor::execute (vector<Nine2OneExecutorInput> &input, Nine2OneCommitPols &pols, vector<vector<FieldElement>> &required)
{
    uint64_t N = pols.degree();
    uint64_t nSlots9 = (N-1)/slotSize;

    uint64_t p = 1;
    FieldElement accField9 = fr.zero();

    for (uint64_t i=0; i<nSlots9; i++)
    {
        vector<FieldElement> keccakFSlot;

        for (uint64_t j=0; j<1600; j++)
        {
            for (uint64_t k=0; k<9; k++)
            {
                pols.bit[p] = getBit(input, i*9 + k, false, j);
                pols.field9[p] = accField9;
                if (k == 0)
                {
                    accField9 = pols.bit[p];
                }
                else
                {
                    accField9 += (pols.bit[p] << 7*k);
                }
                p++;
            }
            keccakFSlot.push_back(accField9);
        }

        for (uint64_t j=0; j<1600; j++)
        {
            for (uint64_t k=0; k<9; k++)
            {
                pols.bit[p] = getBit(input, i*9 + k, true, j);
                pols.field9[p] = accField9;
                if (k == 0)
                {
                    accField9 = pols.bit[p];
                }
                else
                {
                    accField9 += (pols.bit[p] << 7*k);
                }
                p++;
            }
        }

        required.push_back(keccakFSlot);

        //pols.bit[p] = fr.zero();
        pols.field9[p] = accField9;
        accField9 = fr.zero();
        p++;

        p += slotSize - (3200*9+1);
    }
    zkassert(p <= N);

    cout << "Nine2OneExecutor successfully processed " << input.size() << " Keccak hashes" << endl;
}

FieldElement Nine2OneExecutor::getBit (vector<Nine2OneExecutorInput> &input, uint64_t block, bool isOut, uint64_t pos)
{
    if (block >= input.size())
    {
        return 0;
    }
    return bitFromState(isOut ? input[block].st[1] : input[block].st[0], pos);
}

FieldElement Nine2OneExecutor::bitFromState (uint64_t (&st)[5][5][2], uint64_t i)
{
    uint64_t y = i/320;
    uint64_t x = (i%320)/64;
    uint64_t z = i%64;
    uint64_t z1 = z/32;
    uint64_t z2 = z%32;
    return FieldElement((st[x][y][z1] >> z2) & 1);
}
