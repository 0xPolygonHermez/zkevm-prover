#include <iostream>
#include "norm_gate9_executor.hpp"

using namespace std;

void NormGate9Executor::execute (vector<NormGate9ExecutorInput> &input, NormGate9CommitPols & pols)
{
    if (input.size() > nBlocks)
    {
        cerr << "Error: NormGate9 not big enought for this input" << endl;
        exit(-1);
    }

    uint64_t acca = 0;
    uint64_t accb = 0;
    uint64_t accc = 0;
    uint64_t p = 0;

    for (uint64_t i=0; i<input.size(); i++)
    {
        for (uint64_t j=0; j<3; j++)
        {
            pols.freeA[p] = fr.fromU64( (fr.toU64(input[i].a)  >> (uint64_t(21) * j)) & 0b111111111111111111111 );
            pols.freeB[p] = fr.fromU64( (fr.toU64(input[i].b)  >> (uint64_t(21) * j)) & 0b111111111111111111111 );

            pols.freeANorm[p] = fr.fromU64( fr.toU64(pols.freeA[p]) & 0b000000100000010000001 );
            pols.freeBNorm[p] = fr.fromU64( fr.toU64(pols.freeB[p]) & 0b000000100000010000001 );

            if (fr.isZero(input[i].type)) // XORN
            {
                //pols.gateType[p] = 0;
                pols.freeCNorm[p] = fr.fromU64( fr.toU64(pols.freeANorm[p]) ^ fr.toU64(pols.freeBNorm[p]) );
            }
            else if (fr.isOne(input[i].type)) // "ANDP"
            {
                pols.gateType[p] = fr.one();
                pols.freeCNorm[p] = fr.fromU64( (fr.toU64(pols.freeANorm[p]) ^ 0b000000100000010000001) & fr.toU64(pols.freeBNorm[p]) );
            }
            else
            {
                cerr << "NormGate9Executor::execute() Invalid door type=" << fr.toU64(input[i].type) << endl;
                exit(-1);
            }

            pols.a[p] = fr.fromU64(acca);
            pols.b[p] = fr.fromU64(accb);
            pols.c[p] = fr.fromU64(accc);

            acca = acca + (fr.toU64(pols.freeA[p]) << (21*j));
            accb = accb + (fr.toU64(pols.freeB[p]) << (21*j));
            accc = accc + (fr.toU64(pols.freeCNorm[p]) << (21*j));

            if (j==2)
            {
                acca = 0;
                accb = 0;
                accc = 0;
            }

            p++;
        }
    }

    if (p < N)
    {
        pols.a[p] = fr.fromU64(acca);
        pols.b[p] = fr.fromU64(accb);
        pols.c[p] = fr.fromU64(accc);

        acca = 0;
        accb = 0;
        accc = 0;        
    }

    pols.a[0] = fr.fromU64(acca);
    pols.b[0] = fr.fromU64(accb);
    pols.c[0] = fr.fromU64(accc);

    cout << "NormGate9Executor successfully processed " << input.size() << " NormGate9 actions" << endl;
}