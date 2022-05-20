#include "norm_gate9_executor.hpp"

using namespace std;

void NormGate9Executor::execute (vector<NormGate9ExecutorInput> &input, NormGate9CommitPols & pols)
{
    uint64_t degree = pols.degree();
    uint64_t nBlocks = degree/3;
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
            pols.freeA[p] = (input[i].a  >> (uint64_t(21) * j)) & 0b111111111111111111111;
            pols.freeB[p] = (input[i].b  >> (uint64_t(21) * j)) & 0b111111111111111111111;

            pols.freeANorm[p] = pols.freeA[p] & 0b000000100000010000001;
            pols.freeBNorm[p] = pols.freeB[p] & 0b000000100000010000001;

            /* TODO: Is input[i][0] a string?*/
            if (input[i].type == 0) // XORN
            {
                //pols.gateType[p] = 0;
                pols.freeCNorm[p] = pols.freeANorm[p] ^ pols.freeBNorm[p];
            }
            else if (input[i].type == 1) // "ANDP"
            {
                pols.gateType[p] = 1;
                pols.freeCNorm[p] = (pols.freeANorm[p] ^ 0b000000100000010000001) & pols.freeBNorm[p];
            }
            else
            {
                cerr << "NormGate9Executor::execute() Invalid door type=" << input[i].type << endl;
                exit(-1);
            }

            pols.a[p] = acca;
            pols.b[p] = accb;
            pols.c[p] = accc;

            acca = acca + (pols.freeA[p] << (21*j));
            accb = accb + (pols.freeB[p] << (21*j));
            accc = accc + (pols.freeCNorm[p] << (21*j));

            if (j==2)
            {
                acca = 0;
                accb = 0;
                accc = 0;
            }

            p++;
        }
    }

    while (p<degree) {

        pols.freeA[p] = 0;
        pols.freeB[p] = 0;
        pols.freeANorm[p] = 0;
        pols.freeBNorm[p] = 0;
        pols.freeCNorm[p] = 0;
        pols.gateType[p] = 0;

        pols.a[p] = acca;
        pols.b[p] = accb;
        pols.c[p] = accc;

        acca = 0;
        accb = 0;
        accc = 0;

        p+=1;
    }

    pols.a[0] = acca;
    pols.b[0] = accb;
    pols.c[0] = accc;

    cout << "NormGate9Executor successfully processed " << input.size() << " NormGate9 actions" << endl;
}