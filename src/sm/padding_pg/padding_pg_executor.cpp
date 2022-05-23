#include "padding_pg_executor.hpp"
#include "scalar.hpp"

void PaddingPGExecutor::prepareInput (vector<PaddingPGExecutorInput> &input)
{

    for (uint64_t i=0; i<input.size(); i++)
    {
        if (input[i].data.length() > 0)
        {
            for (uint64_t c=0; c<input[i].data.length(); c+=2)
            {
                uint8_t aux;
                aux = 16*char2byte(input[i].data[c]) + char2byte(input[i].data[c+1]);
                input[i].dataBytes.push_back(aux);
            }
        }

        string hashString = keccak256(input[i].dataBytes);
        input[i].hash.set_str(Remove0xIfPresent(hashString), 16);

        input[i].realLen = input[i].dataBytes.size();

        input[i].dataBytes.push_back(0x1);


        while (input[i].dataBytes.size() % bytesPerBlock) input[i].dataBytes.push_back(0);

        input[i].dataBytes[ input[i].dataBytes.size() - 1] |= 0x80;
    }
}

void PaddingPGExecutor::execute (vector<PaddingPGExecutorInput> &input, PaddingPGCommitPols &pols/*, vector<PaddingPGBitExecutorInput> &required*/)
{
    prepareInput(input);

    uint64_t N = pols.degree();

    uint64_t p = 0;

    uint64_t addr = 0;

    FieldElement *crF[8];
    crF[0] = pols.crF0;
    crF[1] = pols.crF1;
    crF[2] = pols.crF2;
    crF[3] = pols.crF3;
    crF[4] = pols.crF4;
    crF[5] = pols.crF5;
    crF[6] = pols.crF6;
    crF[7] = pols.crF7;

    FieldElement *crV[8];
    crV[0] = pols.crV0;
    crV[1] = pols.crV1;
    crV[2] = pols.crV2;
    crV[3] = pols.crV3;
    crV[4] = pols.crV4;
    crV[5] = pols.crV5;
    crV[6] = pols.crV6;
    crV[7] = pols.crV7;

    /*for (uint64_t k=0; k<8; k++)
    {
        crV[k][p] = 0;
    }*/

    for (uint64_t i=0; i<input.size(); i++)
    {

        int64_t curRead = -1;
        uint64_t lastOffset = 0;

        for (uint64_t j=0; j<input[i].dataBytes.size(); j++)
        {

            pols.freeIn[p] = input[i].dataBytes[j];
            
            uint64_t acci = (j % bytesPerBlock) / bytesPerElement;
            uint64_t sh = (j % bytesPerElement)*8;
            for (uint64_t k=0; k<nElements; k++)
            {
                if (k == acci) {
                    pols.acc[k][p+1] = pols.acc[k][p] | (pols.freeIn[p] << sh);
                } else {
                    pols.acc[k][p+1] = pols.acc[k][p];
                }
            }

            pols.prevHash0[p+1] = pols.prevHash0[p];
            pols.prevHash1[p+1] = pols.prevHash1[p];
            pols.prevHash2[p+1] = pols.prevHash2[p];
            pols.prevHash3[p+1] = pols.prevHash3[p];

            pols.len[p] = input[i].realLen;
            pols.addr[p] = addr;
            pols.rem[p] = FieldElement(input[i].realLen - j);
            pols.remInv[p] = pols.rem[p] == 0 ? 0 : fr.inv(pols.rem[p]);
            pols.spare[p] = (pols.rem[p] > 0xFFFF) ? 1 : 0;
            pols.firstHash[p] = (j==0) ? 1 : 0;

            if (lastOffset == 0)
            {
                curRead += 1;
                pols.crLen[p] = (curRead<int64_t(input[i].reads.size())) ? input[i].reads[curRead] : 1;
                pols.crOffset[p] = pols.crLen[p] - 1;
            }
            else
            {
                pols.crLen[p] = pols.crLen[p-1];
                pols.crOffset[p] = pols.crOffset[p-1] - 1;
            }
            pols.crOffsetInv[p] = (pols.crOffset[p] == 0) ? 0 : fr.inv(pols.crOffset[p]);

            uint64_t crAccI = pols.crOffset[p]/4;
            uint64_t crSh = (pols.crOffset[p]%4)*8;

            for (uint64_t k=0; k<8; k++)
            {
                crF[k][p] = (k==crAccI) ? (1<<crSh) : 0;
                if (pols.crOffset[p] == 0)
                {
                    crV[k][p+1] = 0;
                }
                else
                {
                    crV[k][p+1] = (k==crAccI) ? (crV[k][p] + (pols.freeIn[p]<<crSh)) : crV[k][p];
                }
            }

            lastOffset = pols.crOffset[p];

            if ( (j % bytesPerBlock) == (bytesPerBlock -1) )
            {
                uint64_t data[12];
                data[0] = pols.acc[0][p+1];
                data[1] = pols.acc[1][p+1];
                data[2] = pols.acc[2][p+1];
                data[3] = pols.acc[3][p+1];
                data[4] = pols.acc[4][p+1];
                data[5] = pols.acc[5][p+1];
                data[6] = pols.acc[6][p+1];
                data[7] = pols.acc[7][p+1];
                data[8] = pols.prevHash0[p];
                data[9] = pols.prevHash1[p];
                data[10] = pols.prevHash2[p];
                data[11] = pols.prevHash3[p];

                poseidon.hash(data);
                
                pols.curHash0[p] = data[0]; 
                pols.curHash1[p] = data[1];
                pols.curHash2[p] = data[2];
                pols.curHash3[p] = data[3];

                /* TODO: required.PoseidonG.push([
                    pols.acc[0][p+1],
                    pols.acc[1][p+1],
                    pols.acc[2][p+1],
                    pols.acc[3][p+1],
                    pols.acc[4][p+1],
                    pols.acc[5][p+1],
                    pols.acc[6][p+1],
                    pols.acc[7][p+1],
                    pols.prevHash0[p],
                    pols.prevHash1[p],
                    pols.prevHash2[p],
                    pols.prevHash3[p],
                    pols.curHash0[p], 
                    pols.curHash1[p], 
                    pols.curHash2[p], 
                    pols.curHash3[p]
                ]);*/

                pols.acc[0][p+1] = 0;
                pols.acc[1][p+1] = 0;
                pols.acc[2][p+1] = 0;
                pols.acc[3][p+1] = 0;
                pols.acc[4][p+1] = 0;
                pols.acc[5][p+1] = 0;
                pols.acc[6][p+1] = 0;
                pols.acc[7][p+1] = 0;

                for (uint64_t k=1; k<bytesPerBlock; k++)
                {
                    pols.curHash0[p-k] = pols.curHash0[p];
                    pols.curHash1[p-k] = pols.curHash1[p];
                    pols.curHash2[p-k] = pols.curHash2[p];
                    pols.curHash3[p-k] = pols.curHash3[p];
                }
                pols.prevHash0[p+1] = pols.curHash0[p];
                pols.prevHash1[p+1] = pols.curHash1[p];
                pols.prevHash2[p+1] = pols.curHash2[p];
                pols.prevHash3[p+1] = pols.curHash3[p];

                if (j == input[i].dataBytes.size() - 1)
                {
                    pols.prevHash0[p+1] = 0;
                    pols.prevHash1[p+1] = 0;
                    pols.prevHash2[p+1] = 0;
                    pols.prevHash3[p+1] = 0;
                }

            }

            p += 1;
        }
        addr += 1;
    }

    uint64_t nFullUnused = ((N - p - 1)/bytesPerBlock)+1;

    uint64_t data[12];
    memset(data, 0, sizeof(data));
    data[0] = 0x1;
    data[7] = (uint64_t(0x80) << 48);

    poseidon.hash(data);

    uint64_t h0[4];
    h0[0] = data[0];
    h0[1] = data[1];
    h0[2] = data[2];
    h0[3] = data[3];

    // TODO: required.PoseidonG.push([ 0x1n, 0n, 0n, 0n, 0n, 0n, 0n, 0x80n << 48n, 0n, 0n, 0n, 0n, ...h0  ]);

    for (uint64_t i=0; i<nFullUnused; i++)
    {
        uint64_t bytesBlock = ((N-p) > bytesPerBlock) ? bytesPerBlock : (N-p);
        if (bytesBlock < 2)
        {
            cerr << "Error: PaddingPGExecutor::execute() Alignment is not possible" << endl;
        }
        for (uint64_t j=0; j<bytesBlock; j++)
        {
            if (j==0)
            {
                pols.freeIn[p] = 1;
            }
            else if (j==(bytesBlock-1))
            {
                pols.freeIn[p] = 0x80;
            }
            else
            {
                pols.freeIn[p] = 0;
            }
            pols.acc[0][p] = (j==0) ? 0 : 0x1;
            pols.acc[1][p] = 0;
            pols.acc[2][p] = 0;
            pols.acc[3][p] = 0;
            pols.acc[4][p] = 0;
            pols.acc[5][p] = 0;
            pols.acc[6][p] = 0;
            pols.acc[7][p] = 0;
            pols.len[p] = 0;
            pols.addr[p] = addr;
            pols.rem[p] = (j==0) ? fr.zero() : fr.inv(FieldElement(j)); // = -j
            pols.remInv[p] = (pols.rem[p]==0) ? 0 : fr.inv(pols.rem[p]);
            pols.spare[p] = j>0 ? 1 : 0;
            pols.firstHash[p] = (j==0) ? 1 : 0;
            pols.prevHash0[p] = 0;
            pols.prevHash1[p] = 0;
            pols.prevHash2[p] = 0;
            pols.prevHash3[p] = 0;
            pols.curHash0[p] = h0[0];
            pols.curHash1[p] = h0[1];
            pols.curHash2[p] = h0[2];
            pols.curHash3[p] = h0[3];

            pols.crOffset[p] = 0;
            pols.crLen[p] = 1;
            pols.crOffsetInv[p] = 0;
            pols.crF0[p] = 1;
            pols.crF1[p] = 0;
            pols.crF1[p] = 0;
            pols.crF2[p] = 0;
            pols.crF3[p] = 0;
            pols.crF4[p] = 0;
            pols.crF5[p] = 0;
            pols.crF6[p] = 0;
            pols.crF7[p] = 0;

            pols.crV0[p] = 0;
            pols.crV1[p] = 0;
            pols.crV2[p] = 0;
            pols.crV3[p] = 0;
            pols.crV4[p] = 0;
            pols.crV5[p] = 0;
            pols.crV6[p] = 0;
            pols.crV7[p] = 0;

            p += 1;
        }
        addr += 1;
    }

    cout << "PaddingPGExecutor successfully processed " << input.size() << " Keccak hashes" << endl;
}