#include "padding_kk_executor.hpp"
#include "scalar.hpp"

void PaddingKKExecutor::prepareInput (vector<PaddingKKExecutorInput> &input)
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

void PaddingKKExecutor::execute (vector<PaddingKKExecutorInput> &input, PaddingKKCommitPols &pols, vector<PaddingKKBitExecutorInput> &required)
{
    prepareInput(input);

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

            pols.len[p] = input[i].realLen;
            pols.addr[p] = addr;
            pols.connected[p] = j<bytesPerBlock ? 0 : 1;
            pols.rem[p] = fr.sub(FieldElement(input[i].realLen), FieldElement(j));
            pols.remInv[p] = pols.rem[p] == 0 ? 0 : fr.inv(pols.rem[p]);
            pols.spare[p] = (pols.rem[p] > 0xFFFF) ? 1 : 0;
            pols.firstHash[p] = (j==0) ? 1 : 0;

            if (lastOffset == 0)
            {
                curRead += 1;
                pols.crLen[p] = (curRead<int64_t(input[i].reads.size())) ? input[i].reads[curRead] : 1;
                pols.crOffset[p] = fr.sub(pols.crLen[p], fr.one());
            }
            else
            {
                pols.crLen[p] = pols.crLen[p-1];
                pols.crOffset[p] = fr.sub(pols.crOffset[p-1], fr.one());
            }
            pols.crOffsetInv[p] = (pols.crOffset[p] == 0) ? 0 : fr.inv(pols.crOffset[p]);

            uint64_t crAccI = pols.crOffset[p]/4;
            uint64_t crSh = (pols.crOffset[p]%4)*8;

            for (uint64_t k=0; k<8; k++)
            {
                crF[k][p] = (k==crAccI) ? (1<<crSh) : 0;
                if (pols.crOffset[p] == fr.zero())
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
                PaddingKKBitExecutorInput paddingKKBitExecutorInput;
                for (uint64_t k=0; k<bytesPerBlock; k++)
                {
                    paddingKKBitExecutorInput.r[k] = input[i].dataBytes[j - bytesPerBlock + 1 + k];
                }
                paddingKKBitExecutorInput.connected = (j < bytesPerBlock) ? false : true;
                required.push_back(paddingKKBitExecutorInput);

                if (j == input[i].dataBytes.size() - 1)
                {
                    scalar2fea(fr, input[i].hash,
                        pols.hash0[p], 
                        pols.hash1[p], 
                        pols.hash2[p], 
                        pols.hash3[p], 
                        pols.hash4[p], 
                        pols.hash5[p], 
                        pols.hash6[p], 
                        pols.hash7[p]);

                    for (uint64_t k=1; k<input[i].dataBytes.size(); k++)
                    {
                        pols.hash0[p-k] = pols.hash0[p];
                        pols.hash1[p-k] = pols.hash1[p];
                        pols.hash2[p-k] = pols.hash2[p];
                        pols.hash3[p-k] = pols.hash3[p];
                        pols.hash4[p-k] = pols.hash4[p];
                        pols.hash5[p-k] = pols.hash5[p];
                        pols.hash6[p-k] = pols.hash6[p];
                        pols.hash7[p-k] = pols.hash7[p];
                    }
                }

            }

            p += 1;
        }
        addr += 1;
    }

    uint64_t nTotalBlocks = 9*(N/blockSize);
    uint64_t nUsedBlocks = p/bytesPerBlock;

    if (nUsedBlocks > nTotalBlocks)
    {
        cerr << "Error: PaddingKKExecutor::execute() Too many keccak blocks nUsedBlocks=" << nUsedBlocks << " > nTotalBlocks=" << nTotalBlocks << endl;
        exit(-1);
    }

    uint64_t nFullUnused = nTotalBlocks - nUsedBlocks;

    uint8_t bytes0[bytesPerBlock];
    for (uint64_t i=0; i<bytesPerBlock; i++)
    {
        bytes0[i] = (i==0) ? 1 : (  (i==bytesPerBlock-1) ? 0x80 : 0);
    }
    string hashZeroInput = "";
    string hashZero = keccak256(hashZeroInput);
    mpz_class hashZeroScalar(hashZero);
    FieldElement hash0[8];
    scalar2fea(fr, hashZeroScalar, hash0);

    for (uint64_t i=0; i<nFullUnused; i++)
    {
        for (uint64_t j=0; j<bytesPerBlock; j++)
        {
            pols.freeIn[p] = j==0 ? fr.one() : ((j == bytesPerBlock-1) ? FieldElement(0x80) : fr.zero());

            pols.len[p] = fr.zero();
            pols.addr[p] = addr;
            pols.rem[p] = (j==0) ? 0 : fr.neg(FieldElement(j));
            pols.remInv[p] = (pols.rem[p] == 0) ? 0 : fr.inv(pols.rem[p]);
            pols.spare[p] = (pols.rem[p] > 0xFFFF) ? 1 : 0;
            pols.firstHash[p] = (j==0) ? 1 : 0;
            pols.connected[p] = 0;

            pols.crLen[p] =  fr.one();
            pols.crOffset[p] = fr.zero();

            pols.crOffsetInv[p] = (pols.crOffset[p] == fr.zero()) ? fr.zero() : fr.inv(pols.crOffset[p]);

            for (uint64_t k=0; k<8; k++)
            {
                crF[k][p] = (k==0) ? 1 : 0;
                crV[k][p+1] = 0;
            }

            if (j % bytesPerBlock == (bytesPerBlock -1) )
            {
                PaddingKKBitExecutorInput paddingKKBitExecutorInput;
                for (uint64_t k=0; k<bytesPerBlock; k++)
                {
                    paddingKKBitExecutorInput.r[k] = bytes0[k];
                }
                paddingKKBitExecutorInput.connected = false;
                required.push_back(paddingKKBitExecutorInput);
        
                pols.hash0[p] = hash0[0];
                pols.hash1[p] = hash0[1];
                pols.hash2[p] = hash0[2];
                pols.hash3[p] = hash0[3];
                pols.hash4[p] = hash0[4];
                pols.hash5[p] = hash0[5];
                pols.hash6[p] = hash0[6];
                pols.hash7[p] = hash0[7];
                
                for (uint64_t k=1; k<bytesPerBlock; k++)
                {
                    pols.hash0[p-k] = pols.hash0[p];
                    pols.hash1[p-k] = pols.hash1[p];
                    pols.hash2[p-k] = pols.hash2[p];
                    pols.hash3[p-k] = pols.hash3[p];
                    pols.hash4[p-k] = pols.hash4[p];
                    pols.hash5[p-k] = pols.hash5[p];
                    pols.hash6[p-k] = pols.hash6[p];
                    pols.hash7[p-k] = pols.hash7[p];
                }
            }

            p += 1;
        }
        addr += 1;
    }

    uint64_t fp = p;
    while (p<N)
    {
        pols.freeIn[p] = fr.zero();

        pols.len[p] = fr.zero();
        pols.addr[p] = addr;
        pols.connected[p] = 0;

        pols.rem[p] = (p==fp) ? 0 : fr.sub(pols.rem[p-1], fr.one()) ;
        pols.remInv[p] = (pols.rem[p] == 0) ? 0 : fr.inv(pols.rem[p]);
        pols.spare[p] =  (p==fp) ? 0 : 1;
        pols.firstHash[p] = (p==fp) ? 1 : 0;

        pols.crLen[p] =  fr.one();
        pols.crOffset[p] = fr.zero();

        pols.crOffsetInv[p] = (pols.crOffset[p] == 0) ? 0 : fr.inv(pols.crOffset[p]);

        for (uint64_t k=0; k<8; k++)
        {
            crF[k][p] = (k==0) ? 1 : 0;
            crV[k][(p+1)%N] = 0;
        }

        pols.hash0[p] = fr.zero();
        pols.hash1[p] = fr.zero();
        pols.hash2[p] = fr.zero();
        pols.hash3[p] = fr.zero();
        pols.hash4[p] = fr.zero();
        pols.hash5[p] = fr.zero();
        pols.hash6[p] = fr.zero();
        pols.hash7[p] = fr.zero();

        p += 1;
    }

    cout << "PaddingKKExecutor successfully processed " << input.size() << " Keccak hashes" << endl;
}