#include <iostream>
#include "padding_kk_executor.hpp"
#include "scalar.hpp"

using namespace std;

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

    CommitGeneratedPol crF[8];
    crF[0] = pols.crF0;
    crF[1] = pols.crF1;
    crF[2] = pols.crF2;
    crF[3] = pols.crF3;
    crF[4] = pols.crF4;
    crF[5] = pols.crF5;
    crF[6] = pols.crF6;
    crF[7] = pols.crF7;

    CommitGeneratedPol crV[8];
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

            pols.freeIn[p] = fr.fromU64(input[i].dataBytes[j]);

            pols.len[p] = fr.fromU64(input[i].realLen);
            pols.addr[p] = fr.fromU64(addr);
            if (j >= bytesPerBlock) pols.connected[p] = fr.one();
            pols.rem[p] = fr.sub(fr.fromU64(input[i].realLen), fr.fromU64(j));
            if (!fr.isZero(pols.rem[p]))
            {
                pols.remInv[p] = fr.inv(pols.rem[p]);
                if (fr.toU64(pols.rem[p]) > 0xFFFF)
                {
                    pols.spare[p] = fr.one();
                }
            }
            
            if (j == 0) pols.firstHash[p] = fr.one();
            pols.incCounter[p] = fr.fromU64((j / bytesPerBlock) +1);

            if (lastOffset == 0)
            {
                curRead += 1;
                pols.crLen[p] = (curRead<int64_t(input[i].reads.size())) ? fr.fromU64(input[i].reads[curRead]) : fr.one();
                pols.crOffset[p] = fr.sub(pols.crLen[p], fr.one());
            }
            else
            {
                pols.crLen[p] = pols.crLen[p-1];
                pols.crOffset[p] = fr.sub(pols.crOffset[p-1], fr.one());
            }
            if (!fr.isZero(pols.crOffset[p])) pols.crOffsetInv[p] = fr.inv(pols.crOffset[p]);

            uint64_t crAccI = fr.toU64(pols.crOffset[p])/4;
            uint64_t crSh = (fr.toU64(pols.crOffset[p])%4)*8;

            for (uint64_t k=0; k<8; k++)
            {
                if (k == crAccI) crF[k][p] = fr.fromU64(1 << crSh);
                if (!fr.isZero(pols.crOffset[p]))
                {
                    crV[k][p+1] = (k==crAccI) ? fr.fromU64(fr.toU64(crV[k][p]) + (fr.toU64(pols.freeIn[p])<<crSh)) : crV[k][p];
                }
            }

            lastOffset = fr.toU64(pols.crOffset[p]);

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
        cerr << "Error: PaddingKKExecutor::execute() Too many keccak blocks nUsedBlocks=" << nUsedBlocks << " > nTotalBlocks=" << nTotalBlocks << " BlockSize=" << blockSize << endl;
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
    Goldilocks::Element hash0[8];
    scalar2fea(fr, hashZeroScalar, hash0);

    for (uint64_t i=0; i<nFullUnused; i++)
    {
        for (uint64_t j=0; j<bytesPerBlock; j++)
        {
            pols.addr[p] = fr.fromU64(addr);
            if (j == 0)
            {
                pols.freeIn[p] = fr.one();
                pols.firstHash[p] = fr.one();
            }
            else
            {
                if (j == (bytesPerBlock - 1)) pols.freeIn[p] = fr.fromU64(0x80);
                pols.rem[p] = fr.neg(fr.fromU64(j));
                pols.remInv[p] = fr.inv(pols.rem[p]);
                if (fr.toU64(pols.rem[p]) > 0xFFFF) pols.spare[p] = fr.one();
            }

            pols.incCounter[p] = fr.one();
            pols.crLen[p] =  fr.one();

            crF[0][p] = fr.one();

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
        pols.addr[p] = fr.fromU64(addr);
    

        if (p == fp)
        {
            pols.firstHash[p] = fr.one(); 
        }
        else
        {
            pols.rem[p] = fr.sub(pols.rem[p-1], fr.one());
            if (!fr.isZero(pols.rem[p])) pols.remInv[p] = fr.inv(pols.rem[p]);
            pols.spare[p] = fr.one();
        }

        pols.incCounter[p] = fr.one();
        pols.crLen[p] =  fr.one();
        crF[0][p] = fr.one();

        p += 1;
    }

    cout << "PaddingKKExecutor successfully processed " << input.size() << " Keccak hashes" << endl;
}