#include <iostream>
#include "padding_sha256_executor.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "goldilocks_precomputed.hpp"
#include "zklog.hpp"
#include "zkassert.hpp"

using namespace std;

uint64_t PaddingSha256Executor::prepareInput (vector<PaddingSha256ExecutorInput> &input)
{
    uint64_t totalInputBytes = 0;

    for (uint64_t i=0; i<input.size(); i++)
    {
        if (input[i].data.length() > 0)
        {
            // Make sure we got an even number of characters
            if ((input[i].data.length()%2) != 0)
            {
                zklog.error("PaddingSha256Executor::prepareInput() detected at entry i=" + to_string(i) + " a odd data string length=" + to_string(input[i].data.length()));
                exitProcess();
            }

            // Convert string (data) into binary (dataBytes)
            input[i].dataBytes.clear();
            for (uint64_t c=0; c<input[i].data.length(); c+=2)
            {
                uint8_t aux;
                aux = 16*char2byte(input[i].data[c]) + char2byte(input[i].data[c+1]);
                input[i].dataBytes.push_back(aux);
            }
        }

        SHA256(input[i].dataBytes.data(), input[i].dataBytes.size(), input[i].hash);

        input[i].realLen = input[i].dataBytes.size();
        if(input[i].realLen > 1ULL<<29){
            zklog.error("PaddingSha256Executor::prepareInput() detected at entry i=" + to_string(i) + " a data string length=" + to_string(input[i].realLen) + " > 2^29");
            exitProcess();
        }

        // Add padding
        input[i].dataBytes.push_back(0x80);
        while (input[i].dataBytes.size() % bytesPerBlock != 56) input[i].dataBytes.push_back(0);
        for(uint64_t j=0; j<4; j++) input[i].dataBytes.push_back(0);

        uint64_t bitLen = input[i].realLen*8;
        for(int64_t e=3; e>=0; e--)
        {
            input[i].dataBytes.push_back((bitLen >> (8*e)) & 0xFF);
        }

        totalInputBytes += input[i].dataBytes.size();
    }

    return totalInputBytes;
}

void PaddingSha256Executor::execute (vector<PaddingSha256ExecutorInput> &input, PaddingSha256CommitPols &pols, vector<PaddingSha256BitExecutorInput> &required)
{
    uint64_t totalInputBytes = prepareInput(input);

    // Check input size: totalInputBytes/bitsPerElement <= bytesPerBlock* maxBlocks; maxBlocks=N/blockSize
    //                   this condition depends on all the SM used to evaluated the Sha256 hash
    if (totalInputBytes * blockSize > bitsPerElement*bytesPerBlock*N)
    {
        zklog.error("PaddingKKExecutor::execute() Too many entries input.size()=" + to_string(input.size()) + " totalInputBytes=" + to_string(totalInputBytes) + " > bitsPerElement*bytesPerBlock*(N/blockSize)=" + to_string(bitsPerElement*bytesPerBlock*(N/blockSize)));
        exitProcess();
    }

    uint64_t p = 0;
    uint64_t pDone = 0;
    uint64_t addr = 0;

    CommitPol crF[8] = { pols.crF0, pols.crF1, pols.crF2, pols.crF3, pols.crF4, pols.crF5, pols.crF6, pols.crF7 };
    CommitPol crV[8] = { pols.crV0, pols.crV1, pols.crV2, pols.crV3, pols.crV4, pols.crV5, pols.crV6, pols.crV7 };

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
            if (!fr.isZero(pols.rem[p])) pols.remInv[p] = glp.inv(pols.rem[p]);
            
            if (j > input[i].realLen) pols.spare[p] = fr.one(); // padding bytes will meet condition: rem == 0 or spare == 1
            pols.incCounter[p] = fr.fromU64((j / bytesPerBlock) +1);

            uint64_t s = input[i].dataBytes.size()-1-j;
            uint64_t lengthSection = 0;
            if (s < 8)
            {
                lengthSection = 1;
                pols.lengthSection[p] = fr.one();
            }
            if (s < 4) pols.accLength[p] = fr.fromU64((input[i].realLen<<3) & (0xFFFFFFFF<<(s*8)));

            bool lastBlockLatch = (p % bytesPerBlock) == (bytesPerBlock - 1);
            bool lastHashLatch = lastBlockLatch && lengthSection;
            if (lastHashLatch)
            {
                if (input[i].lenCalled)
                {
                    pols.lastHashLen[p] = fr.one();
                }
                if (input[i].digestCalled)
                {
                    pols.lastHashDigest[p] = fr.one();
                }
            }

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
            if (!fr.isZero(pols.crOffset[p])) pols.crOffsetInv[p] = glp.inv(pols.crOffset[p]);

            uint64_t crAccI = fr.toU64(pols.crOffset[p])/4;
            uint64_t crSh = (fr.toU64(pols.crOffset[p])%4)*8;

            for (uint64_t k=0; k<8; k++)
            {
                if (k == crAccI)
                {
                    crF[k][p] = fr.fromU64(1 << crSh);
                }
                if (!fr.isZero(pols.crOffset[p]))
                {
                    crV[k][p+1] = (k==crAccI) ? fr.fromU64(fr.toU64(crV[k][p]) + (fr.toU64(pols.freeIn[p])<<crSh)) : crV[k][p];
                }
            }

            lastOffset = fr.toU64(pols.crOffset[p]);

            if ( (j % bytesPerBlock) == (bytesPerBlock -1) )
            {
                PaddingSha256BitExecutorInput paddingSha256BitExecutorInput;
                for (uint64_t k=0; k<bytesPerBlock; k++)
                {
                    paddingSha256BitExecutorInput.data[k] = input[i].dataBytes[j - bytesPerBlock + 1 + k];
                }
                paddingSha256BitExecutorInput.connected = (j < bytesPerBlock) ? false : true;
                required.push_back(paddingSha256BitExecutorInput);

                if (j == input[i].dataBytes.size() - 1) 
                {
                    // The Sha digest's chunks are ordered from most to
                    // least significant. (input7 => hash0, input6 => hash1, ..)
                    scalar2fea(fr, input[i].hash,
                        pols.hash7[p], 
                        pols.hash6[p], 
                        pols.hash5[p], 
                        pols.hash4[p], 
                        pols.hash3[p], 
                        pols.hash2[p], 
                        pols.hash1[p], 
                        pols.hash0[p]);

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

    pDone = p;

    uint64_t nTotalBlocks = bitsPerElement*(N/blockSize);
    uint64_t nUsedBlocks = p/bytesPerBlock;

    if (nUsedBlocks > nTotalBlocks)
    {
        zklog.error("PaddingKKExecutor::execute() Too many Sha256 blocks nUsedBlocks=" + to_string(nUsedBlocks) + " > nTotalBlocks=" + to_string(nTotalBlocks) + " BlockSize=" + to_string(blockSize));
        exitProcess();
    }

    uint64_t nFullUnused = nTotalBlocks - nUsedBlocks;

    uint8_t bytes0[bytesPerBlock];
    for (uint64_t i=0; i<bytesPerBlock; i++)
    {
        bytes0[i] = (i==0) ?  0x80 : 0;
    }

    for (uint64_t i=0; i<nFullUnused; i++)
    {
        for (uint64_t j=0; j<bytesPerBlock; j++)
        {
            pols.addr[p] = fr.fromU64(addr);
            if (j == 0)
            {
                pols.freeIn[p] = fr.fromU64(0x80);
            }else{
                pols.rem[p] = fr.neg(fr.fromU64(j));
                if (!fr.isZero(pols.rem[p])) pols.remInv[p] = glp.inv(pols.rem[p]);
                pols.spare[p] = fr.one();
            }
            

            pols.incCounter[p] = fr.one();
            if(j>=56) pols.lengthSection[p] = fr.one();
            
            pols.crLen[p] =  fr.one();
            crF[0][p] = fr.one();

            if (j % bytesPerBlock == (bytesPerBlock -1) )
            {
                PaddingSha256BitExecutorInput paddingSha256BitExecutorInput;
                for (uint64_t k=0; k<bytesPerBlock; k++)
                {
                    paddingSha256BitExecutorInput.data[k] = bytes0[k];
                }
                paddingSha256BitExecutorInput.connected = false;
                required.push_back(paddingSha256BitExecutorInput);
                
                // The Sha digest's chunks are ordered from most to
                // least significant. (input7 => hash0, input6 => hash1, ..)                
                pols.hash0[p] = hash0[7];
                pols.hash1[p] = hash0[6];
                pols.hash2[p] = hash0[5];
                pols.hash3[p] = hash0[4];
                pols.hash4[p] = hash0[3];
                pols.hash5[p] = hash0[2];
                pols.hash6[p] = hash0[1];
                pols.hash7[p] = hash0[0];
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
        pols.incCounter[p] = fr.one();


        if (p != fp)
        {
            pols.rem[p] = fr.sub(pols.rem[p-1], fr.one());
            if (!fr.isZero(pols.rem[p])) pols.remInv[p] = glp.inv(pols.rem[p]);
            pols.spare[p] = fr.one();
        } else{
             pols.freeIn[p] = fr.fromU64(0x80);
        }
        if(N-p <= 8) pols.lengthSection[p] = fr.one();
        pols.crLen[p] =  fr.one();
        crF[0][p] = fr.one();

        p += 1;
    }

    zklog.info("PaddingSha256Executor successfully processed " + to_string(input.size()) + " Sha256 hashes p=" + to_string(p) + " pDone=" + to_string(pDone) + " (" + to_string((double(pDone)*100)/N) + "%)");
}