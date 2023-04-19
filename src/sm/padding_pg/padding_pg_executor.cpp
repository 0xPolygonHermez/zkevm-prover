#include <iostream>
#include "padding_pg_executor.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "poseidon_g_permutation.hpp"
#include "goldilocks_precomputed.hpp"
#include "zklog.hpp"

using namespace std;

uint64_t PaddingPGExecutor::prepareInput (vector<PaddingPGExecutorInput> &input)
{
    uint64_t totalInputBytes = 0;
    for (uint64_t i=0; i<input.size(); i++)
    {
        if (input[i].data.length() > 0)
        {
            // Make sure we got an even number of characters
            if ((input[i].data.length()%2) != 0)
            {
                zklog.error("PaddingPGExecutor::prepareInput() detected at entry i=" + to_string(i) + " a odd data string length=" + to_string(input[i].data.length()));
                exitProcess();
            }

            // Convert string (data) into binary (dataBytes)
            for (uint64_t c=0; c<input[i].data.length(); c+=2)
            {
                uint8_t aux;
                aux = 16*char2byte(input[i].data[c]) + char2byte(input[i].data[c+1]);
                input[i].dataBytes.push_back(aux);
            }
        }

        keccak256(input[i].dataBytes, input[i].hash);

        input[i].realLen = input[i].dataBytes.size();

        // Add padding
        input[i].dataBytes.push_back(0x1);
        while (input[i].dataBytes.size() % bytesPerBlock) input[i].dataBytes.push_back(0);
        input[i].dataBytes[ input[i].dataBytes.size() - 1] |= 0x80;

        totalInputBytes += input[i].dataBytes.size();
    }
    return totalInputBytes;
}

void PaddingPGExecutor::execute (vector<PaddingPGExecutorInput> &input, PaddingPGCommitPols &pols, vector<array<Goldilocks::Element, 17>> &required)
{    
    uint64_t totalInputBytes = prepareInput(input);

    // Check input size
    if (totalInputBytes > N)
    {
        zklog.error("PaddingPGExecutor::execute() Too many entries totalInputBytes=" + to_string(totalInputBytes) + " > N=" + to_string(N));
        exitProcess();
    }

    uint64_t p = 0;
    uint64_t pDone = 0;

    uint64_t addr = 0;

    CommitPol crF[8] = { pols.crF0, pols.crF1, pols.crF2, pols.crF3, pols.crF4, pols.crF5, pols.crF6, pols.crF7 };

    CommitPol crV[8] = { pols.crV0, pols.crV1, pols.crV2, pols.crV3, pols.crV4, pols.crV5, pols.crV6, pols.crV7 };

    pols.incCounter[p] = fr.one();

    for (uint64_t i=0; i<input.size(); i++)
    {

        int64_t curRead = -1;
        uint64_t lastOffset = 0;

        for (uint64_t j=0; j<input[i].dataBytes.size(); j++)
        {

            pols.freeIn[p] = fr.fromU64(input[i].dataBytes[j]);
            
            uint64_t acci = (j % bytesPerBlock) / bytesPerElement;
            uint64_t sh = (j % bytesPerElement)*8;
            for (uint64_t k=0; k<nElements; k++)
            {
                if (k == acci) {
                    pols.acc[k][p+1] = fr.fromU64( fr.toU64(pols.acc[k][p]) | (fr.toU64(pols.freeIn[p]) << sh) );
                } else {
                    pols.acc[k][p+1] = pols.acc[k][p];
                }
            }

            pols.prevHash0[p+1] = pols.prevHash0[p];
            pols.prevHash1[p+1] = pols.prevHash1[p];
            pols.prevHash2[p+1] = pols.prevHash2[p];
            pols.prevHash3[p+1] = pols.prevHash3[p];
            pols.incCounter[p+1] = pols.incCounter[p];

            pols.len[p] = fr.fromU64(input[i].realLen);
            pols.addr[p] = fr.fromU64(addr);
            pols.rem[p] = fr.sub(fr.fromU64(input[i].realLen), fr.fromU64(j));
            if (!fr.isZero(pols.rem[p]))
            {
                pols.remInv[p] = glp.inv(pols.rem[p]);
                if (fr.toU64(pols.rem[p]) > 0xFFFF) pols.spare[p] = fr.one();
            }
            
            bool lastBlock = (p % bytesPerBlock) == (bytesPerBlock - 1);
            bool lastHash = lastBlock && ((!fr.isZero(pols.spare[p])) || fr.isZero(pols.rem[p]));
            if (lastHash)
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
                pols.crLen[p] = fr.fromU64( (curRead<int64_t(input[i].reads.size())) ? input[i].reads[curRead] : 1 );
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
                if (k == crAccI) crF[k][p] = fr.fromU64(1 << crSh);
                if (!fr.isZero(pols.crOffset[p]))
                {
                    crV[k][p+1] = (k==crAccI) ? fr.fromU64(fr.toU64(crV[k][p]) + (fr.toU64(pols.freeIn[p])<<crSh)) : crV[k][p];
                }
            }

            lastOffset = fr.toU64(pols.crOffset[p]);

            if ( (j % bytesPerBlock) == (bytesPerBlock -1) )
            {
                Goldilocks::Element data[12];
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

                Goldilocks::Element dataHash[4];
                poseidon.hash(dataHash, data);
                
                pols.curHash0[p] = dataHash[0]; 
                pols.curHash1[p] = dataHash[1];
                pols.curHash2[p] = dataHash[2];
                pols.curHash3[p] = dataHash[3];

                array<Goldilocks::Element,17> aux;
                aux[0] = pols.acc[0][p+1];
                aux[1] = pols.acc[1][p+1];
                aux[2] = pols.acc[2][p+1];
                aux[3] = pols.acc[3][p+1];
                aux[4] = pols.acc[4][p+1];
                aux[5] = pols.acc[5][p+1];
                aux[6] = pols.acc[6][p+1];
                aux[7] = pols.acc[7][p+1];
                aux[8] = pols.prevHash0[p];
                aux[9] = pols.prevHash1[p];
                aux[10] = pols.prevHash2[p];
                aux[11] = pols.prevHash3[p];
                aux[12] = pols.curHash0[p];
                aux[13] = pols.curHash1[p]; 
                aux[14] = pols.curHash2[p]; 
                aux[15] = pols.curHash3[p];
                aux[16] = fr.fromU64(POSEIDONG_PERMUTATION4_ID);
                required.push_back(aux);

                pols.acc[0][p+1] = fr.zero();
                pols.acc[1][p+1] = fr.zero();
                pols.acc[2][p+1] = fr.zero();
                pols.acc[3][p+1] = fr.zero();
                pols.acc[4][p+1] = fr.zero();
                pols.acc[5][p+1] = fr.zero();
                pols.acc[6][p+1] = fr.zero();
                pols.acc[7][p+1] = fr.zero();

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
                pols.incCounter[p+1] = fr.inc(pols.incCounter[p]);

                if (j == (input[i].dataBytes.size() - 1))
                {
                    pols.prevHash0[p+1] = fr.zero(); // TODO: Comment out?
                    pols.prevHash1[p+1] = fr.zero(); // TODO: Comment out?
                    pols.prevHash2[p+1] = fr.zero(); // TODO: Comment out?
                    pols.prevHash3[p+1] = fr.zero(); // TODO: Comment out?
                    pols.incCounter[p+1] = fr.one();
                }

            }

            p += 1;
        }
        addr += 1;
    }

    pDone = p;

    uint64_t nFullUnused = ((N - p - 1)/bytesPerBlock)+1;

    Goldilocks::Element data[12];
    data[0] = fr.one();
    data[1] = fr.zero();
    data[2] = fr.zero();
    data[3] = fr.zero();
    data[4] = fr.zero();
    data[5] = fr.zero();
    data[6] = fr.zero();
    data[7] = fr.fromU64((uint64_t(0x80) << 48));
    data[8] = fr.zero();
    data[9] = fr.zero();
    data[10] = fr.zero();
    data[11] = fr.zero();

    Goldilocks::Element h0[4];
    poseidon.hash(h0, data);

    array<Goldilocks::Element,17> aux;
    aux[0] = fr.one();
    aux[1] = fr.zero();
    aux[2] = fr.zero();
    aux[3] = fr.zero();
    aux[4] = fr.zero();
    aux[5] = fr.zero();
    aux[6] = fr.zero();
    aux[7] = fr.fromU64((uint64_t(0x80) << 48));
    aux[8] = fr.zero();
    aux[9] = fr.zero();
    aux[10] = fr.zero();
    aux[11] = fr.zero();
    aux[12] = h0[0];
    aux[13] = h0[1]; 
    aux[14] = h0[2]; 
    aux[15] = h0[3];
    aux[16] = fr.fromU64(POSEIDONG_PERMUTATION4_ID);
    required.push_back(aux);

    for (uint64_t i=0; i<nFullUnused; i++)
    {
        uint64_t bytesBlock = ((N-p) > bytesPerBlock) ? bytesPerBlock : (N-p);
        if (bytesBlock < 2)
        {
            zklog.error("PaddingPGExecutor::execute() Alignment is not possible");
        }
        for (uint64_t j=0; j<bytesBlock; j++)
        {
            if (j==0)
            {
                pols.freeIn[p] = fr.one();
            }
            else if (j==(bytesBlock-1))
            {
                pols.freeIn[p] = fr.fromU64(0x80);
            }
            if (j != 0) pols.acc[0][p] = fr.one();

            pols.addr[p] = fr.fromU64(addr);
            pols.rem[p] = fr.neg(fr.fromU64(j)); // = -j
            if (!fr.isZero(pols.rem[p])) pols.remInv[p] = glp.inv(pols.rem[p]);
            if (j != 0)
            {
                pols.spare[p] = fr.one();
            }
            pols.prevHash0[p] = fr.zero(); // TODO: Comment out?
            pols.prevHash1[p] = fr.zero(); // TODO: Comment out?
            pols.prevHash2[p] = fr.zero(); // TODO: Comment out?
            pols.prevHash3[p] = fr.zero(); // TODO: Comment out?
            pols.incCounter[p] = fr.one();
            pols.curHash0[p] = h0[0];
            pols.curHash1[p] = h0[1];
            pols.curHash2[p] = h0[2];
            pols.curHash3[p] = h0[3];

            pols.crLen[p] = fr.one();
            pols.crF0[p] = fr.one();
            p += 1;
        }
        addr += 1;
    }

    zklog.info("PaddingPGExecutor successfully processed " + to_string(input.size()) + " Poseidon hashes p=" + to_string(p) + " pDone=" + to_string(pDone) + " (" + to_string((double(pDone)*100)/N) + "%)");
}