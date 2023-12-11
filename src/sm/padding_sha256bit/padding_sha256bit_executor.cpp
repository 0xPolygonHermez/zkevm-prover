#include "padding_sha256bit_executor.hpp"
#include "sha256.hpp"
#include "timer.hpp"
#include "definitions.hpp"
#include "zkmax.hpp"
#include "zklog.hpp"

const uint32_t hIn[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };

inline uint64_t getStateBit ( const uint32_t (&state)[8], uint64_t i )
{
    assert(i < 256);
    uint64_t sh = 31 - (i%32);
    return (state[i/32] >> sh) & 1;
}

void PaddingSha256BitExecutor::execute (vector<PaddingSha256BitExecutorInput> &input, PaddingSha256BitCommitPols &pols, vector<Bits2FieldSha256ExecutorInput> &required)
{

    // Check that input size does not exeed the number of slots
    if (input.size() > nSlots)
    {
        zklog.error("PaddingKKBitExecutor::execute() Too many entries input.size()=" + to_string(input.size()) + " > nSlots=" + to_string(nSlots));
        exitProcess();
    }
    uint64_t p = 0;
    // Convert pols.sOutX to and array, for programming convenience
    CommitPol sOut[8] = { pols.sOut0, pols.sOut1, pols.sOut2, pols.sOut3, pols.sOut4, pols.sOut5, pols.sOut6, pols.sOut7 };

    uint8_t zeroIn[64];
    zeroIn[0] = 0x80; 
    memset(&zeroIn[1], 0, sizeof(zeroIn)-1);

    uint32_t zeroOut[8];
    SHA256F(zeroIn,hIn,zeroOut);

    uint32_t currentState[8]; //rich: check
    currentState[0] = 0x80;
    memset(&currentState[0], 0, sizeof(currentState)-1);


    for (uint64_t i=0; i<nSlots; i++)
    {

        bool connected = true;
        uint32_t stIn[8];
        uint32_t stOut[8];
        uint8_t inR[64];

        // If this is the first block of a message, or we run out of them, start from a reset state
        // optimize avoiding copies
        if ((i>=input.size()) || (input[i].connected == false))
        {
            connected = false;
            memcpy(stIn, hIn, sizeof(hIn));
        }
        else
        {
            memcpy(&stIn, &currentState, sizeof(currentState));
        }

        if(i >= input.size())
        {
            memcpy(inR, zeroIn, sizeof(zeroIn));
            memcpy(stOut, zeroOut, sizeof(zeroOut));
        }
        else
        {
            memcpy(inR, input[i].data, sizeof(inR));
            SHA256F(inR, stIn, stOut);
        }

        for (uint64_t j=0; j<256; j++)
        {
            pols.s1[p] = fr.fromU64(getStateBit(currentState, j));
            if(connected){ 
                pols.connected[p] = fr.one();
                pols.s2[p] = pols.s1[p];
            }else{
                pols.s2[p] = fr.fromU64(getStateBit(hIn, j));
            }
            p++;            
        }
        for(uint64_t j=0; j < 512; ++j){

            uint64_t byteIdx = j / 8;
            uint64_t bitIdx = 7 - (j % 8);
            uint64_t byte = (i < input.size()) ? input[i].data[byteIdx] : 0;
            uint64_t bit = (byte >> bitIdx) & 1;

            if(connected) pols.connected[p] = fr.one();
            pols.s1[p] = fr.fromU64(bit);
            if(j < 256){
                pols.s2[p] = fr.fromU64(getStateBit(stOut, j));
            }

            uint64_t k = 7 - (j % 8);
            uint64_t inc = fr.toU64(pols.s1[p]) << k;
            pols.r8[p] = (k==7) ? fr.fromU64(inc) : fr.add(pols.r8[p-1], fr.fromU64(inc));
            
            for(uint64_t r=0; r < 8; r++){
                if(j>0){
                    sOut[r][p] = sOut[r][p-1];
                }
            }

            uint64_t inc2 = fr.toU64(pols.s2[p]) << (31-(j%32));

            if( j< 32){
                sOut[0][p] = fr.add(sOut[0][p], fr.fromU64(inc2));
            } else if (j<64){
                sOut[1][p] = fr.add(sOut[1][p], fr.fromU64(inc2));
            } else if (j<96){
                sOut[2][p] = fr.add(sOut[2][p], fr.fromU64(inc2));
            } else if (j<128){
                sOut[3][p] = fr.add(sOut[3][p], fr.fromU64(inc2));
            } else if (j<160){
                sOut[4][p] = fr.add(sOut[4][p], fr.fromU64(inc2));
            } else if (j<192){
                sOut[5][p] = fr.add(sOut[5][p], fr.fromU64(inc2));
            } else if (j<224){
                sOut[6][p] = fr.add(sOut[6][p], fr.fromU64(inc2));
            } else if (j<256){
                sOut[7][p] = fr.add(sOut[7][p], fr.fromU64(inc2));
            } 
            p += 1;
        }
        Bits2FieldSha256ExecutorInput b2fInput;
        memcpy(b2fInput.inBlock, inR, sizeof(inR));
        memcpy(b2fInput.inputState, stIn, sizeof(stIn));
        memcpy(b2fInput.outputState, stOut, sizeof(stOut));
        required.push_back(b2fInput);

        memcpy(currentState, stOut, sizeof(currentState));
    }

    // Connect the last state with the first
    uint64_t pp = 0;
    for (uint64_t j=0; j<256; j++)
    {
        pols.s1[pp] = fr.fromU64( getStateBit(currentState, j));
        pp++;
    }

    uint64_t pDone = p;

    zklog.info("PaddingKKBitExecutor successfully processed " + to_string(input.size()) + " Keccak actions p=" + to_string(p) + " pDone=" + to_string(pDone) + " (" + to_string((double(pDone)*100)/N) + "%)");

}

