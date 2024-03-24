#include "mem_align_executor.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "zklog.hpp"

uint8_t getByte (mpz_class value, uint8_t index) {
    mpz_class r = (value >> (8 * index)) & 0xFF;
    return r.get_ui();
}

void MemAlignExecutor::execute (vector<MemAlignAction> &input, MemAlignCommitPols &pols)
{
    // Check input size 
    if (input.size()*32 > N)
    {
        zklog.error("MemAlignExecutor::execute() Too many entries input.size()=" + to_string(input.size()) + " > N/32=" + to_string(N/32));
        exitProcess();
    }

    uint64_t factors[4] = { 2 << 24, 2 << 16, 2 << 8, 1};
    for (uint64_t i=0; i<input.size(); i++) 
    {
        mpz_class m0v = input[i].m0;
        mpz_class m1v = input[i].m1;
        mpz_class v = input[i].v;
        uint8_t mode = input[i].mode;
        uint8_t wr = input[i].wr;
        uint8_t offset = mode & 0x7F;
        uint8_t len = (mode >> 7) & 0x3F;
        uint8_t format = (mode >> 13) & 0x03;
        if (len == 0) {
            len = 32;
        }
        if ((len + offset) >= 64) {
            len = 64 - offset;
        } 

        uint64_t polIndex = i * 32;
        mpz_class vv = v;
        
        // setting index when result was ready
        uint64_t polResultIndex = (i * 32 + 31)%N;
        if (wr) pols.result[polResultIndex] = fr.one();

        for (uint8_t j=0; j<32; j++)
        {
            uint8_t pos = (64 - offset + j);
            uint8_t bytePos = pos % 32;
            const bool selV = (bytePos < len) || wr;
            const bool isM0 = pos >= 64;
            const bool selM0 = (bytePos < len && isM0);
            const bool selM1 = (bytePos < len && !isM0);

            // if format is LEFT_BE bytePos no change.
            switch (format) {
                case 0b00:  bytePos = (bytePos + 32 - len) % 32; break;
                case 0b10:  bytePos = 31 - bytePos; break;
                case 0b11:   bytePos = (31 + len - bytePos) % 32; break;                
            }
            // (bytePos >= 0 && bytePos < 32, "format="+to_string(format)+" bytePos="+to_string(bytePos)+" len="+to_string(len)+" offset="+to_string(offset));
            zkassert(bytePos >= 0 && bytePos < 32);
    
            uint8_t inM0 = getByte(m0v, 31-j);
            uint8_t inM1 = getByte(m1v, 31-j);
            uint8_t inV_M = selV ? getByte(v, 31-bytePos) : 0;
            uint8_t inV_V = getByte(v, 31-j);

            uint64_t curIndex = polIndex + j;
        
            pols.result[curIndex] = j == 31 ? fr.one() : fr.zero();
            pols.wr[curIndex] = wr ? fr.one() : fr.zero();
            pols.mode[curIndex] = fr.fromU64(mode);
            pols.inM[0][curIndex] = fr.fromU64(inM0);
            pols.inM[1][curIndex] = fr.fromU64(inM1);
            pols.bytePos[curIndex] = fr.fromU64(bytePos);

            // divide inV in two part to do range check without extra lookup.
            pols.inV[0][curIndex] = fr.fromU64(inV_M & 0x3F);
            pols.inV[1][curIndex] = fr.fromU64(inV_M >> 6);

            pols.inV_V[curIndex] = fr.fromU64(inV_V);

            pols.selM0[curIndex] = selM0 ? fr.one() : fr.zero();
            pols.selM1[curIndex] = selM1 ? fr.one() : fr.zero();
            pols.selV[curIndex] = selV ? fr.one() : fr.zero();

            uint8_t mIndex = 7 - (j >> 2);
            uint64_t factor = factors[j % 4];

            if (j) {
                uint64_t prevIndex = polIndex + j - 1;
                for (int index = 0; index < 8; ++index) {
                    pols.m0[index][curIndex] = pols.m0[index][prevIndex];
                    pols.m1[index][curIndex] = pols.m1[index][prevIndex];
                    pols.w0[index][curIndex] = pols.w0[index][prevIndex];
                    pols.w1[index][curIndex] = pols.w1[index][prevIndex];
                    pols.v[index][curIndex]  = pols.v[index][prevIndex];
                }
            }

            pols.m0[mIndex][curIndex] = fr.fromU64(fr.toU64(pols.m0[mIndex][curIndex]) + inM0 * factor);
            pols.m1[mIndex][curIndex] = fr.fromU64(fr.toU64(pols.m1[mIndex][curIndex]) + inM1 * factor);
            pols.v[mIndex][curIndex]  = fr.fromU64(fr.toU64(pols.v[mIndex][curIndex]) + inV_V * factor);

            uint8_t inW0 = selM0 ? inV_M : inM0;
            uint8_t inW1 = selM1 ? inV_M : inM1;

            pols.w0[mIndex][curIndex] = fr.fromU64(fr.toU64(pols.w0[mIndex][curIndex]) + inW0 * factor);
            pols.w1[mIndex][curIndex] = fr.fromU64(fr.toU64(pols.w1[mIndex][curIndex]) + inW1 * factor);
        }
    }
    for (uint64_t i = (input.size() * 32); i < N; i++) {
        pols.selM0[i] = fr.one();
        pols.selV[i] = fr.one();
        pols.bytePos[i] = fr.fromU64(i % 32);
    }    

    zklog.info("MemAlignExecutor successfully processed " + to_string(input.size()) + " memory align actions (" + to_string((double(input.size())*32*100)/N) + "%)");
}