#include "mem_align_executor.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "zklog.hpp"

uint8_t STEP (uint64_t i) { return i % 32; }
uint8_t OFFSET (uint64_t i) { return ((i >> 5) % 32); }
uint8_t WR8 (uint64_t i) { return (i % 3072) >= 2048 ?  1 : 0; }
uint8_t V_BYTE (uint64_t i) { return (31 + (OFFSET(i) + WR8(i)) - STEP(i)) % 32; }
uint64_t FACTORV (uint8_t index, uint64_t i) { 
    uint64_t f[4] = {1, 0x100, 0x10000, 0x1000000};
    return (V_BYTE(i) >> 2) == index ? f[V_BYTE(i) % 4] : 0; 
}

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

    uint64_t factors[4] = {1, 1<<8, 1<<16, 1<<24};
    for (uint64_t i=0; i<input.size(); i++) 
    {
        mpz_class m0v = input[i].m0;
        mpz_class m1v = input[i].m1;
        mpz_class v = input[i].v;
        uint8_t offset = input[i].offset;
        uint8_t wr8 = input[i].wr8;
        uint8_t wr256 = input[i].wr256;
        uint64_t polIndex = i * 32;
        mpz_class vv = v;
        
        // setting index when result was ready
        uint64_t polResultIndex = ((i+1) * 32)%N;
        if (!(wr8 || wr256)) pols.resultRd[polResultIndex] = fr.one();
        if (wr8) pols.resultWr8[polResultIndex] = fr.one();
        if (wr256) pols.resultWr256[polResultIndex] = fr.one();

        for (uint8_t j=0; j<32; j++)
        {
            uint8_t vByte = ((31 + (offset + wr8) - j) % 32);
            uint8_t inM0 = getByte(m0v, 31-j);
            uint8_t inM1 = getByte(m1v, 31-j);
            uint8_t inV = getByte(vv, vByte);
            uint8_t selM1 = (wr8 ? (j == offset) :(offset > j)) ? 1:0;

            pols.wr8[polIndex + j + 1] = fr.fromU64(wr8);
            pols.wr256[polIndex + j + 1] = fr.fromU64(wr256);
            pols.offset[polIndex + j + 1] = fr.fromU64(offset);
            pols.inM[0][polIndex + j] = fr.fromU64(inM0);
            pols.inM[1][polIndex + j] = fr.fromU64(inM1);
            pols.inV[polIndex + j] = fr.fromU64(inV);
            pols.selM1[polIndex + j] = fr.fromU64(selM1);
            pols.factorV[vByte >> 2][polIndex + j] = fr.fromU64(factors[(vByte % 4)]);

            uint8_t mIndex = 7 - (j >> 2);

            uint8_t inW0 = ((wr256 * (1 - selM1)) == 1 || (wr8 * selM1) == 1)? inV : ((wr256 + wr8) * inM0);
            uint8_t inW1 = (wr256 * selM1) == 1 ? inV : ((wr256 + wr8) * inM1);

            uint64_t factor = factors[3 - (j % 4)];

            pols.m0[mIndex][polIndex + 1 + j] = fr.fromU64( (( j == 0 ) ? 0 : fr.toU64(pols.m0[mIndex][polIndex + j])) + inM0 * factor );
            pols.m1[mIndex][polIndex + 1 + j] = fr.fromU64( (( j == 0 ) ? 0 : fr.toU64(pols.m1[mIndex][polIndex + j])) + inM1 * factor );

            pols.w0[mIndex][polIndex + 1 + j] = fr.fromU64( (( j == 0 ) ? 0 : fr.toU64(pols.w0[mIndex][polIndex + j])) + inW0 * factor );
            pols.w1[mIndex][polIndex + 1 + j] = fr.fromU64( (( j == 0 ) ? 0 : fr.toU64(pols.w1[mIndex][polIndex + j])) + inW1 * factor );
        }

        for (uint8_t j = 0; j < 32; ++j) {
            for (uint8_t index = 0; index < 8; index++) {
                pols.v[index][polIndex + 1 + j] = fr.add( (( j == 0 ) ? fr.zero() : pols.v[index][polIndex + j]), fr.mul( pols.inV[polIndex + j], pols.factorV[index][polIndex + j] ) );
            }
        }        

        for (uint8_t index = 0; index < 8; index++) {
            for (uint8_t j = 32 - (index  * 4); j < 32; j++) {
                pols.m0[index][polIndex + j + 1] = pols.m0[index][polIndex + j];
                pols.m1[index][polIndex + j + 1] = pols.m1[index][polIndex + j];
                pols.w0[index][polIndex + j + 1] = pols.w0[index][polIndex + j];
                pols.w1[index][polIndex + j + 1] = pols.w1[index][polIndex + j];
            }
        }
    }
    for (uint64_t i = (input.size() * 32); i < N; i++) {
        for (uint8_t index = 0; index < 8; index++) {
            pols.factorV[index][i] = fr.fromU64(FACTORV(index, i % 32));
        }
    }    

    zklog.info("MemAlignExecutor successfully processed " + to_string(input.size()) + " memory align actions (" + to_string((double(input.size())*32*100)/N) + "%)");
}