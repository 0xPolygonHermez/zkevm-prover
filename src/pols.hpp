#ifndef POLS_HPP
#define POLS_HPP

#include "context.hpp"

void createPols(Context &ctx, json &pil);
void mapPols(Context &ctx);
void unmapPols(Context &ctx);

/*
   Polynomials size:
   Today all pols have the size of a finite field element, but some pols will only be 0 or 1 (way smaller).
   It is not efficient to user 256 bits to store just 1 bit.
   The PIL JSON file will specify the type of every polynomial: bit, byte, 32b, 64b, field element.
   pols[n will store the polynomial type, size, ID, and a pointer to the memory area.
*/
/*
   Polynomials memory:
   Using memory mapping to HDD file.
   TODO: Allocate dynamically according to the PIL JSON file contents.
*/

#define INVALID_ID 0xFFFFFFFFFFFFFFFF
extern uint64_t A0;
extern uint64_t A1;
extern uint64_t A2;
extern uint64_t A3;
extern uint64_t B0;
extern uint64_t B1;
extern uint64_t B2;
extern uint64_t B3;
extern uint64_t C0;
extern uint64_t C1;
extern uint64_t C2;
extern uint64_t C3;
extern uint64_t D0;
extern uint64_t D1;
extern uint64_t D2;
extern uint64_t D3;
extern uint64_t E0;
extern uint64_t E1;
extern uint64_t E2;
extern uint64_t E3;
extern uint64_t FREE0;
extern uint64_t FREE1;
extern uint64_t FREE2;
extern uint64_t FREE3;
extern uint64_t CONST;
extern uint64_t CTX;
extern uint64_t GAS;
extern uint64_t JMP;
extern uint64_t JMPC;
extern uint64_t MAXMEM;
extern uint64_t PC;
extern uint64_t SP;
extern uint64_t SR;
extern uint64_t arith;
extern uint64_t assert;
extern uint64_t bin;
extern uint64_t comparator;
extern uint64_t ecRecover;
extern uint64_t hashE;
extern uint64_t hashRD;
extern uint64_t hashWR;
extern uint64_t inA;
extern uint64_t inB;
extern uint64_t inC;
extern uint64_t inD;
extern uint64_t inE;
extern uint64_t inCTX;
extern uint64_t inFREE;
extern uint64_t inGAS;
extern uint64_t inMAXMEM;
extern uint64_t inPC;
extern uint64_t inSP;
extern uint64_t inSR;
extern uint64_t inSTEP;
extern uint64_t inc;
extern uint64_t dec2;
extern uint64_t ind;
extern uint64_t isCode;
extern uint64_t isMaxMem;
extern uint64_t isMem;
extern uint64_t isNeg;
extern uint64_t isStack;
extern uint64_t mRD;
extern uint64_t mWR;
extern uint64_t neg;
extern uint64_t offset;
extern uint64_t opcodeRomMap;
extern uint64_t sRD;
extern uint64_t sWR;
extern uint64_t setA;
extern uint64_t setB;
extern uint64_t setC;
extern uint64_t setD;
extern uint64_t setE;
extern uint64_t setCTX;
extern uint64_t setGAS;
extern uint64_t setMAXMEM;
extern uint64_t setPC;
extern uint64_t setSP;
extern uint64_t setSR;
extern uint64_t shl;
extern uint64_t shr;
extern uint64_t useCTX;
extern uint64_t zkPC;
extern uint64_t byte4_freeIN;
extern uint64_t byte4_out;

#endif