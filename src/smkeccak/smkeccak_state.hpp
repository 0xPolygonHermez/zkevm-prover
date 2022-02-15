#ifndef SMKECCAK_STATE_HPP
#define SMKECCAK_STATE_HPP

#include <stdint.h>
#include "config.hpp"
#include "scalar.hpp"
#include "utils.hpp"

#define Sin 0
#define Sout 1600
#define Rin 3200
#define maxRefs 1000000
#define OP_XOR 1
#define OP_AND 2
#define OP_COPY 3
#define OP_SET 4

class Eval
{
public:
    uint64_t op;
    uint64_t a;
    uint64_t b;
    uint64_t r;
};

class SMKeccakState
{
public:
    uint8_t  * bits;
    uint64_t nextRef; 
    vector<Eval> evals;
    uint64_t SoutRefs[1600];

    uint64_t * carry;
    uint64_t * maxCarry;
    uint64_t totalMaxCarry;

    // Counters
    uint64_t xors;
    uint64_t ands;
    uint64_t copies;

    // Well-known values positions
    uint64_t one;
    uint64_t zero;

    SMKeccakState ()
    {
        bits = (uint8_t *)malloc(maxRefs);
        zkassert(bits != NULL);
        carry = (uint64_t *)malloc(maxRefs*sizeof(uint64_t));
        zkassert(carry!=NULL);
        maxCarry = (uint64_t *)malloc(maxRefs*sizeof(uint64_t));
        zkassert(maxCarry!=NULL);
        for (uint64_t i=0; i<maxRefs; i++)
        {
            bits[i] = 0;
            carry[i] = 1;
            maxCarry[i] = 1;
        }
        totalMaxCarry = 1;
        for (uint64_t i=0; i<1600; i++)
        {
            SoutRefs[i] = Sout + i;
        }

        nextRef = Rin + 1088;

        one = nextRef;
        bits[one] = 1;
        nextRef++;

        zero = nextRef;
        bits[zero] = 0;
        nextRef++;

        xors = 0;
        ands = 0;
        copies = 0;
    }

    ~SMKeccakState ()
    {
        free(bits);
        free(carry);
        free(maxCarry);
    }

    uint64_t getFreeRef (void)
    {
        zkassert(nextRef < maxRefs);
        nextRef++;
        return nextRef - 1;
    }

    uint64_t getBit (uint64_t x, uint64_t y, uint64_t z)
    {
        return 64*x + 320*y + z;
    }

    void setRin (uint8_t * pRin)
    {
        zkassert(pRin != NULL);
        memcpy(bits+Rin, pRin, 1088);
    }

    void getOutput (uint8_t * pOutput)
    {
        for (uint64_t i=0; i<32; i++)
        {
            bits2byte(&bits[Sin+i*8], *(pOutput+i));
        }
    }

    void copySoutToSin (void)
    {
        memcpy(bits+Sin, bits+Sout, 1600);
        memset(bits+Sout, 0, 1600);
    }

    void XOR ( uint64_t a, uint64_t b, uint64_t r)
    {
        zkassert(a<maxRefs);
        zkassert(b<maxRefs);
        zkassert(r<maxRefs);
        zkassert(bits[a]<=1);
        zkassert(bits[b]<=1);
        zkassert(bits[r]<=1);
        bits[r] = bits[a]^bits[b];
        xors++;
        Eval eval;
        eval.op = OP_XOR;
        eval.a = a;
        eval.b = b;
        eval.r = r;
        evals.push_back(eval);

        carry[r] = carry[a] + carry[b];
        maxCarry[r] = zkmax(carry[r], maxCarry[r]);
        totalMaxCarry = zkmax(maxCarry[r], totalMaxCarry);
    }

    void AND ( uint64_t a, uint64_t b, uint64_t r)
    {
        zkassert(a<maxRefs);
        zkassert(b<maxRefs);
        zkassert(r<maxRefs);
        zkassert(bits[a]<=1);
        zkassert(bits[b]<=1);
        zkassert(bits[r]<=1);
        bits[r] = bits[a]&bits[b];
        ands++;
        Eval eval;
        eval.op = OP_AND;
        eval.a = a;
        eval.b = b;
        eval.r = r;
        evals.push_back(eval);

        carry[r] = 1;
    }

    void COPY ( uint64_t a, uint64_t r)
    {
        zkassert(a<maxRefs);
        zkassert(r<maxRefs);
        zkassert(bits[a]<=1);
        zkassert(bits[r]<=1);
        bits[r] = bits[a];
        copies++;
        Eval eval;
        eval.op = OP_COPY;
        eval.a = a;
        eval.b = 0;
        eval.r = r;
        evals.push_back(eval);
    }

    void printCounters (void)
    {
        cout << "bit xors=" << to_string(xors) << endl;
        cout << "bit ands=" << to_string(ands) << endl;
        cout << "bit copies=" << to_string(copies) << endl;
        cout << "nextRef=" << to_string(nextRef) << endl;
        cout << "totalMaxCarry=" << to_string(totalMaxCarry) << endl;
    }
};

#endif