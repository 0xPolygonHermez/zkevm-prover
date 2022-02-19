#ifndef KECCAK_SM_STATE_HPP
#define KECCAK_SM_STATE_HPP

#include <stdint.h>
#include <vector>
#include <nlohmann/json.hpp>
#include "config.hpp"
#include "scalar.hpp"
#include "utils.hpp"

using namespace std;
using json = nlohmann::json;

// Well-known positions
#define SinRef (2)
#define SoutRef (SinRef+1600)
#define RinRef (SoutRef+1600)

#define maxRefs 1000000
#define OP_XOR 1
#define OP_ANDP 2

class Eval
{
public:
    uint64_t op;
    uint64_t a;
    uint64_t b;
    uint64_t r;
};

class KeccakSMState
{
public:
    uint8_t  * bits;
    uint64_t nextRef; 
    vector<Eval> evals;
    uint64_t SinRefs[1600];
    uint64_t SoutRefs[1600];

    uint64_t * carry;
    uint64_t * maxCarry;
    uint64_t totalMaxCarry;

    // Counters
    uint64_t xors;
    uint64_t ands;

    // Fixed positions
    uint64_t zero;
    uint64_t one;

    KeccakSMState ()
    {
        // Allocate arrays
        bits = (uint8_t *)malloc(maxRefs);
        zkassert(bits != NULL);
        carry = (uint64_t *)malloc(maxRefs*sizeof(uint64_t));
        zkassert(carry!=NULL);
        maxCarry = (uint64_t *)malloc(maxRefs*sizeof(uint64_t));
        zkassert(maxCarry!=NULL);

        // Initialize arrays
        for (uint64_t i=0; i<maxRefs; i++)
        {
            bits[i] = 0;
            carry[i] = 1;
            maxCarry[i] = 1;
        }

        // Initialize the max carry
        totalMaxCarry = 1;

        // Initialize the input state references
        for (uint64_t i=0; i<1600; i++)
        {
            SinRefs[i] = SinRef + i;
        }
        
        // Initialize the output state references
        for (uint64_t i=0; i<1600; i++)
        {
            SoutRefs[i] = SoutRef + i;
        }

        // Calculate the next reference (the first free slot)
        nextRef = RinRef + 1088;

        // Init the first 2 references
        zero = 0;
        one = 1;
        bits[zero] = 0;
        bits[one] = 1;

        // Init counters
        xors = 0;
        ands = 0;
    }

    ~KeccakSMState ()
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
        memcpy(bits+RinRef, pRin, 1088);
    }

    void getOutput (uint8_t * pOutput)
    {
        for (uint64_t i=0; i<32; i++)
        {
            bits2byte(&bits[SinRef+i*8], *(pOutput+i));
        }
    }

    void copySoutToSin (void)
    {
        uint8_t localSout[1600];
        for (uint64_t i=0; i<1600; i++)
        {
            localSout[i] = bits[SoutRefs[i]];
        }
        for (uint64_t i=0; i<1600; i++)
        {
            bits[SinRef+i] = localSout[i];
        }
    }

    void resetSoutRefs (void)
    {
        for (uint64_t i=0; i<1600; i++)
        {
            SoutRefs[i] = SoutRef + i;
        }
        memset(bits+SoutRef, 0, 1600);
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

    void ANDP ( uint64_t a, uint64_t b, uint64_t r)
    {
        zkassert(a<maxRefs);
        zkassert(b<maxRefs);
        zkassert(r<maxRefs);
        zkassert(bits[a]<=1);
        zkassert(bits[b]<=1);
        zkassert(bits[r]<=1);
        bits[r] = (1-bits[a])&bits[b];
        ands++;
        Eval eval;
        eval.op = OP_ANDP;
        eval.a = a;
        eval.b = b;
        eval.r = r;
        evals.push_back(eval);

        carry[r] = 1;
        //carry[r] = zkmax(carry[a], carry[b]);
        //maxCarry[r] = zkmax(carry[r], maxCarry[r]);
        //totalMaxCarry = zkmax(maxCarry[r], totalMaxCarry);
    }

    void printCounters (void)
    {
        cout << "bit xors=" << to_string(xors) << endl;
        cout << "bit ands=" << to_string(ands) << endl;
        cout << "nextRef=" << to_string(nextRef) << endl;
        cout << "totalMaxCarry=" << to_string(totalMaxCarry) << endl;
    }

    void saveToJson (json &j)
    {
        json evaluations;
        for (uint64_t i=0; i<evals.size(); i++)
        {
            json evalJson;
            evalJson["op"] = evals[i].op==1? "xor":"andp";
            evalJson["a"] = evals[i].a;
            evalJson["b"] = evals[i].b;
            evalJson["r"] = evals[i].r;
            evaluations[i] = evalJson;
        }
        j["evaluations"] = evaluations;

        json soutRefs;
        for (uint64_t i=0; i<1600; i++)
        {
            soutRefs[i] = SoutRefs[i];
        }
        j["soutRefs"] = soutRefs;
        j["maxRef"] = nextRef-1;
        j["xors"] = xors;
        j["andps"] = ands;
        j["maxCarry"] = totalMaxCarry;
    }
};

#endif