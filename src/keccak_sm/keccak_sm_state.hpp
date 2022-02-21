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

/* Well-known positions:
0: zero bit value
1: one bit value
2...1601: Sin (1600 bits)
1602...2689: Rin (1088 bits)
1690...: available references for XOR and ANDP operations results
*/
#define ZeroRef      (0)
#define OneRef       (1)
#define SinRef0      (2)
#define RinRef0      (SinRef0+1600)
#define SoutRef0     (RinRef0+1088)
#define FirstNextRef (SoutRef0+1600)
#define Bit(x,y,z)   (64*(x) + 320*(y) + (z))

#define maxRefs 160000
#define OP_XOR 1
#define OP_ANDP 2
#define OP_XORN 3
#define MAX_CARRY_BITS 15

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

    uint64_t * value;
    uint64_t * maxValue;
    uint64_t totalMaxValue;

    // Counters
    uint64_t xors;
    uint64_t ands;
    uint64_t xorns;

    void resetBitsAndCounters (void)
    {
        // Initialize arrays
        for (uint64_t i=0; i<maxRefs; i++)
        {
            bits[i] = 0;
            value[i] = 1;
            maxValue[i] = 1;
        }

        // Initialize the max value (worst case, assuming highes values)
        totalMaxValue = 1;

        // Init the first 2 references
        bits[ZeroRef] = 0;
        bits[OneRef] = 1;
        
        // Initialize the input state references
        for (uint64_t i=0; i<1600; i++)
        {
            SinRefs[i] = SinRef0 + i;
        }
        
        // Initialize the output state references
        for (uint64_t i=0; i<1600; i++)
        {
            SoutRefs[i] = ZeroRef; //SoutRef + i;
        }

        // Calculate the next reference (the first free slot)
        nextRef = FirstNextRef;

        // Init counters
        xors = 0;
        ands = 0;
        xorns = 0;
    }

    KeccakSMState ()
    {
        // Allocate arrays
        bits = (uint8_t *)malloc(maxRefs);
        zkassert(bits != NULL);
        value = (uint64_t *)malloc(maxRefs*sizeof(uint64_t));
        zkassert(value!=NULL);
        maxValue = (uint64_t *)malloc(maxRefs*sizeof(uint64_t));
        zkassert(maxValue!=NULL);

        // Reset
        resetBitsAndCounters();
    }

    ~KeccakSMState ()
    {
        // Free arrays
        free(bits);
        free(value);
        free(maxValue);
    }

    // Get a free reference (the next one) and increment counter
    uint64_t getFreeRef (void)
    {
        zkassert(nextRef < maxRefs);
        nextRef++;
        return nextRef - 1;
    }

    // Set Rin data into bits array at RinRef0 position
    void setRin (uint8_t * pRin)
    {
        zkassert(pRin != NULL);
        memcpy(bits+RinRef0, pRin, 1088);
    }

    // Get 32-bytes output from SinRef0
    void getOutput (uint8_t * pOutput)
    {
        for (uint64_t i=0; i<32; i++)
        {
            bits2byte(&bits[SinRef0+i*8], *(pOutput+i));
        }
    }

    // Copy Sout data to Sin buffer, and reset
    void copySoutToSinAndResetRefs (void)
    {
        uint8_t localSout[1600];
        for (uint64_t i=0; i<1600; i++)
        {
            localSout[i] = bits[SoutRefs[i]];
        }
        resetBitsAndCounters();
        for (uint64_t i=0; i<1600; i++)
        {
            bits[SinRef0+i] = localSout[i];
        }
    }

    // Copy Sout references to Sin references
    void copySoutRefsToSinRefs (void)
    {
        for (uint64_t i=0; i<1600; i++)
        {
            SinRefs[i] = SoutRefs[i];
        }
    }

    // XOR operation
    void XOR ( uint64_t a, uint64_t b, uint64_t r)
    {
        zkassert(a<maxRefs);
        zkassert(b<maxRefs);
        zkassert(r<maxRefs);
        zkassert(bits[a]<=1);
        zkassert(bits[b]<=1);
        zkassert(bits[r]<=1);
        bits[r] = bits[a]^bits[b];
        Eval eval;
        if (value[a]+value[b]>=(1<<(MAX_CARRY_BITS+1)))
        {
            xorns++;
            eval.op = OP_XORN;
            value[r] = 1;
        }
        else
        {
            xors++;
            eval.op = OP_XOR;
            value[r] = value[a] + value[b];
            maxValue[r] = zkmax(value[r], maxValue[r]);
            totalMaxValue = zkmax(maxValue[r], totalMaxValue);
        }
        eval.a = a;
        eval.b = b;
        eval.r = r;
        evals.push_back(eval);
    }

    // ANDP operation
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

        value[r] = 1;
    }

    // Print statistics
    void printCounters (void)
    {
        cout << "Max carry bits=" << MAX_CARRY_BITS << endl;
        cout << "#xors=" << to_string(xors) << endl;
        cout << "#ands=" << to_string(ands) << endl;
        cout << "#xorns=" << to_string(xorns) << endl;
        cout << "nextRef=" << to_string(nextRef) << endl;
        cout << "totalMaxValue=" << to_string(totalMaxValue) << endl;
    }

    // Refs must be an array of 1600 bits
    void printRefs (uint64_t * pRefs, string name)
    {
        uint8_t aux[1600];
        for (uint64_t i=0; i<1600; i++)
        {
            aux[i] = bits[pRefs[i]];
        }
        printBits(aux, 1600, name);
    }

    // Generate a JSON object containing all data required for the script file
    void saveToJson (json &j)
    {
        json evaluations;
        json polA; // primeres 1088 XORS manuals sin a a, rin a b
        json polB;
        json polR;
        json polOp;
        for (uint64_t i=0; i<evals.size(); i++)
        {
            json evalJson;
            switch (evals[i].op)
            {
                case OP_XOR:
                    evalJson["op"] = "xor";
                    break;
                case OP_ANDP:
                    evalJson["op"] = "andp";
                    break;
                case OP_XORN:
                    evalJson["op"] = "xorn";
                    break;
                default:
                    cerr << "KeccakSMState::saveToJson() found invalid op value:" << evals[i].op << " in i:" << i << endl;
                    exit(-1);
            }
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
        j["maxValue"] = totalMaxValue;
    }
};

#endif