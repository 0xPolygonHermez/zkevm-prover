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
#define OP_UNKNOWN 0
#define OP_XOR 1
#define OP_ANDP 2
#define OP_XORN 3
#define MAX_CARRY_BITS 15

class Gate
{
public:
    uint64_t op;
    uint64_t a;
    uint64_t b;
    uint64_t r;
    uint64_t fanOut;
    uint64_t value;
    uint64_t maxValue;
    Gate () : op(OP_UNKNOWN), a(0), b(0), r(0), fanOut(0), value(1), maxValue(1) {};
    void reset (void)
    {
        op=OP_UNKNOWN;
        a=0;
        b=0;
        r=0;
        fanOut=0;
        value=1;
        maxValue=1;
    }
};

class KeccakSMState
{
public:
    uint8_t  * bits;
    uint64_t nextRef; 
    uint64_t SinRefs[1600];
    uint64_t SoutRefs[1600];

    // Evaluations, i.e. a chronological list of operations
    vector<Gate *> evals;

    // Gates, i.e. an ordered list of gates
    Gate * gates;

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
            gates[i].reset();
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

        // Add initial evaluations and gates
        ANDP(ZeroRef, ZeroRef, ZeroRef);
        ANDP(ZeroRef, OneRef, OneRef);
        for (uint64_t i=SinRef0+1088; i<SinRef0+1600; i++)
        {
            XOR(ZeroRef, i, i);
        }
        for (uint64_t i=RinRef0; i<RinRef0+1088; i++)
        {
            XOR(ZeroRef, i, i);
        }
    }

    KeccakSMState ()
    {
        // Allocate arrays
        bits = (uint8_t *)malloc(maxRefs);
        zkassert(bits != NULL);
        gates = new Gate[maxRefs];
        zkassert(gates!=NULL);

        // Reset
        resetBitsAndCounters();
    }

    ~KeccakSMState ()
    {
        // Free arrays
        free(bits);
        delete[] gates;
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
        zkassert(gates[r].op == OP_UNKNOWN);

        if (gates[a].value+gates[b].value>=(1<<(MAX_CARRY_BITS+1)))
        {
            return XORN(a, b, r);
        }

        bits[r] = bits[a]^bits[b];
        xors++;

        gates[a].fanOut++;
        gates[b].fanOut++;

        gates[r].op = OP_XOR;
        gates[r].a = a;
        gates[r].b = b;
        gates[r].r = r;
        gates[r].value = gates[a].value + gates[b].value;
        gates[r].maxValue = zkmax(gates[r].value, gates[r].maxValue);
        totalMaxValue = zkmax(gates[r].maxValue, totalMaxValue);
        evals.push_back(&gates[r]);
    }

    // XORN operation
    void XORN ( uint64_t a, uint64_t b, uint64_t r)
    {
        zkassert(a<maxRefs);
        zkassert(b<maxRefs);
        zkassert(r<maxRefs);
        zkassert(bits[a]<=1);
        zkassert(bits[b]<=1);
        zkassert(bits[r]<=1);
        zkassert(gates[r].op == OP_UNKNOWN);

        bits[r] = bits[a]^bits[b];
        xorns++;

        gates[a].fanOut++;
        gates[b].fanOut++;
        
        gates[r].op = OP_XORN;
        gates[r].a = a;
        gates[r].b = b;
        gates[r].r = r;
        gates[r].value = 1;
        evals.push_back(&gates[r]);
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
        zkassert(gates[r].op == OP_UNKNOWN);

        bits[r] = (1-bits[a])&bits[b];
        ands++;
        
        gates[a].fanOut++;
        gates[b].fanOut++;
        
        gates[r].op = OP_ANDP;
        gates[r].a = a;
        gates[r].b = b;
        gates[r].r = r;
        gates[r].value = 1;
        evals.push_back(&gates[r]);
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

    string op2string (uint64_t op)
    {
        switch (op)
        {
            case OP_XOR:
                return "xor";
            case OP_ANDP:
                return "andp";
            case OP_XORN:
                return "xorn";
            default:
                cerr << "KeccakSMState::op2string() found invalid op value:" << op << endl;
                exit(-1);
        }
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
            evalJson["op"] = op2string(evals[i]->op);
            evalJson["a"] = evals[i]->a;
            evalJson["b"] = evals[i]->b;
            evalJson["r"] = evals[i]->r;
            evaluations[i] = evalJson;
        }
        j["evaluations"] = evaluations;

        json gatesJson;
        for (uint64_t i=0; i<nextRef; i++)
        {
            json gateJson;
            gateJson["rindex"] = i;
            gateJson["r"] = gates[i].r;
            gateJson["a"] = gates[i].a;
            gateJson["b"] = gates[i].b;
            gateJson["op"] = op2string(gates[i].op);
            gateJson["fanOut"] = gates[i].fanOut;
            gatesJson[i] = gateJson;
        }
        j["gates"] = gatesJson;

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