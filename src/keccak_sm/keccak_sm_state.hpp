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
#define MAX_CARRY_BITS 6

class Gate
{
public:
    enum Operation
    {
        op_unknown = 0,
        op_xor     = 1,
        op_andp    = 2,
        op_xorn    = 3
    };
    Operation op;
    uint64_t a;
    uint64_t b;
    uint64_t r;
    uint64_t fanOut;
    uint64_t value;
    uint64_t maxValue;
    vector<uint64_t> connectionsToA;
    vector<uint64_t> connectionsToB;
    Gate () : op(op_unknown), a(0), b(0), r(0), fanOut(0), value(1), maxValue(1) {};
    void reset (void)
    {
        op=op_unknown;
        a=0;
        b=0;
        r=0;
        fanOut=0;
        value=1;
        maxValue=1;
        connectionsToA.clear();
        connectionsToB.clear();
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

    KeccakSMState ();
    ~KeccakSMState ();
    void resetBitsAndCounters (void);

    // Set Rin data into bits array at RinRef0 position
    void setRin (uint8_t * pRin);

    // Get 32-bytes output from SinRef0
    void getOutput (uint8_t * pOutput);
    
    // Get a free reference (the next one) and increment counter
    uint64_t getFreeRef (void);

    // Copy Sout references to Sin references
    void copySoutRefsToSinRefs (void);
    
    // Copy Sout data to Sin buffer, and reset
    void copySoutToSinAndResetRefs (void);

    // XOR operation: r = XOR(a,b), r.value = a.value + b.value
    void XOR ( uint64_t a, uint64_t b, uint64_t r);

    // XORN operation: r = XOR(a,b), r.value = 1
    void XORN ( uint64_t a, uint64_t b, uint64_t r);

    // ANDP operation: r = AND( NOT(a), b), r.value = 1
    void ANDP ( uint64_t a, uint64_t b, uint64_t r);

    // Print statistics, for development purposes
    void printCounters (void);

    // Refs must be an array of 1600 bits
    void printRefs (uint64_t * pRefs, string name);

    // Map an operation code into a string
    string op2string (Gate::Operation op);

    // Generate a JSON object containing all data required for the executor script file
    void saveScriptToJson (json &j);

    // Generate a JSON object containing all a, b, r, and op polynomials values, with length 2^parity
    void savePolsToJson (json &j);
};

#endif