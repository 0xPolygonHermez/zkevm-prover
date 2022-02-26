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
1602...3201: Sout (1600 bits)
3201...: available references for XOR and ANDP operations results
*/
#define ZeroRef      (0)
#define OneRef       (1)
#define SinRef0      (2)
#define SoutRef0     (SinRef0+1600)
#define FirstNextRef (SoutRef0+1600)
#define Bit(x,y,z)   (64*(x) + 320*(y) + (z))

#define maxRefs 160000
#define MAX_CARRY_BITS 6

enum GateOperation
{
    gop_unknown = 0,
    gop_xor     = 1,
    gop_andp    = 2,
    gop_xorn    = 3
};

enum Pin
{
    pin_input_a = 0,
    pin_input_b = 1,
    pin_output  = 2
};

class Gate
{
public:
    GateOperation op;
    uint64_t a;
    uint64_t b;
    uint64_t r;
    uint64_t pinA;
    uint64_t pinB;
    uint8_t bit[3];
    uint64_t fanOut;
    uint64_t value;
    uint64_t maxValue;
    vector<uint64_t> connectionsToA;
    vector<uint64_t> connectionsToB;
    Gate () { reset(); };
    void reset (void)
    {
        op=gop_unknown;
        a=0;
        b=0;
        r=0;
        pinA=0;
        pinB=0;
        bit[pin_input_a]=0;
        bit[pin_input_b]=0;
        bit[pin_output]=0;
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
    uint64_t nextRef; 
    uint64_t SinRefs[1600];
    uint64_t SoutRefs[1600];

    // Evaluations, i.e. a chronological list of operations
    vector<Gate *> evals;

    // Gates, i.e. an ordered list of gates
    Gate * gate;

    uint64_t totalMaxValue;

    // Counters
    uint64_t xors;
    uint64_t andps;
    uint64_t xorns;

    KeccakSMState ();
    ~KeccakSMState ();
    void resetBitsAndCounters (void);

    // Set Rin data into bits array at RinRef0 position
    void setRin (uint8_t * pRin);
    
    // Mix Rin data with Sin data
    void mixRin (void);

    // Get 32-bytes output from SinRef0
    void getOutput (uint8_t * pOutput);
    
    // Get a free reference (the next one) and increment counter
    uint64_t getFreeRef (void);

    // Copy Sout references to Sin references
    void copySoutRefsToSinRefs (void);
    
    // Copy Sout data to Sin buffer, and reset
    void copySoutToSinAndResetRefs (void);

    // XOR operation: r = XOR(a,b), r.value = a.value + b.value
    void XOR (uint64_t a, Pin pa, uint64_t b, Pin pb, uint64_t r);
    void XOR (uint64_t a, uint64_t b, uint64_t r) { XOR(a, pin_output, b, pin_output, r); };

    // XORN operation: r = XOR(a,b), r.value = 1
    void XORN (uint64_t a, Pin pa, uint64_t b, Pin pb, uint64_t r);
    void XORN (uint64_t a, uint64_t b, uint64_t r) { XORN(a, pin_output, b, pin_output, r); };

    // ANDP operation: r = AND( NOT(a), b), r.value = 1
    void ANDP (uint64_t a, Pin pa, uint64_t b, Pin pb, uint64_t r);
    void ANDP (uint64_t a, uint64_t b, uint64_t r) { ANDP(a, pin_output, b, pin_output, r); };

    // Print statistics, for development purposes
    void printCounters (void);

    // Refs must be an array of 1600 bits
    void printRefs (uint64_t * pRefs, string name);

    // Map an operation code into a string
    string op2string (GateOperation op);

    // Generate a JSON object containing all data required for the executor script file
    void saveScriptToJson (json &j);

    // Generate a JSON object containing all a, b, r, and op polynomials values, with length 2^parity
    void savePolsToJson (json &j);
};

#endif