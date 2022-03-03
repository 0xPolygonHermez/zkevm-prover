#ifndef KECCAK_SM_STATE_HPP
#define KECCAK_SM_STATE_HPP

#include <stdint.h>
#include <vector>
#include <nlohmann/json.hpp>
#include "gate.hpp"
#include "config.hpp"
#include "scalar.hpp"
#include "utils.hpp"

using namespace std;
using json = nlohmann::json;

/* Well-known positions:
0: zero bit value in input a, one bit value in input b
1...1600: Sin (1600 bits)
1601...3200: Sout (1600 bits)
3201...: available references for XOR and ANDP operations results
*/
#define ZeroRef      (0)
#define SinRef0      (1)
#define SoutRef0     (SinRef0+1600)
#define FirstNextRef (SoutRef0+1600)
#define Bit(x,y,z)   (64*(x) + 320*(y) + (z))

#define maxRefs 160000
#define MAX_CARRY_BITS 6

class KeccakSMState
{
public:
    uint64_t nextRef; 
    uint64_t SinRefs[1600];
    uint64_t SoutRefs[1600];

    // Evaluations, i.e. a chronological list of operations to implement a keccak-f()
    vector<Gate *> program;

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

    // Perform the gate operation
    void OP (GateOperation op, uint64_t refA, PinId pinA, uint64_t refB, PinId pinB, uint64_t refR);

    // XOR operation: r = XOR(a,b), r.value = a.value + b.value
    void XOR (uint64_t refA, PinId pinA, uint64_t refB, PinId pinB, uint64_t refR) { OP(gop_xor, refA, pinA, refB, pinB, refR); };
    void XOR (uint64_t refA, uint64_t refB, uint64_t refR) { XOR(refA, pin_r, refB, pin_r, refR); };

    // XORN operation: r = XOR(a,b), r.value = 1
    void XORN (uint64_t refA, PinId pinA, uint64_t refB, PinId pinB, uint64_t refR) { OP(gop_xorn, refA, pinA, refB, pinB, refR); };
    void XORN (uint64_t refA, uint64_t refB, uint64_t refR) { XORN(refA, pin_r, refB, pin_r, refR); };

    // ANDP operation: r = AND( NOT(a), b), r.value = 1
    void ANDP (uint64_t refA, PinId pinA, uint64_t refB, PinId pinB, uint64_t refR) { OP(gop_andp, refA, pinA, refB, pinB, refR); };
    void ANDP (uint64_t refA, uint64_t refB, uint64_t refR) { ANDP(refA, pin_r, refB, pin_r, refR); };

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

    // Aux method to convert relateve references to absolute references
    uint64_t relRef2AbsRef (uint64_t ref, uint64_t slot, uint64_t numberOfSlots, uint64_t slotSize);
};

#endif