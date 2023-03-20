#ifndef SHA256_SM_STATE_HPP
#define SHA256_SM_STATE_HPP

#include <stdint.h>
#include <vector>
#include <nlohmann/json.hpp>
#include "gate.hpp"
#include "config.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "gate_config.hpp"

using namespace std;
using json = nlohmann::json;

class GateState
{
public:
    const GateConfig &gateConfig;

    uint64_t nextRef; 
    uint64_t * SinRefs;
    uint64_t * SoutRefs;

    // Evaluations, i.e. a chronological list of operations to implement a keccak-f()
    vector<Gate *> program;

    // Gates, i.e. an ordered list of gates
    Gate * gate;

    // Counters
    uint64_t xors;
    uint64_t ors;
    uint64_t andps;
    uint64_t ands;

    GateState (const GateConfig &gateConfig);
    ~GateState ();
    void resetBitsAndCounters (void);

    // Set Rin data into bits array at RinRef0 position (used by Keccak-f)
    void setRin (uint8_t * pRin);
    
    // Mix Rin data with Sin data (used by Keccak-f)
    void mixRin (void);

    // Get 32-bytes output from SinRef0 (used by Keccak-f)
    void getOutput (uint8_t * pOutput);
    
    // Get a free reference (the next one) and increment counter
    uint64_t getFreeRef (void);

    // Copy Sout references to Sin references (they must have the same size)
    void copySoutRefsToSinRefs (void);
    
    // Copy Sout data to Sin buffer, and reset (they must have the same size)
    void copySoutToSinAndResetRefs (void);

    // Perform the gate operation
    void OP (GateOperation op, uint64_t refA, PinId pinA, uint64_t refB, PinId pinB, uint64_t refR);

    // XOR operation: r = XOR(a,b), r.value = a.value + b.value
    void XOR (uint64_t refA, PinId pinA, uint64_t refB, PinId pinB, uint64_t refR) { OP(gop_xor, refA, pinA, refB, pinB, refR); };
    void XOR (uint64_t refA, uint64_t refB, uint64_t refR) { XOR(refA, pin_r, refB, pin_r, refR); };

    // ANDP operation: r = AND( NOT(a), b), r.value = !a.value && b.value
    void ANDP (uint64_t refA, PinId pinA, uint64_t refB, PinId pinB, uint64_t refR) { OP(gop_andp, refA, pinA, refB, pinB, refR); };
    void ANDP (uint64_t refA, uint64_t refB, uint64_t refR) { ANDP(refA, pin_r, refB, pin_r, refR); };

    // OR operation: r = OR(a,b), r.value = a.value or b.value
    void OR (uint64_t refA, PinId pinA, uint64_t refB, PinId pinB, uint64_t refR) { OP(gop_or, refA, pinA, refB, pinB, refR); };
    void OR (uint64_t refA, uint64_t refB, uint64_t refR) { OR(refA, pin_r, refB, pin_r, refR); };

    // AND operation: r = AND( a, b), r.value = a.value && b.value
    void AND (uint64_t refA, PinId pinA, uint64_t refB, PinId pinB, uint64_t refR) { OP(gop_and, refA, pinA, refB, pinB, refR); };
    void AND (uint64_t refA, uint64_t refB, uint64_t refR) { AND(refA, pin_r, refB, pin_r, refR); };

    // Print statistics, for development purposes
    void printCounters (void);

    // Refs must be an array of references: SinRef or SoutRef
    void printRefs (uint64_t * pRefs, string name);

    // Generate a JSON object containing all data required for the executor script file
    void saveScriptToJson (json &j);

    // Generate a JSON object containing all a, b, r, and op polynomials values, with length 2^parity
    void savePolsToJson (json &pols);

    // Generate a JSON object containing all wired pin connections, with length 2^parity
    void saveConnectionsToJson (json &pols);

    // Converts relative references to absolute references, based on the slot
    inline uint64_t relRef2AbsRef (uint64_t ref, uint64_t slot)
    {
        uint64_t result;

        // ZeroRef is the same for all the slots, and it is at reference 0
        if (ref==gateConfig.zeroRef) result = gateConfig.zeroRef;

        // Next references have an offset of one slot size per slot
        else result = slot*gateConfig.slotSize + ref;

        return result;
    }
};

#endif