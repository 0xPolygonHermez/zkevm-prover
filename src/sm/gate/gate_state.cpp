#include "gate_state.hpp"
#include "pols_identity_constants.hpp"
#include "zkassert.hpp"
#include "zklog.hpp"

// Constructor
GateState::GateState(const GateConfig &gateConfig) : gateConfig(gateConfig)
{
    // Allocate Sin references
    SinRefs = new uint64_t[gateConfig.sinRefNumber];
    if (SinRefs == NULL)
    {
        zklog.info("Error: GateState::GateState() failed calling new for SinRefs, N=" + to_string(gateConfig.sinRefNumber));
        exitProcess();
    }

    // Allocate Sout references
    SoutRefs = new uint64_t[gateConfig.soutRefNumber];
    if (SinRefs == NULL)
    {
        zklog.info("Error: GateState::GateState() failed calling new for SoutRefs, N=" + to_string(gateConfig.soutRefNumber));
        exitProcess();
    }

    // Allocate array of gates
    gate = new Gate[gateConfig.maxRefs];
    if (gate == NULL)
    {
        zklog.info("Error: GateState::GateState() failed calling new for gate, N=" + to_string(gateConfig.maxRefs));
        exitProcess();
    }

    // Reset
    resetBitsAndCounters();
}

// Destructor
GateState::~GateState()
{
    // Free memory
    delete[] SinRefs;
    delete[] SoutRefs;
    delete[] gate;
}

void GateState::resetBitsAndCounters(void)
{
    // Initialize array
    for (uint64_t i = 0; i < gateConfig.maxRefs; i++)
    {
        gate[i].reset();
    }

    // Initialize the input state references
    for (uint64_t i = 0; i < gateConfig.sinRefNumber; i++)
    {
        SinRefs[i] = gateConfig.sinRef0 + gateConfig.sinRefDistance * i;
    }

    // Initialize the output state references
    for (uint64_t i = 0; i < gateConfig.soutRefNumber; i++)
    {
        SoutRefs[i] = gateConfig.soutRef0 + gateConfig.soutRefDistance * i;
    }

    // Calculate the next reference (the first free slot)
    nextRef = gateConfig.firstNextRef;

    // Init counters
    xors = 0;
    ors = 0;
    andps = 0;
    ands = 0;

    // Init ZeroRef and OneRef gates as 1 = XOR(0,1)
    gate[gateConfig.zeroRef].op = gop_xor;
    gate[gateConfig.zeroRef].pin[pin_a].bit = 0;
    gate[gateConfig.zeroRef].pin[pin_b].bit = 1;
    gate[gateConfig.zeroRef].pin[pin_r].bit = 1;
}

// Set Rin data into bits array at SinRef0 position
void GateState::setRin(uint8_t *pRin)
{
    if (gateConfig.sinRefNumber < 1088)
    {
        zklog.error("GateState::setRin() called with gateConfig.sinRefNumber=" + to_string(gateConfig.sinRefNumber) + " < 1088");
        exitProcess();
    }

    zkassert(pRin != NULL);

    for (uint64_t i = 0; i < 1088; i++)
    {
        uint64_t ref = gateConfig.sinRef0 + i * gateConfig.sinRefDistance;
        gate[ref].pin[pin_b].bit = pRin[i];
        gate[ref].pin[pin_b].source = external;
    }
}

// Mix Rin data with Sin data
void GateState::mixRin(void)
{
    if (gateConfig.sinRefNumber < 1088)
    {
        zklog.error("GateState::mixRin() called with gateConfig.sinRefNumber=" + to_string(gateConfig.sinRefNumber) + " < 1088");
        exitProcess();
    }

    for (uint64_t i = 0; i < 1088; i++)
    {
        uint64_t ref = gateConfig.sinRef0 + i * gateConfig.sinRefDistance;
        XOR(ref, pin_a, ref, pin_b, ref);
    }
}

// Get 32-bytes output from SinRef0
void GateState::getOutput(uint8_t *pOutput)
{
    if (gateConfig.sinRefNumber < (32 * 8))
    {
        zklog.error("GateState::getOutput() called with gateConfig.sinRefNumber=" + to_string(gateConfig.sinRefNumber) + " < 32*8");
        exitProcess();
    }

    for (uint64_t i = 0; i < 32; i++)
    {
        uint8_t aux[8];
        for (uint64_t j = 0; j < 8; j++)
        {
            aux[j] = gate[gateConfig.sinRef0 + (i * 8 + j) * gateConfig.sinRefDistance].pin[pin_a].bit;
        }
        bits2byte(aux, *(pOutput + i));
    }
}

// Get a free reference (the next one) and increment counter
uint64_t GateState::getFreeRef(void)
{
    zkassert(nextRef < gateConfig.maxRefs);
    uint64_t result = nextRef;
    nextRef++;

    while (true)
    {
        // Skip ZeroRef
        if (nextRef == gateConfig.zeroRef)
        {
            nextRef++;
            continue;
        }

        // Skip Sin gates
        if ((nextRef >= gateConfig.sinRef0) &&
            (nextRef <= gateConfig.sinRef0 + (gateConfig.sinRefNumber - 1) * gateConfig.sinRefDistance) &&
            (((nextRef - gateConfig.sinRef0) % gateConfig.sinRefDistance) == 0))
        {
            nextRef++;
            continue;
        }

        // Skip Sout gates
        if ((nextRef >= gateConfig.soutRef0) &&
            (nextRef <= gateConfig.soutRef0 + (gateConfig.soutRefNumber - 1) * gateConfig.soutRefDistance) &&
            (((nextRef - gateConfig.soutRef0) % gateConfig.soutRefDistance) == 0))
        {
            nextRef++;
            continue;
        }

        break;
    }

    zkassert(nextRef < gateConfig.maxRefs);

    return result;
}

// Copy Sout references to Sin references
void GateState::copySoutRefsToSinRefs(void)
{
    // Check sizes
    if (gateConfig.sinRefNumber != gateConfig.soutRefNumber)
    {
        zklog.error("GateState::copySoutRefsToSinRefs() called with gateConfig.sinRefNumber=" + to_string(gateConfig.sinRefNumber) + " different from gateConfig.soutRefNumber=" + to_string(gateConfig.soutRefNumber));
        exitProcess();
    }

    // Copy SoutRefs into SinRefs
    for (uint64_t i = 0; i < gateConfig.sinRefNumber; i++)
    {
        SinRefs[i] = SoutRefs[i];
    }
}

// Copy Sout data to Sin buffer, and reset
void GateState::copySoutToSinAndResetRefs(void)
{
    // Check sizes
    if (gateConfig.sinRefNumber != gateConfig.soutRefNumber)
    {
        zklog.error("GateState::copySoutToSinAndResetRefs() called with gateConfig.sinRefNumber=" + to_string(gateConfig.sinRefNumber) + " different from gateConfig.soutRefNumber=" + to_string(gateConfig.soutRefNumber));
        exitProcess();
    }

    // Allocate a local set of references
    uint8_t *localSout = new uint8_t[gateConfig.sinRefNumber];
    if (localSout == NULL)
    {
        zklog.error("GateState::copySoutToSinAndResetRefs() failed allocating uint8_t of size=" + to_string(gateConfig.sinRefNumber));
        exitProcess();
    }

    // Copy Sout into local
    for (uint64_t i = 0; i < gateConfig.sinRefNumber; i++)
    {
        localSout[i] = gate[SoutRefs[i]].pin[pin_r].bit;
    }

    // Reset
    resetBitsAndCounters();

    // Restore local to Sin
    for (uint64_t i = 0; i < gateConfig.sinRefNumber; i++)
    {
        gate[gateConfig.sinRef0 + i * gateConfig.sinRefDistance].pin[pin_a].bit = localSout[i];
    }

    // Free memory
    delete[] localSout;
}

void GateState::OP(GateOperation op, uint64_t refA, PinId pinA, uint64_t refB, PinId pinB, uint64_t refR)
{
    zkassert(refA < gateConfig.maxRefs);
    zkassert(refB < gateConfig.maxRefs);
    zkassert(refR < gateConfig.maxRefs);
    zkassert(pinA == pin_a || pinA == pin_b || pinA == pin_r);
    zkassert(pinB == pin_a || pinB == pin_b || pinB == pin_r);
    zkassert(gate[refA].pin[pinA].bit <= 1);
    zkassert(gate[refB].pin[pinB].bit <= 1);
    zkassert(gate[refR].pin[pin_r].bit <= 1);
    // zkassert(refA==refR || refB==refR || gate[refR].op == gop_xor);
    zkassert(op == gop_xor || op == gop_or || op == gop_andp || op == gop_and);

    // Update gate type and connections
    gate[refR].op = op;
    gate[refR].pin[pin_a].source = wired;
    gate[refR].pin[pin_a].wiredRef = refA;
    gate[refR].pin[pin_a].wiredPinId = pinA;
    gate[refR].pin[pin_a].bit = gate[refA].pin[pinA].bit;
    gate[refR].pin[pin_b].source = wired;
    gate[refR].pin[pin_b].wiredRef = refB;
    gate[refR].pin[pin_b].wiredPinId = pinB;
    gate[refR].pin[pin_b].bit = gate[refB].pin[pinB].bit;
    gate[refR].pin[pin_r].source = gated;
    gate[refR].pin[pin_r].wiredRef = refR;

    switch (op)
    {
    case gop_xor:
        // r = XOR(a,b)
        gate[refR].pin[pin_r].bit = gate[refA].pin[pinA].bit ^ gate[refB].pin[pinB].bit;
        xors++;
        break;
    case gop_or:
        // r = OR(a,b)
        gate[refR].pin[pin_r].bit = gate[refA].pin[pinA].bit | gate[refB].pin[pinB].bit;
        ors++;
        break;
    case gop_andp:
        // r = AND(!a,b)
        gate[refR].pin[pin_r].bit = (1 - gate[refA].pin[pinA].bit) & gate[refB].pin[pinB].bit;
        andps++;
        break;
    case gop_and:
        // r = AND(a,b)
        gate[refR].pin[pin_r].bit = gate[refA].pin[pinA].bit & gate[refB].pin[pinB].bit;
        ands++;
        break;
    default:
        zklog.error("GateState::OP() got invalid op=" + to_string(op));
        exitProcess();
    }

    // Increase the operands fan-out counters and add r to their connections
    if (refA != refR)
    {
        gate[refA].pin[pinA].fanOut++;
        gate[refA].pin[pinA].connectionsToInputA.push_back(refR);
    }
    if (refB != refR)
    {
        gate[refB].pin[pinB].fanOut++;
        gate[refB].pin[pinB].connectionsToInputB.push_back(refR);
    }

    // Add this gate to the chronological list of operations
    program.push_back(&gate[refR]);
}

// Print statistics, for development purposes
void GateState::printCounters(void)
{
    double totalOperations = xors + ors + andps + ands;
    zklog.info("xors      = " + to_string(xors) + " = " + to_string(double(xors) * 100 / totalOperations) + "%");
    zklog.info("ors       = " + to_string(ors) + " = " + to_string(double(ors) * 100 / totalOperations) + "%");
    zklog.info("andps     = " + to_string(andps) + " = " + to_string(double(andps) * 100 / totalOperations) + "%");
    zklog.info("ands      = " + to_string(ands) + " = " + to_string(double(ands) * 100 / totalOperations) + "%");
    zklog.info("nextRef-1 = " + to_string(nextRef - 1));
}

// Refs must be an array of references
void GateState::printRefs(uint64_t *pRefs, string name)
{
    uint64_t size = 0;

    // Find the size
    if (pRefs == SinRefs)
    {
        size = gateConfig.sinRefNumber;
    }
    else if (pRefs == SoutRefs)
    {
        size = gateConfig.soutRefNumber;
    }
    else
    {
        zklog.error("GateState::printRefs() got invalid value of pRefs=" + to_string(uint64_t(pRefs)));
        exitProcess();
    }

    // Allocate memory
    uint8_t *aux = new uint8_t[size];
    if (aux == NULL)
    {
        zklog.error("GateState::printRefs() failed allocating " + to_string(size) + " bytes");
        exitProcess();
    }

    // Copy the bits
    for (uint64_t i = 0; i < size; i++)
    {
        aux[i] = gate[pRefs[i]].pin[pin_r].bit;
    }

    // Print the bits
    printBits(aux, size, name);

    // Free  memory
    delete[] aux;
}

// Generate a JSON object containing all data required for the executor script file
void GateState::saveScriptToJson(json &j)
{
    // In order of execution, add the operations data
    json programJson;
    for (uint64_t i = 0; i < program.size(); i++)
    {
        // Root elements
        json evalJson;
        evalJson["op"] = gateop2string(program[i]->op);
        evalJson["ref"] = program[i]->pin[pin_r].wiredRef;

        // Input a elements
        json a;
        uint64_t refa = program[i]->pin[pin_a].wiredRef;
        if ((refa <= (gateConfig.sinRefNumber * gateConfig.sinRefDistance + 1)) && (((refa - 1) % gateConfig.sinRefDistance) == 0) && (refa > gateConfig.sinRefDistance) && (program[i]->pin[pin_a].wiredPinId == PinId::pin_a))
        {
            a["type"] = "input";
            a["bit"] = (refa / gateConfig.sinRefDistance) - 1;
        }
        else
        {
            a["type"] = "wired";
            a["gate"] = program[i]->pin[pin_a].wiredRef;
            a["pin"] = pin2string(program[i]->pin[pin_a].wiredPinId);
        }
        evalJson["a"] = a;

        // Input b elements
        json b;
        uint64_t refb = program[i]->pin[pin_b].wiredRef;
        if ((refb <= (gateConfig.sinRefNumber * gateConfig.sinRefDistance + 1)) && (((refb - 1) % gateConfig.sinRefDistance) == 0) && (refb > gateConfig.sinRefDistance) && (program[i]->pin[pin_b].wiredPinId == PinId::pin_a))
        {
            b["type"] = "input";
            b["bit"] = (refb / gateConfig.sinRefDistance) - 1;
        }
        else
        {
            b["type"] = "wired";
            b["gate"] = program[i]->pin[pin_b].wiredRef;
            b["pin"] = pin2string(program[i]->pin[pin_b].wiredPinId);
        }
        evalJson["b"] = b;

        // Add to program
        programJson[i] = evalJson;
    }
    j["program"] = programJson;

    // Add counters
    j["maxRef"] = nextRef - 1;
    j["xors"] = xors;
    j["ors"] = ors;
    j["andps"] = andps;
    j["ands"] = ands;
}

// Generate a JSON object containing all a, b, r, and op polynomials values
void GateState::savePolsToJson(json &pols)
{
#if 0
    // TODO: Activate KeccakSMState::savePolsToJson() after clarifying how to deal with 64-b FE
    RawFr fr;
    zkassert(Keccak_SlotSize == nextRef - 1);

    // Get the polynomial constant used to generate the polynomials based on arity
    // It is the 2^arity'th root of the unit
    RawFr::Element identityConstant;
    fr.fromString(identityConstant, GetPolsIdentityConstant(Keccak_Arity));

    // Generate polynomials
    pols["a"] = json::array();
    pols["b"] = json::array();
    pols["r"] = json::array();
    pols["op"] = json::array();


    zklog.info("KeccakSMState::savePolsToJson() arity=" + to_string(Keccak_Arity) + " length=" + to_string(Keccak_PolLength) + " slotSize=" + to_string(Keccak_SlotSize) + " numberOfSlots=" + to_string(KeccakSM_NumberOfSlots) + " constant=" + fr.toString(identityConstant));

    // Initialize all polynomials to the corresponding default values, without permutations
    RawFr::Element acc;
    fr.fromUI(acc, 1);
    RawFr::Element k1;
    fr.fromUI(k1, 2);
    RawFr::Element k2;
    fr.fromUI(k2, 3);
    RawFr::Element aux;

    // Init polynomials a, b, and r with the corresponding constants
    for (uint64_t i=0; i<Keccak_PolLength; i++)
    {
        // Log a trace every one million loops
        if ((i%1000000==0) || i==(Keccak_PolLength-1))
        {
            zklog.info("KeccakSMState::savePolsToJson() initializing evaluation " + to_string(i));
        }

        // Polynomial input a
        fr.mul(acc, acc, identityConstant);
        pols["a"][i] = fr.toString(acc);// fe value = 2^23th roots of unity: a, aa, aaa, aaaa ... a^2^23=1

        // Polynomial input b
        fr.mul(aux, acc, k1);
        pols["b"][i] = fr.toString(aux);// fe value = k1*a, k1*aa, ... , k1
        
        // Polynomial output r
        fr.mul(aux, acc, k2);
        pols["r"][i] = fr.toString(aux);// fe value = k2*a, k2*aa, ... , k2
    }

    // After the whole round, the acc value must be the unit
    zklog.info("KeccakSMState::savePolsToJson() final acc=" + fr.toString(acc));
    zkassert(fr.toString(acc)=="1");

    // Init polynomial op (operation)
    pols["op"][ZeroRef] = gate[ZeroRef].op;

    // For all slots
    for (uint64_t slot=0; slot<KeccakSM_NumberOfSlots; slot++)
    {
        // For all gates
        for (uint64_t ref=1; ref<nextRef; ref++)
        {
            // Get the absolute reference, according to the current slot            
            int64_t absRef = relRef2AbsRef(ref, slot);

            // Set the operation polynomial value
            pols["op"][absRef] = gate[ref].op;
        }
    }

    // Init the ending, reminding gates (not part of any slot) as xor
    for (uint64_t absRef=KeccakSM_NumberOfSlots*Keccak_SlotSize; absRef<Keccak_PolLength; absRef++)
    {
        pols["op"][absRef] = gop_xor;
    }

    // Perform the polynomials permutations by rotating all inter-connected connections (except the ZeroRef, which is done later since it is shared by all slots)
    for (uint64_t slot=0; slot<KeccakSM_NumberOfSlots; slot++)
    {
        zklog.info("KeccakSMState::savePolsToJson() permuting non-zero references of slot " + to_string(slot+1) + " of " + to_string(KeccakSM_NumberOfSlots));
        
        // For all gates
        for (uint64_t ref=1; ref<nextRef; ref++)
        {
            // Get the absolute reference, according to the current slot
            int64_t absRef = relRef2AbsRef(ref, slot);

            // For all gate pins: input a, input b and output r
            for (uint64_t pin=0; pin<3; pin++)
            {
                // Get the initialized value of that pin and reference
                string pinString = (pin==0) ? "a" : (pin==1) ? "b" : "r";
                string aux = pols[pinString][absRef];

                // Rotate the value by all its connections to an input a pin
                for (uint64_t con=0; con<gate[ref].pin[pin].connectionsToInputA.size(); con++)
                {
                    // Get the connected gate absolute reference
                    uint64_t relRefA = gate[ref].pin[pin].connectionsToInputA[con];
                    uint64_t absRefA = relRef2AbsRef(relRefA, slot);

                    // Swap the current aux value by the gate pin_a value where it is connected to
                    string auxA = pols["a"][absRefA];
                    pols["a"][absRefA] = aux;
                    aux = auxA;
                }

                // Rotate the value by all its connections to an input b pin
                for (uint64_t con=0; con<gate[ref].pin[pin].connectionsToInputB.size(); con++)
                {
                    // Get the connected gate absolute reference
                    uint64_t relRefB = gate[ref].pin[pin].connectionsToInputB[con];
                    uint64_t absRefB = relRef2AbsRef(relRefB, slot);
                    
                    // Swap the current aux value by the gate pin_b value where it is connected to
                    string auxB = pols["b"][absRefB];
                    pols["b"][absRefB] = aux;
                    aux = auxB;
                }

                // When the rotation is complete, store the last value into this pin and reference
                pols[pinString][absRef] = aux;
            }
        }
    }

    // Perform the permutations for the ZeroRef inputs a(bit=0) and b(bit=1)
    // The zero reference is shares among all the slots, so the rotation will imply the whole set of slots
    // For the pins a and b of the ZeroRef
    for (uint64_t pin=0; pin<2; pin++)
    {
        zklog.info("KeccakSMState::savePolsToJson() permuting zero references of pin " + to_string(pin));

        // Get the initialized value of that pin for reference ZeroRef
        string pinString = (pin==0) ? "a" : "b";
        string aux = pols[pinString][ZeroRef];

        // Rotate the value by all its connections to an input a pin
        for (uint64_t con=0; con<gate[ZeroRef].pin[pin].connectionsToInputA.size(); con++)
        {
            // Interate for all the slots, since ZeroRef is shared
            for (uint64_t slot=0; slot<KeccakSM_NumberOfSlots; slot++)
            {
                // Get the connected gate absolute reference
                uint64_t relRefA = gate[ZeroRef].pin[pin].connectionsToInputA[con];
                uint64_t absRefA = relRef2AbsRef(relRefA, slot);

                // Swap the current aux value by the gate pin_a value where it is connected to
                string auxA = pols["a"][absRefA];
                pols["a"][absRefA] = aux;
                aux = auxA;
            }
        }

        // Rotate the value by all its connections to an input b pin
        for (uint64_t con=0; con<gate[ZeroRef].pin[pin].connectionsToInputB.size(); con++)
        {   
            // Interate for all the slots, since ZeroRef is shared
            for (uint64_t slot=0; slot<KeccakSM_NumberOfSlots; slot++)
            {
                // Get the connected gate absolute reference
                uint64_t relRefB = gate[ZeroRef].pin[pin].connectionsToInputB[con];
                uint64_t absRefB = relRef2AbsRef(relRefB, slot);
                
                // Swap the current aux value by the gate pin_b value where it is connected to
                string auxB = pols["b"][absRefB];
                pols["b"][absRefB] = aux;
                aux = auxB;
            }
        }

        // When the rotation is complete, store the last value into this pin for reference ZeroRef
        pols[pinString][ZeroRef] = aux;
    }
#endif
}

// Generate a JSON object containing all wired connections
void GateState::saveConnectionsToJson(json &j)
{
    // In order of position, add the gates data
    j = json::array();
    for (uint64_t i = 0; i < nextRef; i++)
    {
        json gateJson = json::object();
        for (uint64_t j = 0; j < 3; j++)
        {
            json connections = json::array();
            uint64_t connectionIndex = 0;
            for (uint64_t k = 0; k < gate[i].pin[j].connectionsToInputA.size(); k++)
            {
                json c = json::array();
                c[0] = "A";
                c[1] = gate[i].pin[j].connectionsToInputA[k];
                connections[connectionIndex] = c;
                connectionIndex++;
            }
            for (uint64_t k = 0; k < gate[i].pin[j].connectionsToInputB.size(); k++)
            {
                json c = json::array();
                c[0] = "B";
                c[1] = gate[i].pin[j].connectionsToInputB[k];
                connections[connectionIndex] = c;
                connectionIndex++;
            }
            if (connections.size() > 0)
            {
                string pinName = (j == 0) ? "A" : (j == 1) ? "B"
                                                           : "C";
                gateJson[pinName] = connections;
            }
        }
        j[i] = gateJson;
    }
}