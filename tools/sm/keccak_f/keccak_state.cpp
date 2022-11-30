#include "keccak_state.hpp"
#include "pols_identity_constants.hpp"
#include "zkassert.hpp"

// Constructor
KeccakState::KeccakState ()
{
    // Allocate array of gates
    gate = new Gate[maxRefs];
    zkassert(gate!=NULL);

    // Reset
    resetBitsAndCounters();
}

// Destructor
KeccakState::~KeccakState ()
{
    // Free array of gates
    delete[] gate;
}

void KeccakState::resetBitsAndCounters (void)
{
    // Initialize array
    for (uint64_t i=0; i<maxRefs; i++)
    {
        gate[i].reset();
    }
    
    // Initialize the input state references
    for (uint64_t i=0; i<1600; i++)
    {
        SinRefs[i] = SinRef0 + 44*i;
    }
    
    // Initialize the output state references
    for (uint64_t i=0; i<1600; i++)
    {
        SoutRefs[i] = SoutRef0 + 44*i;
    }

    // Calculate the next reference (the first free slot)
    nextRef = FirstNextRef;

    // Init counters
    xors = 0;
    andps = 0;

    // Init ZeroRef and OneRef gates
    gate[ZeroRef].pin[pin_a].bit = 0;
    gate[ZeroRef].pin[pin_b].bit = 1;
    gate[ZeroRef].op = gop_xor;
    gate[ZeroRef].pin[pin_r].bit = 1;
}

// Set Rin data into bits array at SinRef0 position
void KeccakState::setRin (uint8_t * pRin)
{
    zkassert(pRin != NULL);
    for (uint64_t i=0; i<1088; i++)
    {
        gate[SinRef0+i*44].pin[pin_b].bit = pRin[i];
        gate[SinRef0+i*44].pin[pin_b].source = external;
    }
}

// Mix Rin data with Sin data
void KeccakState::mixRin (void)
{
    for (uint64_t i=0; i<1088; i++)
    {
        XOR(SinRef0+i*44, pin_a, SinRef0+i*44, pin_b, SinRef0+i*44);
    }
}

// Get 32-bytes output from SinRef0
void KeccakState::getOutput (uint8_t * pOutput)
{
    for (uint64_t i=0; i<32; i++)
    {
        uint8_t aux[8];
        for (uint64_t j=0; j<8; j++)
        {
            aux[j] = gate[SinRef0+(i*8+j)*44].pin[pin_a].bit;
        }
        bits2byte(aux, *(pOutput+i));
    }
}

// Get a free reference (the next one) and increment counter
uint64_t KeccakState::getFreeRef (void)
{
    zkassert(nextRef < maxRefs);
    uint64_t result = nextRef;
    nextRef++;

    // Skip Sin and Sout gates, every 44 slots
    if ( (nextRef<=(3200*44+1)) && (((nextRef-1)%44)==0) )
    {
        nextRef++;
    }

    return result;
}

// Copy Sout references to Sin references
void KeccakState::copySoutRefsToSinRefs (void)
{
    for (uint64_t i=0; i<1600; i++)
    {
        SinRefs[i] = SoutRefs[i];
    }
}

// Copy Sout data to Sin buffer, and reset
void KeccakState::copySoutToSinAndResetRefs (void)
{
    uint8_t localSout[1600];
    for (uint64_t i=0; i<1600; i++)
    {
        localSout[i] = gate[SoutRefs[i]].pin[pin_r].bit;
    }
    resetBitsAndCounters();
    for (uint64_t i=0; i<1600; i++)
    {
        gate[SinRef0+i*44].pin[pin_a].bit = localSout[i];
    }
}

void KeccakState::OP (GateOperation op, uint64_t refA, PinId pinA, uint64_t refB, PinId pinB, uint64_t refR)
{
    zkassert(refA < maxRefs);
    zkassert(refB < maxRefs);
    zkassert(refR < maxRefs);
    zkassert(pinA==pin_a || pinA==pin_b || pinA==pin_r);
    zkassert(pinB==pin_a || pinB==pin_b || pinB==pin_r);
    zkassert(gate[refA].pin[pinA].bit <= 1);
    zkassert(gate[refB].pin[pinB].bit <= 1);
    zkassert(gate[refR].pin[pin_r].bit <= 1);
    zkassert(refA==refR || refB==refR || gate[refR].op == gop_xor);
    zkassert(op==gop_xor || op==gop_andp);

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

    if (op==gop_xor)
    {
        // r = XOR(a,b)
        gate[refR].pin[pin_r].bit = gate[refA].pin[pinA].bit^gate[refB].pin[pinB].bit;
        xors++;
    }
    else //if (op==gop_andp)
    {
        // r = AND(a,b)
        gate[refR].pin[pin_r].bit = (1-gate[refA].pin[pinA].bit)&gate[refB].pin[pinB].bit;
        andps++;
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
void KeccakState::printCounters (void)
{
    double totalOperations = xors + andps;
    cout << "Max carry bits=" << MAX_CARRY_BITS << endl;
    cout << "xors=" << xors << "=" << double(xors)*100/totalOperations << "%" << endl;
    cout << "andps=" << andps << "=" << double(andps)*100/totalOperations  << "%" << endl;
    cout << "nextRef-1=" << nextRef-1 << endl;
}

// Refs must be an array of 1600 bits
void KeccakState::printRefs (uint64_t * pRefs, string name)
{
    // Get a local copy of the 1600 bits by reference
    uint8_t aux[1600];
    for (uint64_t i=0; i<1600; i++)
    {
        aux[i] = gate[pRefs[i]].pin[pin_r].bit;
    }

    // Print the bits
    printBits(aux, 1600, name);
}

// Map an operation code into a string
string KeccakState::op2string (GateOperation op)
{
    switch (op)
    {
        case gop_xor:
            return "xor";
        case gop_andp:
            return "andp";
        default:
            cerr << "KeccakSMState::op2string() found invalid op value:" << op << endl;
            exit(-1);
    }
}

// Generate a JSON object containing all data required for the executor script file
void KeccakState::saveScriptToJson (json &j)
{
    // In order of execution, add the operations data
    json programJson;
    for (uint64_t i=0; i<program.size(); i++)
    {
        // Root elements
        json evalJson;
        evalJson["op"] = op2string(program[i]->op);
        evalJson["ref"] = program[i]->pin[pin_r].wiredRef;

        // Input a elements
        json a;
        uint64_t refa = program[i]->pin[pin_a].wiredRef;
        if ( (refa<=(1600*44+1)) && (((refa-1)%44)==0) && (refa>44) && (program[i]->pin[pin_a].wiredPinId==PinId::pin_a) )
        {
            a["type"] = "input";
            a["bit"] = (refa/44) - 1;
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
        if ( (refb<=(1600*44+1)) && (((refb-1)%44)==0) && (refb>44) && (program[i]->pin[pin_b].wiredPinId==PinId::pin_a) )
        {
            b["type"] = "input";
            b["bit"] = (refb/44) - 1;
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
    j["maxRef"] = nextRef-1;
    j["xors"] = xors;
    j["andps"] = andps;
}

// Generate a JSON object containing all a, b, r, and op polynomials values
void KeccakState::savePolsToJson (json &pols)
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


    cout << "KeccakSMState::savePolsToJson() arity=" << Keccak_Arity << " length=" << Keccak_PolLength << " slotSize=" << Keccak_SlotSize << " numberOfSlots=" << KeccakSM_NumberOfSlots << " constant=" << fr.toString(identityConstant) << endl;

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
            cout << "KeccakSMState::savePolsToJson() initializing evaluation " << i << endl;
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
    cout << "KeccakSMState::savePolsToJson() final acc=" << fr.toString(acc) << endl;
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
        cout << "KeccakSMState::savePolsToJson() permuting non-zero references of slot " << slot+1 << " of " << KeccakSM_NumberOfSlots << endl;
        
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
        cout << "KeccakSMState::savePolsToJson() permuting zero references of pin " << pin << endl;

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
void KeccakState::saveConnectionsToJson (json &j)
{
    // In order of position, add the gates data
    j = json::array();
    for (uint64_t i=0; i<nextRef; i++)
    {
        json gateJson = json::object();
        for (uint64_t j=0; j<3; j++)
        {
            json connections = json::array();
            uint64_t connectionIndex = 0;
            for (uint64_t k=0; k<gate[i].pin[j].connectionsToInputA.size(); k++)
            {
                json c = json::array();
                c[0] = "A";
                c[1] = gate[i].pin[j].connectionsToInputA[k];
                connections[connectionIndex] = c;
                connectionIndex++;
            }
            for (uint64_t k=0; k<gate[i].pin[j].connectionsToInputB.size(); k++)
            {
                json c = json::array();
                c[0] = "B";
                c[1] = gate[i].pin[j].connectionsToInputB[k];
                connections[connectionIndex] = c;
                connectionIndex++;
            }
            if (connections.size() > 0)
            {
                string pinName = (j==0) ? "A" : (j==1) ? "B" : "C";
                gateJson[pinName] = connections;
            }
        }
        j[i] = gateJson;
    }
}