#include "keccak_sm_state.hpp"
#include "pols_identity_constants.hpp"

// Constructor
KeccakSMState::KeccakSMState ()
{
    // Allocate array of gates
    gate = new Gate[maxRefs];
    zkassert(gate!=NULL);

    // Reset
    resetBitsAndCounters();
}

// Destructor
KeccakSMState::~KeccakSMState ()
{
    // Free array of gates
    delete[] gate;
}

void KeccakSMState::resetBitsAndCounters (void)
{
    // Initialize array
    for (uint64_t i=0; i<maxRefs; i++)
    {
        gate[i].reset();
    }

    // Initialize the max value (worst case, assuming highes values)
    totalMaxValue = 1;
    
    // Initialize the input state references
    for (uint64_t i=0; i<1600; i++)
    {
        SinRefs[i] = SinRef0 + i;
    }
    
    // Initialize the output state references
    for (uint64_t i=0; i<1600; i++)
    {
        SoutRefs[i] = SoutRef0 + i;
    }

    // Calculate the next reference (the first free slot)
    nextRef = FirstNextRef;

    // Init counters
    xors = 0;
    andps = 0;
    xorns = 0;

    // Init ZeroRef and OneRef gates
    gate[ZeroRef].bit[pin_input_a] = 0;
    gate[ZeroRef].bit[pin_input_b] = 0;
    XOR(ZeroRef, pin_input_a, ZeroRef, pin_input_b, ZeroRef);
    gate[OneRef].bit[pin_input_a] = 1;
    gate[OneRef].bit[pin_input_b] = 0;
    XOR(OneRef, pin_input_a, OneRef, pin_input_b, OneRef);
}

// Set Rin data into bits array at SinRef0 position
void KeccakSMState::setRin (uint8_t * pRin)
{
    zkassert(pRin != NULL);
    for (uint64_t i=0; i<1088; i++)
    {
        gate[SinRef0+i].bit[pin_input_b] = pRin[i];
    }
}

// Mix Rin data with Sin data
void KeccakSMState::mixRin (void)
{
    for (uint64_t i=0; i<1088; i++)
    {
        XOR(SinRef0+i, pin_input_a, SinRef0+i, pin_input_b, SinRef0+i);
    }
    for (uint64_t i=SinRef0+1088; i<SinRef0+1600; i++)
    {
        XOR(i, pin_input_a, ZeroRef, pin_output, i);
    }
}

// Get 32-bytes output from SinRef0
void KeccakSMState::getOutput (uint8_t * pOutput)
{
    for (uint64_t i=0; i<32; i++)
    {
        uint8_t aux[8];
        for (uint64_t j=0; j<8; j++)
        {
            aux[j] = gate[SinRef0+i*8+j].bit[pin_input_a];
        }
        bits2byte(aux, *(pOutput+i));
    }
}

// Get a free reference (the next one) and increment counter
uint64_t KeccakSMState::getFreeRef (void)
{
    zkassert(nextRef < maxRefs);
    nextRef++;
    return nextRef - 1;
}

// Copy Sout references to Sin references
void KeccakSMState::copySoutRefsToSinRefs (void)
{
    for (uint64_t i=0; i<1600; i++)
    {
        SinRefs[i] = SoutRefs[i];
    }
}

// Copy Sout data to Sin buffer, and reset
void KeccakSMState::copySoutToSinAndResetRefs (void)
{
    uint8_t localSout[1600];
    for (uint64_t i=0; i<1600; i++)
    {
        localSout[i] = gate[SoutRefs[i]].bit[pin_output];
    }
    resetBitsAndCounters();
    for (uint64_t i=0; i<1600; i++)
    {
        gate[SinRef0+i].bit[pin_input_a] = localSout[i];
    }
}

// XOR operation: r = XOR(a,b), r.value = a.value + b.value
void KeccakSMState::XOR (uint64_t a, Pin pina, uint64_t b, Pin pinb, uint64_t r)
{
    zkassert(a<maxRefs);
    zkassert(b<maxRefs);
    zkassert(r<maxRefs);
    zkassert(pina==pin_input_a || pina==pin_input_b || pina==pin_output);
    zkassert(pinb==pin_input_a || pinb==pin_input_b || pinb==pin_output);
    zkassert(gate[a].bit[pina]<=1);
    zkassert(gate[b].bit[pinb]<=1);
    zkassert(gate[r].bit[pin_output]<=1);
    //zkassert(gate[r].op == gop_unknown);
    if (gate[r].op != gop_unknown)
    {
        cout << "error" << endl;
    }

    // If the resulting value will exceed the max carry, perform a normalized XOR
    if (gate[a].value+gate[b].value>=(1<<(MAX_CARRY_BITS+1)))
    {
        return XORN(a, pina, b, pinb, r);
    }

    // r=XOR(a,b)
    gate[r].bit[pin_output] = gate[a].bit[pina]^gate[b].bit[pinb];
    xors++;

    // Increase the operands fan-out counters and add r to their connections
    if (a != r)
    {
        gate[a].fanOut++;
        gate[a].connectionsToA.push_back(r);
    }
    if (b != r)
    {
        gate[b].fanOut++;
        gate[b].connectionsToB.push_back(r);
    }

    // Update gate type and connections
    gate[r].op = gop_xor;
    gate[r].a = a;
    gate[r].b = b;
    gate[r].r = r;
    gate[r].pinA = pina;
    gate[r].pinB = pinb;
    gate[r].value = gate[a].value + gate[b].value;
    gate[r].maxValue = zkmax(gate[r].value, gate[r].maxValue);
    totalMaxValue = zkmax(gate[r].maxValue, totalMaxValue);

    // Add this gate to the chronological list of operations
    evals.push_back(&gate[r]);
}

// XORN operation: r = XOR(a,b), r.value = 1
void KeccakSMState::XORN (uint64_t a, Pin pina, uint64_t b, Pin pinb, uint64_t r)
{
    zkassert(a<maxRefs);
    zkassert(b<maxRefs);
    zkassert(r<maxRefs);
    zkassert(pina==pin_input_a || pina==pin_input_b || pina==pin_output);
    zkassert(pinb==pin_input_a || pinb==pin_input_b || pinb==pin_output);
    zkassert(gate[a].bit[pina]<=1);
    zkassert(gate[b].bit[pinb]<=1);
    zkassert(gate[r].bit[pin_output]<=1);
    zkassert(gate[r].op == gop_unknown);

    // r=XOR(a,b)
    gate[r].bit[pin_output] = gate[a].bit[pina]^gate[b].bit[pinb];
    xorns++;

    // Increase the operands fan-out counters and add r to their connections
    if (a != r)
    {
        gate[a].fanOut++;
        gate[a].connectionsToA.push_back(r);
    }
    if (b != r)
    {
        gate[b].fanOut++;
        gate[b].connectionsToB.push_back(r);
    }

    // Update gate type and connections
    gate[r].op = gop_xorn;
    gate[r].a = a;
    gate[r].b = b;
    gate[r].r = r;
    gate[r].pinA = pina;
    gate[r].pinB = pinb;
    gate[r].value = 1;

    // Add this gate to the chronological list of operations
    evals.push_back(&gate[r]);
}

// ANDP operation: r = AND( NOT(a), b), r.value = 1
void KeccakSMState::ANDP (uint64_t a, Pin pina, uint64_t b, Pin pinb, uint64_t r)
{
    zkassert(a<maxRefs);
    zkassert(b<maxRefs);
    zkassert(r<maxRefs);
    zkassert(pina==pin_input_a || pina==pin_input_b || pina==pin_output);
    zkassert(pinb==pin_input_a || pinb==pin_input_b || pinb==pin_output);
    zkassert(gate[a].bit[pina]<=1);
    zkassert(gate[b].bit[pinb]<=1);
    zkassert(gate[r].bit[pin_output]<=1);
    zkassert(gate[r].op == gop_unknown);

    // r=AND(a,b)
    gate[r].bit[pin_output] = (1-gate[a].bit[pina])&gate[b].bit[pinb];
    andps++;

    // Increase the operands fan-out counters and add r to their connections
    if (a != r)
    {
        gate[a].fanOut++;
        gate[a].connectionsToA.push_back(r);
    }
    if (b != r)
    {
        gate[b].fanOut++;
        gate[b].connectionsToB.push_back(r);
    }

    // Update gate type and connections
    gate[r].op = gop_andp;
    gate[r].a = a;
    gate[r].b = b;
    gate[r].r = r;
    gate[r].pinA = pina;
    gate[r].pinB = pinb;
    gate[r].value = 1;

    // Add this gate to the chronological list of operations
    evals.push_back(&gate[r]);
}

// Print statistics, for development purposes
void KeccakSMState::printCounters (void)
{
    double totalOperations = xors + andps + xorns;
    cout << "Max carry bits=" << MAX_CARRY_BITS << endl;
    cout << "xors=" << xors << "=" << double(xors)*100/totalOperations << "%" << endl;
    cout << "andps=" << andps << "=" << double(andps)*100/totalOperations  << "%" << endl;
    cout << "xorns=" << xorns << "=" << double(xorns)*100/totalOperations  << "%" << endl;
    cout << "andps+xorns=" << andps+xorns << "=" << double(andps+xorns)*100/totalOperations  << "%" << endl;
    cout << "xors/(andps+xorns)=" << double(xors)/double(andps+xorns)  << endl;
    cout << "nextRef=" << nextRef << endl;
    cout << "totalMaxValue=" << totalMaxValue << endl;
}

// Refs must be an array of 1600 bits
void KeccakSMState::printRefs (uint64_t * pRefs, string name)
{
    // Get a local copy of the 1600 bits by reference
    uint8_t aux[1600];
    for (uint64_t i=0; i<1600; i++)
    {
        aux[i] = gate[pRefs[i]].bit[pin_output];
    }

    // Print the bits
    printBits(aux, 1600, name);
}

// Map an operation code into a string
string KeccakSMState::op2string (GateOperation op)
{
    switch (op)
    {
        case gop_xor:
            return "xor";
        case gop_andp:
            return "andp";
        case gop_xorn:
            return "xorn";
        default:
            cerr << "KeccakSMState::op2string() found invalid op value:" << op << endl;
            exit(-1);
    }
}

// Generate a JSON object containing all data required for the executor script file
void KeccakSMState::saveScriptToJson (json &j)
{
    // In order of execution, add the operations data
    json evaluations;
    for (uint64_t i=0; i<evals.size(); i++)
    {
        json evalJson;
        evalJson["op"] = op2string(evals[i]->op);
        evalJson["a"] = evals[i]->a;
        evalJson["b"] = evals[i]->b;
        evalJson["r"] = evals[i]->r;
        evalJson["pina"] = evals[i]->pinA;
        evalJson["pinb"] = evals[i]->pinB;
        evaluations[i] = evalJson;
    }
    j["evaluations"] = evaluations;

    // In order of position, add the gates data
    json gatesJson;
    for (uint64_t i=0; i<nextRef; i++)
    {
        json gateJson;
        gateJson["rindex"] = i;
        gateJson["r"] = gate[i].r;
        gateJson["a"] = gate[i].a;
        gateJson["b"] = gate[i].b;
        gateJson["pina"] = gate[i].pinA;
        gateJson["pinb"] = gate[i].pinB;
        gateJson["op"] = op2string(gate[i].op);
        gateJson["fanOut"] = gate[i].fanOut;
        string connections;
        for (uint64_t j=0; j<gate[i].connectionsToA.size(); j++)
        {
            if (connections.size()!=0) connections +=",";
            connections += "A[" + to_string(gate[i].connectionsToA[j]) + "]";
        }
        for (uint64_t j=0; j<gate[i].connectionsToB.size(); j++)
        {
            if (connections.size()!=0) connections +=",";
            connections += "B[" + to_string(gate[i].connectionsToB[j]) + "]";
        }
        gateJson["connections"] = connections;
        gatesJson[i] = gateJson;
    }
    j["gates"] = gatesJson;

    // Add counters
    j["maxRef"] = nextRef-1;
    j["xors"] = xors;
    j["andps"] = andps;
    j["maxValue"] = totalMaxValue;
}

// Generate a JSON object containing all a, b, r, and op polynomials values
void KeccakSMState::savePolsToJson (json &j)
{
    RawFr fr;
    uint64_t parity = 23;
    uint64_t length = 1<<parity;
    uint64_t numberOfSlots = length / nextRef;

    RawFr::Element identityConstant;
    fr.fromString(identityConstant, GetPolsIdentityConstant(parity));

    // Generate polynomials
    json polA;
    json polB;
    json polR;
    json polOp;

    cout << "KeccakSMState::savePolsToJson() parity=" << parity << " length=" << length << " numberOfSlots=" << numberOfSlots << " constant=" << fr.toString(identityConstant) << endl;

    // Initialize all polynomials to the corresponding default values, without permutations
    RawFr::Element acc;
    fr.fromUI(acc, 1);
    RawFr::Element k1;
    fr.fromUI(k1, 2);
    RawFr::Element k2;
    fr.fromUI(k2, 3);
    RawFr::Element aux;

    for (uint64_t i=0; i<length; i++)
    {
        if ((i%1000000==0) || i==(length-1))
        {
            cout << "KeccakSMState::savePolsToJson() initializing evaluation " << i << endl;
        }
        fr.mul(acc, acc, identityConstant);
        polA[i] = fr.toString(acc);// fe value = 2^23th roots of unity: a, aa, aaa, aaaa ... 2^23
        fr.mul(aux, acc, k1);
        polB[i] = fr.toString(aux);// fe value = k1*a, k1*aa, ...
        fr.mul(aux, acc, k2);
        polR[i] = fr.toString(aux);// fe value = k2*a, k2*aa, ...
        polOp[i] = gate[i%nextRef].op;
    }
    cout << "KeccakSMState::savePolsToJson() final acc=" << fr.toString(acc) << endl;

    // Perform the polynomials permutations by rotating all inter-connected connections
    for (uint64_t slot=0; slot<numberOfSlots; slot++)
    {
        cout << "KeccakSMState::savePolsToJson() permuting slot " << slot << " of " << numberOfSlots << endl;
        uint64_t offset = slot*nextRef;
        for (uint64_t i=0; i<nextRef; i++)
        {
            string aux = polR[offset+i];
            for (uint64_t j=0; j<gate[i].connectionsToA.size(); j++)
            {
                string aux2 = polA[offset+gate[i].connectionsToA[j]];
                polA[offset+gate[i].connectionsToA[j]] = aux;
                aux = aux2;
            }
            for (uint64_t j=0; j<gate[i].connectionsToB.size(); j++)
            {
                string aux2 = polB[offset+gate[i].connectionsToB[j]];
                polB[offset+gate[i].connectionsToB[j]] = aux;
                aux = aux2;
            }
            polR[offset+i] = aux;
        }
    }

    // Create the JSON object structure
    json pols;
    pols["a"] = polA;
    pols["b"] = polB;
    pols["r"] = polR;
    pols["op"] = polOp;
    j["pols"] = pols;
}