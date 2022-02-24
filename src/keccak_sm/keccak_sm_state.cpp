#include "keccak_sm_state.hpp"

// Constructor
KeccakSMState::KeccakSMState ()
{
    // Allocate arrays
    bits = (uint8_t *)malloc(maxRefs);
    zkassert(bits != NULL);
    gates = new Gate[maxRefs];
    zkassert(gates!=NULL);

    // Reset
    resetBitsAndCounters();
}

// Destructor
KeccakSMState::~KeccakSMState ()
{
    // Free arrays
    free(bits);
    delete[] gates;
}

void KeccakSMState::resetBitsAndCounters (void)
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

// Set Rin data into bits array at RinRef0 position
void KeccakSMState::setRin (uint8_t * pRin)
{
    zkassert(pRin != NULL);
    memcpy(bits+RinRef0, pRin, 1088);
}

// Get 32-bytes output from SinRef0
void KeccakSMState::getOutput (uint8_t * pOutput)
{
    for (uint64_t i=0; i<32; i++)
    {
        bits2byte(&bits[SinRef0+i*8], *(pOutput+i));
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
        localSout[i] = bits[SoutRefs[i]];
    }
    resetBitsAndCounters();
    for (uint64_t i=0; i<1600; i++)
    {
        bits[SinRef0+i] = localSout[i];
    }
}

// XOR operation: r = XOR(a,b), r.value = a.value + b.value
void KeccakSMState::XOR ( uint64_t a, uint64_t b, uint64_t r)
{
    zkassert(a<maxRefs);
    zkassert(b<maxRefs);
    zkassert(r<maxRefs);
    zkassert(bits[a]<=1);
    zkassert(bits[b]<=1);
    zkassert(bits[r]<=1);
    zkassert(gates[r].op == Gate::op_unknown);

    // If the resulting value will exceed the max carry, perform a normalized XOR
    if (gates[a].value+gates[b].value>=(1<<(MAX_CARRY_BITS+1)))
    {
        return XORN(a, b, r);
    }

    // r=XOR(a,b)
    bits[r] = bits[a]^bits[b];
    xors++;

    // Increase the operands fan-out counters and add r to their connections
    gates[a].fanOut++;
    gates[a].connectionsToA.push_back(r);
    gates[b].fanOut++;
    gates[b].connectionsToB.push_back(r);

    // Update gate type and connections
    gates[r].op = Gate::op_xor;
    gates[r].a = a;
    gates[r].b = b;
    gates[r].r = r;
    gates[r].value = gates[a].value + gates[b].value;
    gates[r].maxValue = zkmax(gates[r].value, gates[r].maxValue);
    totalMaxValue = zkmax(gates[r].maxValue, totalMaxValue);

    // Add this gate to the chronological list of operations
    evals.push_back(&gates[r]);
}

// XORN operation: r = XOR(a,b), r.value = 1
void KeccakSMState::XORN ( uint64_t a, uint64_t b, uint64_t r)
{
    zkassert(a<maxRefs);
    zkassert(b<maxRefs);
    zkassert(r<maxRefs);
    zkassert(bits[a]<=1);
    zkassert(bits[b]<=1);
    zkassert(bits[r]<=1);
    zkassert(gates[r].op == Gate::op_unknown);

    // r=XOR(a,b)
    bits[r] = bits[a]^bits[b];
    xorns++;

    // Increase the operands fan-out counters and add r to their connections
    gates[a].fanOut++;
    gates[a].connectionsToA.push_back(r);
    gates[b].fanOut++;
    gates[b].connectionsToB.push_back(r);
    
    // Update gate type and connections
    gates[r].op = Gate::op_xorn;
    gates[r].a = a;
    gates[r].b = b;
    gates[r].r = r;
    gates[r].value = 1;

    // Add this gate to the chronological list of operations
    evals.push_back(&gates[r]);
}

// ANDP operation: r = AND( NOT(a), b), r.value = 1
void KeccakSMState::ANDP ( uint64_t a, uint64_t b, uint64_t r)
{
    zkassert(a<maxRefs);
    zkassert(b<maxRefs);
    zkassert(r<maxRefs);
    zkassert(bits[a]<=1);
    zkassert(bits[b]<=1);
    zkassert(bits[r]<=1);
    zkassert(gates[r].op == Gate::op_unknown);

    // r=AND(NOT(a),b)
    bits[r] = (1-bits[a])&bits[b];
    ands++;
    
    // Increase the operands fan-out counters and add r to their connections
    gates[a].fanOut++;
    gates[a].connectionsToA.push_back(r);
    gates[b].fanOut++;
    gates[b].connectionsToB.push_back(r);
    
    // Update gate type and connections
    gates[r].op = Gate::op_andp;
    gates[r].a = a;
    gates[r].b = b;
    gates[r].r = r;
    gates[r].value = 1;

    // Add this gate to the chronological list of operations
    evals.push_back(&gates[r]);
}

// Print statistics, for development purposes
void KeccakSMState::printCounters (void)
{
    cout << "Max carry bits=" << MAX_CARRY_BITS << endl;
    cout << "#xors=" << to_string(xors) << endl;
    cout << "#ands=" << to_string(ands) << endl;
    cout << "#xorns=" << to_string(xorns) << endl;
    cout << "nextRef=" << to_string(nextRef) << endl;
    cout << "totalMaxValue=" << to_string(totalMaxValue) << endl;
}

// Refs must be an array of 1600 bits
void KeccakSMState::printRefs (uint64_t * pRefs, string name)
{
    // Get a local copy of the 1600 bits by reference
    uint8_t aux[1600];
    for (uint64_t i=0; i<1600; i++)
    {
        aux[i] = bits[pRefs[i]];
    }

    // Print the bits
    printBits(aux, 1600, name);
}

// Map an operation code into a string
string KeccakSMState::op2string (Gate::Operation op)
{
    switch (op)
    {
        case Gate::op_xor:
            return "xor";
        case Gate::op_andp:
            return "andp";
        case Gate::op_xorn:
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
        evaluations[i] = evalJson;
    }
    j["evaluations"] = evaluations;

    // In order of position, add the gates data
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
        string connections;
        for (uint64_t j=0; j<gates[i].connectionsToA.size(); j++)
        {
            if (connections.size()!=0) connections +=",";
            connections += "A[" + to_string(gates[i].connectionsToA[j]) + "]";
        }
        for (uint64_t j=0; j<gates[i].connectionsToB.size(); j++)
        {
            if (connections.size()!=0) connections +=",";
            connections += "B[" + to_string(gates[i].connectionsToB[j]) + "]";
        }
        gateJson["connections"] = connections;
        gatesJson[i] = gateJson;
    }
    j["gates"] = gatesJson;

    // Add the output state references list
    json soutRefs;
    for (uint64_t i=0; i<1600; i++)
    {
        soutRefs[i] = SoutRefs[i];
    }
    j["soutRefs"] = soutRefs;

    // Add counters
    j["maxRef"] = nextRef-1;
    j["xors"] = xors;
    j["andps"] = ands;
    j["maxValue"] = totalMaxValue;
}

// Generate a JSON object containing all a, b, r, and op polynomials values
void KeccakSMState::savePolsToJson (json &j)
{
    // Generate polynomials
    json polA;
    json polB;
    json polR;
    json polOp;

    // Initialize all polynomials to the corresponding default values, without permutations
    for (uint64_t i=0; i<nextRef; i++)
    {
        polA[i] = 1000000 + i; // TODO: fe = roots of unity: a, aa, aaa, aaaa ... 2^23
        polB[i] = 2000000 + i; // SinRinSout SinRinSout ... les intermitges ... XOR(0,0) sense connectar  k1a, k1aa, ...
        polR[i] = 3000000 + i; // k2a, k2aa, ...
        polOp[i] = gates[i].op;
    }

    // Perform the polynomials permutations by rotating all inter-connected connections
    for (uint64_t i=0; i<nextRef; i++)
    {
        uint64_t aux = polR[i];
        for (uint64_t j=0; j<gates[i].connectionsToA.size(); j++)
        {
            uint64_t aux2 = polA[gates[i].connectionsToA[j]];
            polA[gates[i].connectionsToA[j]] = aux;
            aux = aux2;
        }
        for (uint64_t j=0; j<gates[i].connectionsToB.size(); j++)
        {
            uint64_t aux2 = polB[gates[i].connectionsToB[j]];
            polB[gates[i].connectionsToB[j]] = aux;
            aux = aux2;
        }
        polR[i] = aux;
    }

    // Create the JSON object structure
    json pols;
    pols["a"] = polA;
    pols["b"] = polB;
    pols["r"] = polR;
    pols["op"] = polOp;
    j["pols"] = pols;
}