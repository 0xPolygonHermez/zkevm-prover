#include "keccak_sm_executor.hpp"
#include "utils.hpp"

void KeccakSMExecutor::loadScript (json j)
{
    if ( !j.contains("evaluations") ||
            !j["evaluations"].is_array())
    {
        cerr << "KeccakSMExecutor::loadEvals() found JSON object does not contain not an evaluations array" << endl;
        exit(-1);
    }
    for (uint64_t i=0; i<j["evaluations"].size(); i++)
    {
        if ( !j["evaluations"][i].is_object() )
        {
            cerr << "KeccakSMExecutor::loadEvals() found JSON array's element is not an object" << endl;
            exit(-1);
        }
        if ( !j["evaluations"][i].contains("op") ||
                !j["evaluations"][i]["op"].is_string() )
        {
            cerr << "KeccakSMExecutor::loadEvals() found JSON array's element does not contain string op field" << endl;
            exit(-1);
        }
        if ( !j["evaluations"][i].contains("a") ||
                !j["evaluations"][i]["a"].is_number_unsigned() )
        {
            cerr << "KeccakSMExecutor::loadEvals() found JSON array's element does not contain unsigned number a field" << endl;
            exit(-1);
        }
        if ( !j["evaluations"][i].contains("b") ||
                !j["evaluations"][i]["b"].is_number_unsigned() )
        {
            cerr << "KeccakSMExecutor::loadEvals() found JSON array's element does not contain unsigned number b field" << endl;
            exit(-1);
        }
        if ( !j["evaluations"][i].contains("r") ||
                !j["evaluations"][i]["r"].is_number_unsigned() )
        {
            cerr << "KeccakSMExecutor::loadEvals() found JSON array's element does not contain unsigned number r field" << endl;
            exit(-1);
        }
        Gate gate;
        if (j["evaluations"][i]["op"] == "xor")
        {
            gate.op = OP_XOR;
        }
        else if (j["evaluations"][i]["op"] == "andp")
        {
            gate.op = OP_ANDP;
        }
        else if (j["evaluations"][i]["op"] == "xorn")
        {
            gate.op = OP_XORN;
        }
        else
        {
            cerr << "KeccakSMExecutor::loadEvals() found invalid op value: " << j[i]["op"] << endl;
            exit(-1);
        }
        gate.a = j["evaluations"][i]["a"];
        gate.b = j["evaluations"][i]["b"];
        gate.r = j["evaluations"][i]["r"];
        evals.push_back(gate);
    }

    if ( !j.contains("soutRefs") ||
            !j["soutRefs"].is_array() || 
            ( j["soutRefs"].size() != 1600 ) ||
            !j["soutRefs"][0].is_number_unsigned() )
    {
        cerr << "KeccakSMExecutor::loadEvals() found JSON object does not contain not an 1600 evaluations array" << endl;
        exit(-1);
    }
    for (uint64_t i=0; i<1600; i++)
    {
        SoutRefs[i] = j["soutRefs"][i];
    }

    bLoaded = true;
}

void KeccakSMExecutor::execute (uint8_t * bits)
{
    zkassert(bLoaded);

    for (uint64_t i=0; i<evals.size(); i++)
    {
        if ( (evals[i].op == OP_XOR) ||
             (evals[i].op == OP_XORN) )
        {
            bits[evals[i].r] = bits[evals[i].a]^bits[evals[i].b];
        }
        else if (evals[i].op == OP_ANDP)
        {
            bits[evals[i].r] = (1-bits[evals[i].a])&bits[evals[i].b];
        }
        else
        {
            cerr << "Error: KeccakSMExecutor::execute() found invalid op: " << evals[i].op << " in evaluation: " << i << endl;
            exit(-1);
        }
    }
}

void KeccakSMExecutor::KeccakSM (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput)
{
    Keccak2Input input;
    input.init(pInput, inputSize);
    KeccakSMState S;

    uint8_t r[1088];
    while (input.getNextBits(r))
    {
        S.setRin(r);
        execute(S.bits);
        for (uint64_t i=0; i<1600; i++)
        {
            S.SoutRefs[i] = SoutRefs[i];
        }
        S.copySoutToSinAndResetRefs();
    }
    S.getOutput(pOutput);
}

void KeccakSMExecutorTest (const Config &config)
{
    cout << "KeccakSMExecutorTest() starting" << endl;

    KeccakSMExecutor executor(config);
    json j;
    file2json(config.keccakScriptFile, j);
    executor.loadScript(j);
    
    /* Use a well-known input */
    uint8_t input[188] = {
        0x09, 0x0B, 0xCA, 0xF7, 0x34, 0xC4, 0xF0, 0x6C, 0x93, 0x95,
        0x4A, 0x82, 0x7B, 0x45, 0xA6, 0xE8, 0xC6, 0x7B, 0x8E, 0x0F, 
        0xD1, 0xE0, 0xA3, 0x5A, 0x1C, 0x59, 0x82, 0xD6, 0x96, 0x18, 
        0x28, 0xF9, 0x09, 0x0B, 0xCA, 0xF7, 0x34, 0xC4, 0xF0, 0x6C, 
        0x93, 0x95, 0x4A, 0x82, 0x7B, 0x45, 0xA6, 0xE8, 0xC6, 0x7B, 

        0x8E, 0x0F, 0xD1, 0xE0, 0xA3, 0x5A, 0x1C, 0x59, 0x82, 0xD6, 
        0x96, 0x18, 0x28, 0xF9, 0x09, 0x0B, 0xCA, 0xF7, 0x34, 0xC4, 
        0xF0, 0x6C, 0x93, 0x95, 0x4A, 0x82, 0x7B, 0x45, 0xA6, 0xE8, 
        0xC6, 0x7B, 0x8E, 0x0F, 0xD1, 0xE0, 0xA3, 0x5A, 0x1C, 0x59, 
        0x82, 0xD6, 0x96, 0x18, 0x28, 0xF9, 0x17, 0xC0, 0x4C, 0x37, 

        0x60, 0x51, 0x0B, 0x48, 0xC6, 0x01, 0x27, 0x42, 0xC5, 0x40, 
        0xA8, 0x1A, 0xBA, 0x4B, 0xCA, 0x2F, 0x78, 0xB9, 0xD1, 0x4B, 
        0xFD, 0x2F, 0x12, 0x3E, 0x2E, 0x53, 0xEA, 0x3E, 0x61, 0x7B, 
        0x3A, 0x35, 0x28, 0xF9, 0xCD, 0xD6,   0x63, 0x0F, 0xD3, 0x30, 
        0x1B, 0x9C, 0x89, 0x11, 0xF7, 0xBF, 0x06, 0x3D, 0x29, 0x90,

        0x27, 0xCC, 0x1E, 0xE6, 0x56, 0x7E, 0x0F, 0xE5, 0xD6, 0x64, 
        0x87, 0x11, 0x82, 0xE4, 0xC6, 0xEA, 0xDA, 0xE6, 0x1A, 0x17, 
        0x06, 0xD8, 0x6D, 0x27, 0x32, 0x1A, 0xC3, 0x24, 0x6F, 0x98, 
        0x00, 0x00, 0x03, 0xE9, 0x00, 0x00, 0x00, 0x01};

    uint64_t inputSize = 188; // 188

    /* Call Keccak to get the hash of the input */
    TimerStart(KECCAK_SM_EXECUTOR);
    uint8_t hash[32];
    executor.KeccakSM(input, inputSize, hash);
    TimerStopAndLog(KECCAK_SM_EXECUTOR);
    printBa(hash, 32, "hash");    // Expected result: hash:0x1AFD6EAF13538380D99A245C2ACC4A25481B54556AE080CF07D1FACC0638CD8E

    /* Call the current Keccak to compare */
    TimerStart(CURRENT_KECCAK);
    string aux = keccak256(input, inputSize);
    TimerStopAndLog(CURRENT_KECCAK);
    cout << "Current Keccak: " << aux << endl;
}
