#include "keccak_sm_executor.hpp"
#include "utils.hpp"

void KeccakSMExecutor::loadScript (json j)
{
    if ( !j.contains("program") ||
         !j["program"].is_array())
    {
        cerr << "KeccakSMExecutor::loadEvals() found JSON object does not contain not a program array" << endl;
        exit(-1);
    }
    for (uint64_t i=0; i<j["program"].size(); i++)
    {
        if ( !j["program"][i].is_object() )
        {
            cerr << "KeccakSMExecutor::loadEvals() found JSON array's element is not an object" << endl;
            exit(-1);
        }
        if ( !j["program"][i].contains("op") ||
             !j["program"][i]["op"].is_string() )
        {
            cerr << "KeccakSMExecutor::loadEvals() found JSON array's element does not contain string op field" << endl;
            exit(-1);
        }
        if ( !j["program"][i].contains("refa") ||
             !j["program"][i]["refa"].is_number_unsigned() ||
              j["program"][i]["refa"]>=maxRefs )
        {
            cerr << "KeccakSMExecutor::loadEvals() found JSON array's element does not contain unsigned number refa field" << endl;
            exit(-1);
        }
        if ( !j["program"][i].contains("refb") ||
             !j["program"][i]["refb"].is_number_unsigned() ||
              j["program"][i]["refb"]>=maxRefs )
        {
            cerr << "KeccakSMExecutor::loadEvals() found JSON array's element does not contain unsigned number refb field" << endl;
            exit(-1);
        }
        if ( !j["program"][i].contains("refr") ||
             !j["program"][i]["refr"].is_number_unsigned() ||
              j["program"][i]["refr"]>=maxRefs )
        {
            cerr << "KeccakSMExecutor::loadEvals() found JSON array's element does not contain unsigned number refr field" << endl;
            exit(-1);
        }
        if ( !j["program"][i].contains("pina") ||
             !j["program"][i]["pina"].is_number_unsigned() ||
              j["program"][i]["pina"]>2 )
        {
            cerr << "KeccakSMExecutor::loadEvals() found JSON array's element does not contain unsigned number pina field" << endl;
            exit(-1);
        }
        if ( !j["program"][i].contains("pinb") ||
             !j["program"][i]["pinb"].is_number_unsigned() ||
              j["program"][i]["pinb"]>2 )
        {
            cerr << "KeccakSMExecutor::loadEvals() found JSON array's element does not contain unsigned number pinb field" << endl;
            exit(-1);
        }
        KeccakInstruction instruction;
        Gate gate;
        if (j["program"][i]["op"] == "xor")
        {
            gate.op = gop_xor;
            instruction.op = gop_xor;
        }
        else if (j["program"][i]["op"] == "andp")
        {
            gate.op = gop_andp;
            instruction.op = gop_andp;
        }
        else if (j["program"][i]["op"] == "xorn")
        {
            gate.op = gop_xorn;
            instruction.op = gop_xorn;
        }
        else
        {
            cerr << "KeccakSMExecutor::loadEvals() found invalid op value: " << j[i]["op"] << endl;
            exit(-1);
        }
        instruction.refa = j["program"][i]["refa"];
        instruction.refb = j["program"][i]["refb"];
        instruction.refr = j["program"][i]["refr"];
        instruction.pina = j["program"][i]["pina"];
        instruction.pinb = j["program"][i]["pinb"];
        program.push_back(instruction);
    }

    slotSize = j["maxRef"];

    bLoaded = true;
}

void KeccakSMExecutor::execute (KeccakSMState &S)
{
    zkassert(bLoaded);

    for (uint64_t i=0; i<program.size(); i++)
    {
        if ( (program[i].op == gop_xor) ||
             (program[i].op == gop_xorn) )
        {
            S.gate[program[i].refr].pin[pin_r].bit = 
            S.gate[program[i].refa].pin[program[i].pina].bit ^
            S.gate[program[i].refb].pin[program[i].pinb].bit;
        }
        else if (program[i].op == gop_andp)
        {
            S.gate[program[i].refr].pin[pin_r].bit =
            ( 1 - S.gate[program[i].refa].pin[program[i].pina].bit ) &
            S.gate[program[i].refb].pin[program[i].pinb].bit;
        }
        else
        {
            cerr << "Error: KeccakSMExecutor::execute() found invalid op: " << program[i].op << " in evaluation: " << i << endl;
            exit(-1);
        }
    }
}

void KeccakSMExecutor::execute (uint8_t * bit)
{
    zkassert(bLoaded);

    // Allocate the gate array
    uint64_t length = 1<<arity;
    Gate *gate;
    gate = new Gate[length];
    if (gate == NULL)
    {
        cout << "Error: KeccakSMExecutor::execute() failed calling malloc" << endl;
        exit(-1);
    }

    // Init the array
    gate[ZeroRef].pin[pin_a].bit = 0;
    gate[ZeroRef].pin[pin_b].bit = 1;
    for (uint64_t slot=0; slot<54; slot++)
    {
        for (uint64_t i=0; i<1088; i++)
        {
            gate[1 + slot*3200 + i].pin[pin_b].bit = bit[1 + slot*3200 + i];
        }
    }

    // Calculate the number of slots
    uint64_t numberOfSlots = (length-1) / slotSize;

    // Do the work
    for (uint64_t slot=0; slot<numberOfSlots; slot++)
    {
        for (uint64_t i=0; i<program.size(); i++)
        {
            uint64_t absRefa = relRef2AbsRef(program[i].refa, slot, numberOfSlots, slotSize);
            uint64_t absRefb = relRef2AbsRef(program[i].refb, slot, numberOfSlots, slotSize);
            uint64_t absRefr = relRef2AbsRef(program[i].refr, slot, numberOfSlots, slotSize);

            if ( (program[i].op == gop_xor) ||
                 (program[i].op == gop_xorn) )
            {
                gate[absRefr].pin[pin_r].bit = 
                gate[absRefa].pin[program[i].pina].bit ^
                gate[absRefb].pin[program[i].pinb].bit;
            }
            else if (program[i].op == gop_andp)
            {
                gate[absRefr].pin[pin_r].bit =
                ( 1 - gate[absRefa].pin[program[i].pina].bit ) &
                gate[absRefb].pin[program[i].pinb].bit;
            }
            else
            {
                cerr << "Error: KeccakSMExecutor::execute() found invalid op: " << program[i].op << " in evaluation: " << i << endl;
                exit(-1);
            }
        }
    }

    // Copy Sout
    for (uint64_t slot=0; slot<numberOfSlots; slot++)
    {
        for (uint64_t i=0; i<1600; i++)
        {
            bit[1 + slot*3200 + 1600 + i] = gate[1 + slot*3200 + 1600 + i].pin[pin_r].bit;
        }
    }

    delete [] gate;
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
        execute(S);
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

    cout << "Starting 54-slots testing..." << endl;
    uint8_t Sin[54][136];
    uint8_t hashSout[54][32];
    string hashString[54];

    // Init Sin and hashString
    for (uint64_t slot=0; slot<54; slot++)
    {
        for (uint64_t i=0; i<136; i++)
        {
            Sin[slot][i] = rand();
        }
        Sin[slot][135] = 0b10000001;
        hashString[slot] = keccak256(&Sin[slot][0], 135);
    }

    uint8_t *bit;
    uint64_t length = 1<<arity;
    bit = (uint8_t *)malloc(length);
    if (bit==NULL)
    {
        cerr << "ERROR: KeccakSMExecutorTest() failed calling malloc of length:" << length << endl;
        exit(-1);
    }
    memset(bit, 0, length);
    for (uint64_t slot=0; slot<54; slot++)
    {
        for (uint64_t i=0; i<136; i++)
        {
            byte2bits(Sin[slot][i], &bit[1 + slot*3200 + i*8]);
        }
    }
    executor.execute(bit);
    for (uint64_t slot=0; slot<54; slot++)
    {
        for (uint64_t i=0; i<32; i++)
        {
            bits2byte(&bit[1 + slot*3200 + 1600 + i*8], hashSout[slot][i]);
        }
        printBa(&hashSout[slot][0], 32, "Sout"+to_string(slot));
        cout << "Hash" << slot << " = " << hashString[slot] << endl;
    }

    free(bit);

}
