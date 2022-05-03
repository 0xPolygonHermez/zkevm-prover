#include "keccak_executor.hpp"
#include "utils.hpp"

void KeccakExecutor::loadScript (json j)
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

    zkassert(j["maxRef"] == Keccak_SlotSize);

    bLoaded = true;
}

void KeccakExecutor::execute (KeccakState &S)
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
            cerr << "Error: KeccakExecutor::execute() found invalid op: " << program[i].op << " in evaluation: " << i << endl;
            exit(-1);
        }
    }
}

void KeccakExecutor::execute (uint8_t * bit)
{
    zkassert(bLoaded);

    // Allocate the gate array
    Gate *gate;
    gate = new Gate[Keccak_PolLength];
    if (gate == NULL)
    {
        cout << "Error: KeccakExecutor::execute() failed calling malloc" << endl;
        exit(-1);
    }

    // Init the array
    gate[ZeroRef].pin[pin_a].bit = 0;
    gate[ZeroRef].pin[pin_b].bit = 1;
    for (uint64_t slot=0; slot<Keccak_NumberOfSlots; slot++)
    {
        for (uint64_t i=0; i<1088; i++)
        {
            gate[1 + slot*3200 + i].pin[pin_b].bit = bit[1 + slot*3200 + i];
        }
    }

    // Execute the program
    for (uint64_t slot=0; slot<Keccak_NumberOfSlots; slot++)
    {
        for (uint64_t i=0; i<program.size(); i++)
        {
            uint64_t absRefa = relRef2AbsRef(program[i].refa, slot);
            uint64_t absRefb = relRef2AbsRef(program[i].refb, slot);
            uint64_t absRefr = relRef2AbsRef(program[i].refr, slot);

            if ( (program[i].op == gop_xor) ||
                 (program[i].op == gop_xorn) )
            {
                gate[absRefr].pin[pin_r].bit = gate[absRefa].pin[program[i].pina].bit ^ gate[absRefb].pin[program[i].pinb].bit;
            }
            else if (program[i].op == gop_andp)
            {
                gate[absRefr].pin[pin_r].bit = ( 1 - gate[absRefa].pin[program[i].pina].bit ) & gate[absRefb].pin[program[i].pinb].bit;
            }
            else
            {
                cerr << "Error: KeccakExecutor::execute() found invalid op: " << program[i].op << " in evaluation: " << i << endl;
                exit(-1);
            }
        }
    }

    // Copy Sout
    for (uint64_t slot=0; slot<Keccak_NumberOfSlots; slot++)
    {
        for (uint64_t i=0; i<1600; i++)
        {
            bit[1 + slot*3200 + 1600 + i] = gate[1 + slot*3200 + 1600 + i].pin[pin_r].bit;
        }
    }

    delete [] gate;
}

void KeccakExecutor::execute (KeccakExecuteInput &input, KeccakExecuteOutput &output)
{
    // Reset polynomials
    memset(output.pol, 0 , sizeof(output.pol));

    // Set ZeroRef values
    output.pol[pin_a][ZeroRef] = 0;
    output.pol[pin_b][ZeroRef] = Keccak_Mask;

    // Set Sin and Rin values
    for (uint64_t slot=0; slot<Keccak_NumberOfSlots; slot++)
    {
        uint64_t offset = 1 + slot*3200;
        for (uint64_t row=0; row<9; row++)
        {
            uint64_t mask = uint64_t(1)<<(row*7);
            for (uint64_t i=0; i<1600; i++)
            {
                if (input.Sin[slot][row][i]==1)
                {
                    output.pol[pin_a][offset+i] |= mask;
                }
                if ((i<1088) && (input.Rin[slot][row][i]==1))
                {
                    output.pol[pin_b][offset+i] |= mask;
                }
            }
        }
    }

    // Execute the program
    KeccakInstruction instruction;
    for (uint64_t slot=0; slot<Keccak_NumberOfSlots; slot++)
    {
        for (uint64_t i=0; i<program.size(); i++)
        {
            instruction = program[i];
            uint64_t absRefa = relRef2AbsRef(instruction.refa, slot);
            uint64_t absRefb = relRef2AbsRef(instruction.refb, slot);
            uint64_t absRefr = relRef2AbsRef(instruction.refr, slot);

            switch (program[i].op)
            {
                case gop_xor:
                    output.pol[pin_r][absRefr] = output.pol[instruction.pina][absRefa] + output.pol[instruction.pinb][absRefb];
                    break;

                case gop_xorn:
                    output.pol[pin_r][absRefr] = (output.pol[instruction.pina][absRefa] ^ output.pol[instruction.pinb][absRefb]) & Keccak_Mask;
                    break;

                case gop_andp:
                    output.pol[pin_r][absRefr] = ((~output.pol[instruction.pina][absRefa] ) & output.pol[instruction.pinb][absRefb]) & Keccak_Mask;
                    break;

                default:
                    cerr << "Error: KeccakSMExecutor::execute() found invalid op: " << program[i].op << " in evaluation: " << i << endl;
                    exit(-1);
            }
        }
    }
}

void KeccakExecutor::Keccak (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput)
{
    Keccak2Input input;
    input.init(pInput, inputSize);
    KeccakState S;

    uint8_t r[1088];
    while (input.getNextBits(r))
    {
        S.setRin(r);
        execute(S);
        S.copySoutToSinAndResetRefs();
    }
    S.getOutput(pOutput);
}