#include "keccak_f_executor.hpp"
#include "utils.hpp"

void KeccakFExecutor::loadScript (json j)
{
    if ( !j.contains("program") ||
         !j["program"].is_array())
    {
        cerr << "KeccakFExecutor::loadEvals() found JSON object does not contain not a program array" << endl;
        exit(-1);
    }
    for (uint64_t i=0; i<j["program"].size(); i++)
    {
        if ( !j["program"][i].is_object() )
        {
            cerr << "KeccakFExecutor::loadEvals() found JSON array's element is not an object" << endl;
            exit(-1);
        }
        if ( !j["program"][i].contains("op") ||
             !j["program"][i]["op"].is_string() )
        {
            cerr << "KeccakFExecutor::loadEvals() found JSON array's element does not contain string op field" << endl;
            exit(-1);
        }
        if ( !j["program"][i].contains("ref") ||
             !j["program"][i]["ref"].is_number_unsigned() ||
              j["program"][i]["ref"]>=maxRefs )
        {
            cerr << "KeccakFExecutor::loadEvals() found JSON array's element does not contain unsigned number ref field" << endl;
            exit(-1);
        }
        if ( !j["program"][i].contains("a") ||
             !j["program"][i]["a"].is_object() )
        {
            cerr << "KeccakFExecutor::loadEvals() found JSON array's element does not contain object a field" << endl;
            exit(-1);
        }
        if ( !j["program"][i].contains("b") ||
             !j["program"][i]["b"].is_object() )
        {
            cerr << "KeccakFExecutor::loadEvals() found JSON array's element does not contain object b field" << endl;
            exit(-1);
        }
        if ( !j["program"][i]["a"].contains("type") ||
             !j["program"][i]["a"]["type"].is_string() )
        {
            cerr << "KeccakFExecutor::loadEvals() found JSON array's element does not contain string a type field" << endl;
            exit(-1);
        }
        if ( !j["program"][i]["b"].contains("type") ||
             !j["program"][i]["b"]["type"].is_string() )
        {
            cerr << "KeccakFExecutor::loadEvals() found JSON array's element does not contain string b type field" << endl;
            exit(-1);
        }
        
        KeccakInstruction instruction;

        // Get gate operation and reference
        if (j["program"][i]["op"] == "xor")
        {
            instruction.op = gop_xor;
        }
        else if (j["program"][i]["op"] == "andp")
        {
            instruction.op = gop_andp;
        }
        else if (j["program"][i]["op"] == "xorn")
        {
            instruction.op = gop_xorn;
        }
        else
        {
            cerr << "Error: KeccakFExecutor::loadEvals() found invalid op value: " << j[i]["op"] << endl;
            exit(-1);
        }
        instruction.refr = j["program"][i]["ref"];

        // Get input a pin data
        string typea = j["program"][i]["a"]["type"];
        if (typea=="wired")
        {
            instruction.refa = j["program"][i]["a"]["gate"];
            string pina = j["program"][i]["a"]["pin"];
            instruction.pina = string2pin(pina);
        }
        else if (typea=="input")
        {
            uint64_t bit = j["program"][i]["a"]["bit"];
            instruction.refa = SinRef0 + bit*9;
            instruction.pina = PinId::pin_a;
        }
        else
        {
            cerr << "Error: KeccakFExecutor::loadEvals() found invalid a type value: " << typea << endl;
            exit(-1);
        }
        
        // Get input b pin data
        string typeb = j["program"][i]["b"]["type"];
        if (typeb=="wired")
        {
            instruction.refb = j["program"][i]["b"]["gate"];
            string pinb = j["program"][i]["b"]["pin"];
            instruction.pinb = string2pin(pinb);
        }
        else if (typeb=="input")
        {
            uint64_t bit = j["program"][i]["b"]["bit"];
            instruction.refb = SinRef0 + bit*9;
            instruction.pinb = PinId::pin_a;
        }
        else
        {
            cerr << "Error: KeccakFExecutor::loadEvals() found invalid b type value: " << typeb << endl;
            exit(-1);
        }
        
        program.push_back(instruction);
    }

    zkassert(j["maxRef"] == Keccak_SlotSize);

    bLoaded = true;
}

void KeccakFExecutor::execute (KeccakState &S)
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
            cerr << "Error: KeccakFExecutor::execute() found invalid op: " << program[i].op << " in evaluation: " << i << endl;
            exit(-1);
        }
    }
}

void KeccakFExecutor::execute (uint8_t * bit)
{
    zkassert(bLoaded);

    // Allocate the gate array
    Gate *gate;
    gate = new Gate[Keccak_PolLength];
    if (gate == NULL)
    {
        cout << "Error: KeccakFExecutor::execute() failed calling malloc" << endl;
        exit(-1);
    }

    // Init the array
    gate[ZeroRef].pin[pin_a].bit = 0;
    gate[ZeroRef].pin[pin_b].bit = 1;
    for (uint64_t slot=0; slot<Keccak_NumberOfSlots; slot++)
    {
        for (uint64_t i=0; i<1088; i++)
        {
            gate[relRef2AbsRef(SinRef0+i*9, slot)].pin[pin_a].bit = bit[relRef2AbsRef(SinRef0+i*9, slot)];
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

            /*if (program[i].refr==(3200*9+1) || program[i].refr==((3200*9)+2) || program[i].refr==1 || program[i].refr==Keccak_SlotSize )
            {
                cout << "slot=" << slot << " i=" << i << "/" << program.size() << " refa=" << program[i].refa << " absRefa=" << absRefa << " refb=" << program[i].refb << " absRefb=" << absRefb << " refr=" << program[i].refr << " absRefr=" << absRefr << endl;
            }*/

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
                cerr << "Error: KeccakFExecutor::execute() found invalid op: " << program[i].op << " in evaluation: " << i << endl;
                exit(-1);
            }
        }
    }

    // Copy Sout
    for (uint64_t slot=0; slot<Keccak_NumberOfSlots; slot++)
    {
        for (uint64_t i=0; i<1600; i++)
        {
            bit[relRef2AbsRef(SoutRef0+i*9, slot)] = gate[relRef2AbsRef(SoutRef0+i*9, slot)].pin[pin_r].bit;
        }
    }

    delete [] gate;
}

void KeccakFExecutor::execute (KeccakFExecuteInput &input, KeccakFExecuteOutput &output)
{
    // Reset polynomials
    memset(output.pol, 0 , sizeof(output.pol));

    // Set ZeroRef values
    output.pol[pin_a][ZeroRef] = 0;
    output.pol[pin_b][ZeroRef] = Keccak_Mask;

    // Set Sin and Rin values
    for (uint64_t slot=0; slot<Keccak_NumberOfSlots; slot++)
    {
        for (uint64_t row=0; row<9; row++)
        {
            uint64_t mask = uint64_t(1)<<(row*7);
            for (uint64_t i=0; i<1600; i++)
            {
                if (input.Sin[slot][row][i]==1)
                {
                    output.pol[pin_a][relRef2AbsRef(SinRef0+i*9, slot)] |= mask;
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

            output.pol[pin_a][absRefr] = output.pol[instruction.pina][absRefa];
            output.pol[pin_b][absRefr] = output.pol[instruction.pinb][absRefb];

            /*if (instruction.refr==(3200*9+1) || instruction.refr==((3200*9)+2) || instruction.refr==1 || instruction.refr==Keccak_SlotSize)
            {
                cout << "slot=" << slot << " i=" << i << "/" << program.size() << " refa=" << instruction.refa << " absRefa=" << absRefa << " refb=" << instruction.refb << " absRefb=" << absRefb << " refr=" << instruction.refr << " absRefr=" << absRefr << endl;
            }*/

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
                    cerr << "Error: KeccakFExecutor::execute() found invalid op: " << program[i].op << " in evaluation: " << i << endl;
                    exit(-1);
            }
        }
    }
}

/* Input is fe[54][1600], output is KeccakPols */
void KeccakFExecutor::execute (const FieldElement *input, const uint64_t inputLength, KeccakFCommitPols &pols)
{
    uint64_t numberOfSlots = (pols.degree()-1)/Keccak_SlotSize;
    if (inputLength != numberOfSlots*1600)
    {
        cerr << "Error: KeccakFExecutor::execute() got input size=" << inputLength << " different from numberOfSlots=" << numberOfSlots << "x1600" << endl;
        exit(-1);
    }
    vector<vector<FieldElement>> inputVector;
    for (uint64_t slot=0; slot<numberOfSlots; slot++)
    {
        vector<FieldElement> aux;
        for (uint64_t i=0; i<1600; i++)
        {
            aux.push_back(input[slot*1600+i]);
        }
        inputVector.push_back(aux);
    }
    vector<NormGate9ExecutorInput> required;
    execute(inputVector, pols, required);
}

/* Input is a vector of numberOfSlots*1600 fe, output is KeccakPols */
void KeccakFExecutor::execute (const vector<vector<FieldElement>> &input, KeccakFCommitPols &pols, vector<NormGate9ExecutorInput> &required)
{
    uint64_t numberOfSlots = (pols.degree()-1)/Keccak_SlotSize;
    if (input.size() != numberOfSlots)
    {
        cerr << "Error: KeccakFExecutor::execute() got input size=" << input.size() << " different from numberOfSlots=" << numberOfSlots << endl;
        exit(-1);
    }
    for (uint64_t i=0; i<numberOfSlots; i++)
    {
        if (input[i].size() != 1600)
        {
            cerr << "Error: KeccakFExecutor::execute() got input i=" << i << " size=" << input[i].size() << " different from 1600" << endl;
            exit(-1);
        }
    }

    // Set ZeroRef values
    pols.a[ZeroRef] = 0;
    pols.b[ZeroRef] = Keccak_Mask;

    // Set Sin values
    for (uint64_t slot=0; slot<numberOfSlots; slot++)
    {
        for (uint64_t i=0; i<1600; i++)
        {
            pols.a[relRef2AbsRef(SinRef0 + i*9, slot)] = input[slot][i];
        }
    }

    // Execute the program
    KeccakInstruction instruction;
    for (uint64_t slot=0; slot<numberOfSlots; slot++)
    {
        for (uint64_t i=0; i<program.size(); i++)
        {
            instruction = program[i];
            uint64_t absRefa = relRef2AbsRef(instruction.refa, slot);
            uint64_t absRefb = relRef2AbsRef(instruction.refb, slot);
            uint64_t absRefr = relRef2AbsRef(instruction.refr, slot);
            
            switch (instruction.pina)
            {
                case pin_a:
                    pols.a[absRefr] = pols.a[absRefa];
                    break;
                case pin_b:
                    pols.a[absRefr] = pols.b[absRefa];
                    break;
                case pin_r:
                    pols.a[absRefr] = pols.c[absRefa];
                    break;
                default:
                    cerr << "Error: KeccakFExecutor() found invalid instruction.pina=" << instruction.pina << endl;
                    exit(-1);
            }
            switch (instruction.pinb)
            {
                case pin_a:
                    pols.b[absRefr] = pols.a[absRefb];
                    break;
                case pin_b:
                    pols.b[absRefr] = pols.b[absRefb];
                    break;
                case pin_r:
                    pols.b[absRefr] = pols.c[absRefb];
                    break;
                default:
                    cerr << "Error: KeccakFExecutor() found invalid instruction.pinb=" << instruction.pinb << endl;
                    exit(-1);
            }

            /*if (instruction.refr==(3200*9+1) || instruction.refr==((3200*9)+2) || instruction.refr==1 || instruction.refr==Keccak_SlotSize)
            {
                cout << "slot=" << slot << " i=" << i << "/" << program.size() << " refa=" << instruction.refa << " absRefa=" << absRefa << " refb=" << instruction.refb << " absRefb=" << absRefb << " refr=" << instruction.refr << " absRefr=" << absRefr << endl;
            }*/

            switch (program[i].op)
            {
                case gop_xor:
                {
                    pols.c[absRefr] = pols.a[absRefr] + pols.b[absRefr];
                    break;
                }
                case gop_xorn:
                {
                    pols.c[absRefr] = (pols.a[absRefr] ^ pols.b[absRefr]) & Keccak_Mask;
                    NormGate9ExecutorInput norm;
                    norm.type = 0;
                    norm.a = pols.a[absRefr];
                    norm.b = pols.b[absRefr];
                    required.push_back(norm);
                    break;
                }
                case gop_andp:
                {
                    pols.c[absRefr] = ((~pols.a[absRefr]) & pols.b[absRefr]) & Keccak_Mask;
                    NormGate9ExecutorInput norm;
                    norm.type = 1;
                    norm.a = pols.a[absRefr];
                    norm.b = pols.b[absRefr];
                    required.push_back(norm);
                    break;
                }
                default:
                {
                    cerr << "Error: KeccakFExecutor::execute() found invalid op: " << program[i].op << " in evaluation: " << i << endl;
                    exit(-1);
                }
            }
        }
    }

    cout << "KeccakFExecutor successfully processed " << numberOfSlots << " Keccak-F actions" << endl;
}

void KeccakFExecutor::Keccak (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput)
{
    Keccak2Input input;
    input.init(pInput, inputSize);
    KeccakState S;

    uint8_t r[1088];
    while (input.getNextBits(r))
    {
        for (uint64_t i=0; i<1088; i++)
        {
            S.gate[SinRef0 + i*9].pin[pin_a].bit ^= r[i];
        }
        execute(S);
        S.copySoutToSinAndResetRefs();
    }
    S.getOutput(pOutput);
}