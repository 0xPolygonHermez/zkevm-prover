#include "sha256_f_executor.hpp"
#include "utils.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "zklog.hpp"

inline uint64_t ch(uint64_t a, uint64_t b, uint64_t c) {
    return ((a & b) ^ (~a & c));
}

inline uint64_t maj(uint64_t a, uint64_t b, uint64_t c) {
    return ((a & b) ^ (a & c) ^ (b & c));
}

inline uint64_t carry(uint64_t a, uint64_t b, uint64_t c) {
    return (~a & 0xFFFFFFFF & b & c) | (a & b) | (a & c);
}

TypeSha256Gate string2typeSha (string s)
{
    if (s=="wired") return TypeSha256Gate::type_wired;
    if (s=="input") return TypeSha256Gate::type_input;
    if (s=="inputState") return TypeSha256Gate::type_inputState;

    zklog.error("string2typeSha() got an invalid type id string=" + s);
    exitProcess();
    return TypeSha256Gate::type_unknown;
}

uint64_t string2pinSha(string s)
{
    if (s=="in1") return 0;
    if (s=="in2") return 1;
    if (s=="in3") return 2;
    if (s=="out") return 3;

    zklog.error("string2pin() got an invalid pin id string=" + s);
    exitProcess();
    return 0;
}

void Sha256FExecutor::loadScript(json j)
{
    if (!j.contains("program") ||
        !j["program"].is_array())
    {
        zklog.error("Sha256FExecutor::loadEvals() found JSON object does not contain not a program array");
        exitProcess();
    }
    for (uint64_t i = 0; i < j["program"].size(); i++)
    {
        if (!j["program"][i].is_object())
        {
            zklog.error("Sha256FExecutor::loadEvals() found JSON array's element is not an object");
            exitProcess();
        }
        if (!j["program"][i].contains("op") ||
            !j["program"][i]["op"].is_string())
        {
            zklog.error("Sha256FExecutor::loadEvals() found JSON array's element does not contain string op field");
            exitProcess();
        }
        if (!j["program"][i].contains("ref") ||
            !j["program"][i]["ref"].is_number_unsigned() ||
            j["program"][i]["ref"] >= SHA256GateConfig.maxRefs)
        {
            zklog.error("Sha256FExecutor::loadEvals() found JSON array's element does not contain a valid unsigned number ref field");
            exitProcess();
        }
        
        //check in1 object
        if (!j["program"][i].contains("in1") ||
            !j["program"][i]["in1"].is_object() ||
            !j["program"][i]["in1"].contains("type") ||
            !j["program"][i]["in1"]["type"].is_string())
        {
            zklog.error("Sha256FExecutor::loadEvals() found JSON array's element does not contain a valid in1 object as field");
            exitProcess();
        }
        
        //check in2 object
        if (!j["program"][i].contains("in2") ||
            !j["program"][i]["in2"].is_object() ||
            !j["program"][i]["in2"].contains("type"))
        {
            zklog.error("Sha256FExecutor::loadEvals() found JSON array's element does not contain a valid in2 object as field");
            exitProcess();
        }
        //check in3 object
        if(j["program"][i].contains("in3")){
            if (!j["program"][i]["in3"].is_object() ||
                !j["program"][i]["in3"].contains("type"))
            {
                zklog.error("Sha256FExecutor::loadEvals() found JSON array's element does not contain a valid in3 object as field");
                exitProcess();
            }
        }
        
        Sha256Instruction instruction;

        // Get gate operation and reference
        if (j["program"][i]["op"] == "xor")
        {
            instruction.op = GateOperation::gop_xor;
        }
        else if (j["program"][i]["op"] == "ch")
        {
            instruction.op = GateOperation::gop_ch;
        }
        else if (j["program"][i]["op"] == "maj")
        {
            instruction.op = GateOperation::gop_maj;
        }
        else if (j["program"][i]["op"] == "add")
        {
            instruction.op = GateOperation::gop_add;
        }
        else
        {
            string opString = j[i]["op"];
            zklog.error("Sha256FExecutor::loadEvals() found invalid op value: " + opString);
            exitProcess();
        }
        instruction.ref = j["program"][i]["ref"];

        // Get input in1 pin data
        instruction.type[0] = string2typeSha(j["program"][i]["in1"]["type"]);
        instruction.in[0] = true;
        if(j["program"][i]["in1"].contains("bit")) instruction.bit[0] = j["program"][i]["in1"]["bit"];
        if(j["program"][i]["in1"].contains("gate")) instruction.gate[0] = j["program"][i]["in1"]["gate"];
        if(j["program"][i]["in1"].contains("pin"))  instruction.pin[0] = string2pinSha(j["program"][i]["in1"]["pin"]);


        // Get input in2 pin data
        instruction.type[1] = string2typeSha(j["program"][i]["in2"]["type"]);
        instruction.in[1] = true;
        if(j["program"][i]["in2"].contains("bit")) instruction.bit[1] = j["program"][i]["in2"]["bit"];
        if(j["program"][i]["in2"].contains("gate")) instruction.gate[1] = j["program"][i]["in2"]["gate"];
        if(j["program"][i]["in2"].contains("pin"))  instruction.pin[1] = string2pinSha(j["program"][i]["in2"]["pin"]);


        // Get input in3 pin data
        if(j["program"][i].contains("in3"))
        {   
            instruction.in[2] = true;
            instruction.type[2] = string2typeSha(j["program"][i]["in3"]["type"]);
            if(j["program"][i]["in3"].contains("bit")) instruction.bit[2] = j["program"][i]["in3"]["bit"];
            if(j["program"][i]["in3"].contains("gate")) instruction.gate[2] = j["program"][i]["in3"]["gate"];
            if(j["program"][i]["in3"].contains("pin"))  instruction.pin[2] = string2pinSha(j["program"][i]["in3"]["pin"]);

        }
        
        program.push_back(instruction);
    }

    bLoaded = true;
}

void Sha256FExecutor::execute(const vector<Sha256FExecutorInput> &input, Sha256FCommitPols &pols)
{
    zkassertpermanent(bLoaded);

    // Check input size
    if (input.size() != nSlots)
    {
        zklog.error("Sha256FExecutor::execute() got input.size()=" + to_string(input.size()) + " different from numberOfSlots=" + to_string(nSlots));
        exitProcess();
    }

    // Check number of slots, per input
    for (uint64_t i = 0; i < nSlots; i++)
    {
        if (input[i].stIn.size() != 256)
        {
            zklog.error("Sha256FExecutor::execute() got input i=" + to_string(i) + " stIn size=" + to_string(input[i].stIn.size()) + " different from 256");
            exitProcess();
        }
        if (input[i].rIn.size() != 512)
        {
            zklog.error("Sha256FExecutor::execute() got input i=" + to_string(i) + " rIn size=" + to_string(input[i].stIn.size()) + " different from 512");
            exitProcess();
        }
    }

    pols.input[1][0] = fr.fromU64((1 << bitsPerElement) - 1);
    pols.output[0] = fr.fromU64((1 << bitsPerElement) - 1);

    // Execute the program
//#pragma omp parallel for
    for (uint64_t i = 0; i < nSlots; i++)
    {
        uint64_t offset = i * slotSize;
        for (uint64_t j = 0; j < program.size(); j++)
        {
            if(program[j].in[0]) pols.input[0][program[j].ref + offset] = getVal(input, pols, i, j, 0);
            if(program[j].in[1]) pols.input[1][program[j].ref + offset] = getVal(input, pols, i, j, 1);
            if(program[j].in[2]) pols.input[2][program[j].ref + offset] = getVal(input, pols, i, j, 2);
            uint64_t a = fr.toU64(pols.input[0][program[j].ref + offset]);
            uint64_t b = fr.toU64(pols.input[1][program[j].ref + offset]);
            uint64_t c = fr.toU64(pols.input[2][program[j].ref + offset]);
            if (program[j].op == GateOperation::gop_xor) {
                pols.output[program[j].ref + offset] = fr.fromU64(a ^ b ^ c);
            } else if (program[j].op == GateOperation::gop_ch) {
                pols.output[program[j].ref + offset] = fr.fromU64(ch(a, b, c));
            } else if (program[j].op == GateOperation::gop_maj) {
                pols.output[program[j].ref + offset] = fr.fromU64(maj(a, b, c));
            } else if (program[j].op == GateOperation::gop_add) {
                pols.output[program[j].ref + offset] = fr.fromU64(a ^ b ^ c);
                pols.input[2][program[j].ref + offset + 1] = fr.fromU64(carry(a, b, c));
            } else {
                zklog.error("Sha256FExecutor::execute() found invalid op value: " + to_string(program[j].op));
                exitProcess();   
            }           
        }
    }

    zklog.info("Sha256FExecutor successfully processed " + to_string(nSlots) + " Sha256-F actions (" + to_string((double(input.size()) * slotSize * 100) / N) + "%)");
}

Goldilocks::Element Sha256FExecutor::getVal(const vector<Sha256FExecutorInput> &input, Sha256FCommitPols &pols, uint64_t block, uint64_t j, uint16_t i){

    if(program[j].type[i] == TypeSha256Gate::type_wired){
        uint64_t gateNum = (program[j].gate[i] > 0) ? program[j].gate[i] + (slotSize * block) : program[j].gate[i];
        if (program[j].pin[i] == 0) {
            return pols.input[0][gateNum];
        }
        if (program[j].pin[i] == 1) {
            return pols.input[1][gateNum];
        }
        if (program[j].pin[i] == 2) {
            return pols.input[2][gateNum];
        }
        if (program[j].pin[i] == 3) {
            return pols.output[gateNum];
        }
        zklog.error("Sha256FExecutor::getVal() found invalid pin value: " + to_string(program[j].pin[i]));
    }
    if(program[j].type[i] == TypeSha256Gate::type_input){
        return input[block].rIn[program[j].bit[i]];
    }
    if(program[j].type[i] == TypeSha256Gate::type_inputState){
        return input[block].stIn[program[j].bit[i]];
    }
    zklog.error("Sha256FExecutor::getVal() found invalid reference type: " + to_string(program[j].type[i]));
    return fr.zero();
}


