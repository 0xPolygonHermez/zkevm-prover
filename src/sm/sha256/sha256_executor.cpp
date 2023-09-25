#include "sha256_executor.hpp"
#include "utils.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "zklog.hpp"

void Sha256Executor::loadScript(json j)
{
    if (!j.contains("program") ||
        !j["program"].is_array())
    {
        zklog.error("Sha256Executor::loadEvals() found JSON object does not contain not a program array");
        exitProcess();
    }
    for (uint64_t i = 0; i < j["program"].size(); i++)
    {
        if (!j["program"][i].is_object())
        {
            zklog.error("Sha256Executor::loadEvals() found JSON array's element is not an object");
            exitProcess();
        }
        if (!j["program"][i].contains("op") ||
            !j["program"][i]["op"].is_string())
        {
            zklog.error("Sha256Executor::loadEvals() found JSON array's element does not contain string op field");
            exitProcess();
        }
        if (!j["program"][i].contains("ref") ||
            !j["program"][i]["ref"].is_number_unsigned() ||
            j["program"][i]["ref"] >= SHA256GateConfig.maxRefs)
        {
            zklog.error("Sha256Executor::loadEvals() found JSON array's element does not contain unsigned number ref field");
            exitProcess();
        }
        if (!j["program"][i].contains("a") ||
            !j["program"][i]["a"].is_object())
        {
            zklog.error("Sha256Executor::loadEvals() found JSON array's element does not contain object a field");
            exitProcess();
        }
        if (!j["program"][i].contains("b") ||
            !j["program"][i]["b"].is_object())
        {
            zklog.error("Sha256Executor::loadEvals() found JSON array's element does not contain object b field");
            exitProcess();
        }
        if (!j["program"][i]["a"].contains("type") ||
            !j["program"][i]["a"]["type"].is_string())
        {
            zklog.error("Sha256Executor::loadEvals() found JSON array's element does not contain string a type field");
            exitProcess();
        }
        if (!j["program"][i]["b"].contains("type") ||
            !j["program"][i]["b"]["type"].is_string())
        {
            zklog.error("Sha256Executor::loadEvals() found JSON array's element does not contain string b type field");
            exitProcess();
        }

        Sha256Instruction instruction;

        // Get gate operation and reference
        if (j["program"][i]["op"] == "xor")
        {
            instruction.op = gop_xor;
        }
        else if (j["program"][i]["op"] == "and")
        {
            instruction.op = gop_and;
        }
        else
        {
            string opString = j[i]["op"];
            zklog.error("Sha256Executor::loadEvals() found invalid op value: " + opString);
            exitProcess();
        }
        instruction.refr = j["program"][i]["ref"];

        // Get input a pin data
        string typea = j["program"][i]["a"]["type"];
        if (typea == "wired")
        {
            instruction.refa = j["program"][i]["a"]["gate"];
            string pina = j["program"][i]["a"]["pin"];
            instruction.pina = string2pin(pina);
        }
        else if (typea == "input")
        {
            uint64_t bit = j["program"][i]["a"]["bit"];
            instruction.refa = SHA256GateConfig.sinRef0 + bit * SHA256GateConfig.sinRefDistance;
            instruction.pina = PinId::pin_a;
        }
        else
        {
            zklog.error("Sha256Executor::loadEvals() found invalid a type value: " + typea);
            exitProcess();
        }

        // Get input b pin data
        string typeb = j["program"][i]["b"]["type"];
        if (typeb == "wired")
        {
            instruction.refb = j["program"][i]["b"]["gate"];
            string pinb = j["program"][i]["b"]["pin"];
            instruction.pinb = string2pin(pinb);
        }
        else if (typeb == "input")
        {
            uint64_t bit = j["program"][i]["b"]["bit"];
            instruction.refb = SHA256GateConfig.sinRef0 + bit * SHA256GateConfig.sinRefDistance;
            instruction.pinb = PinId::pin_a;
        }
        else
        {
            zklog.error("Sha256Executor::loadEvals() found invalid b type value: " + typeb);
            exitProcess();
        }

        program.push_back(instruction);
    }

    zkassert(j["maxRef"] == SHA256GateConfig.slotSize);

    bLoaded = true;
}

#if 1
/* Input is a vector of numberOfSlots*SHA256GateConfig.sinRefNumber fe, output is Sha256Pols */
void Sha256Executor::execute(const vector<vector<Goldilocks::Element>> &input, Sha256CommitPols &pols)
{
    zkassertpermanent(bLoaded);
    const uint64_t sha256Mask = 0x7FF;

    // Check input size
    if (input.size() != numberOfSlots)
    {
        zklog.error("Sha256Executor::execute() got input.size()=" + to_string(input.size()) + " different from numberOfSlots=" + to_string(numberOfSlots));
        exitProcess();
    }

    // Check number of slots, per input
    for (uint64_t i = 0; i < numberOfSlots; i++)
    {
        if (input[i].size() != SHA256GateConfig.sinRefNumber)
        {
            zklog.error("Sha256Executor::execute() got input i=" + to_string(i) + " size=" + to_string(input[i].size()) + " different from SHA256GateConfig.sinRefNumber");
            exitProcess();
        }
    }

    // Set SHA256GateConfig.zeroRef values
	pols.inputs[0][SHA256GateConfig.zeroRef] = fr.zero();
    pols.inputs[1][SHA256GateConfig.zeroRef] = fr.fromU64(sha256Mask);
	pols.output[SHA256GateConfig.zeroRef] = fr.fromU64(fr.toU64(pols.inputs[0][SHA256GateConfig.zeroRef]) ^ fr.toU64(pols.inputs[1][SHA256GateConfig.zeroRef]));

    // Set Sin values
    for (uint64_t slot = 0; slot < numberOfSlots; slot++)
    {
        for (uint64_t i = 0; i < SHA256GateConfig.sinRefNumber; i++)
        {
            setPol(pols.inputs[0], SHA256GateConfig.relRef2AbsRef(SHA256GateConfig.sinRef0 + i * SHA256GateConfig.sinRefDistance, slot), fr.toU64(input[slot][i]));
        }
    }

    // Execute the program
#pragma omp parallel for
    for (uint64_t slot = 0; slot < numberOfSlots; slot++)
    {
        for (uint64_t i = 0; i < program.size(); i++)
        {
            uint64_t absRefa = SHA256GateConfig.relRef2AbsRef(program[i].refa, slot);
            uint64_t absRefb = SHA256GateConfig.relRef2AbsRef(program[i].refb, slot);
            uint64_t absRefr = SHA256GateConfig.relRef2AbsRef(program[i].refr, slot);

            switch (program[i].pina)
            {
            case pin_a:
                setPol(pols.inputs[0], absRefr, getPol(pols.inputs[0], absRefa));
                break;
            case pin_b:
                setPol(pols.inputs[0], absRefr, getPol(pols.inputs[1], absRefa));
                break;
            case pin_r:
                setPol(pols.inputs[0], absRefr, getPol(pols.output, absRefa));
                break;
            default:
                zklog.error("Sha256Executor() found invalid program[i].pina=" + to_string(program[i].pina));
                exitProcess();
            }
            switch (program[i].pinb)
            {
            case pin_a:
                setPol(pols.inputs[1], absRefr, getPol(pols.inputs[0], absRefb));
                break;
            case pin_b:
                setPol(pols.inputs[1], absRefr, getPol(pols.inputs[1], absRefb));
                break;
            case pin_r:
                setPol(pols.inputs[1], absRefr, getPol(pols.output, absRefb));
                break;
            default:
                zklog.error("Sha256Executor() found invalid program[i].pinb=" + to_string(program[i].pinb));
                exitProcess();
            }

            switch (program[i].op)
            {
            case gop_xor:
            {
                setPol(pols.output, absRefr, (getPol(pols.inputs[0], absRefr) ^ getPol(pols.inputs[1], absRefr)) & sha256Mask);
                break;
            }
            case gop_and:
            {
                setPol(pols.output, absRefr, ((getPol(pols.inputs[0], absRefr)) & getPol(pols.output, absRefr)) & sha256Mask);
                break;
            }
            default:
            {
                zklog.error("Sha256Executor::execute() found invalid op: " + to_string(program[i].op) + " in evaluation: " + to_string(i));
                exitProcess();
            }
            }
        }
    }

    zklog.info("Sha256Executor successfully processed " + to_string(numberOfSlots) + " Sha256-F actions (" + to_string((double(input.size()) * SHA256GateConfig.slotSize * 100) / N) + "%)");
}
#endif

void Sha256Executor::setPol(CommitPol (&pol), uint64_t index, uint64_t value)
{
    pol[index] = fr.fromU64(value & 0x7FF);
}

uint64_t Sha256Executor::getPol(CommitPol (&pol), uint64_t index)
{
    return fr.toU64(pol[index]);
}
