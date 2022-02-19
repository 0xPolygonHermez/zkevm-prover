#ifndef KECCAK_SM_EXECUTOR_HPP
#define KECCAK_SM_EXECUTOR_HPP

#include "config.hpp"
#include "keccak_sm_state.hpp"
#include "keccak2/keccak2.hpp"

class KeccakSMExecutor
{
    const Config &config;
    vector<Eval> evals;
    uint64_t SoutRefs[1600];
public:
    KeccakSMExecutor (const Config &config) : config(config)
    {

    }
    void loadScript (json j)
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
            Eval eval;
            if (j["evaluations"][i]["op"] == "xor")
            {
                eval.op = OP_XOR;
            }
            else if (j["evaluations"][i]["op"] == "andp")
            {
                eval.op = OP_ANDP;
            }
            else
            {
                cerr << "KeccakSMExecutor::loadEvals() found invalid op value: " << j[i]["op"] << endl;
                exit(-1);
            }
            eval.a = j["evaluations"][i]["a"];
            eval.b = j["evaluations"][i]["b"];
            eval.r = j["evaluations"][i]["r"];
            evals.push_back(eval);
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
    }
    /* bits must be an array of u8 long enough to store all references */
    /* SoutRefs must be an array of 1600 u64 */
    void execute (uint8_t * bits)
    {
        for (uint64_t i=0; i<evals.size(); i++)
        {
            if (evals[i].op == OP_XOR)
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
    void copySoutToSin (uint8_t * bits)
    {
        zkassert(bits!=NULL);
        uint8_t localSout[1600];
        for (uint64_t i=0; i<1600; i++)
        {
            localSout[i] = bits[SoutRefs[i]];
        }
        for (uint64_t i=0; i<1600; i++)
        {
            bits[SinRef+i] = localSout[i];
        }
        memset(bits+SoutRef, 0, 1600);
    }
    void KeccakSM (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput)
    {
        Keccak2Input input;
        input.init(pInput, inputSize);
        KeccakSMState S;

        uint8_t r[1088];
        while (input.getNextBits(r))
        {
            S.setRin(r);
            S.resetSoutRefs();
            execute(S.bits);
            copySoutToSin(S.bits);
        }
        S.getOutput(pOutput);
    }
};

void KeccakSMExecutorTest (const Config &config);

#endif