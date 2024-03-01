#ifndef SHA256_SM_EXECUTOR_HPP
#define SHA256_SM_EXECUTOR_HPP

#include <array>
#include "definitions.hpp"
#include "config.hpp"
#include "sha256_instruction.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "timer.hpp"
#include "gate_state.hpp"
#include "sha256.hpp"
#include "sha256_config.hpp"

USING_PROVER_FORK_NAMESPACE;

using namespace std;

class Sha256FExecutorInput
{
public:
    typedef struct{
        uint64_t * stIn;
        uint64_t stIn_size;
        uint64_t * rIn;
        uint64_t rIn_size;
    } DTO;
    inline void toDTO(DTO* dto) const{
        dto->stIn = (uint64_t*) stIn.data();
        dto->stIn_size = stIn.size();
        dto->rIn = (uint64_t*) rIn.data();
        dto->rIn_size = rIn.size();
    }
    inline void fromDTO(DTO* dto){
        stIn.resize(dto->stIn_size);
        if(dto->stIn_size > 0)
            stIn.assign((Goldilocks::Element*)dto->stIn, (Goldilocks::Element*)dto->stIn + dto->stIn_size);
        rIn.resize(dto->rIn_size);
        if(dto->rIn_size > 0)
            rIn.assign((Goldilocks::Element*)dto->rIn, (Goldilocks::Element*)dto->rIn + dto->rIn_size);
    }
    static inline DTO * toDTO(const vector<Sha256FExecutorInput> &input){
        DTO * dto = new DTO[input.size()];
        for (uint64_t i = 0; i < input.size(); i++){
            input[i].toDTO(dto + i);
        }
        return dto;
    }
    static inline void fromDTO(DTO* dto, uint64_t dto_size, vector<Sha256FExecutorInput> &output){
        output.resize(dto_size);
        for (uint64_t i = 0; i < dto_size; i++){
            output[i].fromDTO(dto +i);
        }
    }

    vector<Goldilocks::Element> stIn;
    vector<Goldilocks::Element> rIn;

    /* Constructor */
    Sha256FExecutorInput() {};
};


class Sha256FExecutor
{
    Goldilocks &fr;
    const Config &config;
    const uint64_t N;
    const uint64_t slotSize;
    const uint64_t bitsPerElement;
    const uint64_t nSlots;
    vector<Sha256Instruction> program;
    bool bLoaded;
public:

    /* Constructor */
    Sha256FExecutor (Goldilocks &fr, const Config &config) :
        fr(fr),
        config(config),
        N(PROVER_FORK_NAMESPACE::PaddingSha256BitCommitPols::pilDegree()),
        slotSize(31488),
        bitsPerElement(7),
        nSlots((N-1)/slotSize)
    {
        bLoaded = false;

        // Avoid initialization if we are not going to generate any proof
        if (!config.generateProof() && !config.runFileExecute) return;

        TimerStart(SHA256_F_SM_EXECUTOR_LOAD);
        json j;
        file2json(config.sha256ScriptFile, j);
        loadScript(j);
        bLoaded = true;
        TimerStopAndLog(SHA256_F_SM_EXECUTOR_LOAD);
    }

    /* Loads evaluations and SoutRefs from a json object */
    void loadScript (json j);

    void execute (const vector<Sha256FExecutorInput> &input, PROVER_FORK_NAMESPACE::Sha256FCommitPols &pols);
    inline void execute (vector<Sha256FExecutorInput> &input, Goldilocks::Element *pAddress){
        PROVER_FORK_NAMESPACE::Sha256FCommitPols pols(pAddress, N);
        execute(input, pols);
    }

private:
    Goldilocks::Element getVal(const vector<Sha256FExecutorInput> &input, Sha256FCommitPols &pols, uint64_t block, uint64_t j, uint16_t i);
};
#endif