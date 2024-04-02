#include "transcriptGL.hpp"
#include "math.h"

void TranscriptGL::put(Goldilocks::Element *input, uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        _add1(input[i]);
    }
}

void TranscriptGL::_updateState() 
{
    while(pending_cursor < TRANSCRIPT_PENDING_SIZE) {
        pending[pending_cursor] = Goldilocks::zero();
        pending_cursor++;
    }
    Goldilocks::Element inputs[TRANSCRIPT_OUT_SIZE];
    std::memcpy(inputs, pending, TRANSCRIPT_PENDING_SIZE * sizeof(Goldilocks::Element));
    std::memcpy(&inputs[TRANSCRIPT_PENDING_SIZE], state, TRANSCRIPT_STATE_SIZE * sizeof(Goldilocks::Element));
    PoseidonGoldilocks::hash_full_result(out, inputs);
    out_cursor = TRANSCRIPT_OUT_SIZE;
    std::memset(pending, 0, TRANSCRIPT_PENDING_SIZE * sizeof(Goldilocks::Element));
    pending_cursor = 0;
    std::memcpy(state, out, TRANSCRIPT_STATE_SIZE * sizeof(Goldilocks::Element));
}

void TranscriptGL::_add1(Goldilocks::Element input)
{
    pending[pending_cursor] = input;
    pending_cursor++;
    out_cursor = 0;
    if (pending_cursor == TRANSCRIPT_PENDING_SIZE)
    {
        _updateState();
    }
}

void TranscriptGL::getField(uint64_t* output)
{
    for (int i = 0; i < 3; i++)
    {
        Goldilocks::Element val = getFields1();
        output[i] = val.fe;
    }
    zklog.info("Challenge: [ " + std::to_string(output[0]) + " " + std::to_string(output[1]) + " " + std::to_string(output[2]) + " ]");

}

void TranscriptGL::getState(Goldilocks::Element* output) {
    if(pending_cursor > 0) {
        _updateState();
    }
    std::memcpy(output, state, TRANSCRIPT_STATE_SIZE * sizeof(Goldilocks::Element));
}

Goldilocks::Element TranscriptGL::getFields1()
{
    if (out_cursor == 0)
    {
        _updateState();
    }
    Goldilocks::Element res = out[(TRANSCRIPT_OUT_SIZE - out_cursor) % TRANSCRIPT_OUT_SIZE];
    out_cursor--;
    return res;
}

void TranscriptGL::getPermutations(uint64_t *res, uint64_t n, uint64_t nBits)
{
    uint64_t totalBits = n * nBits;

    uint64_t NFields = floor((float)(totalBits - 1) / 63) + 1;
    Goldilocks::Element fields[NFields];

    for (uint64_t i = 0; i < NFields; i++)
    {
        fields[i] = getFields1();
    }
    
    std::string permutation = " ";

    uint64_t curField = 0;
    uint64_t curBit = 0;
    for (uint64_t i = 0; i < n; i++)
    {
        uint64_t a = 0;
        for (uint64_t j = 0; j < nBits; j++)
        {
            uint64_t bit = (Goldilocks::toU64(fields[curField]) >> curBit) & 1;
            if (bit)
                a = a + (1 << j);
            curBit++;
            if (curBit == 63)
            {
                curBit = 0;
                curField++;
            }
        }
        res[i] = a;
        permutation += std::to_string(a) + " ";
    }

    zklog.info("Permutation: [ " + permutation + " ]");
}