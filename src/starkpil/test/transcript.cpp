#include "transcript.hpp"

void Transcript::put(Goldilocks::Element *input, uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        _add1(input[i]);
    }
}

void Transcript::_add1(Goldilocks::Element input)
{
    pending[pending_cursor] = input;
    pending_cursor++;
    out_cursor = 0;
    if (pending_cursor == TRANSCRIPT_PENDING_SIZE)
    {
        Goldilocks::Element inputs[TRANSCRIPT_OUT_SIZE];
        std::memcpy(inputs, pending, TRANSCRIPT_PENDING_SIZE * sizeof(Goldilocks::Element));
        std::memcpy(&inputs[TRANSCRIPT_PENDING_SIZE], state, TRANSCRIPT_STATE_SIZE * sizeof(Goldilocks::Element));

        PoseidonGoldilocks::hash_full_result(out, inputs);
        out_cursor = TRANSCRIPT_OUT_SIZE;
        std::memset(pending, 0, TRANSCRIPT_PENDING_SIZE * sizeof(Goldilocks::Element));
        pending_cursor = 0;
        std::memcpy(state, out, TRANSCRIPT_STATE_SIZE * sizeof(Goldilocks::Element));
    }
}

void Transcript::getField(Goldilocks::Element *output)
{
    for (int i = 0; i < 3; i++)
    {
        output[i] = getFields1();
    }
}

Goldilocks::Element Transcript::getFields1()
{
    if (out_cursor == 0)
    {
        Goldilocks::Element inputs[TRANSCRIPT_OUT_SIZE];
        std::memcpy(inputs, pending, TRANSCRIPT_PENDING_SIZE * sizeof(Goldilocks::Element));
        std::memcpy(&inputs[TRANSCRIPT_PENDING_SIZE], state, TRANSCRIPT_STATE_SIZE * sizeof(Goldilocks::Element));
        PoseidonGoldilocks::hash_full_result(out, inputs);
        std::memset(pending, 0, TRANSCRIPT_PENDING_SIZE * sizeof(Goldilocks::Element));
        pending_cursor = 0;
        std::memcpy(state, out, TRANSCRIPT_STATE_SIZE * sizeof(Goldilocks::Element));
    }
    Goldilocks::Element res = out[out_cursor];
    out_cursor++;
    return res;
}