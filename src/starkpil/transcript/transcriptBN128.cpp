#include "transcriptBN128.hpp"
#include "math.h"
#include "zklog.hpp"

void TranscriptBN128::put(Goldilocks::Element *input, uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        RawFr::Element tmp = RawFr::field.zero();
        tmp.v[0] = Goldilocks::toU64(input[i]);
        RawFr::field.toMontgomery(tmp, tmp);
        _add1(tmp);
    }
}

void TranscriptBN128::put(RawFr::Element *input, uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        _add1(input[i]);
    }
}

void TranscriptBN128::_add1(RawFr::Element input)
{
    pending.push_back(input);
    out.clear();
    if (pending.size() == 16)
    {
        _updateState();
    }
}

void TranscriptBN128::getField(uint64_t *output)
{
    for (int i = 0; i < 3; i++)
    {
        output[i] = getFields1();
    }
}

RawFr::Element TranscriptBN128::getFields253()
{
    if (out.size() > 0)
    {
        RawFr::Element res = out[0];
        out.erase(out.begin());
        return res;
    }
    _updateState();
    return getFields253();
}

uint64_t TranscriptBN128::getFields1()
{
    if (out3.size() > 0)
    {
        uint64_t res = out3[0];
        out3.erase(out3.begin());
        return res;
    }
    if (out.size() > 0)
    {
        RawFr::Element res;
        RawFr::field.fromMontgomery(res, out[0]);
        out.erase(out.begin());
        out3.push_back(res.v[0]);
        out3.push_back(res.v[1]);
        out3.push_back(res.v[2]);

        return getFields1();
    }
    _updateState();
    return getFields1();
}

void TranscriptBN128::_updateState()
{
    while (pending.size() < 16)
    {
        pending.push_back(RawFr::field.zero());
    }

    Poseidon_opt p;
    out.insert(out.end(), state.begin(), state.end());
    out.insert(out.end(), pending.begin(), pending.end());

    p.hash(out);

    state[0] = out[0];
    out3.clear();
    pending.clear();
}

void TranscriptBN128::getPermutations(uint64_t *res, uint64_t n, uint64_t nBits)
{
    uint64_t totalBits = n * nBits;

    uint64_t NFields = floor((float)(totalBits - 1) / 253) + 1;
    RawFr::Element fields[NFields];

    for (uint64_t i = 0; i < NFields; i++)
    {
        fields[i] = getFields253();
    }

    zklog.info("fields[0]: " + RawFr::field.toString(fields[0], 10));

    uint64_t curField = 0;
    uint64_t curBit = 0;

    for (uint64_t i = 0; i < n; i++)
    {
        uint64_t a = 0;
        for (uint64_t j = 0; j < nBits; j++)
        {
            mpz_t n2;
            mpz_init(n2);
            RawFr::field.toMpz(n2, fields[curField]);
            uint64_t bit = mpz_tstbit(n2, curBit);
            mpz_clear(n2);

            if (bit) {
                a = a + (1 << j);
            }

            curBit++;

            if (curBit == 253)
            {
                curBit = 0;
                curField++;
            }
        }
        res[i] = a;
    }
}