#ifndef TRANSCRIPT_BN128_CLASS
#define TRANSCRIPT_BN128_CLASS

#include "fr.hpp"
#include "poseidon_opt.hpp"
#include <cstring>
#include "goldilocks_base_field.hpp"

#define TRANSCRIPT_STATE_SIZE 4
#define TRANSCRIPT_PENDING_SIZE 8
#define TRANSCRIPT_OUT_SIZE 12

// TODO: Pending to review and re-factor
class TranscriptBN128
{
private:
    void _add1(RawFr::Element input);
    void _updateState();

public:
    uint typeSize = 2;

    std::vector<RawFr::Element> state;
    std::vector<RawFr::Element> pending;
    std::vector<RawFr::Element> out;
    std::vector<uint64_t> out3;

    TranscriptBN128() : state(1, RawFr::field.zero()), out(1, RawFr::field.zero()) {}
    void put(Goldilocks::Element *input, uint64_t size);
    void put(RawFr::Element *input, uint64_t size);
    void getField(uint64_t *output);

    void getPermutations(uint64_t *res, uint64_t n, uint64_t nBits);
    uint64_t getFields1();
    RawFr::Element getFields253();
};

#endif