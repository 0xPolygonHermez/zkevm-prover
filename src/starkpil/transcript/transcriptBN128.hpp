#ifndef TRANSCRIPT_BN128_CLASS
#define TRANSCRIPT_BN128_CLASS

#include "fr.hpp"
#include "poseidon_opt.hpp"
#include <cstring>
#include "goldilocks_base_field.hpp"

// TODO: Pending to review and re-factor
class TranscriptBN128
{
private:
    void _add1(RawFr::Element input);
    void _updateState();

public:
    uint typeSize = 2;

    uint64_t transcriptArity;

    std::vector<RawFr::Element> state;
    std::vector<RawFr::Element> pending;
    std::vector<RawFr::Element> out;
    std::vector<uint64_t> out3;

    TranscriptBN128(uint64_t arity, bool custom) : state(1, RawFr::field.zero()), out(1, RawFr::field.zero()) {
        transcriptArity = custom ? arity : 16;
    }
    
    void put(Goldilocks::Element *input, uint64_t size);
    void put(RawFr::Element *input, uint64_t size);
    void getState(RawFr::Element* output);
    void getField(uint64_t *output);

    void getPermutations(uint64_t *res, uint64_t n, uint64_t nBits);
    uint64_t getFields1();
    RawFr::Element getFields253();
};

#endif