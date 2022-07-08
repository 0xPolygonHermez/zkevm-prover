#ifndef POSEIDON_GOLDILOCKS
#define POSEIDON_GOLDILOCKS

#include "poseidon_goldilocks_constants.hpp"
#include "goldilocks_base_field.hpp"

#define RATE 8
#define CAPACITY 4
#define HASH_SIZE 4
#define SPONGE_WIDTH (RATE + CAPACITY)
#define HALF_N_FULL_ROUNDS 4
#define N_FULL_ROUNDS_TOTAL (2 * HALF_N_FULL_ROUNDS)
#define N_PARTIAL_ROUNDS 22
#define N_ROUNDS (N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS)

class PoseidonGoldilocks
{

private:
    inline void static pow7(Goldilocks::Element &x)
    {
        Goldilocks::Element x2 = x * x;
        Goldilocks::Element x3 = x * x2;
        Goldilocks::Element x4 = x2 * x2;
        x = x3 * x4;
    };

public:
    void static hash_full_result(Goldilocks::Element (&state)[SPONGE_WIDTH], Goldilocks::Element const (&input)[SPONGE_WIDTH]);
    void static hash(Goldilocks::Element (&state)[CAPACITY], const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    void static linear_hash(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    void static merkletree(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows);
};

#endif