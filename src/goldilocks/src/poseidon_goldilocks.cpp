#include "poseidon_goldilocks.hpp"
#include <math.h> /* floor */
#include "omp.h"

#include "merklehash_goldilocks.hpp"

void PoseidonGoldilocks::hash(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    hash_full_result(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}
void PoseidonGoldilocks::hash_full_result(Goldilocks::Element (&state)[SPONGE_WIDTH], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    std::memcpy(state, input, SPONGE_WIDTH * sizeof(Goldilocks::Element));
    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = state[i] + PoseidonGoldilocksConstants::C[i];
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            pow7(state[j]);
            state[j] = state[j] + PoseidonGoldilocksConstants::C[(r + 1) * SPONGE_WIDTH + j];
        }

        Goldilocks::Element old_state[SPONGE_WIDTH];
        std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);

        for (int i = 0; i < SPONGE_WIDTH; i++)
        {
            state[i] = Goldilocks::zero();
            for (int j = 0; j < SPONGE_WIDTH; j++)
            {
                Goldilocks::Element mji = PoseidonGoldilocksConstants::M[j][i];
                mji = mji * old_state[j];
                state[i] = state[i] + mji;
            }
        }
    }

    for (int j = 0; j < SPONGE_WIDTH; j++)
    {
        pow7(state[j]);
        state[j] = state[j] + PoseidonGoldilocksConstants::C[j + (HALF_N_FULL_ROUNDS * SPONGE_WIDTH)];
    }

    Goldilocks::Element old_state[SPONGE_WIDTH];
    std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = Goldilocks::zero();
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            state[i] = state[i] + (PoseidonGoldilocksConstants::P[j][i] * old_state[j]);
        }
    }

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        pow7(state[0]);
        state[0] = state[0] + PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];

        Goldilocks::Element s0 = state[0] * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];

        for (int j = 1; j < SPONGE_WIDTH; j++)
        {
            s0 = s0 + (state[j] * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + j]);
            state[j] = state[j] + (state[0] * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH + j - 1]);
        }
        state[0] = s0;
    }
    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            pow7(state[j]);
            state[j] = state[j] + PoseidonGoldilocksConstants::C[j + (HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH];
        }

        Goldilocks::Element old_state[SPONGE_WIDTH];
        std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);

        for (int i = 0; i < SPONGE_WIDTH; i++)
        {
            state[i] = Goldilocks::zero();
            for (int j = 0; j < SPONGE_WIDTH; j++)
            {
                state[i] = state[i] + (old_state[j] * PoseidonGoldilocksConstants::M[j][i]);
            }
        }
    }

    for (int j = 0; j < SPONGE_WIDTH; j++)
    {
        pow7(state[j]);
    }

    std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = Goldilocks::zero();
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            state[i] = state[i] + (old_state[j] * PoseidonGoldilocksConstants::M[j][i]);
        }
    }
}

void PoseidonGoldilocks::linear_hash(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[SPONGE_WIDTH];

    while (remaining)
    {
        if (remaining == size)
        {
            memset(state + RATE, 0, CAPACITY * sizeof(Goldilocks::Element));
        }
        else
        {
            std::memcpy(state + RATE, state, CAPACITY * sizeof(Goldilocks::Element));
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;

        memset(&state[n], 0, (RATE - n) * sizeof(Goldilocks::Element));

        std::memcpy(state, input + (size - remaining), n * sizeof(Goldilocks::Element));

        hash_full_result(state, state);

        remaining -= n;
    }
    std::memcpy(output, state, CAPACITY * sizeof(uint64_t));
}

void PoseidonGoldilocks::merkletree(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t dim)
{
    tree[0] = Goldilocks::fromU64(num_cols * dim);
    tree[1] = Goldilocks::fromU64(num_rows);

    std::cout << "Starting copy" << std::endl;
    double st_copy_start = omp_get_wtime();

    Goldilocks::parcpy(&tree[MERKLEHASHGOLDILOCKS_HEADER_SIZE], input, dim * num_cols * num_rows, 64);
    double st_copy_end = omp_get_wtime();
    std::cout << "Copy finished! " << st_copy_end - st_copy_start << " bytes: " << dim * num_cols * num_rows * sizeof(Goldilocks::Element) << std::endl;

    Goldilocks::Element *cursor = &tree[MERKLEHASHGOLDILOCKS_HEADER_SIZE + num_cols * num_rows * dim];

#pragma omp parallel for
    for (uint64_t i = 0; i < num_rows; i++)
    {
        Goldilocks::Element intermediate[num_cols * dim];
        std::memcpy(&intermediate[0], &input[i * num_cols * dim], dim * num_cols * sizeof(Goldilocks::Element));
        linear_hash(&cursor[i * CAPACITY], intermediate, num_cols * dim);
    }

    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = floor((pending - 1) / 2) + 1;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
            std::memcpy(pol_input, &cursor[nextIndex + i * RATE], RATE * sizeof(Goldilocks::Element));
            hash((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + i) * CAPACITY], pol_input);
        }
        nextIndex += pending * CAPACITY;
        pending = pending / 2;
        nextN = floor((pending - 1) / 2) + 1;
    }
}
