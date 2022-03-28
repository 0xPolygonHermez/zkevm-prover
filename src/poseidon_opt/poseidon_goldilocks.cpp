#include "poseidon_goldilocks.hpp"
#include <cstring> //memset

void Poseidon_goldilocks::hash(uint64_t (&state)[SPONGE_WIDTH])
{
	uint8_t round_ctr = 0;
	full_rounds(state, round_ctr);
	partial_rounds_naive(state, round_ctr);
	full_rounds(state, round_ctr);
}

void Poseidon_goldilocks::full_rounds(uint64_t (&state)[SPONGE_WIDTH], uint8_t &round_ctr)
{
	for (uint8_t i = 0; i < HALF_N_FULL_ROUNDS; i++)
	{
		constant_layer(state, round_ctr);
		sbox_layer(state);
		mds_layer(state);
		round_ctr += 1;
	}
}

void Poseidon_goldilocks::constant_layer(uint64_t (&state)[SPONGE_WIDTH], uint8_t &round_ctr)
{
	for (uint8_t i = 0; i < SPONGE_WIDTH; i++)
	{
		state[i] = ((uint128_t)state[i] + (uint128_t)ALL_ROUND_CONSTANTS[i + SPONGE_WIDTH * round_ctr]) % GOLDILOCKS_PRIME;
	}
}

void Poseidon_goldilocks::sbox_layer(uint64_t (&state)[SPONGE_WIDTH])
{
	for (uint8_t i = 0; i < SPONGE_WIDTH; i++)
	{
		if (i < SPONGE_WIDTH)
		{
			sbox_monomial(state[i]);
		}
	}
}

void Poseidon_goldilocks::sbox_monomial(uint64_t &x)
{
	uint128_t x2 = ((uint128_t)x * (uint128_t)x) % GOLDILOCKS_PRIME;
	uint128_t x4 = (x2 * x2) % GOLDILOCKS_PRIME;
	uint128_t x3 = ((uint128_t)x * x2) % GOLDILOCKS_PRIME;
	x = (x3 * x4) % GOLDILOCKS_PRIME;
}

void Poseidon_goldilocks::mds_layer(uint64_t (&state_)[SPONGE_WIDTH])
{
	uint64_t state[SPONGE_WIDTH] = {0};
	std::memcpy(state, state_, SPONGE_WIDTH * sizeof(uint64_t));

	for (uint8_t r = 0; r < SPONGE_WIDTH; r++)
	{
		state_[r] = mds_row_shf(r, state);
	}
}

uint64_t Poseidon_goldilocks::mds_row_shf(uint64_t r, uint64_t (&v)[SPONGE_WIDTH])
{
	uint128_t res = 0;
	res += (uint128_t)v[r] * (uint128_t)MDS_MATRIX_DIAG[r];

	for (uint8_t i = 0; i < SPONGE_WIDTH; i++)
	{
		res += (((uint128_t)v[(i + r) % SPONGE_WIDTH] * (uint128_t)MDS_MATRIX_CIRC[i]));
	}
	return res % GOLDILOCKS_PRIME;
}

void Poseidon_goldilocks::partial_rounds_naive(uint64_t (&state)[SPONGE_WIDTH], uint8_t &round_ctr)
{
	for (uint8_t i = 0; i < N_PARTIAL_ROUNDS; i++)
	{
		constant_layer(state, round_ctr);
		sbox_monomial(state[0]);
		mds_layer(state);
		round_ctr += 1;
	}
}
