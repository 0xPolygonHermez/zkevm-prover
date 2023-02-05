#ifndef POSEIDON_G_EXECUTOR_HPP
#define POSEIDON_G_EXECUTOR_HPP

#include <vector>
#include <array>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"

USING_PROVER_FORK_NAMESPACE;

using namespace std;

class PoseidonGExecutor
{
private:
    Goldilocks &fr;
    PoseidonGoldilocks &poseidon;
    const uint64_t N;
    const uint64_t t;
    const uint64_t nRoundsF;
    const uint64_t nRoundsP;
    const uint64_t maxHashes;
    const array<Goldilocks::Element,12> MCIRC;
    const array<Goldilocks::Element,12> MDIAG;
    array<array<Goldilocks::Element,12>,12> M;
public:
    PoseidonGExecutor(Goldilocks &fr, PoseidonGoldilocks &poseidon) :
        fr(fr),
        poseidon(poseidon),
        N(PROVER_FORK_NAMESPACE::PoseidonGCommitPols::pilDegree()),
        t(12),
        nRoundsF(8),
        nRoundsP(22),
        maxHashes(N / (nRoundsF + nRoundsP + 1)),
        MCIRC({17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20}),
        MDIAG({8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
    {
        for (uint64_t i = 0; i < 12; i++)
        {
            for (uint64_t j = 0; j < 12; j++)
            {
                M[i][j] = MCIRC[(-i + j + 12) % 12];
                if (i==j)
                {
                    M[i][j] = fr.add(M[i][j], MDIAG[i]);
                }
            }
        }
    };
    void execute (vector<array<Goldilocks::Element, 17>> &input, PROVER_FORK_NAMESPACE::PoseidonGCommitPols &pols);
    Goldilocks::Element pow7(Goldilocks::Element &a);
};

#endif