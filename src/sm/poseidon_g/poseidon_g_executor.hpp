#ifndef POSEIDON_G_EXECUTOR_HPP
#define POSEIDON_G_EXECUTOR_HPP

#include <vector>
#include "commit_pols.hpp"
#include "ff/ff.hpp"
#include "poseidon_opt/poseidon_goldilocks.hpp"

using namespace std;

class PoseidonGExecutor
{
private:
    FiniteField &fr;
    Poseidon_goldilocks &poseidon;
    const uint64_t N;
    const uint64_t t;
    const uint64_t nRoundsF;
    const uint64_t nRoundsP;
    const uint64_t maxHashes;
    const array<FieldElement,12> MCIRC;
    const array<FieldElement,12> MDIAG;
    array<array<FieldElement,12>,12> M;
public:
    PoseidonGExecutor(FiniteField &fr, Poseidon_goldilocks &poseidon) :
        fr(fr),
        poseidon(poseidon),
        N(PoseidonGCommitPols::degree()),
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
    void execute (vector<array<FieldElement, 16>> &input, PoseidonGCommitPols &pols);
    FieldElement pow7(FieldElement &a);
};

#endif