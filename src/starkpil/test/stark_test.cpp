#include <iostream>
#include "stark_test.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "commit_pols_fibonacci.hpp"
#include "constant_pols_fibonacci.hpp"

#include "precalculated_pols_fibonacci_file.hpp"
#include "precalculated_pols2ns_fibonacci_file.hpp"

#include "pols_fibonacci.hpp"
#include "pols2ns_fibonacci.hpp"

#include "utils.hpp"

#include "calculateExpsStep1.hpp"
#include "calculateExpsStep12ns.hpp"
#include "calculateExpsStep2prev.hpp"
#include "calculateExpsStep2.hpp"
#include "calculateExpsStep22ns.hpp"

#include "transcript.hpp"
#include "poseidon_goldilocks.hpp"

#define commited_file "fibonacci.commit.bin"
#define constant_file "fibonacci.commit.bin"

using namespace std;

void StarkTest(void)
{
    // Allocate an area of memory, mapped to file, to store all the committed polynomials,
    // and create them using the allocated address
    void *pCommitedAddress = NULL;
    void *pConstantAddress = NULL;

    Transcript transcript;

    starkStruct structStark{10, 11, 8};
    structStark.extendBits = structStark.nBitsExt - structStark.nBits;
    structStark.N = (1 << structStark.nBits);
    structStark.N_Extended = (1 << structStark.nBitsExt);

    starkInfo infoStark{2, 2, 1, 0, 0, 1};

    infoStark.qs1 = (qs *)malloc(sizeof(qs) * infoStark.nQ1);
    infoStark.qs1[0].idExp = 1;
    infoStark.qs1[0].idQ = 0;

    infoStark.qs2 = (qs *)malloc(sizeof(qs) * infoStark.nQ2);
    infoStark.qs3 = (qs *)malloc(sizeof(qs) * infoStark.nQ3);

    infoStark.qs4 = (qs *)malloc(sizeof(qs) * infoStark.nQ4);
    infoStark.qs4[0].idExp = 6;
    infoStark.qs4[0].idQ = 1;

    pCommitedAddress = mapFile(commited_file, structStark.N * infoStark.nCm1 * sizeof(Goldilocks::Element), false);
    pConstantAddress = mapFile(constant_file, structStark.N * infoStark.nConst * sizeof(Goldilocks::Element), false);

    Goldilocks::Element *cm2ns = (Goldilocks::Element *)malloc(structStark.N_Extended * infoStark.nCm1 * sizeof(Goldilocks::Element));

    PolsFibonacci pols(pCommitedAddress, pConstantAddress, structStark, infoStark);

    Pols2nsFibonacci pols2ns(cm2ns, structStark, infoStark);

    step1::calculateExps(pols);

    //pols.computeCommited(pols2ns.cm.Fibonacci.pData);
    NTT_Goldilocks ntt(structStark.N);
    ntt.extendPol((Goldilocks::Element *)pols2ns.cm.Fibonacci.pData, (Goldilocks::Element *)pCommitedAddress, structStark.N_Extended, structStark.N, infoStark.nCm1);

    pols.extendCms(pols2ns.cm.Fibonacci.pData, pols2ns.exps.pData, infoStark.nQ1, infoStark.qs1);

    step12ns::calculateExps(pols2ns);

    Goldilocks::Element *tree = (Goldilocks::Element *)malloc((pols2ns.cm.length + infoStark.nQ1) * structStark.N_Extended * sizeof(Goldilocks::Element));

    // Reorg tree for linear hasing
    for (uint64_t j = 0; j < structStark.N_Extended; j++)
    {
        std::memcpy(&tree[j * (pols2ns.cm.length + infoStark.nQ1)], &pols2ns.cm.Fibonacci.pData[j * pols2ns.cm.length], pols2ns.cm.length * sizeof(Goldilocks::Element));
        std::memcpy(&tree[j * (pols2ns.cm.length + infoStark.nQ1) + pols2ns.cm.length], &pols2ns.q[j * infoStark.nQ1], infoStark.nQ1 * sizeof(Goldilocks::Element));
    }

    Goldilocks::Element rootHash[4];

    PoseidonGoldilocks::merkletree(rootHash, tree, pols2ns.cm.length + infoStark.nQ1, structStark.N_Extended);
    std::cout << Goldilocks::toString(rootHash, 4, 10) << std::endl;

    transcript.put(rootHash, 4);

    ///////////
    // 2.- Caluculate plookups h1 and h2
    ///////////
    Goldilocks::Element challanges_0[3];
    Goldilocks::Element challanges_1[3];
    transcript.getField(challanges_0); // u
    transcript.getField(challanges_1); // defVal

    step2prev::calculateExps(pols);
    step2::calculateExps(pols);

    pols.extendCms(pols2ns.cm.Fibonacci.pData, pols2ns.exps.pData, infoStark.nQ2, infoStark.qs2);
    std::cout << "Merkelizing 2...." << std::endl;

    unmapFile(pCommitedAddress, structStark.N * infoStark.nCm1 * sizeof(Goldilocks::Element));
    unmapFile(pConstantAddress, structStark.N * infoStark.nConst * sizeof(Goldilocks::Element));
    free(tree);
}
