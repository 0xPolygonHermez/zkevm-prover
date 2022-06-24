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

#define commited_file "fibonacci.commit.bin"
#define constant_file "fibonacci.commit.bin"

using namespace std;

void StarkTest(void)
{
    // Allocate an area of memory, mapped to file, to store all the committed polynomials,
    // and create them using the allocated address
    void *pCommitedAddress = NULL;
    void *pConstantAddress = NULL;

    starkStruct structStark{10, 11, 8};
    structStark.extendBits = structStark.nBitsExt - structStark.nBits;
    structStark.N = (1 << structStark.nBits);
    structStark.N_Extended = (1 << structStark.nBitsExt);

    starkInfo infoStark{2, 2, 1, 0, 0, 1};

    infoStark.qs1 = (qs *)malloc(sizeof(qs) * infoStark.nQ1);
    infoStark.qs1[0].idExp = 1;
    infoStark.qs1[0].idQ = 0;

    pCommitedAddress = mapFile(commited_file, structStark.N * infoStark.nCm1 * sizeof(Goldilocks::Element), false);
    pConstantAddress = mapFile(constant_file, structStark.N * infoStark.nConst * sizeof(Goldilocks::Element), false);

    Goldilocks::Element *cm2ns = (Goldilocks::Element *)malloc(structStark.N_Extended * infoStark.nCm1 * sizeof(Goldilocks::Element));

    PolsFibonacci pols(pCommitedAddress, pConstantAddress, structStark, infoStark);

    Pols2nsFibonacci pols2ns(cm2ns, structStark, infoStark);

    step1::calculateExps(pols);
    std::cout << Goldilocks::toString(&pols.exps[1024], structStark.N, 10) << std::endl;

    pols.extendCms(pols2ns.cm.Fibonacci.pData, pols2ns.exps.pData, infoStark.nQ1, infoStark.qs1);

    step12ns::calculateExps(pols2ns);
    std::cout << Goldilocks::toString(&pols2ns.exps[2048], structStark.N_Extended, 10) << std::endl;

    unmapFile(pCommitedAddress, structStark.N * infoStark.nCm1 * sizeof(Goldilocks::Element));
    unmapFile(pConstantAddress, structStark.N * infoStark.nConst * sizeof(Goldilocks::Element));
}
