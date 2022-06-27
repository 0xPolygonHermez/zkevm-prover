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

#define commited_file "fibonacci.commit.bin"

using namespace std;

void StarkTest(void)
{
    Goldilocks::Element a;
    Goldilocks::add(a, Goldilocks::one(), Goldilocks::one());
    // Allocate an area of memory, mapped to file, to store all the committed polynomials,
    // and create them using the allocated address
    void *pCommitedAddress = NULL;
    void *pConstantAddress = NULL;

    pCommitedAddress = mapFile("fibonacci.commit.bin", FibCommitPols::size(), false);
    pConstantAddress = mapFile("fibonacci.const.bin", FibConstantPols::size(), false);

    cout << "Prover::prove() successfully mapped " << FibCommitPols::size() << " bytes to file " << commited_file << endl;

    PolsFibonacci pols(pCommitedAddress, pConstantAddress);

    step1::calculateExps(pols);

    cout << "StarkTest: pretending to succeed: " << Goldilocks::toString(PreCalculatedPols2ns::x[0]) << " " << Goldilocks::toString(PreCalculatedPols2ns::Zi(1000)) << endl;
}
