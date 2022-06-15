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

#include "utils.hpp"

#define commited_file "fibonacci.commit.bin"

using namespace std;

void StarkTest(void)
{
    // Allocate an area of memory, mapped to file, to store all the committed polynomials,
    // and create them using the allocated address
    void *pAddress = NULL;
    pAddress = mapFile("fibonacci.commit.bin", FibCommitPols::size(), false);

    cout << "Prover::prove() successfully mapped " << FibCommitPols::size() << " bytes to file " << commited_file << endl;
    FibCommitPols cmPols(pAddress);
    cout << "StarkTest: pretending to succeed" << endl;
}
