#ifndef R1CS_BINFILE_HPP
#define R1CS_BINFILE_HPP

#include <binfile_utils.hpp>
#include <alt_bn128.hpp>

using namespace std;

namespace R1cs {
const int R1CS_HEADER_SECTION = 1;
const int R1CS_CONSTRAINTS_SECTION = 2;

using FrElement = typename AltBn128::Engine::FrElement;

struct R1csHeader {
    // Number of bytes used for the prime number
    uint64_t n8;
    // Prime number
    mpz_t prime;
    // Number of wires in the circuit1
    uint64_t nVars;
    // Number of outputs
    uint64_t nOutputs;
    // Number of public inputs
    uint64_t nPubInputs;
    // Number of private inputs
    uint64_t nPrvInputs;
    // Number of labels
    uint64_t nLabels;
    // Number of constraints
    uint64_t nConstraints;
};

class R1csBinFile {
public:
    static R1csHeader readR1csHeader(BinFileUtils::BinFile &binfile);
};
}  // namespace R1cs

#endif