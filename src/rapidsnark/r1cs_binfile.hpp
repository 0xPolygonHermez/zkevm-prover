#ifndef R1CS_BINFILE_HPP
#define R1CS_BINFILE_HPP

// #include <iostream>
// #include <string.h>
#include <binfile_utils.hpp>
// #include <binfile_writer.hpp>
// #include <nlohmann/json.hpp>
// #include "compare_fe_fr.hpp"
// #include <sodium.h>
// #include "zkey_fflonk.hpp"
// #include "polynomial/polynomial.hpp"
// #include "ntt_bn128.hpp"
#include <alt_bn128.hpp>
// #include "fft.hpp"
// #include "utils.hpp"


// using json = nlohmann::json;

using namespace std;

namespace R1cs {
    const int R1CS_HEADER_SECTION = 1;
    const int R1CS_CONSTRAINTS_SECTION = 1;

    using FrElement = typename AltBn128::Engine::FrElement;

    struct R1csHeader {
        // Number of bytes used for the prime number
        uint64_t n8;
        // Prime number
        mpz_t prime;
        // Number of wires in the circuit1
        FrElement nVars;
        // Number of outputs
        FrElement nOutputs;
        // Number of public inputs
        FrElement nPubInputs;
        // Number of private inputs
        FrElement nPrvInputs;
        // Number of labels
        FrElement nLabels;
        // Number of constraints
        FrElement nConstraints;
    };

    
    class R1csBinfile
    {
    public:
        static R1csHeader readR1csHeader(Binfile &binfile);
    };
}

#endif
