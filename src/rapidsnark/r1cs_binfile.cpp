#include "r1cs_binfile.hpp"

using FrElement = typename AltBn128::Engine::FrElement;
namespace R1cs
{
static R1csHeader R1csBinfile::readR1csHeader(Binfile &binfile) {
    R1csHeader header;

    binfile.startReadSection(R1CS_HEADER_SECTION);

    header.n8 = binfile.readU32LE();

    mpz_init(header.prime);
    mpz_import(header.prime, header.n8, -1, 1, -1, 0, binfile.read(header.n8));

    mpz_t altBbn128r;
    mpz_init(altBbn128r);
    mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);

    // We assume that the prime is the BN128 prime
    if (mpz_cmp(header.prime, altBbn128r) != 0)
    {
        throw std::invalid_argument("prime not supported");
    }

    mpz_clear(altBbn128r);

    header.nVars = binfile.readU32LE();
    header.nOutputs = binfile.readU32LE();
    header.nPubInputs = binfile.readU32LE();
    header.nPrvInputs = binfile.readU32LE();
    header.nLabels = binfile.readU64LE();
    header.nConstraints = binfile.readU32LE();
    
    binfile.endReadSection();

    return header;
}
}
