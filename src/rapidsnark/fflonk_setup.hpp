#ifndef FFLONK_SETUP_HPP
#define FFLONK_SETUP_HPP

#include <alt_bn128.hpp>
#include <fft.hpp>
#include <iostream>

#include "binfile_utils.hpp"
#include "binfile_writer.hpp"
#include "fflonk_setup.hpp"
#include "fflonk_setup_settings.hpp"
#include "polynomial/cpolynomial.c.hpp"
#include "polynomial/polynomial.hpp"
#include "r1cs_binfile.hpp"
#include "r1cs_constraint_processor.hpp"
#include "utils.hpp"
#include "zkey_fflonk.hpp"

using namespace std;
using namespace R1cs;
using namespace BinFileUtils;

namespace Fflonk {
struct CommitmentAndPolynomial {
    AltBn128::Engine::G1PointAffine commitment;
    Polynomial<AltBn128::Engine> *polynomial;
};

class FflonkSetup {
    using FrElement = typename AltBn128::Engine::FrElement;
    using G1Point = typename AltBn128::Engine::G1Point;
    using G1PointAffine = typename AltBn128::Engine::G1PointAffine;
    using G2PointAffine = typename AltBn128::Engine::G2PointAffine;

    AltBn128::Engine &E;

    // G1PointAffine *PTau;

    FFT<AltBn128::Engine::Fr> *fft = NULL;
    FflonkSetupSettings settings;

    map<string, Polynomial<AltBn128::Engine> *> polynomials;
    FrElement k1, k2;
    FrElement w3, w4, w8, wr;
    G1PointAffine *PTau;

    vector<ConstraintCoefficients> plonkConstraints;
    vector<AdditionCoefficients> plonkAdditions;

    void computeFFConstraints(BinFile &r1cs, R1csHeader &r1csHeader);

    array<vector<R1csConstraint>, 3> readR1csConstraint(R1csHeader &r1csHeader, BinFile &r1cs);
    vector<R1csConstraint> readR1csConstraintLC(R1csHeader &r1csHeader, BinFile &r1cs);

    void computeK1K2();
    bool isIncluded(FrElement k, vector<FrElement> &kArr);
    FrElement computeW3();
    FrElement computeW4();
    FrElement computeW8();
    FrElement getOmegaCubicRoot();
    void writeZkeyFile(string &zkeyFilename, BinFile &ptauFile);
    void writeZkeyHeader(BinFileWriter &zkeyFile);
    void writeAdditions(BinFileWriter &zkeyFile);
    void writeWitnessMap(BinFileWriter &zkeyFile, uint32_t sectionNum, uint32_t posConstraint);
    void writeQMap(BinFileWriter &zkeyFile, uint32_t sectionNum, uint32_t posConstraint);
    void writeSigma(BinFileWriter &zkeyFile);
    void buildSigma(FrElement *sigma, FrElement w, unordered_map<uint64_t, FrElement> &lastSeen,
                    unordered_map<uint64_t, uint64_t> &firstPos, uint64_t signalId, uint64_t idx);
    void writeLagrangePolynomials(BinFileWriter &zkeyFile);
    void writePtau(BinFileWriter &zkeyFile, BinFile &ptauFile);
    void writeC0(BinFileWriter &zkeyFile);
    void writeFflonkHeader(BinFileWriter &zkeyFile, BinFile &ptauFile);

    FrElement *polynomialFromMontgomery(Polynomial<AltBn128::Engine> *polynomial);
    G1Point multiExponentiation(Polynomial<AltBn128::Engine> *polynomial, u_int32_t nx, u_int64_t x[]);
    void scalar2bytes(mpz_class s, uint8_t (&bytes)[32]);

    void reset();

public:
    FflonkSetup(AltBn128::Engine &_E)
        : E(_E){};
    ~FflonkSetup();

    void generateZkey(string r1csFilename, string pTauFilename, string zkeyFilename);
};
}  // namespace Fflonk

#endif