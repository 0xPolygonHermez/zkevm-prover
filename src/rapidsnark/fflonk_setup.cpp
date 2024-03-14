#include "fflonk_setup.hpp"

#include <math.h>
#include <omp.h>

#include "polynomial/evaluations.hpp"
#include "thread_utils.hpp"
#include "zkey.hpp"

namespace Fflonk {

using FrElement = typename AltBn128::Engine::FrElement;
using G1Point = typename AltBn128::Engine::G1Point;
using G1PointAffine = typename AltBn128::Engine::G1PointAffine;
using G2PointAffine = typename AltBn128::Engine::G2PointAffine;

FflonkSetup::~FflonkSetup() {
    reset();
}

void FflonkSetup::reset() {
    delete fft;

    polynomials.clear();
}

void FflonkSetup::generateZkey(string r1csFilename, string pTauFilename, string zkeyFilename) {
    LOG_INFO("FFLONK SETUP STARTED");

    // STEP 1. Read PTau file
    LOG_INFO("> Opening PTau file");
    auto fdPtau = openExisting(pTauFilename, "ptau", 1);
    if (!fdPtau->sectionExists(12)) {
        throw new runtime_error("Powers of Tau file is not well prepared. Section 12 missing.");
    }

    // STEP 2. Read r1cs file
    LOG_INFO("> Opening r1cs file");
    auto fdR1cs = openExisting(r1csFilename, "r1cs", 1);

    // Read r1cs header file
    auto r1csHeader = R1csBinFile::readR1csHeader(*fdR1cs);

    const auto sG1 = sizeof(G1PointAffine);
    const auto sG2 = sizeof(G2PointAffine);

    settings.nVars = r1csHeader.nVars;
    settings.nPublics = r1csHeader.nOutputs + r1csHeader.nPubInputs;

    // Process constraints inside r1cs
    LOG_INFO("> Processing FFlonk constraints");
    computeFFConstraints(*fdR1cs, r1csHeader);

    // As the t polynomial is n+5 whe need at least a power of 4
    // TODO check!!!!
    // NOTE : plonkConstraints + 2 = #constraints + blinding coefficients for each wire polynomial
    double FF_T_POL_DEG_MIN = 3;
    settings.cirPower = max(FF_T_POL_DEG_MIN, log2((plonkConstraints.size() + 2) - 1) + 1);
    settings.domainSize = 1 << settings.cirPower;

    fft = new FFT<AltBn128::Engine::Fr>(settings.domainSize * 4);

    if (fdPtau->getSectionSize(2) < settings.domainSize * 9 * sG1) {
        throw new runtime_error("Powers of Tau is not big enough for this circuit size. Section 2 too small.");
    }
    if (fdPtau->getSectionSize(3) < sG2) {
        throw new runtime_error("Powers of Tau is not well prepared. Section 3 too small.");
    }

    ostringstream ss;
    LOG_INFO("----------------------------");
    LOG_INFO("  FFLONK SETUP SETTINGS");
    LOG_INFO("  Curve:         BN128");
    ss.str("");
    ss << "  Circuit power: " << settings.cirPower;
    LOG_INFO(ss);
    ss.str("");
    ss << "  Domain size:   " << settings.domainSize;
    LOG_INFO(ss);
    ss.str("");
    ss << "  Vars:          " << settings.nVars;
    LOG_INFO(ss);
    ss.str("");
    ss << "  Public vars:   " << settings.nPublics;
    LOG_INFO(ss);
    ss.str("");
    ss << "  Constraints:   " << plonkConstraints.size();
    LOG_INFO(ss);
    ss.str("");
    ss << "  Additions:     " << plonkAdditions.size();
    LOG_INFO(ss);
    ss.str("");
    LOG_INFO("----------------------------");

    // Compute k1 and k2 to be used in the permutation checks
    LOG_INFO("> computing k1 and k2");
    computeK1K2();

    // Compute omega 3 (w3) and omega 4 (w4) to be used in the prover and the verifier
    // w3^3 = 1 and  w4^4 = 1
    LOG_INFO("> computing w3");
    w3 = computeW3();

    LOG_INFO("> computing w4");
    w4 = computeW4();

    LOG_INFO("> computing w8");
    w8 = computeW8();

    LOG_INFO("> computing wr");
    wr = getOmegaCubicRoot();

    // Write output zkey file
    writeZkeyFile(zkeyFilename, *fdPtau);

    LOG_INFO("FFLONK SETUP FINISHED");
}

void FflonkSetup::computeFFConstraints(BinFile &r1cs, R1csHeader &r1csHeader) {
    // Create r1cs processor
    const auto r1csProcessor = new R1csConstraintProcessor(E);

    // Add public inputs and outputs
    for (uint64_t i = 0; i < settings.nPublics; i++) {
        plonkConstraints.push_back(R1csConstraintProcessor::getFflonkConstantConstraint(E, i + 1));
    }

    if (!r1cs.sectionExists(R1CS_CONSTRAINTS_SECTION)) {
        throw new runtime_error("R1CS file is not well prepared. Section 2 missing.");
    }
    // Start reading r1cs constraints section
    r1cs.startReadSection(R1CS_CONSTRAINTS_SECTION);

    for (uint64_t i = 0; i < r1csHeader.nConstraints; i++) {
        auto lc = readR1csConstraint(r1csHeader, r1cs);
        r1csProcessor->processR1csConstraints(settings, lc[0], lc[1], lc[2], plonkConstraints, plonkAdditions);
    }

    r1cs.endReadSection(false);
}

array<vector<R1csConstraint>, 3> FflonkSetup::readR1csConstraint(R1csHeader &r1csHeader, BinFile &r1cs) {
    array<vector<R1csConstraint>, 3> lc;

    lc[0] = readR1csConstraintLC(r1csHeader, r1cs);
    lc[1] = readR1csConstraintLC(r1csHeader, r1cs);
    lc[2] = readR1csConstraintLC(r1csHeader, r1cs);

    return lc;
}

vector<R1csConstraint> FflonkSetup::readR1csConstraintLC(R1csHeader &r1csHeader, BinFile &r1cs) {
    uint32_t n = r1cs.readU32LE();
    vector<R1csConstraint> lc;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t signal_id = r1cs.readU32LE();

        FrElement value;
        E.fr.fromRprLE(value, (uint8_t *)r1cs.read(E.fr.bytes()), E.fr.bytes());

        lc.push_back(R1csConstraint(signal_id, value));
    }

    sort(lc.begin(), lc.end(), [](const R1csConstraint &a, const R1csConstraint &b) {
        return a.signal_id < b.signal_id;
    });

    return lc;
}

void FflonkSetup::computeK1K2() {
    E.fr.fromUI(k1, 2);
    vector<FrElement> kArr;
    while (isIncluded(k1, kArr)) {
        k1 = E.fr.add(k1, E.fr.one());
    }

    kArr.push_back(k1);
    k2 = E.fr.add(k1, E.fr.one());
    while (isIncluded(k2, kArr)) {
        k2 = E.fr.add(k2, E.fr.one());
    }
}

bool FflonkSetup::isIncluded(FrElement k, vector<FrElement> &kArr) {
    auto w = E.fr.one();
    for (uint64_t i = 0; i < settings.domainSize; i++) {
        if (E.fr.eq(k, w))
            return true;

        for (uint64_t j = 0; j < kArr.size(); j++) {
            if (E.fr.eq(k, E.fr.mul(kArr[j], w)))
                return true;
        }
        w = E.fr.mul(w, fft->root(settings.cirPower, 1));
    }

    return false;
}

FrElement FflonkSetup::computeW3() {
    // Exponent is order(r - 1) / 3
    mpz_t result;
    mpz_init_set_str(result, "21888242871839275217838484774961031246154997185409878258781734729429964517155", 10);

    FrElement w3;
    E.fr.fromMpz(w3, result);

    return w3;
}

FrElement FflonkSetup::computeW4() {
    mpz_t result;
    mpz_init_set_str(result, "21888242871839275217838484774961031246007050428528088939761107053157389710902", 10);

    FrElement w4;
    E.fr.fromMpz(w4, result);

    return w4;
}

FrElement FflonkSetup::computeW8() {
    mpz_t result;
    mpz_init_set_str(result, "19540430494807482326159819597004422086093766032135589407132600596362845576832", 10);

    FrElement w4;
    E.fr.fromMpz(w4, result);

    return w4;
}

FrElement FflonkSetup::getOmegaCubicRoot() {
    // Hardcorded 3th-root of Fr.w[28]
    mpz_t firstRoot;
    mpz_init_set_str(firstRoot, "467799165886069610036046866799264026481344299079011762026774533774345988080", 10);

    FrElement tmp;
    E.fr.fromMpz(tmp, firstRoot);

    uint64_t scalar = 1 << (28 - settings.cirPower);

    FrElement result;
    E.fr.exp(result, tmp, (uint8_t *)&scalar, sizeof(uint64_t));

    return result;
}

void FflonkSetup::writeZkeyFile(string &zkeyFilename, BinFile &fdPtau) {
    LOG_INFO("> Writing the zkey file");

    BinFileWriter fdZKey(zkeyFilename, "zkey", 1, Zkey::ZKEY_FF_NSECTIONS);

    ostringstream ss;
    ss << "··· Writing Section " << Zkey::ZKEY_HEADER_SECTION << ". Zkey Header";
    LOG_INFO(ss);
    ss.str("");
    writeZkeyHeader(fdZKey);

    ss << "··· Writing Section " << Zkey::ZKEY_FF_ADDITIONS_SECTION << ". Additions";
    LOG_INFO(ss);
    ss.str("");
    writeAdditions(fdZKey);

    ss << "··· Writing Section " << Zkey::ZKEY_FF_A_MAP_SECTION << ". A Map";
    LOG_INFO(ss);
    ss.str("");
    writeWitnessMap(fdZKey, Zkey::ZKEY_FF_A_MAP_SECTION, 0);

    ss << "··· Writing Section " << Zkey::ZKEY_FF_B_MAP_SECTION << ". B Map";
    LOG_INFO(ss);
    ss.str("");
    writeWitnessMap(fdZKey, Zkey::ZKEY_FF_B_MAP_SECTION, 1);

    ss << "··· Writing Section " << Zkey::ZKEY_FF_C_MAP_SECTION << ". C Map";
    LOG_INFO(ss);
    ss.str("");
    writeWitnessMap(fdZKey, Zkey::ZKEY_FF_C_MAP_SECTION, 2);

    ss << "··· Writing Section " << Zkey::ZKEY_FF_QL_SECTION << ". QL";
    LOG_INFO(ss);
    ss.str("");
    writeQMap(fdZKey, Zkey::ZKEY_FF_QL_SECTION, 3);

    ss << "··· Writing Section " << Zkey::ZKEY_FF_QR_SECTION << ". QR";
    LOG_INFO(ss);
    ss.str("");
    writeQMap(fdZKey, Zkey::ZKEY_FF_QR_SECTION, 4);

    ss << "··· Writing Section " << Zkey::ZKEY_FF_QM_SECTION << ". QM";
    LOG_INFO(ss);
    ss.str("");
    writeQMap(fdZKey, Zkey::ZKEY_FF_QM_SECTION, 5);

    ss << "··· Writing Section " << Zkey::ZKEY_FF_QO_SECTION << ". QO";
    LOG_INFO(ss);
    ss.str("");
    writeQMap(fdZKey, Zkey::ZKEY_FF_QO_SECTION, 6);

    ss << "··· Writing Section " << Zkey::ZKEY_FF_QC_SECTION << ". QC";
    LOG_INFO(ss);
    ss.str("");
    writeQMap(fdZKey, Zkey::ZKEY_FF_QC_SECTION, 7);

    ss << "··· Writing Section "
       << Zkey::ZKEY_FF_SIGMA1_SECTION << ", "
       << Zkey::ZKEY_FF_SIGMA2_SECTION << ", "
       << Zkey::ZKEY_FF_SIGMA3_SECTION << ". Sigma1, Sigma2 & Sigma 3";
    LOG_INFO(ss);
    ss.str("");
    writeSigma(fdZKey);

    ss << "··· Writing Section " << Zkey::ZKEY_FF_LAGRANGE_SECTION << ". Lagrange Polynomials";
    LOG_INFO(ss);
    ss.str("");
    writeLagrangePolynomials(fdZKey);

    ss << "··· Writing Section " << Zkey::ZKEY_FF_PTAU_SECTION << ". Powers of Tau";
    LOG_INFO(ss);
    ss.str("");
    writePtau(fdZKey, fdPtau);

    ss << "··· Writing Section " << Zkey::ZKEY_FF_C0_SECTION << ". C0";
    LOG_INFO(ss);
    ss.str("");
    writeC0(fdZKey);

    ss << "··· Writing Section " << Zkey::ZKEY_FF_HEADER_SECTION << ". FFlonk Header";
    LOG_INFO(ss);
    ss.str("");
    writeFflonkHeader(fdZKey, fdPtau);

    LOG_INFO("> Writing the zkey file finished");
}

void FflonkSetup::writeZkeyHeader(BinFileWriter &zkeyFile) {
    zkeyFile.startWriteSection(Zkey::ZKEY_HEADER_SECTION);
    zkeyFile.writeU32LE(Zkey::FFLONK_PROTOCOL_ID);
    zkeyFile.endWriteSection();
}

void FflonkSetup::writeAdditions(BinFileWriter &zkeyFile) {
    zkeyFile.startWriteSection(Zkey::ZKEY_FF_ADDITIONS_SECTION);

    for (uint64_t i = 0; i < plonkAdditions.size(); i++) {
        auto addition = plonkAdditions[i];
        zkeyFile.writeU32LE(addition.signal_a);
        zkeyFile.writeU32LE(addition.signal_b);
        zkeyFile.write((void *)&addition.value_a, sizeof(FrElement));
        zkeyFile.write((void *)&addition.value_b, sizeof(FrElement));
    }

    zkeyFile.endWriteSection();
}

void FflonkSetup::writeWitnessMap(BinFileWriter &zkeyFile, uint32_t sectionNum, uint32_t posConstraint) {
    if (posConstraint > 2) {
        throw new runtime_error("Invalid constraint position during writing witness map");
    }

    zkeyFile.startWriteSection(sectionNum);

    for (uint64_t i = 0; i < plonkConstraints.size(); i++) {
        auto constraint = plonkConstraints[i];
        auto value = posConstraint == 0 ? constraint.signal_a : posConstraint == 1 ? constraint.signal_b
                                                                                   : constraint.signal_c;
        zkeyFile.writeU32LE(value);
    }

    zkeyFile.endWriteSection();
}

void FflonkSetup::writeQMap(BinFileWriter &zkeyFile, uint32_t sectionNum, uint32_t posConstraint) {
    if (posConstraint < 3 || posConstraint > 7) {
        throw new runtime_error("Invalid constraint position during writing witness map");
    }

    auto name = posConstraint == 3 ? "QL" : posConstraint == 4 ? "QR"
                                        : posConstraint == 5   ? "QM"
                                        : posConstraint == 6   ? "QO"
                                                               : "QC";

    FrElement *buffer_coefs = new FrElement[settings.domainSize];
    FrElement *buffer_evals = new FrElement[settings.domainSize * 4];
    memset(buffer_evals, 0, settings.domainSize * 4 * sizeof(FrElement));
    for (uint64_t i = 0; i < plonkConstraints.size(); i++) {
        auto constraint = plonkConstraints[i];
        auto value = posConstraint == 3 ? constraint.ql : posConstraint == 4 ? constraint.qr
                                                      : posConstraint == 5   ? constraint.qm
                                                      : posConstraint == 6   ? constraint.qo
                                                                             : constraint.qc;
        buffer_evals[i] = value;
    }

    polynomials[name] = Polynomial<AltBn128::Engine>::fromEvaluations(E, fft, buffer_evals, buffer_coefs, settings.domainSize);
    polynomials[name]->fixDegree();
    Evaluations<AltBn128::Engine>(E, fft, buffer_evals, *polynomials[name], settings.domainSize * 4);

    zkeyFile.startWriteSection(sectionNum);
    zkeyFile.write(buffer_coefs, settings.domainSize * sizeof(FrElement));
    zkeyFile.write(buffer_evals, settings.domainSize * 4 * sizeof(FrElement));
    zkeyFile.endWriteSection();
}

void FflonkSetup::writeSigma(BinFileWriter &zkeyFile) {
    FrElement *sigma = new FrElement[settings.domainSize * 3];
    unordered_map<uint64_t, FrElement> lastSeen;
    unordered_map<uint64_t, uint64_t> firstPos;

    memset(sigma, 0, settings.domainSize * 3 * sizeof(FrElement));
    FrElement w = E.fr.one();
    for (uint64_t i = 0; i < settings.domainSize; i++) {
        auto constraint = plonkConstraints[i];

        if (i < plonkConstraints.size()) {
            buildSigma(sigma, w, lastSeen, firstPos, constraint.signal_a, i);
            buildSigma(sigma, w, lastSeen, firstPos, constraint.signal_b, i + settings.domainSize);
            buildSigma(sigma, w, lastSeen, firstPos, constraint.signal_c, i + settings.domainSize * 2);
        } else if (i < settings.domainSize - 2) {
            buildSigma(sigma, w, lastSeen, firstPos, 0, i);
            buildSigma(sigma, w, lastSeen, firstPos, 0, i + settings.domainSize);
            buildSigma(sigma, w, lastSeen, firstPos, 0, i + settings.domainSize * 2);
        } else {
            sigma[i] = w;
            sigma[i + settings.domainSize] = E.fr.mul(w, k1);
            sigma[i + settings.domainSize * 2] = E.fr.mul(w, k2);
        }

        w = E.fr.mul(w, fft->root(settings.cirPower, 1));
    }

    for (uint64_t i = 0; i < settings.nVars; i++) {
        if (firstPos.find(i) != firstPos.end()) {
            sigma[firstPos[i]] = lastSeen[i];
        } else {
            cout << "Variable not used" << endl;
        }
    }

    for (uint32_t i = 0; i < 3; i++) {
        auto sectionId = i == 0 ? Zkey::ZKEY_FF_SIGMA1_SECTION : i == 1 ? Zkey::ZKEY_FF_SIGMA2_SECTION
                                                                        : Zkey::ZKEY_FF_SIGMA3_SECTION;
        auto name = "S" + to_string(i + 1);

        FrElement *buffer_coefs = new FrElement[settings.domainSize];
        FrElement *buffer_evals = new FrElement[settings.domainSize * 4];
        memset(buffer_evals, 0, settings.domainSize * 4 * sizeof(FrElement));

        polynomials[name] = Polynomial<AltBn128::Engine>::fromEvaluations(E, fft, &sigma[i * settings.domainSize], buffer_coefs, settings.domainSize);
        polynomials[name]->fixDegree();
        Evaluations<AltBn128::Engine>(E, fft, buffer_evals, *polynomials[name], settings.domainSize * 4);

        zkeyFile.startWriteSection(sectionId);
        zkeyFile.write(buffer_coefs, settings.domainSize * sizeof(FrElement));
        zkeyFile.write(buffer_evals, settings.domainSize * 4 * sizeof(FrElement));
        zkeyFile.endWriteSection();
    }
}

void FflonkSetup::buildSigma(FrElement *sigma, FrElement w, unordered_map<uint64_t, FrElement> &lastSeen, unordered_map<uint64_t, uint64_t> &firstPos, uint64_t signalId, uint64_t idx) {
    if (lastSeen.find(signalId) == lastSeen.end()) {
        firstPos[signalId] = idx;
    } else {
        sigma[idx] = lastSeen[signalId];
    }

    FrElement v;
    if (idx < settings.domainSize) {
        v = w;
    } else if (idx < settings.domainSize * 2) {
        v = E.fr.mul(w, k1);
    } else {
        v = E.fr.mul(w, k2);
    }

    lastSeen[signalId] = v;
}

void FflonkSetup::writeLagrangePolynomials(BinFileWriter &zkeyFile) {
    auto l = max(settings.nPublics, (uint64_t)1);

    FrElement *buffer_coefs = new FrElement[settings.domainSize];
    FrElement *buffer_evals = new FrElement[settings.domainSize * 4];

    zkeyFile.startWriteSection(Zkey::ZKEY_FF_LAGRANGE_SECTION);

    for (uint64_t i = 0; i < l; i++) {
        memset(buffer_evals, 0, settings.domainSize * 4 * sizeof(FrElement));

        buffer_evals[i] = E.fr.one();

        auto pol = Polynomial<AltBn128::Engine>::fromEvaluations(E, fft, buffer_evals, buffer_coefs, settings.domainSize);
        pol->fixDegree();
        Evaluations<AltBn128::Engine>(E, fft, buffer_evals, *pol, settings.domainSize * 4);

        zkeyFile.write(buffer_coefs, settings.domainSize * sizeof(FrElement));
        zkeyFile.write(buffer_evals, settings.domainSize * 4 * sizeof(FrElement));
    }

    zkeyFile.endWriteSection();
}

void FflonkSetup::writePtau(BinFileWriter &zkeyFile, BinFile &fdPtau) {
    int nThreads = omp_get_max_threads() / 2;
    PTau = new G1PointAffine[settings.domainSize * 9];

    ThreadUtils::parset(PTau, 0, sizeof(G1PointAffine), nThreads);
    ThreadUtils::parcpy(PTau, fdPtau.getSectionData(2), (settings.domainSize * 9) * sizeof(G1PointAffine), nThreads);

    zkeyFile.startWriteSection(Zkey::ZKEY_FF_PTAU_SECTION);
    zkeyFile.write(PTau, (settings.domainSize * 9) * sizeof(G1PointAffine));
    zkeyFile.endWriteSection();
}

void FflonkSetup::writeC0(BinFileWriter &zkeyFile) {
    CPolynomial<AltBn128::Engine> *C0 = new CPolynomial(E, 8);

    C0->addPolynomial(0, polynomials["QL"]);
    C0->addPolynomial(1, polynomials["QR"]);
    C0->addPolynomial(2, polynomials["QO"]);
    C0->addPolynomial(3, polynomials["QM"]);
    C0->addPolynomial(4, polynomials["QC"]);
    C0->addPolynomial(5, polynomials["S1"]);
    C0->addPolynomial(6, polynomials["S2"]);
    C0->addPolynomial(7, polynomials["S3"]);

    FrElement *bufferC0 = new FrElement[settings.domainSize * 8];
    memset(bufferC0, 0, settings.domainSize * 8 * sizeof(FrElement));

    polynomials["C0"] = C0->getPolynomial(bufferC0);

    // Check degree
    if (polynomials["C0"]->getDegree() >= 8 * settings.domainSize) {
        throw runtime_error("C0 Polynomial is not well calculated");
    }

    zkeyFile.startWriteSection(Zkey::ZKEY_FF_C0_SECTION);
    zkeyFile.write(bufferC0, settings.domainSize * 8 * sizeof(FrElement));
    zkeyFile.endWriteSection();
}

void FflonkSetup::writeFflonkHeader(BinFileWriter &zkeyFile, BinFile &ptauFile) {
    zkeyFile.startWriteSection(Zkey::ZKEY_FF_HEADER_SECTION);

    auto n8q = 32;
    uint8_t bytes[32];
    mpz_class auxScalar;
    auxScalar.set_str("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10);
    scalar2bytes(auxScalar, bytes);
    zkeyFile.writeU32LE(n8q);
    zkeyFile.write((void *)bytes, n8q);

    auto n8r = 32;
    auxScalar.set_str("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);
    scalar2bytes(auxScalar, bytes);
    zkeyFile.writeU32LE(n8r);
    zkeyFile.write((void *)bytes, n8r);

    zkeyFile.writeU32LE(settings.nVars);
    zkeyFile.writeU32LE(settings.nPublics);
    zkeyFile.writeU32LE(settings.domainSize);
    zkeyFile.writeU32LE(plonkAdditions.size());
    zkeyFile.writeU32LE(plonkConstraints.size());

    zkeyFile.write(&k1, sizeof(FrElement));
    zkeyFile.write(&k2, sizeof(FrElement));

    zkeyFile.write(&w3, sizeof(FrElement));
    zkeyFile.write(&w4, sizeof(FrElement));
    zkeyFile.write(&w8, sizeof(FrElement));
    zkeyFile.write(&wr, sizeof(FrElement));

    G2PointAffine bX_2;
    memcpy(&bX_2, (G2PointAffine *)ptauFile.getSectionData(3) + 1, sizeof(G2PointAffine));
    zkeyFile.write(&bX_2, sizeof(G2PointAffine));

    u_int64_t lengths[8] = {polynomials["QL"]->getDegree() + 1,
                            polynomials["QR"]->getDegree() + 1,
                            polynomials["QO"]->getDegree() + 1,
                            polynomials["QM"]->getDegree() + 1,
                            polynomials["QC"]->getDegree() + 1,
                            polynomials["S1"]->getDegree() + 1,
                            polynomials["S2"]->getDegree() + 1,
                            polynomials["S3"]->getDegree() + 1};
    G1Point commitC0 = multiExponentiation(polynomials["C0"], 8, lengths);
    G1PointAffine commitC0Affine;
    E.g1.copy(commitC0Affine, commitC0);
    zkeyFile.write(&commitC0Affine, sizeof(G1PointAffine));

    zkeyFile.endWriteSection();
}

FrElement *FflonkSetup::polynomialFromMontgomery(Polynomial<AltBn128::Engine> *polynomial) {
    const u_int64_t length = polynomial->getLength();

    FrElement *result = new FrElement[length];
    int nThreads = omp_get_max_threads() / 2;
    ThreadUtils::parset(result, 0, length * sizeof(FrElement), nThreads);

#pragma omp parallel for
    for (u_int32_t index = 0; index < length; ++index) {
        E.fr.fromMontgomery(result[index], polynomial->coef[index]);
    }

    return result;
}

G1Point FflonkSetup::multiExponentiation(Polynomial<AltBn128::Engine> *polynomial, u_int32_t nx, u_int64_t x[]) {
    G1Point value;
    FrElement *pol = polynomialFromMontgomery(polynomial);
    E.g1.multiMulByScalar(value, PTau, (uint8_t *)pol, sizeof(pol[0]), polynomial->getDegree() + 1, nx, x);
    return value;
}

void FflonkSetup::scalar2bytes(mpz_class s, uint8_t (&bytes)[32]) {
    mpz_class ScalarMask8("FF", 16);
    mpz_class ScalarZero("0", 16);

    for (uint64_t i = 0; i < 32; i++) {
        mpz_class aux = s & ScalarMask8;
        bytes[i] = aux.get_ui();
        s = s >> 8;
    }
    if (s != ScalarZero) {
        LOG_ERROR("scalar2bytes() run out of space of 32 bytes");
        throw runtime_error("scalar2bytes() run out of space of 32 bytes");
    }
}
}  // namespace Fflonk