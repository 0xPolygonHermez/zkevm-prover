#ifndef SHPLONK_HPP
#define SHPLONK_HPP

#include <iostream>
#include <string.h>
#include <vector>
#include <binfile_utils.hpp>
#include <nlohmann/json.hpp>
#include "polynomial/polynomial.hpp"
#include "polynomial/cpolynomial.hpp"
#include "zkey_pilfflonk.hpp"
#include "shplonk.hpp"
#include "pilfflonk_transcript.hpp"
#include <alt_bn128.hpp>
#include "fft.hpp"

using json = nlohmann::json;

using namespace std;

namespace ShPlonk {
    class ShPlonkProver {

        AltBn128::Engine &E;

        using FrElement = typename AltBn128::FrElement;
        using G1Point = typename AltBn128::G1Point;
        using G1PointAffine = typename AltBn128::G1PointAffine;

        PilFflonkZkey::PilFflonkZkey *zkeyPilFflonk;
        
        PilFflonkTranscript *transcript;

        FrElement challengeXiSeed;
        FrElement challengeXi;
        FrElement challengeAlpha;
        FrElement challengeY;

        std::map<std::string, FrElement *> rootsMap;

        std::map<std::string, Polynomial<AltBn128::Engine> *> polynomialsShPlonk;

        std::map <std::string, AltBn128::G1Point> polynomialCommitments;

        std::map<std::string, AltBn128::FrElement> evaluationCommitments;

        std::map<std::string, std::map<u_int32_t, FrElement>> randomCoefs;
        
        std::vector<u_int32_t> openingPoints;

        std::vector<FrElement> inverseElements;

    public:
        void addPolynomialCommitment(const std::string &key, AltBn128::G1Point commit);

        AltBn128::G1Point getPolynomialCommitment(const std::string &key);

        void addPolynomialShPlonk(const std::string &key, Polynomial<AltBn128::Engine>* pol);

        void addRandomCoef(const std::string &key, u_int32_t pos, FrElement coef);
        
        Polynomial<AltBn128::Engine> * getPolynomialShPlonk(const std::string &key);

        ShPlonkProver(AltBn128::Engine &_E, PilFflonkZkey::PilFflonkZkey *zkey);

        ~ShPlonkProver();

        void reset();

        void commit(u_int32_t stage, FrElement* buffCoefs, G1PointAffine *PTau, std::map<std::string, AltBn128::FrElement *> ptrShPlonk);

        json open(G1PointAffine *PTau, AltBn128::FrElement * buffCoefsConstant, std::map<std::string, AltBn128::FrElement *> ptrCommitted, std::map<std::string, AltBn128::FrElement *> ptrShPlonk, FrElement xiSeed, std::vector<std::string> nonCommittedPols);

        json toJson();

        FrElement getChallengeXi();
    protected:
        void computeR();

        void computeZT();

        void computeL(AltBn128::FrElement *reservedBuffer, AltBn128::FrElement *tmpBuffer);

        void computeZTS2();

        void computeW(AltBn128::FrElement *reservedBuffer, AltBn128::FrElement *tmpBuffer);

        void computeWp(AltBn128::FrElement *reservedBuffer, AltBn128::FrElement *tmpBuffer);

        void computeChallengeXiSeed(FrElement previousChallenge);

        void computeChallengeAlpha(std::vector<std::string> nonCommittedPols);
        
        void computeChallengeY(G1Point W);

        void calculateOpeningPoints();

        void calculateRoots();

        void getMontgomeryBatchedInverse();

        void computeLiSingleOpeningPoint(u_int32_t i);

        void computeLiMultipleOpeningPoints(u_int32_t i);

        void computeLiTwoOpeningPoints(u_int32_t i);

        void calculateEvaluations(FrElement* buffConstantCoefs, std::map<std::string, AltBn128::FrElement *> ptrCommitted);

        AltBn128::FrElement fastEvaluate(u_int32_t stage, FrElement* buffCoefs, u_int32_t nPols, u_int32_t degree, u_int32_t id, std::string polName, FrElement openingPoint);

        u_int32_t findDegree(u_int32_t fIndex, std::string name);

        u_int32_t findPolId(u_int32_t stage, std::string name);

        // void prepareCommits(std::map<std::string, AltBn128::FrElement *> ptrShPlonk);

        // void sumCommits(u_int32_t nCommits, std::string* polynomialsNames, std::string dstName);

        // void sumPolynomials(u_int32_t nPols, std::string* polynomialsNames, std::map<std::string, AltBn128::FrElement *> ptrShPlonk, std::string dstName);

        int find(std::string* arr, u_int32_t n, std::string x);
        int find(u_int32_t* arr, u_int32_t n, u_int32_t x);

        AltBn128::G1Point multiExponentiation(G1PointAffine *PTau, Polynomial<AltBn128::Engine> *polynomial, u_int32_t nx, u_int64_t x[]);
        
        FrElement *polynomialFromMontgomery(Polynomial<AltBn128::Engine> *polynomial);

        void getCommittedPolynomial(u_int32_t stage, FrElement* buffCoefs, FrElement* reservedBuffer, PilFflonkZkey::ShPlonkPol* pol, u_int64_t* degrees, u_int64_t* polsIds);

    };
}

#endif
