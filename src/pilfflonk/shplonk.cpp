#include <sstream>
#include "timer.hpp"
#include "zklog.hpp"
#include "shplonk.hpp"
#include <algorithm>

namespace ShPlonk {

    ShPlonkProver::ShPlonkProver(AltBn128::Engine &_E, PilFflonkZkey::PilFflonkZkey *zkey) : E(_E) {
        zkeyPilFflonk = zkey;

        transcript = new PilFflonkTranscript(_E);

        this->reset();
    }

    ShPlonkProver::~ShPlonkProver() {
        this->reset();

        delete transcript;
        delete zkeyPilFflonk;
    }  

    void ShPlonkProver::reset() {
        this->polynomialsShPlonk.clear();
        this->rootsMap.clear();
        this->evaluationCommitments.clear();
        this->polynomialCommitments.clear();
        this->openingPoints.clear();
        this->inverseElements.clear();

        for (auto& x : this->randomCoefs) {
            x.second.clear();
        }
        this->randomCoefs.clear();
    }

    void ShPlonkProver::addPolynomialShPlonk(const std::string &key, Polynomial<AltBn128::Engine> *pol) {
        polynomialsShPlonk[key] = pol;
    }

    Polynomial<AltBn128::Engine> * ShPlonkProver::getPolynomialShPlonk(const std::string &key) {
        return this->polynomialsShPlonk[key];
    }

    void ShPlonkProver::addRandomCoef(const std::string &key, u_int32_t pos, FrElement coef) {
        this->randomCoefs[key][pos] = coef;
    };

    void ShPlonkProver::addPolynomialCommitment(const std::string &key, AltBn128::G1Point commit) {
        polynomialCommitments[key] = commit;
    }

    AltBn128::G1Point ShPlonkProver::getPolynomialCommitment(const std::string &key) {
        return this->polynomialCommitments[key];
    }

    void ShPlonkProver::computeR() {
        TimerStart(SHPLONK_COMPUTE_R_POLYNOMIALS);

        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            u_int32_t nRoots = zkeyPilFflonk->f[i]->nPols * zkeyPilFflonk->f[i]->nOpeningPoints;
            FrElement* evals = new FrElement[nRoots];

            for(u_int32_t j = 0; j < nRoots; ++j) {
                evals[j] = polynomialsShPlonk["f" + std::to_string(i)]->fastEvaluate(rootsMap["f" + std::to_string(i)][j]);
            }
 
            polynomialsShPlonk["R" + std::to_string(i)] = Polynomial<AltBn128::Engine>::lagrangePolynomialInterpolation(rootsMap["f" + std::to_string(i)], evals, nRoots);
            
            // Check the degree of R0(X) < (degree - 1)
            if (polynomialsShPlonk["R" + std::to_string(i)]->getDegree() > (nRoots - 1))
            {
                throw std::runtime_error("Polynomial is not well calculated");
            }

            delete[] evals;
        }

        TimerStopAndLog(SHPLONK_COMPUTE_R_POLYNOMIALS);
    }

    void ShPlonkProver::computeZT()
    { 
        u_int32_t nRoots = 0;
        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            nRoots +=  zkeyPilFflonk->f[i]->nPols * zkeyPilFflonk->f[i]->nOpeningPoints;
        }

        FrElement* arr = new FrElement[nRoots];

        u_int32_t index = 0;
        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            for(u_int32_t j = 0; j < zkeyPilFflonk->f[i]->nPols * zkeyPilFflonk->f[i]->nOpeningPoints; ++j) {
                arr[index++] = rootsMap["f" + std::to_string(i)][j];
            }

        }

        polynomialsShPlonk["ZT"] = Polynomial<AltBn128::Engine>::zerofierPolynomial(arr, nRoots);

        delete[] arr;
    }

    void ShPlonkProver::computeL(AltBn128::FrElement *reservedBuffer, AltBn128::FrElement *tmpBuffer)
    {
        TimerStart(SHPLONK_COMPUTE_L_POLYNOMIAL);

        FrElement* mulL = new FrElement[zkeyPilFflonk->f.size()];
        FrElement* preL = new FrElement[zkeyPilFflonk->f.size()];
        FrElement* evalRiY = new FrElement[zkeyPilFflonk->f.size()];

        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            FrElement mulLi = E.fr.one();
            for (u_int32_t j = 0; j < zkeyPilFflonk->f[i]->nPols * zkeyPilFflonk->f[i]->nOpeningPoints; j++)
            {
                mulLi = E.fr.mul(mulLi, E.fr.sub(challengeY, rootsMap["f" + std::to_string(i)][j]));
            }
            mulL[i] = mulLi;
        }

       
        FrElement alpha = E.fr.one();
        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            preL[i] = alpha;
            for(u_int32_t j = 0; j < zkeyPilFflonk->f.size(); ++j) {
                if(i != j) {
                    preL[i] = E.fr.mul(preL[i], mulL[j]);
                }
            }
            alpha = E.fr.mul(alpha,challengeAlpha);
        }

        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            evalRiY[i] = polynomialsShPlonk["R" + std::to_string(i)]->fastEvaluate(challengeY);
        }

        u_int64_t maxDegree = 0;
        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            if(zkeyPilFflonk->f[i]->degree > maxDegree) {
                maxDegree = zkeyPilFflonk->f[i]->degree;
            }
        }

        u_int64_t lengthBuffer = maxDegree + 1;

        // COMPUTE L(X)
        polynomialsShPlonk["Wp"] = new Polynomial<AltBn128::Engine>(E, reservedBuffer, lengthBuffer);
        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            auto fTmp = Polynomial<AltBn128::Engine>::fromPolynomial(E, *polynomialsShPlonk["f" + std::to_string(i)], tmpBuffer);
            fTmp->subScalar(evalRiY[i]);
            fTmp->mulScalar(preL[i]);
            polynomialsShPlonk["Wp"]->add(*fTmp);
        }
       
        computeZT();

        FrElement evalZTY = polynomialsShPlonk["ZT"]->fastEvaluate(challengeY);
        polynomialsShPlonk["W"]->mulScalar(evalZTY);
        polynomialsShPlonk["Wp"]->sub(*polynomialsShPlonk["W"]);

        if (polynomialsShPlonk["Wp"]->getDegree() > maxDegree)
        {
            throw std::runtime_error("L Polynomial is not well calculated");
        }

        delete[] mulL;
        delete[] preL;
        delete[] evalRiY;

        TimerStopAndLog(SHPLONK_COMPUTE_L_POLYNOMIAL);      
    }

    void ShPlonkProver::computeZTS2()
    {
        u_int32_t nRoots = 0;
        for(u_int32_t i = 1; i < zkeyPilFflonk->f.size(); ++i) {
            nRoots += zkeyPilFflonk->f[i]->nPols * zkeyPilFflonk->f[i]->nOpeningPoints;
        }

        FrElement* arr = new FrElement[nRoots];

        u_int32_t index = 0;
        for(u_int32_t i = 1; i < zkeyPilFflonk->f.size(); ++i) {
            for(u_int32_t j = 0; j < zkeyPilFflonk->f[i]->nPols * zkeyPilFflonk->f[i]->nOpeningPoints; ++j) {
                arr[index++] = rootsMap["f" + std::to_string(i)][j];
            }
        }


        polynomialsShPlonk["ZTS2"] = Polynomial<AltBn128::Engine>::zerofierPolynomial(arr, nRoots);

        delete[] arr;
    }

    void ShPlonkProver::computeW(AltBn128::FrElement *reservedBuffer, AltBn128::FrElement *tmpBuffer)
    {

        TimerStart(SHPLONK_COMPUTE_W_POLYNOMIAL);

        u_int64_t nTotalRoots = 0;
        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            nTotalRoots += zkeyPilFflonk->f[i]->nPols * zkeyPilFflonk->f[i]->nOpeningPoints;
        }

        u_int64_t maxDegree = 0;
        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            u_int64_t fiDegree = zkeyPilFflonk->f[i]->degree + nTotalRoots - zkeyPilFflonk->f[i]->nPols * zkeyPilFflonk->f[i]->nOpeningPoints;
            if(fiDegree > maxDegree) {
                maxDegree = fiDegree;
            }
        }

        u_int64_t lengthBuffer = maxDegree + 1;

        Polynomial<AltBn128::Engine> * polynomialW = new Polynomial<AltBn128::Engine>(E, reservedBuffer, lengthBuffer);
  
        FrElement* initialOpenValues = new FrElement[openingPoints.size()];
        for(u_int32_t i = 0; i < openingPoints.size(); ++i) {
            initialOpenValues[i] = challengeXi;
            for(u_int32_t j = 0; j < openingPoints[i]; ++j) {
                initialOpenValues[i] = E.fr.mul(initialOpenValues[i], zkeyPilFflonk->omegas["w1_1d1"]);
            }
        } 

        FrElement* alphas = new FrElement[zkeyPilFflonk->f.size()];
        alphas[0] = E.fr.one();
        for(u_int32_t i = 1; i < zkeyPilFflonk->f.size(); ++i) {
            alphas[i] = E.fr.mul(alphas[i - 1], challengeAlpha);
        }

        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            auto fTmp = Polynomial<AltBn128::Engine>::fromPolynomial(E, *polynomialsShPlonk["f" + std::to_string(i)], tmpBuffer);
            fTmp->sub(*polynomialsShPlonk["R" + std::to_string(i)]);
            fTmp->mulScalar(alphas[i]);

            for(u_int32_t j = 0; j < zkeyPilFflonk->f[i]->nOpeningPoints; ++j) {
                u_int32_t openingPoint = zkeyPilFflonk->f[i]->openingPoints[j];
                auto found = std::find(openingPoints.begin(), openingPoints.end(), openingPoint);
                if (found == openingPoints.end()) throw std::runtime_error("Opening point not found");
                FrElement openValue = initialOpenValues[std::distance(openingPoints.begin(), found)];

                fTmp->divByZerofier(zkeyPilFflonk->f[i]->nPols, openValue);
            }
           
            polynomialW->add(*fTmp);           
        }

        if (polynomialW->getDegree() > maxDegree - nTotalRoots)
        {
            throw std::runtime_error("W Polynomial is not well calculated");
        }
        
        delete[] initialOpenValues;

        polynomialsShPlonk["W"] = polynomialW;
        TimerStopAndLog(SHPLONK_COMPUTE_W_POLYNOMIAL);
        
    }

    void ShPlonkProver::computeWp(AltBn128::FrElement *reservedBuffer, AltBn128::FrElement *tmpBuffer) {
        TimerStart(SHPLONK_COMPUTE_WP_POLYNOMIAL);

        computeL(reservedBuffer, tmpBuffer);
        computeZTS2();

        FrElement ZTS2Y = polynomialsShPlonk["ZTS2"]->fastEvaluate(challengeY);
        E.fr.inv(ZTS2Y, ZTS2Y); 
        polynomialsShPlonk["Wp"]->mulScalar(ZTS2Y);
        polynomialsShPlonk["Wp"]->divByXSubValue(challengeY);

        u_int64_t maxDegree = 0; 
        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            if(zkeyPilFflonk->f[i]->degree > maxDegree) {
                maxDegree = zkeyPilFflonk->f[i]->degree;
            }
        }

        maxDegree -= 1;

        if (polynomialsShPlonk["Wp"]->getDegree() > maxDegree)
        {
            throw std::runtime_error("Degree of L(X)/(ZTS2(y)(X-y)) is not correct");
        }

        TimerStopAndLog(SHPLONK_COMPUTE_WP_POLYNOMIAL);
    }

    void ShPlonkProver::computeChallengeXiSeed(FrElement previousChallenge)
    {     
        transcript->reset();
        transcript->addScalar(previousChallenge);

        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            if(zkeyPilFflonk->f[i]->nStages > 1 || zkeyPilFflonk->f[i]->stages[0].stage != 0) {
                G1Point commit = polynomialCommitments["f" + std::to_string(zkeyPilFflonk->f[i]->index)]; 
                transcript->addPolCommitment(commit);
            }
        }

        challengeXiSeed = transcript->getChallenge();
    }


    void ShPlonkProver::computeChallengeAlpha(std::vector<std::string> nonCommittedPols)
    {    
        transcript->reset();
        transcript->addScalar(challengeXiSeed);
        
        //Calculate evaluations size
        u_int32_t nEvaluations = 0;
        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            nEvaluations += zkeyPilFflonk->f[i]->nPols * zkeyPilFflonk->f[i]->nOpeningPoints;
        }

        std::string * evaluationsNames = new std::string[nEvaluations];

        //Calculate evaluations names
        int index = 0;
        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            for(u_int32_t j = 0; j < zkeyPilFflonk->f[i]->nOpeningPoints; ++j) {
            u_int32_t openingPoint = zkeyPilFflonk->f[i]->openingPoints[j];
                std::string wPower = openingPoint == 0 ? "" : (openingPoint == 1 ? "w" : "w" + std::to_string(openingPoint));

                for(u_int32_t k = 0; k < zkeyPilFflonk->f[i]->nPols; ++k) {
                    std::string polName = zkeyPilFflonk->f[i]->pols[k];
                    evaluationsNames[index++] = polName + wPower; 
                }
            }
        }

        for (u_int32_t i = 0; i < nEvaluations; ++i) {
            if(std::find(nonCommittedPols.begin(), nonCommittedPols.end(), evaluationsNames[i]) == nonCommittedPols.end()) {
                transcript->addScalar(evaluationCommitments[evaluationsNames[i]]);
            }
        }

        delete[] evaluationsNames;

        challengeAlpha = transcript->getChallenge();
    }

    void ShPlonkProver::computeChallengeY(G1Point W)
    {    
        transcript->reset();
        transcript->addScalar(challengeAlpha);
        transcript->addPolCommitment(W);
        challengeY = transcript->getChallenge();
    }

    void ShPlonkProver::calculateOpeningPoints() {
     
        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            for(u_int32_t j = 0; j < zkeyPilFflonk->f[i]->nOpeningPoints; ++j) {
                auto it = std::find(openingPoints.begin(), openingPoints.end(), zkeyPilFflonk->f[i]->openingPoints[j]);
                if(it == openingPoints.end()) {
                    openingPoints.push_back(zkeyPilFflonk->f[i]->openingPoints[j]);
                }
            }
        }        
    }

    void ShPlonkProver::calculateRoots() {
        TimerStart(SHPLONK_CALCULATE_ROOTS);

        std::map<std::string, FrElement *> omegasMap;

        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); i++) {
            u_int32_t nPols =  zkeyPilFflonk->f[i]->nPols;
            std::string omega = "w" + std::to_string(nPols);
            FrElement initialOmega =zkeyPilFflonk->omegas[omega];
            
            rootsMap["f" + std::to_string(i)] = new FrElement[nPols*zkeyPilFflonk->f[i]->nOpeningPoints];

            for(u_int32_t k = 0; k < zkeyPilFflonk->f[i]->nOpeningPoints; k++) {

                u_int32_t openingPoint = zkeyPilFflonk->f[i]->openingPoints[k];
                
                FrElement initialValue = openingPoint == 0 ? E.fr.one() : zkeyPilFflonk->omegas[omega + "_" + std::to_string(openingPoint) + "d" + std::to_string(nPols)];
                
                std::string wName = omega + "_" + std::to_string(openingPoint);

                omegasMap[wName] = new FrElement[nPols];
                omegasMap[wName][0] = E.fr.one();
                for (u_int32_t j = 1; j < nPols; j++)
                {   
                    omegasMap[wName][j] = E.fr.mul(omegasMap[wName][j - 1], initialOmega);
                }

                rootsMap["f" + std::to_string(i)][nPols*k] = initialValue;

                for(u_int32_t j = 0; j < zkeyPilFflonk->powerW / nPols; ++j) {
                    rootsMap["f" + std::to_string(i)][nPols*k] = E.fr.mul(rootsMap["f" + std::to_string(i)][nPols*k], challengeXiSeed);
                }

                for (u_int32_t j = 1; j < nPols; j++)
                {
                    rootsMap["f" + std::to_string(i)][j + nPols*k] = E.fr.mul(rootsMap["f" + std::to_string(i)][nPols*k], omegasMap[wName][j]);
                }
            }
        }

        for (auto const &x : omegasMap) {
            delete[] x.second;
        }
        omegasMap.clear();

        TimerStopAndLog(SHPLONK_CALCULATE_ROOTS);
    }

    void ShPlonkProver::getMontgomeryBatchedInverse()
    {   
        TimerStart(SHPLONK_CALCULATE_MONTGOMERY_BATCHED_INVERSE);
     
        std::map<std::string, bool> isAddedMulLi;

        for(u_int32_t i = 1; i < zkeyPilFflonk->f.size(); ++i) {
            std::string nPols = std::to_string(zkeyPilFflonk->f[i]->nPols);

            std::string concatenatedOpenings;

            for(u_int32_t j = 0; j < zkeyPilFflonk->f[i]->nOpeningPoints; ++j) {
                concatenatedOpenings += std::to_string(zkeyPilFflonk->f[i]->openingPoints[j]);
            }

            std::string wName = zkeyPilFflonk->f[i]->openingPoints[0] == 0 
                ? nPols + "_" + concatenatedOpenings
                : nPols + "_" + std::to_string(zkeyPilFflonk->f[i]->openingPoints[0]) + "d" + nPols + "_" +  concatenatedOpenings;

            if (isAddedMulLi.find(wName) == isAddedMulLi.end()) {
                isAddedMulLi[wName] = true;
                FrElement mulLi = E.fr.one();
                for (u_int32_t j = 0; j < zkeyPilFflonk->f[i]->nPols * zkeyPilFflonk->f[i]->nOpeningPoints; j++)
                {
                    mulLi = E.fr.mul(mulLi, E.fr.sub(challengeY, rootsMap["f" + std::to_string(i)][j]));
                }
                inverseElements.push_back(mulLi);
            }
        }

        std::map<std::string, bool> isAddedDen;

        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            std::string nPols = std::to_string(zkeyPilFflonk->f[i]->nPols);

            std::string concatenatedOpenings;

            for(u_int32_t j = 0; j < zkeyPilFflonk->f[i]->nOpeningPoints; ++j) {
                concatenatedOpenings += std::to_string(zkeyPilFflonk->f[i]->openingPoints[j]);
            }

            std::string wName = zkeyPilFflonk->f[i]->openingPoints[0] == 0 
                ? nPols + "_" + concatenatedOpenings
                : nPols + "_" + std::to_string(zkeyPilFflonk->f[i]->openingPoints[0]) + "d" + nPols + "_" +  concatenatedOpenings;

            if (isAddedDen.find(wName) == isAddedDen.end()) {
                isAddedDen[wName] = true;

                if(zkeyPilFflonk->f[i]->nOpeningPoints > 2) {
                    computeLiMultipleOpeningPoints(i);
                } else if(zkeyPilFflonk->f[i]->nOpeningPoints == 2) {
                    computeLiTwoOpeningPoints(i);
                } else if(zkeyPilFflonk->f[i]->nPols > 1) {
                    computeLiSingleOpeningPoint(i);
                }
            }
        }
        
        

        FrElement mulAccumulator = E.fr.one();
        for (u_int32_t i = 0; i < inverseElements.size(); i++)
        {            
            mulAccumulator = E.fr.mul(mulAccumulator, inverseElements[i]);
        }
      
        E.fr.inv(mulAccumulator, mulAccumulator);

        evaluationCommitments["inv"] = mulAccumulator;

        TimerStopAndLog(SHPLONK_CALCULATE_MONTGOMERY_BATCHED_INVERSE);
    }    

    void ShPlonkProver::computeLiMultipleOpeningPoints(u_int32_t i) {
        u_int32_t nRoots = zkeyPilFflonk->f[i]->nPols *  zkeyPilFflonk->f[i]->nOpeningPoints;
        for (u_int64_t j = 0; j < nRoots; j++) {
            u_int32_t idx = j;
            FrElement den = E.fr.one();
            for(u_int32_t k = 0; k < nRoots - 1; k++) {
                idx = (idx + 1) % nRoots;
                den = E.fr.mul(den, E.fr.sub(rootsMap["f" + std::to_string(i)][j], rootsMap["f" + std::to_string(i)][idx]));
            }
            inverseElements.push_back(den);
        }
    }

    void ShPlonkProver::computeLiTwoOpeningPoints(u_int32_t i) {
        u_int32_t len = zkeyPilFflonk->f[i]->nPols;

        FrElement xi0 = challengeXi;
        for(u_int32_t j = 0; j < zkeyPilFflonk->f[i]->openingPoints[0]; ++j) {
            xi0 = E.fr.mul(xi0, zkeyPilFflonk->omegas["w1_1d1"]);
        }

        FrElement xi1 = challengeXi;
        for(u_int32_t j = 0; j < zkeyPilFflonk->f[i]->openingPoints[1]; ++j) {
            xi1 = E.fr.mul(xi1, zkeyPilFflonk->omegas["w1_1d1"]);
        }

        if(len == 1) {
            inverseElements.push_back(E.fr.sub(rootsMap["f" + std::to_string(i)][0], rootsMap["f" + std::to_string(i)][1]));
            inverseElements.push_back(E.fr.sub(rootsMap["f" + std::to_string(i)][1], rootsMap["f" + std::to_string(i)][0]));
            return;
        }

        FrElement den1 = E.fr.set(len);
        if(len > 2) {
            for (u_int64_t j = 0; j < (len - 2); j++)
            {
                den1 = E.fr.mul(den1, rootsMap["f" + std::to_string(i)][0]);
            }
        }

        den1 = E.fr.mul(den1, E.fr.sub(xi0, xi1));
        for (u_int64_t j = 0; j < len; j++) {
            FrElement den2 = rootsMap["f" + std::to_string(i)][(len - 1) * j % len];
            FrElement den3 = E.fr.sub(challengeY, rootsMap["f" + std::to_string(i)][j]);

            inverseElements.push_back(E.fr.mul(den1, E.fr.mul(den2, den3)));
        }

        den1 = E.fr.set(len);
        if(len > 2) {
            for (u_int64_t j = 0; j < (len - 2); j++)
            {
                den1 = E.fr.mul(den1, rootsMap["f" + std::to_string(i)][len]);
            }
        }

        den1 = E.fr.mul(den1, E.fr.sub(xi1, xi0));
        for (u_int64_t j = 0; j < len; j++) {
            FrElement den2 = rootsMap["f" + std::to_string(i)][len + ((len - 1) * j % len)];
            FrElement den3 = E.fr.sub(challengeY, rootsMap["f" + std::to_string(i)][len + j]);

            inverseElements.push_back(E.fr.mul(den1, E.fr.mul(den2, den3)));
        }

       
    }

    void ShPlonkProver::computeLiSingleOpeningPoint(u_int32_t i) {
        u_int32_t len = zkeyPilFflonk->f[i]->nPols *  zkeyPilFflonk->f[i]->nOpeningPoints;

        if(len == 1) return;

        FrElement den1 = E.fr.set(len);
        if(len > 2) {
            for (u_int64_t j = 0; j < (len - 2); j++)
            {
                den1 = E.fr.mul(den1, rootsMap["f" + std::to_string(i)][0]);
            }
        }
        
        for (uint j = 0; j < len; j++) {
            FrElement den2 = rootsMap["f" + std::to_string(i)][((len - 1) * j) % len];
            FrElement den3 = E.fr.sub(challengeY, rootsMap["f" + std::to_string(i)][j]);

            inverseElements.push_back(E.fr.mul(E.fr.mul(den1, den2), den3));
        }
    }

    void ShPlonkProver::calculateEvaluations(AltBn128::FrElement* buffConstantCoefs, std::map<std::string, AltBn128::FrElement *> ptrCommitted) {
        TimerStart(SHPLONK_CALCULATE_EVALUATIONS);
        FrElement* initialOpenValues = new FrElement[openingPoints.size()];
        for(u_int32_t i = 0; i < openingPoints.size(); ++i) {
            initialOpenValues[i] = challengeXi;
            for(u_int32_t j = 0; j < openingPoints[i]; ++j) {
                initialOpenValues[i] = E.fr.mul(initialOpenValues[i], zkeyPilFflonk->omegas["w1_1d1"]);
            }
        }

        //Calculate evaluations
        for(u_int32_t i = 0; i < zkeyPilFflonk->f.size(); ++i) {
            for(u_int32_t j = 0; j <  zkeyPilFflonk->f[i]->nOpeningPoints; ++j) {
                u_int32_t openingPoint = zkeyPilFflonk->f[i]->openingPoints[j];
                std::string wPower = openingPoint == 0 ? "" : (openingPoint == 1 ? "w" : "w" + std::to_string(openingPoint));

                auto found = std::find(openingPoints.begin(), openingPoints.end(), openingPoint);
                if (found == openingPoints.end()) throw std::runtime_error("Opening point not found");
                FrElement openValue = initialOpenValues[std::distance(openingPoints.begin(), found)];

                for(u_int32_t k = 0; k < zkeyPilFflonk->f[i]->nPols; ++k) {
                    std::string polName = zkeyPilFflonk->f[i]->pols[k];

                    u_int32_t stage = zkeyPilFflonk->f[i]->stages[0].stage;

                    u_int32_t polDegree = findDegree(i, polName);

                    FrElement* buffCoefs = stage == 0 
                        ? buffConstantCoefs 
                        : stage == 4 
                            ? ptrCommitted["q_2ns"]
                            : ptrCommitted["cm" + std::to_string(stage) + "_coefs"];

                    u_int32_t polId = findPolId(stage, polName);

                    u_int32_t nPols = zkeyPilFflonk->polsNamesStage[stage]->size();
                    
                    evaluationCommitments[polName + wPower] = fastEvaluate(stage, buffCoefs, nPols, polDegree, polId, polName, openValue);
                }
            }
        }

        delete[] initialOpenValues;

        TimerStopAndLog(SHPLONK_CALCULATE_EVALUATIONS);
    }

    u_int32_t ShPlonkProver::findPolId(u_int32_t stage, std::string polName) {
        for (const auto& [index, name] : *zkeyPilFflonk->polsNamesStage[stage]) {
            if(name == polName) return index;
        }
        throw std::runtime_error("Polynomial name not found");
    }

    u_int32_t ShPlonkProver::findDegree(u_int32_t fIndex, std::string name) {
        for(u_int32_t i = 0; i < zkeyPilFflonk->f[fIndex]->stages[0].nPols; i++) {
            if(zkeyPilFflonk->f[fIndex]->stages[0].pols[i].name == name) {
                return zkeyPilFflonk->f[fIndex]->stages[0].pols[i].degree;
            }
        }
        throw std::runtime_error("Polynomial name not found");
    }

    AltBn128::FrElement ShPlonkProver::fastEvaluate(u_int32_t stage, FrElement* buffCoefs, u_int32_t nPols, u_int32_t degree, u_int32_t id, std::string polName, FrElement openingPoint) {
        int nThreads = omp_get_max_threads();

        uint64_t nCoefs = degree;
        uint64_t coefsThread = nCoefs / nThreads;
        uint64_t residualCoefs = nCoefs - coefsThread * nThreads;

        FrElement res[nThreads * 4];
        FrElement xN[nThreads * 4];

        xN[0] = E.fr.one();

        #pragma omp parallel for
        for (int i = 0; i < nThreads; i++) {
            res[i*4] = E.fr.zero();

            uint64_t nCoefs = i == (nThreads - 1) ? coefsThread + residualCoefs : coefsThread;
            for (u_int64_t j = nCoefs; j > 0; j--) {
                if(stage == 4) {
                    u_int32_t index = (i * coefsThread) + j - 1;
                    u_int32_t pos = (index) + id * zkeyPilFflonk->maxQDegree * (1 << zkeyPilFflonk->power);
                    FrElement coef;
                    if(zkeyPilFflonk->maxQDegree > 0 && index >= zkeyPilFflonk->maxQDegree * (1 << zkeyPilFflonk->power)) {
                        coef = this->randomCoefs[polName][index];
                    } else {
                        coef = buffCoefs[pos];
                    }
                    res[i*4] = E.fr.add(coef, E.fr.mul(res[i*4], openingPoint));
                } else {
                    res[i*4] = E.fr.add(buffCoefs[((i * coefsThread) + j - 1) * nPols + id], E.fr.mul(res[i*4], openingPoint));
                }
                if (i == 0) xN[0] = E.fr.mul(xN[0], openingPoint);
            }
        }

        for (int i = 1; i < nThreads; i++) {
            res[0] = E.fr.add(res[0], E.fr.mul(xN[i - 1], res[i*4]));
            xN[i] = E.fr.mul(xN[i - 1], xN[0]);
        }

        return res[0];
    }

    // void ShPlonkProver::sumCommits(u_int32_t nCommits, std::string* polynomialsNames, std::string dstName) {
    //     // Initialize commit to zero in G1 curve 
    //     G1Point commit = E.g1.zero();

    //     // Add all the commits
    //     for(u_int32_t i = 0; i < nCommits; ++i) {
    //         E.g1.add(commit, polynomialCommitments[polynomialsNames[i]], commit);  
    //     }

    //     polynomialCommitments[dstName] = commit;
    // }

    // void ShPlonkProver::sumPolynomials(u_int32_t nPols, std::string* polynomialsNames, std::map<std::string, AltBn128::FrElement *> ptrShPlonk, std::string dstName) {
    //     if(nPols == 1) {
    //         polynomialsShPlonk[dstName] = polynomialsShPlonk[polynomialsNames[0]];
    //         return;            
    //     }
    
    //     u_int64_t maxDegree = 0;
    //     for(u_int32_t i = 0; i < nPols; ++i) {
    //         polynomialsShPlonk[polynomialsNames[i]]->fixDegree();
    //         if(polynomialsShPlonk[polynomialsNames[i]]->getDegree() > maxDegree) {
    //             maxDegree = polynomialsShPlonk[polynomialsNames[i]]->getDegree();
    //         }
    //     }

    //     u_int64_t lengthBuffer = maxDegree + 1;
        
    //     polynomialsShPlonk[dstName] = new Polynomial<AltBn128::Engine>(E, ptrShPlonk[dstName], lengthBuffer);

    //     #pragma omp parallel for
    //     for (u_int64_t i = 0; i <= maxDegree; i++) {
    //         FrElement coef = E.fr.zero();
    //         for (u_int32_t j = 0; j < nPols; ++j) {
    //             if(polynomialsShPlonk[polynomialsNames[j]] != nullptr && polynomialsShPlonk[polynomialsNames[j]]->getDegree() > 0 && i <= polynomialsShPlonk[polynomialsNames[j]]->getDegree()) {
    //                 coef = E.fr.add(coef, polynomialsShPlonk[polynomialsNames[j]]->coef[i]);
    //             }
    //         }
    //         polynomialsShPlonk[dstName]->coef[i] = coef;
    //     }
    //     polynomialsShPlonk[dstName]->fixDegree();
    // }

    // void ShPlonkProver::prepareCommits(std::map<std::string, AltBn128::FrElement *> ptrShPlonk) {
    //     u_int32_t nPols = zkeyPilFflonk->f.size();
    //     for(u_int64_t i = 0; i < nPols; ++i) {

    //         std::string* polynomialsNames = new std::string[zkeyPilFflonk->f[i]->nStages];

    //         for(u_int64_t j = 0; j < zkeyPilFflonk->f[i]->nStages; ++j) {
        
    //             std::string index = "f" + std::to_string(zkeyPilFflonk->f[i]->index) + "_" + std::to_string(zkeyPilFflonk->f[i]->stages[j].stage);
    //             polynomialsNames[j] = index;

    //             if(polynomialsShPlonk.find(index) == polynomialsShPlonk.end()) throw std::runtime_error("Polynomial " + index + " is not provided");
    //             if(polynomialCommitments.find(index) == polynomialCommitments.end()) throw std::runtime_error("Commit " + index + " is not provided");
    //         }
            
    //         sumCommits(zkeyPilFflonk->f[i]->nStages, polynomialsNames, "f" + std::to_string(zkeyPilFflonk->f[i]->index));
    //         sumPolynomials(zkeyPilFflonk->f[i]->nStages, polynomialsNames, ptrShPlonk,  "f" + std::to_string(zkeyPilFflonk->f[i]->index));

    //         if(polynomialsShPlonk["f" + std::to_string(zkeyPilFflonk->f[i]->index)]->getDegree() > zkeyPilFflonk->f[i]->degree) {
    //             throw std::runtime_error("Polynomial f" + std::to_string(i) + " degree is greater than expected");
    //         }   
            
    //         delete[] polynomialsNames;
    //     }
    // }

    void ShPlonkProver::commit(u_int32_t stage, FrElement* buffCoefs, G1PointAffine *PTau, std::map<std::string, AltBn128::FrElement *> ptrShPlonk) {
        
        if(NULL == zkeyPilFflonk) {
            throw std::runtime_error("Zkey data not set");
        }

        for (auto it = zkeyPilFflonk->f.begin(); it != zkeyPilFflonk->f.end(); ++it) {
            PilFflonkZkey::ShPlonkPol* pol = it->second;

            u_int32_t* stages = new u_int32_t[pol->nStages];
            for(u_int32_t i = 0; i < pol->nStages; ++i) {
                stages[i] = pol->stages[i].stage;
            }
 
            int stagePos = find(stages, pol->nStages, stage);

            if(stagePos != -1) {
                PilFflonkZkey::ShPlonkStage* stagePol = &pol->stages[stagePos];
                
                u_int64_t* lengths = new u_int64_t[pol->nPols]{};
                u_int64_t* polsIds = new u_int64_t[pol->nPols]{};

                for(u_int32_t j = 0; j < stagePol->nPols; ++j) {
                    std::string name = stagePol->pols[j].name;
                    int index = find(pol->pols, pol->nPols, name);
                    if (index == -1) {
                            throw std::runtime_error("Polynomial " + std::string(name) + " missing");
                    }
                    
                    polsIds[j] = findPolId(stage, name);
                    lengths[index] = findDegree(it->first, name);
                }

                std::string index = "f" + std::to_string(pol->index);

                getCommittedPolynomial(stage, buffCoefs, ptrShPlonk[index], pol, lengths, polsIds);        

                // Check degree
                if (polynomialsShPlonk[index]->getDegree() > pol->degree) 
                {
                    throw std::runtime_error("Committed Polynomial is not well calculated");
                }

                G1Point Fi = multiExponentiation(PTau, polynomialsShPlonk[index], pol->nPols, lengths);
                zklog.info("Commit " + index + ": " + E.g1.toString(Fi));
                polynomialCommitments[index] = Fi;
                
                delete[] lengths;
                delete[] polsIds;
            }

            delete[] stages;
        }
    }

    void ShPlonkProver::getCommittedPolynomial(u_int32_t stage, FrElement* buffCoefs, FrElement* reservedBuffer, PilFflonkZkey::ShPlonkPol* pol, u_int64_t* degrees, u_int64_t* polsIds) {
        
        std::string name = "f" + std::to_string(pol->index);

        u_int32_t nPols = pol->nPols;
        u_int32_t polDegree = pol->degree;

        polynomialsShPlonk[name] = new Polynomial<AltBn128::Engine>(E, reservedBuffer, polDegree + 1);

        u_int32_t nPolsStage = zkeyPilFflonk->polsNamesStage[stage]->size();
            
        #pragma omp parallel for
        for (u_int64_t i = 0; i < polDegree; i++) {
            for (u_int32_t j = 0; j < nPols; j++) {
                if (degrees[j] >= 0 && i < degrees[j]) 
                {
                    if(stage == 4) {
                        u_int32_t pos = polsIds[j] * zkeyPilFflonk->maxQDegree * (1 << zkeyPilFflonk->power) + i;
                        if(zkeyPilFflonk->maxQDegree > 0 && i >= zkeyPilFflonk->maxQDegree * (1 << zkeyPilFflonk->power)) {
                            polynomialsShPlonk[name]->coef[i * nPols + j] = this->randomCoefs[pol->pols[j]][i];
                        } else {
                            polynomialsShPlonk[name]->coef[i * nPols + j] = buffCoefs[pos];
                        }
                    } else {
                        polynomialsShPlonk[name]->coef[i * nPols + j] = buffCoefs[polsIds[j] + nPolsStage * i];
                    }
                }
            }
        }

        polynomialsShPlonk[name]->fixDegree();

    }

    json ShPlonkProver::open(G1PointAffine *PTau, AltBn128::FrElement * buffCoefsConstant, std::map<std::string, AltBn128::FrElement *> ptrCommitted, std::map<std::string, AltBn128::FrElement *> ptrShPlonk, FrElement xiSeed, std::vector<std::string> nonCommittedPols) {
        TimerStart(SHPLONK_OPEN);

        if(NULL == zkeyPilFflonk) {
            throw std::runtime_error("Zkey data not set");
        }

        // Calculate opening points
        calculateOpeningPoints();

        challengeXiSeed = xiSeed;

        challengeXi = E.fr.one();
        for(u_int32_t i = 0; i < zkeyPilFflonk->powerW; ++i) {
            challengeXi = E.fr.mul(challengeXi, challengeXiSeed);
        }

        zklog.info("Challenge xi: " + E.fr.toString(challengeXi));

        // Calculate roots
        calculateRoots();

        calculateEvaluations(buffCoefsConstant, ptrCommitted);        

        computeChallengeAlpha(nonCommittedPols);

        zklog.info("Challenge alpha: " + E.fr.toString(challengeAlpha));

        computeR();
        
        computeW(ptrShPlonk["W"], ptrShPlonk["tmp"]);

        u_int64_t* lengthsW = new u_int64_t[1]{polynomialsShPlonk["W"]->getDegree() + 1};
        G1Point W = multiExponentiation(PTau, polynomialsShPlonk["W"], 1, lengthsW);
        polynomialCommitments["W"] = W;

        zklog.info("Commit W: " + E.g1.toString(polynomialCommitments["W"]));

        computeChallengeY(W);

        zklog.info("Challenge Y: " + E.fr.toString(challengeY));

        computeWp(ptrShPlonk["Wp"], ptrShPlonk["tmp"]);
        u_int64_t* lengthsWp = new u_int64_t[1]{polynomialsShPlonk["Wp"]->getDegree() + 1};
        G1Point Wp = multiExponentiation(PTau, polynomialsShPlonk["Wp"], 1, lengthsWp);

        polynomialCommitments["Wp"] = Wp;

        zklog.info("Commit Wp: " + E.g1.toString(polynomialCommitments["Wp"]));

        getMontgomeryBatchedInverse();

        zklog.info("Batched Inverse shplonk: " + E.fr.toString(evaluationCommitments["inv"]));

        delete[] lengthsW;
        delete[] lengthsWp;

        TimerStopAndLog(SHPLONK_OPEN);
        return toJson();
    }

    AltBn128::G1Point ShPlonkProver::multiExponentiation(G1PointAffine *PTau, Polynomial<AltBn128::Engine> *polynomial, u_int32_t nx, u_int64_t x[])
    {
        TimerStart(SHPLONK_CALCULATE_MSM);
        G1Point value;
        FrElement *pol = this->polynomialFromMontgomery(polynomial);
        E.g1.multiMulByScalar(value, PTau, (uint8_t *)pol, sizeof(pol[0]), polynomial->getDegree() + 1, nx, x);
        TimerStopAndLog(SHPLONK_CALCULATE_MSM);
        return value;
    }

        AltBn128::FrElement *ShPlonkProver::polynomialFromMontgomery(Polynomial<AltBn128::Engine> *polynomial)
    {
        const u_int64_t length = polynomial->getDegree() + 1;
        FrElement *result = new FrElement[length];
        int nThreads = omp_get_max_threads() / 2;
        ThreadUtils::parset(result, 0, length * sizeof(FrElement), nThreads);

#pragma omp parallel for
        for (u_int32_t index = 0; index < length; ++index)
        {
            E.fr.fromMontgomery(result[index], polynomial->coef[index]);
        }

        return result;
    }


    int ShPlonkProver::find(std::string* arr, u_int32_t n, std::string x) {
        for(u_int32_t i = 0; i < n; ++i) {
            if(arr[i] == x) {
                return int(i);
            }
        }

        return -1;
    }

    int ShPlonkProver::find(u_int32_t* arr, u_int32_t n, u_int32_t x) {
        for(u_int32_t i = 0; i < n; ++i) {
            if(arr[i] == x) {
                return int(i);
            }
        }

        return -1;
    }

    AltBn128::FrElement ShPlonkProver::getChallengeXi() {
        return challengeXi;
    }

    json ShPlonkProver::toJson() {
        json jsonProof;

        jsonProof["polynomials"] = {};
        jsonProof["evaluations"] = {};

        for (auto &[key, point]: this->polynomialCommitments) {
            G1PointAffine tmp;
            E.g1.copy(tmp, point);

            jsonProof["polynomials"][key] = {};

            std::string x = E.f1.toString(tmp.x);
            std::string y = E.f1.toString(tmp.y);

            jsonProof["polynomials"][key].push_back(x);
            jsonProof["polynomials"][key].push_back(y);
            jsonProof["polynomials"][key].push_back("1");
        }

        for (auto &[key, element]: this->evaluationCommitments) {
            jsonProof["evaluations"][key] = E.fr.toString(element);
        }

        jsonProof["protocol"] = "pilfflonk";
        jsonProof["curve"] = "bn128";

        return jsonProof;
    }

}
