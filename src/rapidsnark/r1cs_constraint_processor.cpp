#include "r1cs_constraint_processor.hpp"
// #include "timer.hpp"
// #include <stdio.h>
// #include <math.h>

// #include <sys/mman.h>
// #include <sys/stat.h>
// #include <fcntl.h>
// #include <unistd.h>

// #include "thread_utils.hpp"
// #include <omp.h>

using FrElement = typename AltBn128::Engine::FrElement;
namespace R1cs
{
const LINEAR_COMBINATION_NULLABLE = 0;
const LINEAR_COMBINATION_CONSTANT = 1;
const LINEAR_COMBINATION_VARIABLE = 2;

    R1csConstraintProcessor::~R1csConstraintProcessor()
    {
    }

    void R1csConstraintProcessor::processR1csConstraints(Binfile &binfile,
                                                        FflonkSetupSettings &settings,
                                                        vector<R1csConstraint> &lcA,
                                                        vector<R1csConstraint> &lcB,
                                                        vector<R1csConstraint> &lcC,
                                                        std::vector<FrElement> &plonkConstraints,
                                                        std::vector<FrElement> &plonkAdditions)
    {
        this->normalizeLinearCombination(lcA);
        this->normalizeLinearCombination(lcB);
        this->normalizeLinearCombination(lcC);

        const auto lctA = this->getLinearCombinationType(lcA);
        const auto lctB = this->getLinearCombinationType(lcB);

        if((lctA == LINEAR_COMBINATION_NULLABLE) || (lctB == LINEAR_COMBINATION_NULLABLE))
        {
            this->processR1csAdditionConstraint(settings, plonkConstraints, plonkAdditions, lcC);
        } else if (lctA == LINEAR_COMBINATION_CONSTANT) {
            const lcCC = this->joinLinearCombinations(lcB, lcC, lcA[0]);
            this->processR1csAdditionConstraint(settings,  plonkConstraints, plonkAdditions, lcCC);
        } else if (lctB == LINEAR_COMBINATION_CONSTANT) {
            const lcCC = this->joinLinearCombinations(lcA, lcC, lcB[0]);
            this->processR1csAdditionConstraint(settings,  plonkConstraints, plonkAdditions, lcCC);
        } else {
            this->processR1csMultiplicationConstraint(settings,  plonkConstraints, plonkAdditions, lcA, lcB, lcC);
        }
    }    

    int R1csConstraintProcessor::getLinearCombinationType(vector<R1csConstraint> &lc) {
        FrElement k = E.fr.zero();
        uint64_t n = 0;

        auto it = lc.begin();
        while (it != lc.end()) {
            if (E.fr.eq(it->value, E.fr.zero())) {
                it = lc.erase(it); // Removes the element and updates the iterator
            } else {
                if(it->signal_id == 0) {
                    k = E.fr.add(k, + it->value);
                } else {
                    n++;
                }
                ++it;
            }
        }

        if(n>0) return LINEAR_COMBINATION_VARIABLE;
        if(!E.fr.eq(k, E.fr.zero())) return LINEAR_COMBINATION_CONSTANT;
        return LINEAR_COMBINATION_NULLABLE;
    }

    vector<R1csConstraint> R1csConstraintProcessor::normalizeLinearCombination(vector<R1csConstraint> &lc) {
        auto it = lc.begin();
        while (it != lc.end()) {
            if (E.fr.eq(it->value, E.fr.zero())) {
                it = lc.erase(it); // Removes the element and updates the iterator
            } else {
                ++it;
            }
        }
    }

    vector<R1csConstraint> R1csConstraintProcessor::joinLinearCombinations(vector<R1csConstraint> &lcA, vector<R1csConstraint> &lcB, FrElement k) {
        std::map<uint32_t, FrElement> res;
        for (auto const &x : lcA) {
            auto result = E.fr.mul(x.value, k);

            auto it = res.find(x.signal_id);
            if (it != res.end()) {
                result = E.fr.add(it->second, result);
            }
            
            res[x.signal_id] = result;
            }
        }

        for (auto const &x : lcB) {
            auto result = x.value;

            auto it = res.find(x.signal_id);
            if (it != res.end()) {
                result = E.fr.add(it->second, result);
            }
            
            res[x.signal_id] = result;
        }


        return res;
    }

    void R1csConstraintProcessor::reduceCoefs(FflonkSetupSettings &settings, 
                                            std::vector<FrElement> &plonkConstraints,
                                            std::vector<FrElement> &plonkAdditions, vector<R1csConstraint> &linCom, uint32_t maxC) {
        ConstraintReduceCoefficients res;
        res.k = E.fr.zero();

        vector<std::tuple<uint64_t, FrElement>> cs;

        for (auto const &x : linCom) {
            if (x.signal_id == 0) {
                res.k = E.fr.add(res.k, x.value);
            } else if (!E.fr.eq(x.value, E.fr.zero())) {
                cs.push_back(std::make_tuple(x.signal_id, x.value));
            }
        }

        while (cs.size() > maxC) {
            const auto c1 = cs.front();
            cs.erase(cs.begin());
            const auto c2 = cs.front();
            cs.erase(cs.begin());
            const auto so = settings.nVars++;

            const auto constraints = this->getFFlonkAdditionConstraint(
                c1[0], c2[0], so,
                E.Fr.neg(c1[1]), E.Fr.neg(c2[1]), E.Fr.zero(), E.Fr.one(), E.Fr.zero());

            plonkConstraints.push_back(constraints);
            plonkAdditions.push_back([c1[0], c2[0], c1[1], c2[1]]);

            cs.push_back([so, this.Fr.one]);
        }

        for (let i = 0; i < cs.length; i++) {
            res.signals[i] = cs[i][0];
            res.coefs[i] = cs[i][1];
        }

        while (res.coefs.length < maxC) {
            res.signals.push(0);
            res.coefs.push(E.Fr.zero);
        }

        return res;
    }

    void R1csConstraintProcessor::processR1csAdditionConstraint(FflonkSetupSettings &settings, 
                                                        std::vector<FrElement> &plonkConstraints,
                                                        std::vector<FrElement> &plonkAdditions,
                                                        vector<R1csConstraint> &linCom) {
        const auto C = this->reduceCoefs(settings, plonkConstraints, plonkAdditions, linCom, 3);

        const constraints = this->getFFlonkAdditionConstraint(
            C.signals[0], C.signals[1], C.signals[2], C.coefs[0], C.coefs[1], this.Fr.zero, C.coefs[2], C.k);

        plonkConstraints.push_back(constraints);
    }

    processR1csMultiplicationConstraint(FflonkSetupSettings &settings, 
                                        std::vector<FrElement> &plonkConstraints,
                                        std::vector<FrElement> &plonkAdditions,
                                        vector<R1csConstraint> &lcA,
                                        vector<R1csConstraint> &lcB,
                                        vector<R1csConstraint> &lcC) {
        const auto A = this->reduceCoefs(settings, plonkConstraints, plonkAdditions, lcA, 1);
        const auto B = this->reduceCoefs(settings, plonkConstraints, plonkAdditions, lcB, 1);
        const auto C = this->reduceCoefs(settings, plonkConstraints, plonkAdditions, lcC, 1);

        const constraints = this->getFFlonkMultiplicationConstraint(
            A.signals[0], B.signals[0], C.signals[0],
            E.Fr.mul(A.coefs[0], B.k),
            E.Fr.mul(A.k, B.coefs[0]),
            E.Fr.mul(A.coefs[0], B.coefs[0]),
            E.Fr.neg(C.coefs[0]),
            E.Fr.sub(this.Fr.mul(A.k, B.k), C.k));

        plonkConstraints.push_back(constraints);
    }}
