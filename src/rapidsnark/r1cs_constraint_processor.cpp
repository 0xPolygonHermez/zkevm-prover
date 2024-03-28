#include "r1cs_constraint_processor.hpp"

namespace R1cs {
using FrElement = typename AltBn128::Engine::FrElement;

const int LINEAR_COMBINATION_NULLABLE = 0;
const int LINEAR_COMBINATION_CONSTANT = 1;
const int LINEAR_COMBINATION_VARIABLE = 2;

void R1csConstraintProcessor::processR1csConstraints(
    Fflonk::FflonkSetupSettings &settings,
    vector<R1csConstraint> &lcA,
    vector<R1csConstraint> &lcB,
    vector<R1csConstraint> &lcC,
    std::vector<ConstraintCoefficients> &plonkConstraints,
    std::vector<R1cs::AdditionCoefficients> &plonkAdditions) {
    this->normalizeLinearCombination(lcA);
    this->normalizeLinearCombination(lcB);
    this->normalizeLinearCombination(lcC);

    const auto lctA = this->getLinearCombinationType(lcA);
    const auto lctB = this->getLinearCombinationType(lcB);

    if ((lctA == LINEAR_COMBINATION_NULLABLE) || (lctB == LINEAR_COMBINATION_NULLABLE)) {
        processR1csAdditionConstraint(settings, plonkConstraints, plonkAdditions, lcC);
    } else if (lctA == LINEAR_COMBINATION_CONSTANT) {
        auto lcCC = this->joinLinearCombinations(lcB, lcC, lcA[0].value);
        processR1csAdditionConstraint(settings, plonkConstraints, plonkAdditions, lcCC);
    } else if (lctB == LINEAR_COMBINATION_CONSTANT) {
        auto lcCC = this->joinLinearCombinations(lcA, lcC, lcB[0].value);
        processR1csAdditionConstraint(settings, plonkConstraints, plonkAdditions, lcCC);
    } else {
        processR1csMultiplicationConstraint(settings, plonkConstraints, plonkAdditions, lcA, lcB, lcC);
    }
}

int R1csConstraintProcessor::getLinearCombinationType(vector<R1csConstraint> &lc) {
    FrElement k = E.fr.zero();
    uint64_t n = 0;

    auto it = lc.begin();
    while (it != lc.end()) {
        if (E.fr.eq(it->value, E.fr.zero())) {
            it = lc.erase(it);  // Removes the element and updates the iterator
        } else {
            if (it->signal_id == 0) {
                k = E.fr.add(k, it->value);
            } else {
                n++;
            }
            ++it;
        }
    }

    if (n > 0)
        return LINEAR_COMBINATION_VARIABLE;
    if (!E.fr.eq(k, E.fr.zero()))
        return LINEAR_COMBINATION_CONSTANT;
    return LINEAR_COMBINATION_NULLABLE;
}

void R1csConstraintProcessor::normalizeLinearCombination(vector<R1csConstraint> &lc) {
    auto it = lc.begin();
    while (it != lc.end()) {
        if (E.fr.eq(it->value, E.fr.zero())) {
            it = lc.erase(it);  // Removes the element and updates the iterator
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

    for (auto const &x : lcB) {
        auto result = x.value;

        auto it = res.find(x.signal_id);
        if (it != res.end()) {
            result = E.fr.add(it->second, result);
        }

        res[x.signal_id] = result;
    }

    // Convert map to vector
    vector<R1csConstraint> resVec;
    for (auto const &x : res) {
        resVec.push_back(R1csConstraint(x.first, x.second));
    }
    return resVec;
}

ConstraintReduceCoefficients R1csConstraintProcessor::reduceCoefs(Fflonk::FflonkSetupSettings &settings,
                                                                  std::vector<R1cs::ConstraintCoefficients> &plonkConstraints,
                                                                  std::vector<R1cs::AdditionCoefficients> &plonkAdditions,
                                                                  vector<R1csConstraint> &linCom,
                                                                  uint32_t maxC) {
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
            std::get<0>(c1), std::get<0>(c2), so, E.fr.neg(std::get<1>(c1)), E.fr.neg(std::get<1>(c2)), E.fr.zero(), E.fr.one(), E.fr.zero());

        plonkConstraints.push_back(constraints);

        auto addition = AdditionCoefficients(std::get<0>(c1), std::get<0>(c2), std::get<1>(c1), std::get<1>(c2));
        plonkAdditions.push_back(addition);

        std::tuple<uint64_t, FrElement> tuple = std::make_tuple(so, E.fr.one());
        cs.push_back(tuple);
    }

    for (uint64_t i = 0; i < cs.size(); i++) {
        res.signals.push_back(std::get<0>(cs[i]));
        res.coefs.push_back(std::get<1>(cs[i]));
    }

    while (res.coefs.size() < maxC) {
        res.signals.push_back(0);
        res.coefs.push_back(E.fr.zero());
    }

    return res;
}

void R1csConstraintProcessor::processR1csAdditionConstraint(Fflonk::FflonkSetupSettings &settings,
                                                            std::vector<R1cs::ConstraintCoefficients> &plonkConstraints,
                                                            std::vector<R1cs::AdditionCoefficients> &plonkAdditions,
                                                            vector<R1csConstraint> &linCom) {
    const auto C = reduceCoefs(settings, plonkConstraints, plonkAdditions, linCom, 3);

    const auto constraints = getFFlonkAdditionConstraint(
        C.signals[0], C.signals[1], C.signals[2], C.coefs[0], C.coefs[1], E.fr.zero(), C.coefs[2], C.k);

    plonkConstraints.push_back(constraints);
}

void R1csConstraintProcessor::processR1csMultiplicationConstraint(Fflonk::FflonkSetupSettings &settings,
                                                                  std::vector<R1cs::ConstraintCoefficients> &plonkConstraints,
                                                                  std::vector<R1cs::AdditionCoefficients> &plonkAdditions,
                                                                  vector<R1csConstraint> &lcA,
                                                                  vector<R1csConstraint> &lcB,
                                                                  vector<R1csConstraint> &lcC) {
    const auto A = this->reduceCoefs(settings, plonkConstraints, plonkAdditions, lcA, 1);
    const auto B = this->reduceCoefs(settings, plonkConstraints, plonkAdditions, lcB, 1);
    const auto C = this->reduceCoefs(settings, plonkConstraints, plonkAdditions, lcC, 1);

    const auto constraints = this->getFFlonkMultiplicationConstraint(
        A.signals[0], B.signals[0], C.signals[0],
        E.fr.mul(A.coefs[0], B.k),
        E.fr.mul(A.k, B.coefs[0]),
        E.fr.mul(A.coefs[0], B.coefs[0]),
        E.fr.neg(C.coefs[0]),
        E.fr.sub(E.fr.mul(A.k, B.k), C.k));

    plonkConstraints.push_back(constraints);
}
}  // namespace R1cs