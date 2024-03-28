#ifndef R1CS_CONSTRAINT_PROCESOR_HPP
#define R1CS_CONSTRAINT_PROCESOR_HPP

#include <stdio.h>
#include <alt_bn128.hpp>
#include <binfile_utils.hpp>

#include "fflonk_setup_settings.hpp"

using namespace std;

namespace R1cs
{
    using FrElement = typename AltBn128::Engine::FrElement;

    struct R1csConstraint {
        uint64_t signal_id;
        FrElement value;

        R1csConstraint(uint64_t id, const FrElement val) : signal_id(id), value(val) {};
    };

    struct ConstraintCoefficients {
        uint64_t signal_a;
        uint64_t signal_b;
        uint64_t signal_c;
        FrElement ql;
        FrElement qr;
        FrElement qm;
        FrElement qo;
        FrElement qc;
    };

    struct AdditionCoefficients {
        uint64_t signal_a;
        uint64_t signal_b;
        FrElement value_a;
        FrElement value_b;

        AdditionCoefficients(uint64_t a, uint64_t b, FrElement va, FrElement vb) : signal_a(a), signal_b(b), value_a(va), value_b(vb) {};
    };

    struct ConstraintReduceCoefficients {
        FrElement k;
        vector<uint64_t> signals;
        vector<FrElement> coefs;
    };

    class R1csConstraintProcessor
    {
        AltBn128::Engine &E;

    public:
        R1csConstraintProcessor(AltBn128::Engine &E): E(E) {};

        void processR1csConstraints(Fflonk::FflonkSetupSettings &settings,
                                    vector<R1csConstraint> &lcA,
                                    vector<R1csConstraint> &lcB,
                                    vector<R1csConstraint> &lcC,
                                    std::vector<ConstraintCoefficients> &plonkConstraints,
                                    std::vector<R1cs::AdditionCoefficients> &plonkAdditions);

        static ConstraintCoefficients getFflonkConstantConstraint(AltBn128::Engine &E, uint64_t signal_a) {
            ConstraintCoefficients cc;
            
            cc.signal_a = signal_a;
            cc.signal_b = 0;
            cc.signal_c = 0;
            cc.ql = E.fr.one();
            cc.qr = E.fr.zero();
            cc.qm = E.fr.zero();
            cc.qo = E.fr.zero();
            cc.qc = E.fr.zero();

            return cc;
        }

        static ConstraintCoefficients getFFlonkAdditionConstraint(uint64_t signal_a, uint64_t signal_b, uint64_t signal_c, FrElement ql, FrElement qr, FrElement qm, FrElement qo, FrElement qc) {
            ConstraintCoefficients cc;
            
            cc.signal_a = signal_a;
            cc.signal_b = signal_b;
            cc.signal_c = signal_c;
            cc.ql = ql;
            cc.qr = qr;
            cc.qm = qm;
            cc.qo = qo;
            cc.qc = qc;

            return cc;
        }

        static ConstraintCoefficients getFFlonkMultiplicationConstraint(uint64_t signal_a, uint64_t signal_b, uint64_t signal_c, FrElement ql, FrElement qr, FrElement qm, FrElement qo, FrElement qc) {
            ConstraintCoefficients cc;
            
            cc.signal_a = signal_a;
            cc.signal_b = signal_b;
            cc.signal_c = signal_c;
            cc.ql = ql;
            cc.qr = qr;
            cc.qm = qm;
            cc.qo = qo;
            cc.qc = qc;

            return cc;
        }

        int getLinearCombinationType(vector<R1csConstraint> &lc);
        void normalizeLinearCombination(vector<R1csConstraint> &lc);
        vector<R1csConstraint> joinLinearCombinations(vector<R1csConstraint> &lcA, vector<R1csConstraint> &lcB, FrElement k);
        ConstraintReduceCoefficients reduceCoefs(Fflonk::FflonkSetupSettings &settings, 
                        std::vector<R1cs::ConstraintCoefficients> &plonkConstraints,
                        std::vector<R1cs::AdditionCoefficients> &plonkAdditions, vector<R1csConstraint> &linCom, uint32_t maxC);
        void processR1csAdditionConstraint(Fflonk::FflonkSetupSettings &settings, 
                                        std::vector<R1cs::ConstraintCoefficients> &plonkConstraints,
                                        std::vector<R1cs::AdditionCoefficients> &plonkAdditions,
                                        vector<R1csConstraint> &linCom);
        void processR1csMultiplicationConstraint(Fflonk::FflonkSetupSettings &settings, 
                                        std::vector<R1cs::ConstraintCoefficients> &plonkConstraints,
                                        std::vector<R1cs::AdditionCoefficients> &plonkAdditions,
                                        vector<R1csConstraint> &lcA,
                                        vector<R1csConstraint> &lcB,
                                        vector<R1csConstraint> &lcC);
    };
}
#endif