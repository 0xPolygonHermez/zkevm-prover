#ifndef R1CS_CONSTRAINT_PROCESOR_HPP
#define R1CS_CONSTRAINT_PROCESOR_HPP

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

using FrElement = typename AltBn128::Engine::FrElement;

namespace R1cs
{
    struct R1csContraint {
        uint32_t signal_id;
        FrElement value;
    };

    struct 

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

    
    class R1csConstraintProcessor
    {
        struct ConstraintReduceCoefficients {
            FrElement k;
            vector<uint64_t> signals;
            vector<FrElement> coefs;
        };

    public:
        R1csConstraintProcessor() : {};

        ~R1csConstraintProcessor();

        //TODO! lcA, lcB and lcC are not int, but some other type
        void processR1csConstraints(Binfile &binfile,
                                    FflonkSetupSettings &settings,
                                    vector<R1csConstraint> &lcA,
                                    vector<R1csConstraint> &lcB,
                                    vector<R1csConstraint> &lcC,
                                    std::vector<FrElement> &plonkConstraints,
                                    std::vector<FrElement> &plonkAdditions);

        static ConstraintCoefficients getFflonkConstantConstraint(uint64_t signal_a) {
            ConstraintCoefficients cc;
            
            cc.signal_a = signal_a;
            cc.signal_b = 0;
            cc.signal_c = 0;
            cc.ql = FrElement::one();
            cc.qr = FrElement::zero();
            cc.qm = FrElement::zero();
            cc.qo = FrElement::zero();
            cc.qc = FrElement::zero();

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
    };
}
}
#endif
