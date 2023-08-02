#ifndef PILFFLONK_STEPS_HPP
#define PILFFLONK_STEPS_HPP

#include <alt_bn128.hpp>

struct PilFflonkStepsParams
{
    AltBn128::FrElement* cm1_n;
    AltBn128::FrElement* cm2_n;
    AltBn128::FrElement* cm3_n;
    AltBn128::FrElement* tmpExp_n;
    AltBn128::FrElement* cm1_2ns;
    AltBn128::FrElement* cm2_2ns;
    AltBn128::FrElement* cm3_2ns;
    AltBn128::FrElement* const_n;
    AltBn128::FrElement* const_2ns;
    AltBn128::FrElement* challenges;
    AltBn128::FrElement* x_n;
    AltBn128::FrElement* x_2ns;
    AltBn128::FrElement* constValues;
    AltBn128::FrElement* publicInputs;
    AltBn128::FrElement* q_2ns;
};

class PilFflonkSteps {
    
    public:
        
        u_int64_t getNumConstValues();

        void setConstValues(AltBn128::Engine &E, PilFflonkStepsParams &params);

        void publics_first(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i, uint64_t pub);
        void publics_i(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i, uint64_t pub);
        void publics_last(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i, uint64_t pub);

        void step2prev_first(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i);
        void step2prev_i(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i);
        void step2prev_last(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i);

        void step3prev_first(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i);
        void step3prev_i(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i);
        void step3prev_last(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i);

        void step3_first(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i);
        void step3_i(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i);
        void step3_last(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i);

        void step42ns_first(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i);
        void step42ns_i(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i);
        void step42ns_last(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i);
};

#endif // STARKS_RECURSIVE_1_STEPS_HPP
