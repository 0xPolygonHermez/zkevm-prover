#include "goldilocks_cubic_extension.hpp"
#include "zhInv.hpp"
#include "starks.hpp"
#include "constant_pols_starks.hpp"
#include "allSteps.hpp"

void AllSteps::step2prev_first(StepsParams &params, uint64_t i) {
     Goldilocks::Element tmp_12;
     Goldilocks::copy(tmp_12, params.pols[12 + i*15]);
     Goldilocks::Element tmp_13;
     Goldilocks::copy(tmp_13, params.pols[13 + ((i + 1)%1024)*15]);
     Goldilocks::Element tmp_14;
     Goldilocks::mul(tmp_14, params.pols[12 + i*15], params.pols[13 + ((i + 1)%1024)*15]);
     Goldilocks::Element tmp_15;
     Goldilocks::copy(tmp_15,  params.pConstPols->getElement(7,i));
     Goldilocks::Element tmp_16;
     Goldilocks::copy(tmp_16,  params.pConstPols->getElement(8,i));
     Goldilocks::Element tmp_17;
     Goldilocks::copy(tmp_17, params.pols[14 + i*15]);
     Goldilocks::Element tmp_18;
     Goldilocks::copy(tmp_18,  params.pConstPols->getElement(6,i));
     Goldilocks3::Element tmp_0;
     Goldilocks3::mul(tmp_0, tmp_15, (Goldilocks3::Element &)*params.challenges[0]);
     Goldilocks3::Element tmp_1;
     Goldilocks3::add(tmp_1, tmp_16, tmp_0);
     Goldilocks3::Element tmp_2;
     Goldilocks3::mul(tmp_2, (Goldilocks3::Element &)*params.challenges[0], tmp_1);
     Goldilocks3::Element tmp_3;
     Goldilocks3::add(tmp_3, tmp_17, tmp_2);
     Goldilocks3::Element tmp_4;
     Goldilocks3::sub(tmp_4, tmp_3, (Goldilocks3::Element &)*params.challenges[1]);
     Goldilocks3::Element tmp_5;
     Goldilocks3::mul(tmp_5, tmp_18, tmp_4);
     Goldilocks3::add((Goldilocks3::Element &)(params.pols[49152 + i*12]), tmp_5, (Goldilocks3::Element &)*params.challenges[1]);
     Goldilocks::Element tmp_19;
     Goldilocks::copy(tmp_19, params.pols[11 + i*15]);
     Goldilocks3::Element tmp_6;
     Goldilocks3::mul(tmp_6, tmp_12, (Goldilocks3::Element &)*params.challenges[0]);
     Goldilocks3::Element tmp_7;
     Goldilocks3::add(tmp_7, tmp_13, tmp_6);
     Goldilocks3::Element tmp_8;
     Goldilocks3::mul(tmp_8, tmp_7, (Goldilocks3::Element &)*params.challenges[0]);
     Goldilocks3::Element tmp_9;
     Goldilocks3::add(tmp_9, tmp_14, tmp_8);
     Goldilocks3::Element tmp_10;
     Goldilocks3::sub(tmp_10, tmp_9, (Goldilocks3::Element &)(params.pols[49152 + i*12]));
     Goldilocks3::Element tmp_11;
     Goldilocks3::mul(tmp_11, tmp_19, tmp_10);
     Goldilocks3::add((Goldilocks3::Element &)(params.pols[21513 + i*21]), tmp_11, (Goldilocks3::Element &)(params.pols[49152 + i*12]));
}

void AllSteps::step2prev_i(StepsParams &params, uint64_t i) {
     Goldilocks::Element tmp_12;
     Goldilocks::copy(tmp_12, params.pols[12 + i*15]);
     Goldilocks::Element tmp_13;
     Goldilocks::copy(tmp_13, params.pols[13 + ((i + 1)%1024)*15]);
     Goldilocks::Element tmp_14;
     Goldilocks::mul(tmp_14, params.pols[12 + i*15], params.pols[13 + ((i + 1)%1024)*15]);
     Goldilocks::Element tmp_15;
     Goldilocks::copy(tmp_15,  params.pConstPols->getElement(7,i));
     Goldilocks::Element tmp_16;
     Goldilocks::copy(tmp_16,  params.pConstPols->getElement(8,i));
     Goldilocks::Element tmp_17;
     Goldilocks::copy(tmp_17, params.pols[14 + i*15]);
     Goldilocks::Element tmp_18;
     Goldilocks::copy(tmp_18,  params.pConstPols->getElement(6,i));
     Goldilocks3::Element tmp_0;
     Goldilocks3::mul(tmp_0, tmp_15, (Goldilocks3::Element &)*params.challenges[0]);
     Goldilocks3::Element tmp_1;
     Goldilocks3::add(tmp_1, tmp_16, tmp_0);
     Goldilocks3::Element tmp_2;
     Goldilocks3::mul(tmp_2, (Goldilocks3::Element &)*params.challenges[0], tmp_1);
     Goldilocks3::Element tmp_3;
     Goldilocks3::add(tmp_3, tmp_17, tmp_2);
     Goldilocks3::Element tmp_4;
     Goldilocks3::sub(tmp_4, tmp_3, (Goldilocks3::Element &)*params.challenges[1]);
     Goldilocks3::Element tmp_5;
     Goldilocks3::mul(tmp_5, tmp_18, tmp_4);
     Goldilocks3::add((Goldilocks3::Element &)(params.pols[49152 + i*12]), tmp_5, (Goldilocks3::Element &)*params.challenges[1]);
     Goldilocks::Element tmp_19;
     Goldilocks::copy(tmp_19, params.pols[11 + i*15]);
     Goldilocks3::Element tmp_6;
     Goldilocks3::mul(tmp_6, tmp_12, (Goldilocks3::Element &)*params.challenges[0]);
     Goldilocks3::Element tmp_7;
     Goldilocks3::add(tmp_7, tmp_13, tmp_6);
     Goldilocks3::Element tmp_8;
     Goldilocks3::mul(tmp_8, tmp_7, (Goldilocks3::Element &)*params.challenges[0]);
     Goldilocks3::Element tmp_9;
     Goldilocks3::add(tmp_9, tmp_14, tmp_8);
     Goldilocks3::Element tmp_10;
     Goldilocks3::sub(tmp_10, tmp_9, (Goldilocks3::Element &)(params.pols[49152 + i*12]));
     Goldilocks3::Element tmp_11;
     Goldilocks3::mul(tmp_11, tmp_19, tmp_10);
     Goldilocks3::add((Goldilocks3::Element &)(params.pols[21513 + i*21]), tmp_11, (Goldilocks3::Element &)(params.pols[49152 + i*12]));
}

void AllSteps::step2prev_last(StepsParams &params, uint64_t i) {
     Goldilocks::Element tmp_12;
     Goldilocks::copy(tmp_12, params.pols[12 + i*15]);
     Goldilocks::Element tmp_13;
     Goldilocks::copy(tmp_13, params.pols[13 + ((i + 1)%1024)*15]);
     Goldilocks::Element tmp_14;
     Goldilocks::mul(tmp_14, params.pols[12 + i*15], params.pols[13 + ((i + 1)%1024)*15]);
     Goldilocks::Element tmp_15;
     Goldilocks::copy(tmp_15,  params.pConstPols->getElement(7,i));
     Goldilocks::Element tmp_16;
     Goldilocks::copy(tmp_16,  params.pConstPols->getElement(8,i));
     Goldilocks::Element tmp_17;
     Goldilocks::copy(tmp_17, params.pols[14 + i*15]);
     Goldilocks::Element tmp_18;
     Goldilocks::copy(tmp_18,  params.pConstPols->getElement(6,i));
     Goldilocks3::Element tmp_0;
     Goldilocks3::mul(tmp_0, tmp_15, (Goldilocks3::Element &)*params.challenges[0]);
     Goldilocks3::Element tmp_1;
     Goldilocks3::add(tmp_1, tmp_16, tmp_0);
     Goldilocks3::Element tmp_2;
     Goldilocks3::mul(tmp_2, (Goldilocks3::Element &)*params.challenges[0], tmp_1);
     Goldilocks3::Element tmp_3;
     Goldilocks3::add(tmp_3, tmp_17, tmp_2);
     Goldilocks3::Element tmp_4;
     Goldilocks3::sub(tmp_4, tmp_3, (Goldilocks3::Element &)*params.challenges[1]);
     Goldilocks3::Element tmp_5;
     Goldilocks3::mul(tmp_5, tmp_18, tmp_4);
     Goldilocks3::add((Goldilocks3::Element &)(params.pols[49152 + i*12]), tmp_5, (Goldilocks3::Element &)*params.challenges[1]);
     Goldilocks::Element tmp_19;
     Goldilocks::copy(tmp_19, params.pols[11 + i*15]);
     Goldilocks3::Element tmp_6;
     Goldilocks3::mul(tmp_6, tmp_12, (Goldilocks3::Element &)*params.challenges[0]);
     Goldilocks3::Element tmp_7;
     Goldilocks3::add(tmp_7, tmp_13, tmp_6);
     Goldilocks3::Element tmp_8;
     Goldilocks3::mul(tmp_8, tmp_7, (Goldilocks3::Element &)*params.challenges[0]);
     Goldilocks3::Element tmp_9;
     Goldilocks3::add(tmp_9, tmp_14, tmp_8);
     Goldilocks3::Element tmp_10;
     Goldilocks3::sub(tmp_10, tmp_9, (Goldilocks3::Element &)(params.pols[49152 + i*12]));
     Goldilocks3::Element tmp_11;
     Goldilocks3::mul(tmp_11, tmp_19, tmp_10);
     Goldilocks3::add((Goldilocks3::Element &)(params.pols[21513 + i*21]), tmp_11, (Goldilocks3::Element &)(params.pols[49152 + i*12]));
}
