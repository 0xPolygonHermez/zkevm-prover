#include "arith_equation.hpp"

string arith2string (uint64_t arithEquation)
{
    switch (arithEquation)
    {
        case ARITH_BASE:            return "ARITH_BASE";
        case ARITH_ECADD_DIFFERENT: return "ARITH_ECADD_DIFFERENT";
        case ARITH_ECADD_SAME:      return "ARITH_ECADD_SAME";
        case ARITH_BN254_MULFP2:    return "ARITH_BN254_MULFP2";
        case ARITH_BN254_ADDFP2:    return "ARITH_BN254_ADDFP2";
        case ARITH_BN254_SUBFP2:    return "ARITH_BN254_SUBFP2";
        case ARITH_MOD:             return "ARITH_MOD";
        case ARITH_384_MOD:         return "ARITH_384_MOD";
        case ARITH_BLS12381_MULFP2: return "ARITH_BLS12381_MULFP2";
        case ARITH_BLS12381_ADDFP2: return "ARITH_BLS12381_ADDFP2";
        case ARITH_BLS12381_SUBFP2: return "ARITH_BLS12381_SUBFP2";
        case ARITH_256TO384:        return "ARITH_256TO384";
        default:                    return "arith2string() unrecognized arith equation = " + to_string(arithEquation);
    }
}