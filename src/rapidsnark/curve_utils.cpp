#include "curve_utils.hpp"

namespace CurveUtils {
    std::string getCurveNameByEngine() {
        //return std::is_base_of_v < AltBn128::Engine, Engine > ? "bn128" : "unknown";
        return "bn128";
    }
}
