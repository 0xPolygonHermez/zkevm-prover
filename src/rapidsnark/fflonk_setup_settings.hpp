#ifndef FFLONK_SETUP_SETTINGS_HPP
#define FFLONK_SETUP_SETTINGS_HPP

#include <iostream>

namespace Fflonk {
struct FflonkSetupSettings {
    uint64_t nVars;
    uint64_t nPublics;
    uint64_t cirPower;
    uint64_t domainSize;
};
}  // namespace Fflonk

#endif