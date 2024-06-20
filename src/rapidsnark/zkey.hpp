#ifndef ZKEY_HPP
#define ZKEY_HPP

#include "binfile_utils.hpp"

namespace Zkey {
    const int GROTH16_PROTOCOL_ID = 1;
    const int PLONK_PROTOCOL_ID = 2;
    const int FFLONK_PROTOCOL_ID = 10;

    const int ZKEY_HEADER_SECTION = 1;

    int getProtocolIdFromZkey(BinFileUtils::BinFile *fd);
}

#endif
