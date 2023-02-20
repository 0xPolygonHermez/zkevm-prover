#include "zkey.hpp"

namespace Zkey {

    int getProtocolIdFromZkey(BinFileUtils::BinFile *fd) {
        fd->startReadSection(ZKEY_HEADER_SECTION);
        uint32_t protocolId = fd->readU32LE();
        fd->endReadSection();

        return protocolId;
    }

}

