#include <stdexcept>

#include "zkey.hpp"
#include "zkey_fflonk.hpp"

namespace Zkey {
    FflonkZkeyHeader::FflonkZkeyHeader() {
        this->protocolId = Zkey::FFLONK_PROTOCOL_ID;
    }

    FflonkZkeyHeader::~FflonkZkeyHeader() {
        mpz_clear(qPrime);
        mpz_clear(rPrime);
    }

    FflonkZkeyHeader* FflonkZkeyHeader::loadFflonkZkeyHeader(BinFileUtils::BinFile *f) {
        auto fflonkZkeyHeader = new FflonkZkeyHeader();

        f->startReadSection(Zkey::ZKEY_FF_HEADER_SECTION);

        fflonkZkeyHeader->n8q = f->readU32LE();
        mpz_init(fflonkZkeyHeader->qPrime);
        mpz_import(fflonkZkeyHeader->qPrime, fflonkZkeyHeader->n8q, -1, 1, -1, 0, f->read(fflonkZkeyHeader->n8q));

        fflonkZkeyHeader->n8r = f->readU32LE();
        mpz_init(fflonkZkeyHeader->rPrime);
        mpz_import(fflonkZkeyHeader->rPrime, fflonkZkeyHeader->n8r, -1, 1, -1, 0, f->read(fflonkZkeyHeader->n8r));

        fflonkZkeyHeader->nVars = f->readU32LE();
        fflonkZkeyHeader->nPublic = f->readU32LE();
        fflonkZkeyHeader->domainSize = f->readU32LE();
        fflonkZkeyHeader->nAdditions = f->readU32LE();
        fflonkZkeyHeader->nConstraints = f->readU32LE();

        fflonkZkeyHeader->k1 = f->read(fflonkZkeyHeader->n8r);
        fflonkZkeyHeader->k2 = f->read(fflonkZkeyHeader->n8r);

        fflonkZkeyHeader->w3 = f->read(fflonkZkeyHeader->n8r);
        fflonkZkeyHeader->w4 = f->read(fflonkZkeyHeader->n8r);
        fflonkZkeyHeader->w8 = f->read(fflonkZkeyHeader->n8r);
        fflonkZkeyHeader->wr = f->read(fflonkZkeyHeader->n8r);

        fflonkZkeyHeader->X2 = f->read(fflonkZkeyHeader->n8q * 4);

        fflonkZkeyHeader->C0 = f->read(fflonkZkeyHeader->n8q * 2);

        f->endReadSection();

        return fflonkZkeyHeader;
    }
}

