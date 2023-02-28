#ifndef ZKEY_FFLONK_H
#define ZKEY_FFLONK_H

#include <gmp.h>

namespace Zkey {
    const int ZKEY_FF_NSECTIONS = 17;

    const int ZKEY_FF_HEADER_SECTION = 2;
    const int ZKEY_FF_ADDITIONS_SECTION = 3;
    const int ZKEY_FF_A_MAP_SECTION = 4;
    const int ZKEY_FF_B_MAP_SECTION = 5;
    const int ZKEY_FF_C_MAP_SECTION = 6;
    const int ZKEY_FF_QL_SECTION = 7;
    const int ZKEY_FF_QR_SECTION = 8;
    const int ZKEY_FF_QM_SECTION = 9;
    const int ZKEY_FF_QO_SECTION = 10;
    const int ZKEY_FF_QC_SECTION = 11;
    const int ZKEY_FF_SIGMA1_SECTION = 12;
    const int ZKEY_FF_SIGMA2_SECTION = 13;
    const int ZKEY_FF_SIGMA3_SECTION = 14;
    const int ZKEY_FF_LAGRANGE_SECTION = 15;
    const int ZKEY_FF_PTAU_SECTION = 16;
    const int ZKEY_FF_C0_SECTION = 17;

    class FflonkZkeyHeader  {
    public:
        int protocolId;

        u_int32_t n8q;
        mpz_t qPrime;
        u_int32_t n8r;
        mpz_t rPrime;

        u_int32_t nVars;
        u_int32_t nPublic;
        u_int32_t domainSize;
        u_int32_t nAdditions;
        u_int32_t nConstraints;

        void *k1;
        void *k2;
        void *w3;
        void *w4;
        void *w8;
        void *wr;
        void *X2;
        void *C0;

        FflonkZkeyHeader();

        ~FflonkZkeyHeader();

        static FflonkZkeyHeader* loadFflonkZkeyHeader(BinFileUtils::BinFile *f);
    };
}

#endif
