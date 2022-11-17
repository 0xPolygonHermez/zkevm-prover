#include "zhInv.hpp"

// TODO: Pending to review and re-factor

ZhInv::ZhInv(){};

ZhInv::ZhInv(uint64_t nBits, uint64_t nBitsExt)
{
    if (nBits == 0 || nBitsExt == 0)
        return;

    zkassert(nBits < nBitsExt);

    Goldilocks::Element w = Goldilocks::one();
    Goldilocks::Element sn = Goldilocks::shift();
    uint64_t extendBits = nBitsExt - nBits;

    zhinvSize = (1 << extendBits);
    zkassert(zhinvSize != 0);

    for (uint64_t i = 0; i < nBits; i++)
    {
        Goldilocks::square(sn, sn);
    }
    for (uint64_t i = 0; i < zhinvSize; i++)
    {
        Goldilocks::Element inv;
        Goldilocks::inv(inv, (sn * w) - Goldilocks::one());
        ZHInv.push_back(inv);
        Goldilocks::mul(w, w, Goldilocks::w(extendBits));
    }
};
