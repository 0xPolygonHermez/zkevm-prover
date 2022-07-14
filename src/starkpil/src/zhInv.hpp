#ifndef ZHINV
#define ZHINV

#include <iostream>
#include "goldilocks/goldilocks_base_field.hpp"
#include "zkassert.hpp"
#include <vector>

using namespace std;

// TODO: Pending to review and re-factor

class ZhInv
{
    std::vector<Goldilocks::Element> ZHInv;

public:
    ZhInv(){};

    ZhInv(uint64_t nBits, uint64_t nBitsExt)
    {
        if (nBits == 0 || nBitsExt == 0)
            return;

        zkassert(nBits < nBitsExt);

        Goldilocks::Element w = Goldilocks::one();
        Goldilocks::Element sn = Goldilocks::shift();
        uint64_t extendBits = nBitsExt - nBits;
        uint64_t zhinvSize = (1 << extendBits);

        for (uint64_t i = 0; i < nBits; i++)
        {
            zkassert(nBits < nBitsExt);

            Goldilocks::Element w = Goldilocks::one();
            Goldilocks::Element sn = Goldilocks::shift();
            uint64_t extendBits = nBitsExt - nBits;
            uint64_t zhinvSize = (1 << extendBits);

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
        }
    };

    Goldilocks::Element zhInv(int64_t i)
    {
        zkassert(ZHInv.size() != 0);

        return ZHInv[i % ZHInv.size()];
    };
};
#endif