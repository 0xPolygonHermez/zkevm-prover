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
    ZhInv();

    ZhInv(uint64_t nBits, uint64_t nBitsExt);

    Goldilocks::Element zhInv(int64_t i);
};
#endif