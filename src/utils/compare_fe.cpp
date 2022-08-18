#include "compare_fe.hpp"

bool CompareFeImpl(const Goldilocks::Element &a, const Goldilocks::Element &b)
{
    return a.fe < b.fe;
}

bool CompareFeVectorImpl(const std::vector<Goldilocks::Element> &a, const std::vector<Goldilocks::Element> &b)
{
    if (a.size() == 1)
    {
        return Goldilocks::toU64(a[0]) < Goldilocks::toU64(b[0]);
    }
    else if (Goldilocks::toU64(a[0]) != Goldilocks::toU64(b[0]))
    {
        return Goldilocks::toU64(a[0]) < Goldilocks::toU64(b[0]);
    }
    else if (Goldilocks::toU64(a[1]) != Goldilocks::toU64(b[1]))
    {
        return Goldilocks::toU64(a[1]) < Goldilocks::toU64(b[1]);
    }
    else
    {
        return Goldilocks::toU64(a[2]) < Goldilocks::toU64(b[2]);
    }
}