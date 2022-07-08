#include "compare_fe.hpp"

bool CompareFeImpl(const Goldilocks::Element &a, const Goldilocks::Element &b)
{
    return a.fe < b.fe;
}