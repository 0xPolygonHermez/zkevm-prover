#include "compare_fe.hpp"

bool CompareFeImpl(const Goldilocks::Element &a, const Goldilocks::Element &b)
{
    return a.fe < b.fe;
}

bool CompareFeVectorImpl(const std::vector<Goldilocks::Element> &a, const std::vector<Goldilocks::Element> &b)
{
    for(uint64_t i = 0; i < a.size(); ++i) {
        uint64_t value1 = Goldilocks::toU64(a[i]);
        uint64_t value2 = Goldilocks::toU64(b[i]);
        if(value1 != value2) {
            return value1 < value2;
        }
    }

    return false;
}