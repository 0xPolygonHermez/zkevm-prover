#ifndef COMPARE_FE_HPP
#define COMPARE_FE_HPP

#include <vector>
#include "goldilocks_base_field.hpp"

// This function is needed to use maps that have a fe key.
// The funtion returns true if a <= b.
// Elements are ordered according to this function, allowing a dichotomic search.

bool CompareFeImpl(const Goldilocks::Element &a, const Goldilocks::Element &b);
bool CompareFeVectorImpl(const std::vector<Goldilocks::Element> &a, const std::vector<Goldilocks::Element> &b);

class CompareFe
{
public:
    bool operator()(const Goldilocks::Element &a, const Goldilocks::Element &b) const
    {
        return CompareFeImpl(a, b);
    }
    bool operator()(const std::vector<Goldilocks::Element> &a, const std::vector<Goldilocks::Element> &b) const
    {
        return CompareFeVectorImpl(a, b);
    }
};

#endif