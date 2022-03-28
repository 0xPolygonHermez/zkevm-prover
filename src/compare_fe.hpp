#ifndef COMPARE_FE_HPP
#define COMPARE_FE_HPP

#include "ff/ff.hpp"

// This function is needed to use maps that have a fe key.
// The funtion returns true if a <= b.
// Elements are ordered according to this function, allowing a dichotomic search.

bool CompareFeImpl(const FieldElement &a, const FieldElement &b);

class CompareFe {
public:
    bool operator()(const FieldElement &a, const FieldElement &b) const
    {
        return CompareFeImpl(a, b);
    }
};

#endif