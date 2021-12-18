#ifndef COMPARE_FE_HPP
#define COMPARE_FE_HPP

#include "ffiasm/fr.hpp"

// This function is needed to use maps that have a fe key.
// The funtion returns true if a <= b.
// Elements are ordered according to this function, allowing a dichotomic search.

bool CompareFeImpl(const RawFr::Element &a, const RawFr::Element &b);

class CompareFe {
public:
    bool operator()(const RawFr::Element &a, const RawFr::Element &b) const
    {
        return CompareFeImpl(a, b);
    }
};

#endif