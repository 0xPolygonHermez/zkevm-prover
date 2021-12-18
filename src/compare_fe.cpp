#include "compare_fe.hpp"

bool CompareFeImpl(const RawFr::Element &a, const RawFr::Element &b)
{
         if (a.v[3] != b.v[3]) return a.v[3] < b.v[3];
    else if (a.v[2] != b.v[2]) return a.v[2] < b.v[2];
    else if (a.v[1] != b.v[1]) return a.v[1] < b.v[1];
    else                       return a.v[0] < b.v[0];
}