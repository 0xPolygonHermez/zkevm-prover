#include "compare_fe.hpp"

bool CompareFeImpl(const FieldElement &a, const FieldElement &b)
{
    return a < b;
}