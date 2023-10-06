#include "child.hpp"
#include "scalar.hpp"

string Child::print (Goldilocks &fr) const
{
    switch (type)
    {
        case UNSPECIFIED:
        {
            return "UNSPECIFIED";
        }
        case ZERO:
        {
            return "ZERO";
        }
        case LEAF:
        {
            return "LEAF level=" + to_string(leaf.level) + " key=" + fea2string(fr, leaf.key) + " value=" + leaf.value.get_str(16) + " hash=" + fea2string(fr, leaf.hash);
        }
        case INTERMEDIATE:
        {
            return "INTERMEDIATE hash=" + fea2string(fr, intermediate.hash);
        }
        default:
        {
            return "INVALID TYPE type=" + to_string(type);
        }
    }
}