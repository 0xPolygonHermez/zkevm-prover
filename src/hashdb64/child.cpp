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

zkresult Child::getHash (Goldilocks::Element (&hash)[4]) const
{
    switch (type)
    {
        case ZERO:
        {
            hash[0] = fr.zero();
            hash[1] = fr.zero();
            hash[2] = fr.zero();
            hash[3] = fr.zero();
            return ZKR_SUCCESS;
        }
        case LEAF:
        {
            hash[0] = leaf.hash[0];
            hash[1] = leaf.hash[1];
            hash[2] = leaf.hash[2];
            hash[3] = leaf.hash[3];
            return ZKR_SUCCESS;
        }
        case INTERMEDIATE:
        {
            hash[0] = intermediate.hash[0];
            hash[1] = intermediate.hash[1];
            hash[2] = intermediate.hash[2];
            hash[3] = intermediate.hash[3];
            return ZKR_SUCCESS;
        }
        default:
        {
            zklog.error("Child::getHash() found unexpected type=" + to_string(type));
            return ZKR_DB_ERROR;
        }
    }
}