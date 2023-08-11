#include "smt_get_result.hpp"
#include "scalar.hpp"

using namespace std;

string SmtGetResult::toString (Goldilocks &fr)
{
    string result;
    result += "root=" + fea2string(fr, root) + "\n";
    result += "key=" + fea2string(fr, key) + "\n";
    result += "insKey=" + fea2string(fr, insKey) + "\n";
    result += "value=" + value.get_str(16) + "\n";
    result += "insValue=" + insValue.get_str(16) + "\n";
    result += "isOld0=" + to_string(isOld0) + "\n";
    result += "proofHashCounter=" + to_string(proofHashCounter) + "\n";
    map< uint64_t, vector<Goldilocks::Element> >::const_iterator it;
    for (it=siblings.begin(); it!=siblings.end(); it++)
    {
        result += "siblings[" + to_string(it->first) + "]=";
        for (uint64_t i=0; i<it->second.size(); i++)
        {
            result += fr.toString(it->second[i], 16) + ":";
        }
        result += "\n";
    }
    return result;
}