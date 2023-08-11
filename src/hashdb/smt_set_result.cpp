#include "smt_set_result.hpp"
#include "scalar.hpp"

using namespace std;

string SmtSetResult::toString (Goldilocks &fr)
{
    string result;
    result += "mode=" + mode + "\n";
    result += "oldRoot=" + fea2string(fr, oldRoot) + "\n";
    result += "newRoot=" + fea2string(fr, newRoot) + "\n";
    result += "key=" + fea2string(fr, key) + "\n";
    result += "insKey=" + fea2string(fr, insKey) + "\n";
    result += "oldValue=" + oldValue.get_str(16) + "\n";
    result += "newValue=" + newValue.get_str(16) + "\n";
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