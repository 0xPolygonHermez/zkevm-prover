#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include <vector>
#include "reference.hpp"

using namespace std;

class Output
{
public:
    Reference ref; // Contains data if array.size()==0
    string name;
    vector<Output> array;
    bool isRef (void) { return array.size()==0; }
    bool isArray (void) { return !isRef(); }
};

#endif