#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include <vector>
#include "reference.hpp"

using namespace std;

class Output
{
public:
    string name;
    Reference ref; // Contains data if array.size()==0
    vector<Output> array;
    vector<Output> objects;
    bool isArray (void) { return array.size() > 0; }
    bool isObject (void) { return objects.size() > 0; }
};

#endif