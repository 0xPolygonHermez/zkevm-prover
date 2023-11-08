#ifndef RLP_DATA_HPP_fork_6
#define RLP_DATA_HPP_fork_6

#include <string>
#include <vector>

using namespace std;

namespace fork_6
{

enum RLPType
{
    rlpTypeUnknown = 0,
    rlpTypeString = 1,
    rlpTypeList = 2
};

// RLP data can be a string (data) or a list of other RLP data elements (rlpData)
class RLPData
{
public:
    RLPType type;
    string data; // Data without length encoding prefix
    string dataWithLength; // Data with length encoding prefix
    // TODO: Store data without length for strings, and data with length for lists
    vector<RLPData> rlpData;
};

}

#endif