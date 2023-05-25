#ifndef MULTI_WRITE_DATA_HPP
#define MULTI_WRITE_DATA_HPP

#include <vector>
#include <string>
#include "flush_data.hpp"

using namespace std;

class MultiWriteData
{
public:
    // Flush data
    vector<FlushData> program;
    vector<FlushData> programUpdate;
    vector<FlushData> nodes;
    vector<FlushData> nodesUpdate;
    string nodesStateRoot;

    // SQL query including all data to store in database
    string query;

    // Indicates if data has been already stored in database
    bool stored;

    void Reset (void)
    {
        // Reset strings
        program.clear();
        programUpdate.clear();
        nodes.clear();
        nodesUpdate.clear();
        nodesStateRoot.clear();
        query.clear();
        stored = false;
    }

    bool IsEmpty (void)
    {
        return (nodes.size() == 0) && (nodesUpdate.size() == 0) && (nodesStateRoot.size() == 0) && (program.size() == 0) && (programUpdate.size() == 0);
    }
};

#endif