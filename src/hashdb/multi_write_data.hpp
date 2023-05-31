#ifndef MULTI_WRITE_DATA_HPP
#define MULTI_WRITE_DATA_HPP

#include <string>
#include <unordered_map>

using namespace std;

class MultiWriteData
{
public:
    // Flush data
    unordered_map<string, string> program;
    unordered_map<string, string> programIntray;
    unordered_map<string, string> nodes;
    unordered_map<string, string> nodesIntray;
    string nodesStateRoot;

    // SQL query including all data to store in database
    string query;

    // Indicates if data has been already stored in database
    bool stored;

    void Reset (void)
    {
        // Reset strings
        program.clear();
        programIntray.clear();
        nodes.clear();
        nodesIntray.clear();
        nodesStateRoot.clear();
        query.clear();
        stored = false;
    }

    bool IsEmpty (void)
    {
        return (nodes.size() == 0) &&
               (nodesIntray.size() == 0) &&
               (program.size() == 0) &&
               (programIntray.size() == 0) &&
               (nodesStateRoot.size() == 0);
    }

    void acceptIntray (void)
    {
        if (programIntray.size() > 0)
        {
            program.merge(programIntray);
            programIntray.clear();
        }
        if (nodesIntray.size() > 0)
        {
            nodes.merge(nodesIntray);
            nodesIntray.clear();
        }
    }
};

#endif