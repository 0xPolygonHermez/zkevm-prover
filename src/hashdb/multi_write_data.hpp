#ifndef MULTI_WRITE_DATA_HPP
#define MULTI_WRITE_DATA_HPP

#include <string>
#include <unordered_map>
#include "definitions.hpp"
#include "zklog.hpp"
#include "multi_query.hpp"

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

    // SQL queries, including all data to store in database
    MultiQuery multiQuery;

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
        multiQuery.reset();
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

    void acceptIntray (bool bSenderCalling = false)
    {
        if (programIntray.size() > 0)
        {
#ifdef LOG_DB_ACCEPT_INTRAY
            if (bSenderCalling)
            {
                zklog.info("MultiWriteData::acceptIntray() rescuing " + to_string(programIntray.size()) + " program hashes");
            }
#endif
            program.merge(programIntray);
            programIntray.clear();
        }
        if (nodesIntray.size() > 0)
        {
#ifdef LOG_DB_ACCEPT_INTRAY
            if (bSenderCalling)
            {
                zklog.info("MultiWriteData::acceptIntray() rescuing " + to_string(nodesIntray.size()) + " nodes hashes");
            }
#endif
            nodes.merge(nodesIntray);
            nodesIntray.clear();
        }
    }
};

#endif