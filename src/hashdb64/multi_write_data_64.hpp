#ifndef MULTI_WRITE_DATA_64_HPP
#define MULTI_WRITE_DATA_64_HPP

#include <string>
#include <unordered_map>
#include "definitions.hpp"
#include "zklog.hpp"
#include "multi_query.hpp"
#include "key_value.hpp"

using namespace std;

class MultiWriteData64
{
public:
    // Flush data
    unordered_map<string, string> program;
    unordered_map<string, string> programIntray;
    unordered_map<string, string> nodes;
    unordered_map<string, string> nodesIntray;
    unordered_map<uint64_t, KeyValue> keyValue;
    unordered_map<uint64_t, KeyValue> keyValueIntray;
    unordered_map<string, uint64_t> version;
    unordered_map<string, uint64_t> versionIntray;
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
        keyValue.clear();
        keyValueIntray.clear();
        version.clear();
        versionIntray.clear();
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
               (nodesStateRoot.size() == 0) &&
               (keyValue.size() == 0) &&
               (keyValueIntray.size() == 0) &&
               (version.size() == 0) &&
               (versionIntray.size() == 0);
    }

    void acceptIntray (bool bSenderCalling = false)
    {
        if (programIntray.size() > 0)
        {
#ifdef LOG_DB_ACCEPT_INTRAY
            if (bSenderCalling)
            {
                zklog.info("MultiWriteData64::acceptIntray() rescuing " + to_string(programIntray.size()) + " program hashes");
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
                zklog.info("MultiWriteData64::acceptIntray() rescuing " + to_string(nodesIntray.size()) + " nodes hashes");
            }
#endif
            nodes.merge(nodesIntray);
            nodesIntray.clear();
        }
        if (keyValueIntray.size() > 0)
        {
#ifdef LOG_DB_ACCEPT_INTRAY
            if (bSenderCalling)
            {
                zklog.info("MultiWriteData64::acceptIntray() rescuing " + to_string(keyValueIntray.size()) + " keyValue pairs");
            }
#endif
            keyValue.merge(keyValueIntray);
            keyValueIntray.clear();
        }
        if (versionIntray.size() > 0)
        {
#ifdef LOG_DB_ACCEPT_INTRAY
            if (bSenderCalling)
            {
                zklog.info("MultiWriteData64::acceptIntray() rescuing " + to_string(versionIntray.size()) + " versions");
            }   
#endif
            version.merge(versionIntray);
            versionIntray.clear();
        }
    }
};

#endif