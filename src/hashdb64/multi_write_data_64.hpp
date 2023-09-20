#ifndef MULTI_WRITE_DATA_64_HPP
#define MULTI_WRITE_DATA_64_HPP

#include <string>
#include <unordered_map>
#include <map>
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
    map<uint64_t, vector<KeyValue>> keyValueA; //keyValue must be inserted to the db in order with respect to version!
    unordered_map<uint64_t, vector<KeyValue>> keyValueAIntray;
    unordered_map<string, vector<uint64_t>>  keyVersions;
    unordered_map<string, vector<uint64_t>>  keyVersionsIntray;
    unordered_map<string, uint64_t> version;
    unordered_map<string, uint64_t> versionIntray;
    uint64_t latestVersion=0;
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
        keyValueA.clear();
        keyValueAIntray.clear();
        keyVersions.clear();
        keyVersionsIntray.clear();
        version.clear();
        versionIntray.clear();
        nodesStateRoot.clear();
        multiQuery.reset();
        stored = false;
        latestVersion = 0;
    }

    bool IsEmpty (void)
    {
        return (nodes.size() == 0) &&
               (nodesIntray.size() == 0) &&
               (program.size() == 0) &&
               (programIntray.size() == 0) &&
               (nodesStateRoot.size() == 0) &&
               (keyValueA.size() == 0) &&
               (keyValueAIntray.size() == 0) &&
               (keyVersions.size() == 0) &&
               (keyVersionsIntray.size() == 0) &&
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
        if (keyValueAIntray.size() > 0)
        {
#ifdef LOG_DB_ACCEPT_INTRAY
            if (bSenderCalling)
            {
                zklog.info("MultiWriteData64::acceptIntray() rescuing " + to_string(keyValueAIntray.size()) + " keyValueA pairs");
            }
#endif      
            for(auto it = keyValueAIntray.begin(); it != keyValueAIntray.end(); ++it){
                if(keyValueA.find(it->first) == keyValueA.end()){
                    keyValueA[it->first] = it->second;
                }else{
                    keyValueA[it->first].insert(keyValueA[it->first].end(), it->second.begin(), it->second.end());
                }
            }
            keyValueAIntray.clear();
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
        if (keyVersionsIntray.size() > 0)
        {
#ifdef LOG_DB_ACCEPT_INTRAY
            if (bSenderCalling)
            {
                zklog.info("MultiWriteData64::acceptIntray() rescuing " + to_string(keyVersionsIntray.size()) + " keyVersions");
            }
#endif
            for(auto it = keyVersionsIntray.begin(); it != keyVersionsIntray.end(); ++it){
                if(keyVersions.find(it->first) == keyVersions.end()){
                    keyVersions[it->first] = it->second;
                }else{
                    keyVersions[it->first].insert(keyVersions[it->first].end(), it->second.begin(), it->second.end());
                }
            }
            keyVersionsIntray.clear();
        }
    }
};

#endif