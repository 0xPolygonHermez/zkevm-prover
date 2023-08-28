#ifndef MULTI_QUERY_HPP
#define MULTI_QUERY_HPP

#include <vector>
#include <string>
#include "single_query.hpp"

using namespace std;

class MultiQuery
{
public:
    vector<SingleQuery> queries;
    bool isEmpty (void) { return queries.size() == 0; }
    uint64_t size (void)
    {
        uint64_t size = 0;
        for (uint64_t i=0; i<queries.size(); i++)
        {
            size += queries[i].size();
        }
        return size;
    }
    void reset (void)
    {
        queries.clear();
    }
};

#endif