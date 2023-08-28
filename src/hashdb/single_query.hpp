#ifndef SINGLE_QUERY_HPP
#define SINGLE_QUERY_HPP

#include <string>

using namespace std;

class SingleQuery
{
public:
    string query;
    bool full;
    bool sent;
    SingleQuery() : full(false), sent(false) {};
    uint64_t size (void) { return query.size(); }
};

#endif