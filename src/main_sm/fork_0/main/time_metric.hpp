#ifndef TIME_METRIC_HPP_fork_0
#define TIME_METRIC_HPP_fork_0

#include <unordered_map>
#include <iostream>
#include "main_sm/fork_0/main/rom_command.hpp"
#include "zkassert.hpp"
#include "zkmax.hpp"

namespace fork_0
{

class TimeMetric
{
    public:
        uint64_t time;
        uint64_t times;
    TimeMetric() : time(0), times(0) {;}
};

class TimeMetricStorage
{
public:
    unordered_map<string, TimeMetric> map;
    void add(RomCommand &cmd, uint64_t time, uint64_t times=1)
    {
        string key = op2String(cmd.op) + "[" + function2String(cmd.function) + "]";
        add(key, time, times);
    }
    void add(const char * pChar, uint64_t time, uint64_t times=1)
    {
        string key = pChar;
        add(key, time, times);
    }
    void add(string &key, uint64_t time, uint64_t times=1);
    void print(const char * pTitle, uint64_t padding = 32);
};

} // namespace
#endif