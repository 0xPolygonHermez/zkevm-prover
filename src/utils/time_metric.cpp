#include <iomanip>
#include "utils/time_metric.hpp"
#include "zklog.hpp"

using namespace std;

void TimeMetricStorage::add(string &key, uint64_t time, uint64_t times)
{
    lock();

    unordered_map<string, TimeMetric>::iterator it;
    it = map.find(key);
    if (map.find(key) == map.end())
    {
        TimeMetric tm;
        tm.time  = time;
        tm.times = times;
        map[key] = tm;
    }
    else
    {
        it->second.time  += time;
        it->second.times += times;
    }

    unlock();
}
void TimeMetricStorage::print(const char * pTitle, uint64_t padding)
{
    lock();

    uint64_t duration = TimeDiff(startTime);

    if (pTitle != NULL)
    {
        zklog.info(string(pTitle) + ":");
    }
    else
    {
        zklog.info("TimeMetricStorage::print():");
    }
    
    uint64_t totalTime = 0;
    uint64_t totalTimes = 0;
    unordered_map<string, TimeMetric>::iterator it;
    for (it = map.begin(); it != map.end(); it++)
    {
        totalTime += it->second.time;
        totalTimes += it->second.times;
    }

    zklog.info("Duration=" + to_string(duration) + " us time/duration=" + to_string(totalTime*1000/zkmax(duration, (uint64_t)1)) + "%");

    for (it = map.begin(); it != map.end(); it++)
    {
        string key = it->first;
        if (key.size() < padding)
        {
            key.insert(0, padding - key.size(), ' ');
        }
        string time = to_string(it->second.time);
        if (time.size() < 10)
        {
            time.insert(0, 10 - time.size(), ' ');
        }
        string percentage1 = to_string(it->second.time*1000/zkmax(totalTime,(uint64_t)1));
        if (percentage1.size() < 3)
        {
            percentage1.insert(0, 3 - percentage1.size(), ' ');
        }
        string percentage2 = to_string(it->second.time*1000/zkmax(duration,(uint64_t)1));
        if (percentage2.size() < 3)
        {
            percentage2.insert(0, 3 - percentage2.size(), ' ');
        }
        string times = to_string(it->second.times);
        if (times.size() < 8)
        {
            times.insert(0, 8 - times.size(), ' ');
        }
        string nsPerTime = to_string(it->second.time*1000/zkmax(it->second.times,(uint64_t)1));
        if (nsPerTime.size() < 7)
        {
            nsPerTime.insert(0, 7 - nsPerTime.size(), ' ');
        }
        zklog.info(key + " time: " + time + " us (" + percentage1 + "%)(" + percentage2 + "%), called " + times + " times, so " + nsPerTime + " ns/time");
    }
    string total = "TOTAL";
    if(total.size() < padding)
    {
        total.insert(0, padding - total.size(), ' ');
    }
    string totalTimeString = to_string(totalTime);
    if (totalTimeString.size() < 10)
    {
        totalTimeString.insert(0, 10 - totalTimeString.size(), ' ');
    }
    string percentage = to_string(totalTime*1000/zkmax(duration, (uint64_t)1));
    if (percentage.size() < 3)
    {
        percentage.insert(0, 3 - percentage.size(), ' ');
    }
    string totalTimesString = to_string(totalTimes);
    if (totalTimesString.size() < 8)
    {
        totalTimesString.insert(0, 8 - totalTimesString.size(), ' ');
    }
    string totalNsPerTime = to_string(totalTime*1000/zkmax(totalTimes,(uint64_t)1));
    if (totalNsPerTime.size() < 7)
    {
        totalNsPerTime.insert(0, 7 - totalNsPerTime.size(), ' ');
    }

    zklog.info(total + " time: " + totalTimeString + " us (1000)(" + percentage + "%), called " + totalTimesString + " times, so " + totalNsPerTime + " ns/time");

    unlock();
}

void TimeMetricStorage::clear (void)
{
    lock();
    map.clear();
    unlock();
}