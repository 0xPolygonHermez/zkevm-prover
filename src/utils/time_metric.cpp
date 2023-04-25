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

    string s;

    if (pTitle != NULL)
    {
        s = string(pTitle) + ": ";

#ifdef TIME_METRIC_TABLE
        s += "\n";
#endif
    }
    else
    {
        s = "TimeMetricStorage::print(): ";
    }
    
    uint64_t totalTime = 0;
    uint64_t totalTimes = 0;
    unordered_map<string, TimeMetric>::iterator it;
    for (it = map.begin(); it != map.end(); it++)
    {
        totalTime += it->second.time;
        totalTimes += it->second.times;
    }

    s += "Duration=" + to_string(duration) + "us time/duration=" + to_string(totalTime*1000/zkmax(duration, (uint64_t)1)) + "% ";

#ifdef TIME_METRIC_TABLE
    s += "\n";
#endif

    for (it = map.begin(); it != map.end(); it++)
    {
        string key = it->first;
#ifdef TIME_METRIC_TABLE
        if (key.size() < padding)
        {
            key.insert(0, padding - key.size(), ' ');
        }
#endif
        string time = to_string(it->second.time);
#ifdef TIME_METRIC_TABLE
        if (time.size() < 10)
        {
            time.insert(0, 10 - time.size(), ' ');
        }
#endif
        string percentage1 = to_string(it->second.time*1000/zkmax(totalTime,(uint64_t)1));
#ifdef TIME_METRIC_TABLE
        if (percentage1.size() < 3)
        {
            percentage1.insert(0, 3 - percentage1.size(), ' ');
        }
#endif
        string percentage2 = to_string(it->second.time*1000/zkmax(duration,(uint64_t)1));
#ifdef TIME_METRIC_TABLE
        if (percentage2.size() < 3)
        {
            percentage2.insert(0, 3 - percentage2.size(), ' ');
        }
#endif
        string times = to_string(it->second.times);
#ifdef TIME_METRIC_TABLE
        if (times.size() < 8)
        {
            times.insert(0, 8 - times.size(), ' ');
        }
#endif
        string nsPerTime = to_string(it->second.time*1000/zkmax(it->second.times,(uint64_t)1));
#ifdef TIME_METRIC_TABLE
        if (nsPerTime.size() < 7)
        {
            nsPerTime.insert(0, 7 - nsPerTime.size(), ' ');
        }
#endif
        s += key + "=" + time + "us(" + percentage1 + "%)(" + percentage2 + "%)=" + times + "times=" + nsPerTime + "ns/time ";
#ifdef TIME_METRIC_TABLE
        s += "\n";
#endif
    }
    string total = "TOTAL";
#ifdef TIME_METRIC_TABLE
    if(total.size() < padding)
    {
        total.insert(0, padding - total.size(), ' ');
    }
#endif
    string totalTimeString = to_string(totalTime);
#ifdef TIME_METRIC_TABLE
    if (totalTimeString.size() < 10)
    {
        totalTimeString.insert(0, 10 - totalTimeString.size(), ' ');
    }
#endif
    string percentage = to_string(totalTime*1000/zkmax(duration, (uint64_t)1));
#ifdef TIME_METRIC_TABLE
    if (percentage.size() < 3)
    {
        percentage.insert(0, 3 - percentage.size(), ' ');
    }
#endif
    string totalTimesString = to_string(totalTimes);
#ifdef TIME_METRIC_TABLE
    if (totalTimesString.size() < 8)
    {
        totalTimesString.insert(0, 8 - totalTimesString.size(), ' ');
    }
#endif
    string totalNsPerTime = to_string(totalTime*1000/zkmax(totalTimes,(uint64_t)1));
#ifdef TIME_METRIC_TABLE
    if (totalNsPerTime.size() < 7)
    {
        totalNsPerTime.insert(0, 7 - totalNsPerTime.size(), ' ');
    }
#endif

    s += total + "=" + totalTimeString + "us(1000)(" + percentage + "%)=" + totalTimesString + "times=" + totalNsPerTime + "ns/time";

    zklog.info(s);

    unlock();
}

void TimeMetricStorage::clear (void)
{
    lock();
    map.clear();
    unlock();
}