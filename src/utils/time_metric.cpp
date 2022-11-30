#include "time_metric.hpp"

void TimeMetricStorage::add(string &key, uint64_t time, uint64_t times)
{
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
}
void TimeMetricStorage::print(const char * pTitle, uint64_t padding)
{
    if (pTitle != NULL)
    {
        cout << pTitle << ":" << endl;
    }
    else
    {
        cout << "TimeMetricStorage::print():" << endl;
    }
    uint64_t totalTime = 0;
    uint64_t totalTimes = 0;
    unordered_map<string, TimeMetric>::iterator it;
    for (it = map.begin(); it != map.end(); it++)
    {
        string key = it->first;
        if(key.size() < padding)
        {
            key.insert(0, padding - key.size(), ' ');
        }
        cout << key << " time: " << setw(10) << it->second.time << " us, called " << setw(6) << it->second.times << " times, so " << setw(4) << it->second.time/zkmax(it->second.times,(uint64_t)1) << " us/time" << endl;
        totalTime += it->second.time;
        totalTimes += it->second.times;
    }
    string total = "TOTAL";
    if(total.size() < padding)
    {
        total.insert(0, padding - total.size(), ' ');
    }

    cout << total << " time: " << setw(10) << totalTime << " us, called " << setw(6) << totalTimes << " times, so " << setw(4) << totalTime/zkmax(totalTimes,(uint64_t)1) << " us/time" << endl;
}