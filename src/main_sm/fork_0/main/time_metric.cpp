#include "main_sm/fork_0/main/time_metric.hpp"

namespace fork_0
{

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
        totalTime += it->second.time;
        totalTimes += it->second.times;
    }
    for (it = map.begin(); it != map.end(); it++)
    {
        string key = it->first;
        if(key.size() < padding)
        {
            key.insert(0, padding - key.size(), ' ');
        }
        cout << key << " time: " << setw(10) << it->second.time << " us (" << setw(3) << it->second.time*1000/totalTime << "%), called " << setw(8) << it->second.times << " times, so " << setw(7) << it->second.time*1000/zkmax(it->second.times,(uint64_t)1) << " ns/time" << endl;
    }
    string total = "TOTAL";
    if(total.size() < padding)
    {
        total.insert(0, padding - total.size(), ' ');
    }

    cout << total << " time: " << setw(10) << totalTime << " us (100%), called " << setw(8) << totalTimes << " times, so " << setw(7) << totalTime*1000/zkmax(totalTimes,(uint64_t)1) << " ns/time" << endl;
}

} // namespace