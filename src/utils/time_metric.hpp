#ifndef TIME_METRIC_HPP
#define TIME_METRIC_HPP

#include <unordered_map>
#include <iostream>
#include "zkassert.hpp"
#include "zkmax.hpp"
#include "timer.hpp"

class TimeMetric
{
    public:
        uint64_t time;
        uint64_t times;
    TimeMetric() : time(0), times(0) {;}
};

class TimeMetricStorage
{
private:

    // Lock
    pthread_mutex_t mutex; // Mutex to protect multithread access
    void lock(void) { pthread_mutex_lock(&mutex); };
    void unlock(void) { pthread_mutex_unlock(&mutex); };

    // Data
    unordered_map<string, TimeMetric> map;

    // Start time
    timeval startTime;

public:

    TimeMetricStorage()
    {
        pthread_mutex_init(&mutex, NULL);
        gettimeofday(&startTime, NULL);
    }
    
    void add(const char * pChar, uint64_t time, uint64_t times=1)
    {
        string key = pChar;
        add(key, time, times);
    }
    
    void add   (string &key, uint64_t time, uint64_t times=1);
    void print (const char * pTitle, uint64_t padding = 32);
    void clear (void);
};

#endif