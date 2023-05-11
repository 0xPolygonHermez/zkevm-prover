#ifndef WATHDOG_HPP
#define WATHDOG_HPP

#include "timer.hpp"

class Watchdog
{
private:
    struct timeval startTime;
    bool bStarted;
    pthread_mutex_t mutex;
    uint64_t timeout; // In micro-seconds
    pthread_t watchdogPthread;
    void lock(void) { pthread_mutex_lock(&mutex); };
    void unlock(void) { pthread_mutex_unlock(&mutex); };

public:
    Watchdog();
    void start (uint64_t timeout);
    void stop (void);
    void restart (void);
    bool timeoutPassed (void);
};

#endif