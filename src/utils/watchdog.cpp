#include <unistd.h>
#include "watchdog.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "timer.hpp"

void *watchdogThread (void *arg)
{
    Watchdog * pWatchdog = (Watchdog *)arg;
    while (true)
    {
        sleep(1);
        if (pWatchdog->timeoutPassed())
        {
            zklog.error("watchdogThread() timeout passed");
            exitProcess();
        }
    }
}

Watchdog::Watchdog() : bStarted(false), timeout(0)
{
    pthread_mutex_init(&mutex, NULL);
    pthread_create(&watchdogPthread, NULL, watchdogThread, this);
};

void Watchdog::start(uint64_t _timeout)
{
    lock();

    if (bStarted)
    {
        zklog.error("Watchdog::start() called with bStarted=true");
        exitProcess();
    }
    
    timeout = _timeout;
    gettimeofday(&startTime, NULL);
    bStarted = true;

    unlock();
}

void Watchdog::restart(void)
{
    lock();

    if (!bStarted)
    {
        zklog.error("Watchdog::restart() called with bStarted=false");
        exitProcess();
    }

    gettimeofday(&startTime, NULL);

    unlock();
}

void Watchdog::stop(void)
{
    lock();

    if (!bStarted)
    {
        zklog.error("Watchdog::stop() called with bStarted=false");
        exitProcess();
    }
    
    bStarted = false;

    unlock();
}

bool Watchdog::timeoutPassed (void)
{
    lock();
    bool bResult = bStarted && (TimeDiff(startTime) > timeout);
    unlock();
    return bResult;
}