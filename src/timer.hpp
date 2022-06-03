#ifndef TIMER_HPP
#define TIMER_HPP

#include <cstdint>
#include <sys/time.h>
#include "definitions.hpp"

// Returns the time difference in us
uint64_t TimeDiff(const struct timeval &startTime, const struct timeval &endTime);
uint64_t TimeDiff(const struct timeval &startTime); // End time is now

#ifdef LOG_TIME
#define TimerStart(name) cout << "--> " + string(#name) + " starting..." << endl; struct timeval name##_start; gettimeofday(&name##_start,NULL)
#define TimerStop(name) cout << "<-- " + string(#name) + " done" << endl; struct timeval name##_stop; gettimeofday(&name##_stop,NULL)
#define TimerLog(name) cout << "" + string(#name) + ": " << double(TimeDiff(name##_start,name##_stop))/1000000 << " s" << endl
#define TimerStopAndLog(name) cout << "<-- " + string(#name) + " done: " << double(TimeDiff(name##_start))/1000000 << " s" << endl
#else
#define TimerStart(name)
#define TimerStop(name)
#define TimerLog(name)
#define TimerStopAndLog(name)
#endif

#endif