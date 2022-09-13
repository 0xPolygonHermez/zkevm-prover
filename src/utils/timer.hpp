#ifndef TIMER_HPP
#define TIMER_HPP

#include <cstdint>
#include <sys/time.h>
#include <string>
#include "definitions.hpp"

// Returns the time difference in us
uint64_t TimeDiff(const struct timeval &startTime, const struct timeval &endTime);
uint64_t TimeDiff(const struct timeval &startTime); // End time is now

// Returns date and time in a string
std::string DateAndTime(struct timeval &tv);

#ifdef LOG_TIME
#define TimerStart(name) struct timeval name##_start; gettimeofday(&name##_start,NULL); cout << DateAndTime(name##_start) << " --> " + string(#name) + " starting..." << endl
#define TimerStop(name) struct timeval name##_stop; gettimeofday(&name##_stop,NULL); cout << DateAndTime(name##_stop) << " <-- " + string(#name) + " done" << endl
#define TimerLog(name) cout << DateAndTime(name##_stop) << " " + string(#name) + ": " << double(TimeDiff(name##_start, name##_stop))/1000000 << " s" << endl
#define TimerStopAndLog(name) struct timeval name##_stop; gettimeofday(&name##_stop,NULL); cout << DateAndTime(name##_stop) << " <-- " + string(#name) + " done: " << double(TimeDiff(name##_start, name##_stop))/1000000 << " s" << endl
#else
#define TimerStart(name)
#define TimerStop(name)
#define TimerLog(name)
#define TimerStopAndLog(name)
#endif

#endif