#ifndef UTILS_HPP
#define UTILS_HPP

#include <sys/time.h>
#include "ffiasm/fr.hpp"
#include "context.hpp"
#include "config.hpp"
#include "reference.hpp"

/*********/
/* Print */
/*********/

// These functions log information into the console

void printRegs    (Context &ctx);
void printVars    (Context &ctx);
void printMem     (Context &ctx);
#ifdef USE_LOCAL_STORAGE
void printStorage (Context &ctx);
#endif
void printDb      (Context &ctx);

void printReg  (Context &ctx, string name, RawFr::Element &V, bool h = false, bool bShort = false);
void printU64  (Context &ctx, string name, uint64_t v);
void printU32  (Context &ctx, string name, uint32_t v);
void printU16  (Context &ctx, string name, uint16_t v);

void printReference (RawFr &fr,Reference &ref);

string printFea (Context &ctx, Fea &fea);

// Returns the time difference in us
uint64_t TimeDiff (const struct timeval &startTime, const struct timeval &endTime);
uint64_t TimeDiff (const struct timeval &startTime); // End time is now

#define zkmin(a,b) ((a>=b)?b:a)
#define zkmax(a,b) ((a>=b)?a:b)

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