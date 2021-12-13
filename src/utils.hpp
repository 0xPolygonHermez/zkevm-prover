#ifndef UTILS_HPP
#define UTILS_HPP

#include <sys/time.h>
#include "ffiasm/fr.hpp"
#include "context.hpp"
#include "config.hpp"

/*********/
/* Print */
/*********/

// These functions log information into the console

void printRegs    (Context &ctx);
void printVars    (Context &ctx);
void printMem     (Context &ctx);
void printStorage (Context &ctx);
void printDb      (Context &ctx);
void printDb      (RawFr &fr, map< RawFr::Element, vector<RawFr::Element>, CompareFe > &db);

void printReg  (Context &ctx, string name, RawFr::Element &V, bool h = false, bool bShort = false);
void printU64  (Context &ctx, string name, uint64_t v);
void printU32  (Context &ctx, string name, uint32_t v);
void printU16  (Context &ctx, string name, uint16_t v);

string printFea (Context &ctx, Fea &fea);

// Returns the time difference in us
uint64_t TimeDiff (const struct timeval &startTime, const struct timeval &endTime);
uint64_t TimeDiff (const struct timeval &startTime); // End time is now

#ifdef LOG_TIME
#define TimerStart(name) struct timeval name##_start; gettimeofday(&name##_start,NULL)
#define TimerStop(name) struct timeval name##_stop; gettimeofday(&name##_stop,NULL)
#define TimerLog(name) cout << "TIMER: " + string(#name) + ": " << double(TimeDiff(name##_start,name##_stop))/1000 << " ms" << endl
#else
#define TimerStart(name)
#define TimerStop(name)
#define TimerLog(name)
#endif

#endif