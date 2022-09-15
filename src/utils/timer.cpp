#include <iostream>
#include "timer.hpp"
#include "utils.hpp"

using namespace std;


uint64_t TimeDiff(const struct timeval &startTime, const struct timeval &endTime)
{
    struct timeval diff = {0, 0};

    // Calculate the time difference
    diff.tv_sec = endTime.tv_sec - startTime.tv_sec;
    if (endTime.tv_usec >= startTime.tv_usec)
        diff.tv_usec = endTime.tv_usec - startTime.tv_usec;
    else if (diff.tv_sec > 0)
    {
        diff.tv_usec = 1000000 + endTime.tv_usec - startTime.tv_usec;
        diff.tv_sec--;
    }
    else
    {
        cerr << "Error: TimeDiff() got startTime > endTime" << endl;
        exitProcess();
    }

    // Return the total number of us
    return diff.tv_usec + 1000000 * diff.tv_sec;
}

uint64_t TimeDiff(const struct timeval &startTime)
{
    struct timeval endTime;
    gettimeofday(&endTime, NULL);
    return TimeDiff(startTime, endTime);
}

std::string DateAndTime(struct timeval &tv)
{
    struct tm *pTm;
    pTm = localtime(&tv.tv_sec);
    char cResult[32];
    strftime(cResult, sizeof(cResult), "%Y/%m/%d %H:%M:%S", pTm);
    std::string sResult(cResult);
    return sResult;
}