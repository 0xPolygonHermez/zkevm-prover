#ifndef __THREAD_UTILS_H__
#define __THREAD_UTILS_H__

#include <omp.h>
#include <cstring>

class ThreadUtils {
public:
    static void parcpy(void *dst, const void *src, uint64_t nBytes, uint32_t nThreads);
    static void parset(void *dst, int value, uint64_t nBytes, uint32_t nThreads);
};

inline void ThreadUtils::parcpy(void *dst, const void *src, uint64_t nBytes, uint32_t nThreads) {
    if (nThreads < 1) nThreads = 1;

    uint64_t bytesThread = nBytes / nThreads;
    uint64_t residualBytes = nBytes - bytesThread * nThreads;

    #pragma omp parallel for
    for(uint32_t i=0; i < nThreads; i++)
    {
        uint64_t offset = i * bytesThread;
        memcpy((uint8_t *)dst + offset, (uint8_t *)src + offset, i == nThreads - 1 ? bytesThread + residualBytes : bytesThread);
    }
}

inline void ThreadUtils::parset(void *dst, int value, uint64_t nBytes, uint32_t nThreads) {
    if (nThreads < 1) nThreads = 1;

    uint64_t bytesThread = nBytes / nThreads;
    uint64_t residualBytes = nBytes - bytesThread * nThreads;

    #pragma omp parallel for
    for(uint32_t i=0; i < nThreads; i++)
    {
        uint64_t offset = i * bytesThread;
        memset((uint8_t *)dst + offset, value, i == nThreads - 1 ? bytesThread + residualBytes : bytesThread);
    }
}

#endif //__THREAD_UTILS_H__
