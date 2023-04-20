#ifndef GOLDILOCKS_PRECOMPUTED_HPP
#define GOLDILOCKS_PRECOMPUTED_HPP

#include <iostream>
#include <pthread.h>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"
#include "zklog.hpp"

using namespace std;

#define GOLDILOCKS_PRECOMPUTED_MAX (8*1024*1024)
//#define GOLDILOCKS_PRECOMPUTED_DEBUG

class GoldilocksPrecomputed
{
private:
    Goldilocks fr;
    Goldilocks::Element invPos[GOLDILOCKS_PRECOMPUTED_MAX];
    Goldilocks::Element invNeg[GOLDILOCKS_PRECOMPUTED_MAX];
    bool bInitialized;
    
#ifdef GOLDILOCKS_PRECOMPUTED_DEBUG
    pthread_mutex_t mutex;    // Mutex to protect the requests queues
    uint64_t succeeded;
    uint64_t failed;
#endif

public:

    GoldilocksPrecomputed() : bInitialized(false)
    {
#ifdef GOLDILOCKS_PRECOMPUTED_DEBUG
        pthread_mutex_init(&mutex, NULL);
        succeeded = 0;
        failed = 0;
#endif
    }

    ~GoldilocksPrecomputed()
    {
#ifdef GOLDILOCKS_PRECOMPUTED_DEBUG
        uint64_t invComputedSize = GOLDILOCKS_PRECOMPUTED_MAX * 8 * 2;
        zklog.info("GoldilocksPrecomputed got inv() computed size= " + to_string(invComputedSize) + " succeeded=" + to_string(succeeded) + " failed=" + to_string(failed) + " total=" + to_string(succeeded+failed) + " percentage=" + to_string(double(succeeded)*100/(succeeded+failed)) + "%");
#endif
    }
#ifdef GOLDILOCKS_PRECOMPUTED_DEBUG
    void lock(void) { pthread_mutex_lock(&mutex); };
    void unlock(void) { pthread_mutex_unlock(&mutex); };
#endif

    void init (void);

    inline Goldilocks::Element inv (const Goldilocks::Element &fe)
    {
        zkassert(bInitialized);

        if (fe.fe<GOLDILOCKS_PRECOMPUTED_MAX)
        {
#ifdef GOLDILOCKS_PRECOMPUTED_DEBUG
            lock();
            succeeded++;
            unlock();
            if (!fr.equal(fr.inv(fe), invPos[fe.fe]))
            {
                zklog.error("GoldilocksPrecomputed::inv() pos fe=" + fr.toString(fe,16) + " fr.inv(fe)=" + fr.toString(fr.inv(fe),16) + " invPos[fe.fe]=" + fr.toString(invPos[fe.fe],16));
                exit(-1);
            }
#endif
            return invPos[fe.fe];
        }
        if ( (fe.fe <= GOLDILOCKS_PRIME) && (fe.fe > (GOLDILOCKS_PRIME-GOLDILOCKS_PRECOMPUTED_MAX)) )
        {
#ifdef GOLDILOCKS_PRECOMPUTED_DEBUG
            lock();
            succeeded++;
            unlock();
            if (!fr.equal(fr.inv(fe), invNeg[GOLDILOCKS_PRIME - fe.fe]))
            {
                zklog.error("GoldilocksPrecomputed::inv() neg fe=" + fr.toString(fe,16) + " fr.inv(fe)=" + fr.toString(fr.inv(fe),16) + " invNeg[GOLDILOCKS_PRIME - fe.fe]=" + fr.toString(invNeg[GOLDILOCKS_PRIME - fe.fe],16));
                exit(-1);
            }
#endif
            return invNeg[GOLDILOCKS_PRIME - fe.fe];
        }
#ifdef GOLDILOCKS_PRECOMPUTED_DEBUG
        lock();
        failed++;
        unlock();
        //zklog.info("GoldilocksPrecomputed::inv() called with fe=" + fr.toString(fe,10));
#endif
        return fr.inv(fe);
    }
};

extern GoldilocksPrecomputed glp;

#endif