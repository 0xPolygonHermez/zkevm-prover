#include <string>
#include <gmpxx.h>
#include "ecrecover.hpp"
#include "ecrecover_test.hpp"
#include "zklog.hpp"
#include "timer.hpp"

using namespace std;

struct ECRecoverTestVector
{
    string hash;
    string r;
    string s;
    string v;
    bool precompiled;
    ECRecoverResult result;
};

ECRecoverTestVector ecrecoverTestVectors[] = {
    { "57e7a14390cd9b6d76032484b266b8263bf59c4c0c0cbbf8e2b3f9e66c825bf8", "1cee7e01dc62f69a12c3510c6d64de04ee6346d84b6a017f3e786c7d87f963e7", "5d8cc91fa983cd6d9cf55fff80d73bd26cd333b0f098acc1e58edb1fd484ad73", "1b", false, ECR_NO_ERROR }
};

void ECRecoverTest (void)
{
    TimerStart(ECRECOVER_TEST);

    mpz_class hash, r, s, v;

    ECRecoverResult result;

    for (uint64_t i=0; i<1; i++)
    {
        hash.set_str(ecrecoverTestVectors[i].hash, 16);
        r.set_str(ecrecoverTestVectors[i].r, 16);
        s.set_str(ecrecoverTestVectors[i].s, 16);
        v.set_str(ecrecoverTestVectors[i].v, 16);
        result = ECRecover(hash, r, s, v, ecrecoverTestVectors[i].precompiled);
        if (result != ecrecoverTestVectors[i].result)
        {
            zklog.error("ECRecoverTest() failed i=" + to_string(i) + " hash=" + ecrecoverTestVectors[i].hash + " result=" + to_string(result) + " expectedResult=" + to_string(ecrecoverTestVectors[i].result));
        }
    }

    TimerStopAndLog(ECRECOVER_TEST);
}
