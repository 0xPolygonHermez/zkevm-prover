#include <string>
#include <gmpxx.h>
#include "ecrecover.hpp"
#include "ecrecover_test.hpp"
#include "zklog.hpp"
#include "timer.hpp"
#include <iostream>

using namespace std;

struct ECRecoverTestVector
{
    string signature;
    string r;
    string s;
    string v;
    bool precompiled;
    ECRecoverResult result;
    string address;

};

ECRecoverTestVector ecrecoverTestVectors[] = {
    { "d9eba16ed0ecae432b71fe008c98cc872bb4cc214d3220a36f365326cf807d68", "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", "265e99e47ad31bb2cab9646c504576b3abc6939a1710afc08cbf3034d73214b8", "1c", false, ECR_NO_ERROR,"BC44674AD5868F642EAD3FDF94E2D9C9185EAFB7" }
};

void ECRecoverTest (void)
{
    TimerStart(ECRECOVER_TEST);

    mpz_class signature, r, s, v, address;

    ECRecoverResult result;
    for (uint64_t i=0; i<1; i++)
    {
        signature.set_str(ecrecoverTestVectors[i].signature, 16);
        r.set_str(ecrecoverTestVectors[i].r, 16);
        s.set_str(ecrecoverTestVectors[i].s, 16);
        v.set_str(ecrecoverTestVectors[i].v, 16);
        address.set_str(ecrecoverTestVectors[i].address, 16);

        result = ECRecover(signature, r, s, v, ecrecoverTestVectors[i].precompiled, address);
        
        if (result != ecrecoverTestVectors[i].result)
        {
            zklog.error("ECRecoverTest() failed i=" + to_string(i) + " signature=" + ecrecoverTestVectors[i].signature + " result=" + to_string(result) + " expectedResult=" + to_string(ecrecoverTestVectors[i].result));
        }
        
        if (address != mpz_class(ecrecoverTestVectors[i].address,16))
        {
            zklog.error("ECRecoverTest() failed i=" + to_string(i) + " signature=" + ecrecoverTestVectors[i].signature + " address=" + address.get_str(16) + " expectedAddress=" + ecrecoverTestVectors[i].address);
        }
    }

    TimerStopAndLog(ECRECOVER_TEST);
}
