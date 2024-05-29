#include "pols_test.hpp"
#include "../../src/sm/pols_generated/commit_pols.hpp"
#include "../../src/utils/timer.hpp"
#include "../../src/utils/exit_process.hpp"
#include "../../src/config/config.hpp"
#include "../../src/config/zkglobals.hpp"

#define POLS_TEST_NUMBER_OF_LOOPS_1 10000
#define POLS_TEST_NUMBER_OF_LOOPS_2 10000

#pragma GCC push_options
#pragma GCC optimize ("O0")

void PolsTest (void)
{
    // Allocate committed polynomials for only 1 evaluation
    void * pAddress = calloc(PROVER_FORK_NAMESPACE::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
    if (pAddress == NULL)
    {
        zklog.error("Executor::processBatch() failed calling calloc(" + to_string(PROVER_FORK_NAMESPACE::CommitPols::pilSize()) + ")");
        exitProcess();
    }
    PROVER_FORK_NAMESPACE::CommitPols pols(pAddress,1);

    uint64_t value = rand();
    Goldilocks::Element fe;

    cout << "value=" << value << endl;

    TimerStart(POLS_TEST);
    for (uint64_t i=0; i<POLS_TEST_NUMBER_OF_LOOPS_1; i++)
    {
        //cout << "j=" << j << endl;
        for (uint64_t j=0; j<POLS_TEST_NUMBER_OF_LOOPS_2; j++)
        {
            pols.Main.A0[0] = fr.fromU64(value+i+j);
            pols.Main.A1[0] = fr.fromU64(value+i+j);
            pols.Main.A2[0] = fr.fromU64(value+i+j);
            pols.Main.A3[0] = fr.fromU64(value+i+j);
            pols.Main.A4[0] = fr.fromU64(value+i+j);
            pols.Main.A5[0] = fr.fromU64(value+i+j);
            pols.Main.A6[0] = fr.fromU64(value+i+j);
            pols.Main.A7[0] = fr.fromU64(value+i+j);
            pols.Main.B0[0] = fr.fromU64(value+i+j);
            pols.Main.B1[0] = fr.fromU64(value+i+j);
            pols.Main.B2[0] = fr.fromU64(value+i+j);
            pols.Main.B3[0] = fr.fromU64(value+i+j);
            pols.Main.B4[0] = fr.fromU64(value+i+j);
            pols.Main.B5[0] = fr.fromU64(value+i+j);
            pols.Main.B6[0] = fr.fromU64(value+i+j);
            pols.Main.B7[0] = fr.fromU64(value+i+j);
            pols.Main.C0[0] = fr.fromU64(value+i+j);
            pols.Main.C1[0] = fr.fromU64(value+i+j);
            pols.Main.C2[0] = fr.fromU64(value+i+j);
            pols.Main.C3[0] = fr.fromU64(value+i+j);
            pols.Main.C4[0] = fr.fromU64(value+i+j);
            pols.Main.C5[0] = fr.fromU64(value+i+j);
            pols.Main.C6[0] = fr.fromU64(value+i+j);
            pols.Main.C7[0] = fr.fromU64(value+i+j);
            pols.Main.D0[0] = fr.fromU64(value+i+j);
            pols.Main.D1[0] = fr.fromU64(value+i+j);
            pols.Main.D2[0] = fr.fromU64(value+i+j);
            pols.Main.D3[0] = fr.fromU64(value+i+j);
            pols.Main.D4[0] = fr.fromU64(value+i+j);
            pols.Main.D5[0] = fr.fromU64(value+i+j);
            pols.Main.D6[0] = fr.fromU64(value+i+j);
            pols.Main.D7[0] = fr.fromU64(value+i+j);
            pols.Main.E0[0] = fr.fromU64(value+i+j);
            pols.Main.E1[0] = fr.fromU64(value+i+j);
            pols.Main.E2[0] = fr.fromU64(value+i+j);
            pols.Main.E3[0] = fr.fromU64(value+i+j);
            pols.Main.E4[0] = fr.fromU64(value+i+j);
            pols.Main.E5[0] = fr.fromU64(value+i+j);
            pols.Main.E6[0] = fr.fromU64(value+i+j);
            pols.Main.E7[0] = fr.fromU64(value+i+j);
        }
        for (uint64_t j=0; j<POLS_TEST_NUMBER_OF_LOOPS_2; j++)
        {
            fe = pols.Main.A0[0];
            fe = pols.Main.A1[0];
            fe = pols.Main.A2[0];
            fe = pols.Main.A3[0];
            fe = pols.Main.A4[0];
            fe = pols.Main.A5[0];
            fe = pols.Main.A6[0];
            fe = pols.Main.A7[0];
            fe = pols.Main.B0[0];
            fe = pols.Main.B1[0];
            fe = pols.Main.B2[0];
            fe = pols.Main.B3[0];
            fe = pols.Main.B4[0];
            fe = pols.Main.B5[0];
            fe = pols.Main.B6[0];
            fe = pols.Main.B7[0];
            fe = pols.Main.C0[0];
            fe = pols.Main.C1[0];
            fe = pols.Main.C2[0];
            fe = pols.Main.C3[0];
            fe = pols.Main.C4[0];
            fe = pols.Main.C5[0];
            fe = pols.Main.C6[0];
            fe = pols.Main.C7[0];
            fe = pols.Main.D0[0];
            fe = pols.Main.D1[0];
            fe = pols.Main.D2[0];
            fe = pols.Main.D3[0];
            fe = pols.Main.D4[0];
            fe = pols.Main.D5[0];
            fe = pols.Main.D6[0];
            fe = pols.Main.D7[0];
            fe = pols.Main.E0[0];
            fe = pols.Main.E1[0];
            fe = pols.Main.E2[0];
            fe = pols.Main.E3[0];
            fe = pols.Main.E4[0];
            fe = pols.Main.E5[0];
            fe = pols.Main.E6[0];
            fe = pols.Main.E7[0];
        }
    }
    TimerStopAndLog(POLS_TEST);

    free(pAddress);
}

#pragma GCC pop_options