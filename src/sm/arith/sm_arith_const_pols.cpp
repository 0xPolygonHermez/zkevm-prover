#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include "sm_arith_const_pols.hpp"

class MyPols {
    public:
        ArithClockConstPol ck;
        ArithBit19ConstPol bit19;
};
/*
int main (int argc, char *argv[])
{
    MyPols pols;  

    for(int index=0; index < 100; ++index) {
        printf("#%02d", index);
        for (int clock=0; clock < 32; ++clock) {
            printf(" %1ld", pols.ck[clock][index]);
        }
        printf("\n");
    }

    for(int index=0; index < 100; ++index) {
        printf("#%02d %3ld\n", index, pols.bit19[index]);
    }
}
*/