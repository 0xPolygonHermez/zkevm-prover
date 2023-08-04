#include <unistd.h>
#include "exit_process.hpp"
#include "utils.hpp"

bool bExitingProcess = false;

void exitProcess(void)
{
    // Notify other threads that we are exiting
    bExitingProcess = true;

    // Log information
    printCallStack();
    printMemoryInfo();
    printProcessInfo();

    // Wait
    sleep(5);

    // Exit the process
    exit(-1);
}