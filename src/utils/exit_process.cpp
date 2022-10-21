#include "exit_process.hpp"
#include "utils.hpp"

void exitProcess(void)
{
    printCallStack();
    printMemoryInfo();
    printProcessInfo();
    exit(-1);
}