#include "exit_process.hpp"
#include "utils.hpp"

void exitProcess(void)
{
    printCallStack();
    cout << endl;
    printMemoryInfo();
    cout << endl;
    printProcessInfo();
    cout << endl;
    exit(-1);
}