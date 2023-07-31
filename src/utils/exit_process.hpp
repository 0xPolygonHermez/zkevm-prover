#ifndef EXIT_PROCESS_HPP
#define EXIT_PROCESS_HPP

// Exit process with an error, printing call stack
void exitProcess(void);

// If true, process is exiting, so do not start anything, just finish
extern bool bExitingProcess;

#endif