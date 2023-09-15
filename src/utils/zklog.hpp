#ifndef ZKLOG_HPP
#define ZKLOG_HPP

#include <string>

using namespace std;

enum zkLogType
{
    logTypeInfo = 0,
    logTypeWarning = 1,
    logTypeError = 2
};

class zkLog
{
private:
    // Mutex attributes
    pthread_mutex_t mutex;
    void lock(void) { pthread_mutex_lock(&mutex); };
    void unlock(void) { pthread_mutex_unlock(&mutex); };

    // Log pid
    string pid;

    // Configuration
    bool jsonLogs;

    string getThreadID (void);

    void log (const zkLogType type, const string &message);

public:
    zkLog ();
    void setPID (const string &_pid) { pid = _pid; };
    void setJsonLogs (bool bJsonLogs) { jsonLogs = bJsonLogs; };

    void info      (const string &message) { log(logTypeInfo,    message); };
    void warning   (const string &message) { log(logTypeWarning, message); };
    void error     (const string &message) { log(logTypeError,   message); };
};

extern zkLog zklog;

#endif