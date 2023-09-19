#ifndef ZKLOG_HPP
#define ZKLOG_HPP

#include <string>
#include <pthread.h>
#include <vector>

using namespace std;

enum zkLogType
{
    logTypeInfo = 0,
    logTypeWarning = 1,
    logTypeError = 2
};

class LogTag
{
public:
    string name;
    string value;
    LogTag(const char * name, string&value) : name(name), value(value) {;};
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

    void log (const zkLogType type, const string &message, const vector<LogTag> *tags);

public:
    zkLog ();
    void setPID (const string &_pid) { pid = _pid; };
    void setJsonLogs (bool bJsonLogs) { jsonLogs = bJsonLogs; };

    void info      (const string &message, const vector<LogTag> *tags = NULL) { log(logTypeInfo,    message, tags); };
    void warning   (const string &message, const vector<LogTag> *tags = NULL) { log(logTypeWarning, message, tags); };
    void error     (const string &message, const vector<LogTag> *tags = NULL) { log(logTypeError,   message, tags); };
};

extern zkLog zklog;

#endif