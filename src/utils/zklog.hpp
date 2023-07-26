#ifndef ZKLOG_HPP
#define ZKLOG_HPP

#include <string>

using namespace std;

class zkLog
{
private:
    // Mutex attributes
    pthread_mutex_t mutex;
    void lock(void) { pthread_mutex_lock(&mutex); };
    void unlock(void) { pthread_mutex_unlock(&mutex); };

    // Log prefix
    string prefix;

    string getThreadID (void);

public:
    zkLog ();
    void setPrefix (const string &_prefix);
    void info      (const string &message);
    void warning   (const string &message);
    void error     (const string &message);
};

extern zkLog zklog;

#endif