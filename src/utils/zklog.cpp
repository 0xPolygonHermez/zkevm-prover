#include "zklog.hpp"
#include "utils.hpp"

zkLog zklog;

zkLog::zkLog ()
{
    pthread_mutex_init(&mutex, NULL);
}

void zkLog::info (const string &message)
{
    lock();
    cout << getTimestamp() << " " << message << endl;
    unlock();
}

void zkLog::warning (const string &message)
{
    lock();
    cout << getTimestamp() << " zkWarning: " << message << endl;
    unlock();
}

void zkLog::error (const string &message)
{
    lock();
    cerr << getTimestamp() << " zkError: " << message << endl << flush;
    unlock();
}