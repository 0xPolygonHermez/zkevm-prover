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
    cout << getTimestamp() << " Info: " << message << endl;
    unlock();
}

void zkLog::warning (const string &message)
{
    lock();
    cout << getTimestamp() << " Warning: " << message << endl;
    unlock();
}

void zkLog::error (const string &message)
{
    lock();
    cerr << getTimestamp() << " Error: " << message << endl;
    unlock();
}