#include "zklog.hpp"
#include "utils.hpp"

zkLog zklog;

zkLog::zkLog ()
{
    pthread_mutex_init(&mutex, NULL);
}

void zkLog::setPrefix ( const string &_prefix)
{
    prefix = _prefix;
}

void zkLog::info (const string &message)
{
    lock();
    cout << getTimestamp() << " " << prefix << message << endl;
    unlock();
}

void zkLog::warning (const string &message)
{
    lock();
    cout << getTimestamp() << " " << prefix << "zkWarning: " << message << endl;
    unlock();
}

void zkLog::error (const string &message)
{
    lock();
    cerr << getTimestamp() << " " << prefix << "zkError: " << message << endl << flush;
    unlock();
}