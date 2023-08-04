#include "zklog.hpp"
#include "utils.hpp"

zkLog zklog;

string zkLog::getThreadID (void)
{
    pthread_t selfThreadID = pthread_self();
    mpz_class selfThreadIDScalar(selfThreadID);
    string selfThreadIDString = selfThreadIDScalar.get_str(16);
    uint64_t offset = selfThreadIDString.size() > 7 ? selfThreadIDString.size() - 7 : 0;
    return selfThreadIDString.substr(offset, 7);
}

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
    cout << getTimestamp() << " " << prefix << getThreadID() << " " << message << endl;
    unlock();
}

void zkLog::warning (const string &message)
{
    lock();
    cout << getTimestamp() << " " << prefix << getThreadID() << " " << "zkWarning: " << message << endl;
    unlock();
}

void zkLog::error (const string &message)
{
    lock();
    cerr << getTimestamp() << " " << prefix << getThreadID() << " " << "zkError: " << message << endl << flush;
    unlock();
}