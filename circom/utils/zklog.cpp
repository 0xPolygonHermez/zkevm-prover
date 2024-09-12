#include "zklog.hpp"

zkLog zklog;

string zkLog::getThreadID (void)
{
    pthread_t selfThreadID = pthread_self();
    mpz_class selfThreadIDScalar(selfThreadID);
    string selfThreadIDString = selfThreadIDScalar.get_str(16);
    uint64_t offset = selfThreadIDString.size() > 7 ? selfThreadIDString.size() - 7 : 0;
    return selfThreadIDString.substr(offset, 7);
}

zkLog::zkLog () : jsonLogs(true)
{
    pthread_mutex_init(&mutex, NULL);
}

inline string getTimestamp(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    char tmbuf[64], buf[256];
    strftime(tmbuf, sizeof(tmbuf), "%Y%m%d_%H%M%S", gmtime(&tv.tv_sec));
    snprintf(buf, sizeof(buf), "%s_%06ld", tmbuf, tv.tv_usec);
    return buf;
}

inline string getTimestampWithPeriod(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    char buf[256];
    snprintf(buf, sizeof(buf), "%ld.%06ld", tv.tv_sec, tv.tv_usec);
    return buf;
}

void zkLog::log (const zkLogType type, const string &message, const vector<LogTag> *tags)
{
    lock();
    if (jsonLogs)
    {
        string s = "{";
        switch (type)
        {
            case logTypeInfo:
            {
                s+= "\"level\":\"info\"";
                break;
            }
            case logTypeWarning:
            {
                s+= "\"level\":\"warn\"";
                break;
            }
            case logTypeError:
            {
                s+= "\"level\":\"error\"";
                break;
            }
            default:
            {
                cerr << "zkLog::log() invalid type=" << type << endl << flush;
                exit(-1);
            }
        }
        s += ",\"ts\":\"" + getTimestampWithPeriod() + "\"";
        s += ",\"msg\":\"" + message + "\"";
        s += ",\"pid\":\"" + pid + "\"";
        s += ",\"tid\":\"" + getThreadID() + "\"";
        if (tags != NULL)
        {
            for (uint64_t i=0; i<tags->size(); i++)
            {
                s += ",\"" + (*tags)[i].name + "\":\"" + (*tags)[i].value + "\"";
            }
        }
        s += "}";

        if (type == logTypeError)
        {
            cerr << s << endl << flush;
        }
        else
        {
            cout << s << endl << flush;
        }
    }
    else
    {
        string s = getTimestamp() + " " + pid + " " + getThreadID() + " ";
        switch (type)
        {
            case logTypeInfo:
            {
                s += message;
                cout << s << endl << flush;
                break;
            }
            case logTypeWarning:
            {
                s += "zkWarning: " + message;
                cout << s << endl << flush;
                break;
            }
            case logTypeError:
            {
                s += "zkError: " + message;
                cerr << s << endl << flush;
                break;
            }
            default:
            {
                cerr << "zkLog::log() invalid type=" << type << endl << flush;
                exit(-1);
            }
        }
    }
    unlock();
}