#include "version_data_page.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"

string version2key (const uint64_t version)
{
    string key;
    ba2ba(key, version);
    return key;
}

string versionData2value (const VersionDataStruct &versionData)
{
    string value;
    value.append((char *)&versionData, sizeof(VersionDataStruct));
    return value;
}

void value2versionData (VersionDataStruct &versionData, const string &value)
{
    if (value.size() != sizeof(VersionDataStruct))
    {
        zklog.error("value2versionData() found invalid value.size=" + to_string(value.size()) + " != sizeof(VersionDataStruct)=" + to_string(sizeof(VersionDataStruct)));
        exitProcess();
    }

    memcpy(&versionData, value.c_str(), sizeof(VersionDataStruct));
}