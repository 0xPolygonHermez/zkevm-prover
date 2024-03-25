#ifndef ROOT_VERSION_PAGE_HPP
#define ROOT_VERSION_PAGE_NPP

#include <string>
#include <stdint.h>

using namespace std;

/* Root version page is built on top of a generic KeyValuePage,
   in which the value is a U64 version converted to/from a string */

string version2value (const uint64_t version);

uint64_t value2version (const string &value);

#endif