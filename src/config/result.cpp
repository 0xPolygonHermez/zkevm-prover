#include "result.hpp"

struct {
    int code;
    const char* message;
} resultdesc[] = {
    { R_UNSPECIFIED, "Unspecified error" },
    { R_SUCCESS, "Success" },
    { R_DB_KEY_NOT_FOUND, "Key not found in the database" },
    { R_DB_ERROR, "Database error" },
    { R_INTERNAL_ERROR, "Internal error" }
};

const char* result2string (int code)
{
    for (int i = 0; resultdesc[i].message; i++)
        if (resultdesc[i].code == code)
            return resultdesc[i].message;
    return "unknown";
}