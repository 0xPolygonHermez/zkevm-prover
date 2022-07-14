#include "zkresult.hpp"

struct {
    int code;
    const char* message;
} resultdesc[] = {
    { ZKR_UNSPECIFIED, "Unspecified error" },
    { ZKR_SUCCESS, "Success" },
    { ZKR_DB_KEY_NOT_FOUND, "Key not found in the database" },
    { ZKR_DB_ERROR, "Database error" },
    { ZKR_INTERNAL_ERROR, "Internal error" }
};

const char* zkresult2string (int code)
{
    for (int i = 0; resultdesc[i].message; i++)
        if (resultdesc[i].code == code)
            return resultdesc[i].message;
    return "unknown";
}