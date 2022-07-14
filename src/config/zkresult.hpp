#ifndef ZKRESULT_HPP
#define ZKRESULT_HPP

typedef enum : int
{
    ZKR_UNSPECIFIED = 0,
    ZKR_SUCCESS = 1,
    ZKR_DB_KEY_NOT_FOUND = 2,
    ZKR_DB_ERROR = 3,
    ZKR_INTERNAL_ERROR = 4
} zkresult;

const char* zkresult2string (int code);

#endif