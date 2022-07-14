#ifndef RESULT_HPP
#define RESULT_HPP

typedef enum : int
{
    R_UNSPECIFIED = 0,
    R_SUCCESS = 1,
    R_DB_KEY_NOT_FOUND = 2,
    R_DB_ERROR = 3,
    R_INTERNAL_ERROR = 4
} result_t;

const char* result2string (int code);

#endif