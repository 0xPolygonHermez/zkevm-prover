#ifndef ZKRESULT_HPP
#define ZKRESULT_HPP

typedef enum : int
{
    ZKR_UNSPECIFIED = 0,
    ZKR_SUCCESS = 1,
    ZKR_DB_KEY_NOT_FOUND = 2, // Requested key was not found in database
    ZKR_DB_ERROR = 3, // Error connecting to database, or processing request
    ZKR_INTERNAL_ERROR = 4,
    ZKR_SM_MAIN_ASSERT = 5, // Main state machine executor assert failed
    ZKR_SM_MAIN_ARITH = 6, // Main state machine executor arith condition failed
    ZKR_SM_MAIN_BINARY = 7, // Main state machine executor binary condition failed
    ZKR_SM_MAIN_HASHP = 8, // Main state machine executor hash poseidon condition failed
    ZKR_SM_MAIN_HASHK = 9, // Main state machine executor hash Keccak condition failed
    ZKR_SM_MAIN_STORAGE = 10, // Main state machine executor storage condition failed
    ZKR_SM_MAIN_MEMORY = 11, // Main state machine executor memory condition failed
    ZKR_SM_MAIN_MEMALIGN = 12, // Main state machine executor memalign condition failed
    ZKR_SM_MAIN_ADDRESS = 13, // Main state machine executor address condition failed
    ZKR_SMT_INVALID_DATA_SIZE = 14, // Invalid size data for a MT node
    ZKR_SM_MAIN_BATCH_L2_DATA_TOO_BIG = 15 // Input batch L2 data is too big

} zkresult;

const char* zkresult2string (int code);

#endif