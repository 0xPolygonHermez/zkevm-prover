#include <iostream>
#include <fstream>
#include <string>
#include <gmpxx.h>
#include <cstdio>
#include <sys/stat.h>
#include <nlohmann/json.hpp>
#include "../src/goldilocks/src/goldilocks_base_field.hpp"
#include "../src/goldilocks/src/poseidon_goldilocks.hpp"

using namespace std;
using json = nlohmann::json;

/***********************************************************/
/* Declaration of replicas of zkEVM required functionality */
/***********************************************************/

typedef unordered_map<string, vector<Goldilocks::Element>> MTMap;
typedef unordered_map<string, vector<uint8_t>> ProgramMap;

void json2file (const json &j, const string &fileName);
void file2json (const string &fileName, json &j);
void db2json (json &input, const MTMap &db, string name);
void contractsBytecode2json (json &input, const ProgramMap &contractsBytecode, string name);
string NormalizeToNFormat (const string &s, uint64_t n);
string NormalizeTo0xNFormat (const string &s, uint64_t n);
string PrependZeros (const string &s, uint64_t n);
string Remove0xIfPresent(const string &s);
void exitProcess(void) { exit(-1); };
char byte2char (uint8_t b);
string byte2string(uint8_t b);
void string2ba (const string &textString, string &baString);
inline void fea2scalar (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element (&fea)[4])
{
    scalar = fr.toU64(fea[3]);
    scalar <<= 64;
    scalar += fr.toU64(fea[2]);
    scalar <<= 64;
    scalar += fr.toU64(fea[1]);
    scalar <<= 64;
    scalar += fr.toU64(fea[0]);
}
string fea2string (Goldilocks &fr, const Goldilocks::Element(&fea)[4]);
inline void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element &fe0, Goldilocks::Element &fe1, Goldilocks::Element &fe2, Goldilocks::Element &fe3, Goldilocks::Element &fe4, Goldilocks::Element &fe5, Goldilocks::Element &fe6, Goldilocks::Element &fe7);
mpz_class ScalarMask32  ("FFFFFFFF", 16);
inline void ba2scalar(mpz_class &s, const string &ba)
{
    mpz_import(s.get_mpz_t(), ba.size(), 1, 1, 0, 0, ba.c_str());
};
mpz_class ScalarTwoTo8  ("100", 16);
void ba2scalar (const uint8_t *pData, uint64_t dataSize, mpz_class &s)
{
    s = 0;
    for (uint64_t i=0; i<dataSize; i++)
    {
        s *= ScalarTwoTo8;
        s += pData[i];
    }
}
Goldilocks fr;
PoseidonGoldilocks poseidon;
void poseidonLinearHash (const vector<uint8_t> &_data, Goldilocks::Element (&result)[4]);
typedef enum : int
{
    ZKR_UNSPECIFIED = 0,
    ZKR_SUCCESS = 1,
    ZKR_DB_KEY_NOT_FOUND = 2, // Requested key was not found in database
    ZKR_DB_ERROR = 3, // Error connecting to database, or processing request
    ZKR_INTERNAL_ERROR = 4,
    ZKR_SM_MAIN_ASSERT = 5, // Main state machine executor assert failed
    ZKR_SM_MAIN_STORAGE_INVALID_KEY = 6, // Main state machine executor storage condition failed
    ZKR_SM_MAIN_MEMORY = 7, // Main state machine executor memory condition failed
    ZKR_SMT_INVALID_DATA_SIZE = 8, // Invalid size data for a MT node
    ZKR_AGGREGATED_PROOF_INVALID_INPUT = 9, // Aggregated proof input is incorrect
    ZKR_SM_MAIN_OOC_ARITH = 10, // Incremented arith counters exceeded the maximum
    ZKR_SM_MAIN_OOC_BINARY = 11, // Incremented binary counters exceeded the maximum
    ZKR_SM_MAIN_OOC_MEM_ALIGN = 12, // Incremented mem align counters exceeded the maximum
    ZKR_SM_MAIN_OOC_KECCAK_F = 13, // Incremented keccak-f counters exceeded the maximum
    ZKR_SM_MAIN_OOC_PADDING_PG = 14, // Incremented padding pg counters exceeded the maximum
    ZKR_SM_MAIN_OOC_SHA256_F = 15, // Incremented SHA-256-f counters exceeded the maximum
    ZKR_SM_MAIN_OOC_POSEIDON_G = 16, // Incremented poseidon g counters exceeded the maximum
    ZKR_HASHDB_GRPC_ERROR = 17, // Error making GRPC call to hash DB service
    ZKR_SM_MAIN_OUT_OF_STEPS = 18, // Main state machine executor did not complete the execution within available steps
    ZKR_SM_MAIN_INVALID_FORK_ID = 19, // Main state machine executor does not support the requested fork ID
    ZKR_SM_MAIN_INVALID_UNSIGNED_TX = 20, // Main state machine executor cannot process unsigned TXs in prover mode
    ZKR_SM_MAIN_BALANCE_MISMATCH = 21, // Main state machine executor found that total tranferred balances are not zero
    ZKR_SM_MAIN_FEA2SCALAR = 22, // Main state machine executor failed calling fea2scalar()
    ZKR_SM_MAIN_TOS32 = 23, // Main state machine executor failed calling fr.toS32()
    ZKR_SM_MAIN_S33 = 24, // Main state machine executor failed getting an S33 value from op
    ZKR_STATE_MANAGER = 25, // State root error
    ZKR_SM_MAIN_INVALID_NO_COUNTERS = 26, // No counters received outside of a process batch request
    ZKR_SM_MAIN_ARITH_ECRECOVER_DIVIDE_BY_ZERO = 27, // Main state machine arith operation during ECRecover found a divide by zero situation
    ZKR_SM_MAIN_ADDRESS_OUT_OF_RANGE = 28, // Main state machine address out of valid memory space range
    ZKR_SM_MAIN_ADDRESS_NEGATIVE = 29, // Main state machine address is negative
    ZKR_SM_MAIN_MEMALIGN_OFFSET_OUT_OF_RANGE = 30, // Main state machine mem align offset is out of range
    ZKR_SM_MAIN_MULTIPLE_FREEIN = 31, // Main state machine got multiple free inputs in one sigle ROM instruction
    ZKR_SM_MAIN_STORAGE_READ_MISMATCH = 32, // Main state machine read ROM operation check failed
    ZKR_SM_MAIN_STORAGE_WRITE_MISMATCH = 33, // Main state machine write ROM operation check failed
    ZKR_SM_MAIN_ARITH_MISMATCH = 34, // Main state machine arithmetic ROM operation check failed
    ZKR_SM_MAIN_ARITH_ECRECOVER_MISMATCH = 35, // Main state machine arithmetic ECRecover ROM operation check failed
    ZKR_SM_MAIN_BINARY_ADD_MISMATCH = 36, // Main state machine binary add ROM operation check failed
    ZKR_SM_MAIN_BINARY_SUB_MISMATCH = 37, // Main state machine binary sub ROM operation check failed
    ZKR_SM_MAIN_BINARY_LT_MISMATCH = 38, // Main state machine binary less than ROM operation check failed
    ZKR_SM_MAIN_BINARY_SLT_MISMATCH = 39, // Main state machine binary signed less than ROM operation check failed
    ZKR_SM_MAIN_BINARY_EQ_MISMATCH = 40, // Main state machine binary equal ROM operation check failed
    ZKR_SM_MAIN_BINARY_AND_MISMATCH = 41, // Main state machine binary and ROM operation check failed
    ZKR_SM_MAIN_BINARY_OR_MISMATCH = 42, // Main state machine binary or ROM operation check failed
    ZKR_SM_MAIN_BINARY_XOR_MISMATCH = 43, // Main state machine binary XOR ROM operation check failed
    ZKR_SM_MAIN_BINARY_LT4_MISMATCH = 44, // Main state machine binary less than 4 ROM operation check failed
    ZKR_SM_MAIN_MEMALIGN_WRITE_MISMATCH = 45, // Main state machine memory align write ROM operation check failed
    ZKR_SM_MAIN_MEMALIGN_WRITE8_MISMATCH = 46, // Main state machine memory align write 8 ROM operation check failed
    ZKR_SM_MAIN_MEMALIGN_READ_MISMATCH = 47, // Main state machine memory align read ROM operation check failed
    ZKR_DB_VERSION_NOT_FOUND_KVDB = 48, // Version not found in KeyValue database
    ZKR_DB_VERSION_NOT_FOUND_GLOBAL = 49, // Version not found in KeyValue database and not present in hashDB neither

    ZKR_SM_MAIN_HASHK = 50, // Main state machine executor hash Keccak condition failed
    ZKR_SM_MAIN_HASHK_SIZE_OUT_OF_RANGE = 51, // Main state machine Keccak hash size is out of range
    ZKR_SM_MAIN_HASHK_POSITION_NEGATIVE = 52, // Main state machine Keccak hash position is negative
    ZKR_SM_MAIN_HASHK_POSITION_PLUS_SIZE_OUT_OF_RANGE = 53, // Main state Keccak hash position + size is out of range
    ZKR_SM_MAIN_HASHKDIGEST_ADDRESS_NOT_FOUND = 54, // Main state machine Keccak hash digest address is not found
    ZKR_SM_MAIN_HASHKDIGEST_NOT_COMPLETED = 55, // Main state machine Keccak hash digest called when hash has not been completed
    ZKR_SM_MAIN_HASHK_VALUE_MISMATCH = 56, // Main state machine Keccak hash ROM operation value check failed
    ZKR_SM_MAIN_HASHK_PADDING_MISMATCH = 57, // Main state machine Keccak hash ROM operation padding check failed
    ZKR_SM_MAIN_HASHK_SIZE_MISMATCH = 58, // Main state machine Keccak hash ROM operation size check failed
    ZKR_SM_MAIN_HASHKLEN_LENGTH_MISMATCH = 59, // Main state machine Keccak hash ROM operation data length check failed
    ZKR_SM_MAIN_HASHKLEN_CALLED_TWICE = 60, // Main state machine Keccak hash ROM operation called once check failed
    ZKR_SM_MAIN_HASHKDIGEST_NOT_FOUND = 61, // Main state machine Keccak hash digest ROM operation slot not found
    ZKR_SM_MAIN_HASHKDIGEST_DIGEST_MISMATCH = 62, // Main state machine Keccak hash digest ROM operation digest mismatch
    ZKR_SM_MAIN_HASHKDIGEST_CALLED_TWICE = 63, // Main state machine Keccak hash digest ROM operation called twice
    ZKR_SM_MAIN_HASHK_READ_OUT_OF_RANGE = 64, // Main state machine Keccak hash check found read out of range
    
    ZKR_SM_MAIN_HASHP = 65, // Main state machine executor hash poseidon condition failed
    ZKR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE = 66, // Main state machine Poseidon hash size is out of range
    ZKR_SM_MAIN_HASHP_POSITION_NEGATIVE = 67, // Main state machine Poseidon hash position is negative
    ZKR_SM_MAIN_HASHP_POSITION_PLUS_SIZE_OUT_OF_RANGE = 68, // Main state machine Poseidon hash position + size is out of range
    ZKR_SM_MAIN_HASHPDIGEST_ADDRESS_NOT_FOUND = 69, // Main state machine Poseidon hash digest address is not found
    ZKR_SM_MAIN_HASHPDIGEST_NOT_COMPLETED = 70, // Main state machine Poseidon hash digest called when hash has not been completed
    ZKR_SM_MAIN_HASHP_VALUE_MISMATCH = 71, // Main state machine Poseidon hash ROM operation value check failed
    ZKR_SM_MAIN_HASHP_PADDING_MISMATCH = 72, // Main state machine Poseidon hash ROM operation padding check failed
    ZKR_SM_MAIN_HASHP_SIZE_MISMATCH = 73, // Main state machine Poseidon hash ROM operation size check failed
    ZKR_SM_MAIN_HASHPLEN_LENGTH_MISMATCH = 74, // Main state machine Poseidon hash ROM operation data length check failed
    ZKR_SM_MAIN_HASHPLEN_CALLED_TWICE = 75, // Main state machine Poseidon hash ROM operation called once check failed
    ZKR_SM_MAIN_HASHPDIGEST_DIGEST_MISMATCH = 76, // Main state machine Poseidon hash digest ROM operation digest mismatch
    ZKR_SM_MAIN_HASHPDIGEST_CALLED_TWICE = 77, // Main state machine Poseidon hash digest ROM operation called twice
    ZKR_SM_MAIN_HASHP_READ_OUT_OF_RANGE = 78, // Main state machine Poseidon hash check found read out of range
    
    ZKR_SM_MAIN_HASHS = 79, // Main state machine executor hash SHA-256 condition failed
    ZKR_SM_MAIN_HASHS_SIZE_OUT_OF_RANGE = 80, // Main state machine SHA-256 hash size is out of range
    ZKR_SM_MAIN_HASHS_POSITION_NEGATIVE = 81, // Main state machine SHA-256 hash position is negative
    ZKR_SM_MAIN_HASHS_POSITION_PLUS_SIZE_OUT_OF_RANGE = 82, // Main state machine SHA-256 hash position + size is out of range
    ZKR_SM_MAIN_HASHSDIGEST_ADDRESS_NOT_FOUND = 83, // Main state machine SHA-256 hash digest address is not found
    ZKR_SM_MAIN_HASHSDIGEST_NOT_COMPLETED = 84, // Main state machine SHA-256 hash digest called when hash has not been completed
    ZKR_SM_MAIN_HASHS_VALUE_MISMATCH = 85, // Main state machine SHA-256 hash ROM operation value check failed
    ZKR_SM_MAIN_HASHS_PADDING_MISMATCH = 86, // Main state machine SHA-256 hash ROM operation padding check failed
    ZKR_SM_MAIN_HASHS_SIZE_MISMATCH = 87, // Main state machine SHA-256 hash ROM operation size check failed
    ZKR_SM_MAIN_HASHSLEN_LENGTH_MISMATCH = 88, // Main state machine SHA-256 hash ROM operation data length check failed
    ZKR_SM_MAIN_HASHSLEN_CALLED_TWICE = 89, // Main state machine SHA-256 hash ROM operation called once check failed
    ZKR_SM_MAIN_HASHSDIGEST_NOT_FOUND = 90, // Main state machine SHA-256 hash digest ROM operation slot not found
    ZKR_SM_MAIN_HASHSDIGEST_DIGEST_MISMATCH = 91, // Main state machine SHA-256 hash digest ROM operation digest mismatch
    ZKR_SM_MAIN_HASHSDIGEST_CALLED_TWICE = 92, // Main state machine SHA-256 hash digest ROM operation called twice
    ZKR_SM_MAIN_HASHS_READ_OUT_OF_RANGE = 93, // Main state machine SHA-256 hash check found read out of range

    ZKR_SM_MAIN_INVALID_L1_INFO_TREE_INDEX = 94, // Main state machine ROM requests a L1InfoTree not present in input
    ZKR_SM_MAIN_INVALID_L1_INFO_TREE_SMT_PROOF_VALUE = 95, // Main state machine ROM requests a L1InfoTree SMT proof value not present in input
    ZKR_SM_MAIN_INVALID_WITNESS = 96, // Main state machine input witness is invalid or corrupt
    ZKR_CBOR_INVALID_DATA = 97, // CBOR data is invalid
    ZKR_DATA_STREAM_INVALID_DATA = 98, // Data stream data is invalid
    
    ZKR_SM_MAIN_INVALID_TX_STATUS_ERROR = 99, // Invalid TX status-error combination

    ZKR_SM_MAIN_POINT_Z_MISMATCH = 100, // Point Z was calculated and different from the one provided as an input via JSON
    ZKR_SM_MAIN_BLOB_L2_HASH_DATA_MISMATCH = 101, // Blob L1 data hash was calculated and different from the one provided as an input via JSON
    ZKR_SM_MAIN_BATCH_HASH_DATA_MISMATCH = 102, // Batch L2 data hash was calculated and different from the one provided as an input via JSON
    ZKR_SM_MAIN_INVALID_BLOB_TYPE = 103, // Blob type is invalid
    ZKR_SM_MAIN_UNRESTORED_SAVED_CONTEXT = 104, // At least one saved context was not restored after execution was completed

    ZKR_BLOB_OUTER_PROOF_INVALID_INPUT = 105, // Invalid Blob outer proof input
    ZKR_SM_MAIN_INVALID_MEMORY_CTX = 106, // Invalid memory address context
} zkresult;
string zkresult2string(zkresult zkr) { return to_string(zkr); };
void removeKeyBits (Goldilocks &fr, const Goldilocks::Element (&key)[4], uint64_t nBits, Goldilocks::Element (&rkey)[4]);
class zkLog
{
public:
    void error (const string &s) { cerr << "ERROR: " << s << endl; };
    void info (const string &s) { cout << "INFO: " << s << endl; };
};
zkLog zklog;
zkresult witness2db (const string &witness, MTMap &db, ProgramMap &programs, mpz_class &stateRoot);

/********/
/* MAIN */
/********/

int main (int argc, char **argv)
{
    cout << "witness2db" << endl;

    // Check arguments list
    if (argc != 3)
    {
        cerr << "Error: expected 2 arguments but got " << argc - 1 << " Usage: witness2db <input.json> <output.json>" << endl;
        return -1;
    }

    // Get file names
    string inputFileName = argv[1];
    string outputFileName = argv[2];

    cout << "Converting witness from file " << inputFileName <<  " to db to file " << outputFileName << endl;

    json j;
    file2json(inputFileName, j);

    if (!j.contains("witness"))
    {
        cerr << "Input JSON file does not contain an witness field" << endl;
        exit(-1);
    }
    if (!j["witness"].is_string())
    {
        cerr << "Input JSON file contains a witness field, but it is not a string" << endl;
        exit(-1);
    }
    string witnessString = j["witness"];
    string witness;
    string2ba(witnessString, witness);

    // Parse the witness data
    MTMap db;
    ProgramMap contractsBytecode;
    mpz_class oldStateRoot;
    zkresult zkr = witness2db(witness, db, contractsBytecode, oldStateRoot);
    if (zkr != ZKR_SUCCESS)
    {
        cerr << "Failed calling witness2db()" << endl;
        exit(-1);
    }

    // Store the db element
    if (j.contains("db"))
    {
        cerr << "ERROR: input file already contains a \"db\" element" << endl;
        exit(-1);
    }
    db2json(j, db, "db");

    // Store the contractsBytecode element
    if (j.contains("contractsBytecode"))
    {
        cerr << "ERROR: input file already contains a \"contractsBytecode\" element" << endl;
        exit(-1);
    }
    contractsBytecode2json(j, contractsBytecode, "contractsBytecode");

    // Store the oldStateRoot element
    if (j.contains("oldStateRoot"))
    {
        cerr << "ERROR: input file already contains a \"oldStateRoot\" element" << endl;
        exit(-1);
    }
    j["oldStateRoot"] = NormalizeTo0xNFormat(oldStateRoot.get_str(16), 64);

    json2file(j, outputFileName);

    cout << "Success" << endl;

    return 0;
}

/**********************************************************/
/* Definition of replicas of zkEVM required functionality */
/**********************************************************/

void json2file (const json &j, const string &fileName)
{
    ofstream outputStream(fileName);
    if (!outputStream.good())
    {
        cerr << "Failed creating output JSON file " << fileName << endl;
        exit(-1);
    }
    outputStream << setw(4) << j << endl;
    outputStream.close();
}

void file2json (const string &fileName, json &j)
{
    j.clear();
    cout << "Loading JSON file " << fileName << endl;
    std::ifstream inputStream(fileName);
    if (!inputStream.good())
    {
        cerr << "Failed loading input JSON file " << fileName << "; does this file exist?" << endl;
        exit(-1);
    }
    try
    {
        inputStream >> j;
    }
    catch (exception &e)
    {
        cerr << "file2json() failed parsing input JSON file " << fileName << " exception=" << e.what() << endl;
        exit(-1);
    }
    inputStream.close();
}

void db2json (json &input, const MTMap &db, string name)
{
    input[name] = json::object();
    for(MTMap::const_iterator iter = db.begin(); iter != db.end(); iter++)
    {
        string key = NormalizeTo0xNFormat(iter->first, 64);
        vector<Goldilocks::Element> dbValue = iter->second;
        json value;
        for (uint64_t i=0; i<dbValue.size(); i++)
        {
            value[i] = PrependZeros(fr.toString(dbValue[i], 16), 16);
        }
        input[name][key] = value;
    }
}

void contractsBytecode2json (json &input, const ProgramMap &contractsBytecode, string name)
{
    input[name] = json::object();
    for(ProgramMap::const_iterator iter = contractsBytecode.begin(); iter != contractsBytecode.end(); iter++)
    {
        string key = NormalizeTo0xNFormat(iter->first, 64);
        vector<uint8_t> dbValue = iter->second;
        string value = "";
        for (uint64_t i=0; i<dbValue.size(); i++)
        {
            value += byte2string(dbValue[i]);
        }
        input[name][key] = "0x" + value;
    }
}

string NormalizeToNFormat (const string &s, uint64_t n)
{
    return PrependZeros(Remove0xIfPresent(s), n);
}

string NormalizeTo0xNFormat (const string &s, uint64_t n)
{
    return "0x" + NormalizeToNFormat(s, n);
}

// A set of strings with zeros is available in memory for performance reasons
string sZeros[65] = {
    "",
    "0",
    "00",
    "000",
    "0000",
    "00000",
    "000000",
    "0000000",
    "00000000",
    "000000000",
    "0000000000",
    "00000000000",
    "000000000000",
    "0000000000000",
    "00000000000000",
    "000000000000000",
    "0000000000000000",
    "00000000000000000",
    "000000000000000000",
    "0000000000000000000",
    "00000000000000000000",
    "000000000000000000000",
    "0000000000000000000000",
    "00000000000000000000000",
    "000000000000000000000000",
    "0000000000000000000000000",
    "00000000000000000000000000",
    "000000000000000000000000000",
    "0000000000000000000000000000",
    "00000000000000000000000000000",
    "000000000000000000000000000000",
    "0000000000000000000000000000000",
    "00000000000000000000000000000000",
    "000000000000000000000000000000000",
    "0000000000000000000000000000000000",
    "00000000000000000000000000000000000",
    "000000000000000000000000000000000000",
    "0000000000000000000000000000000000000",
    "00000000000000000000000000000000000000",
    "000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000000"
};

string PrependZeros (const string &s, uint64_t n)
{
    // Check that n is not too big
    if (n > 64)
    {
        zklog.error("PrependZeros() called with an that is too big n=" + to_string(n));
        exitProcess();
    }
    // Check that string size is not too big
    uint64_t stringSize = s.size();
    if ( (stringSize > n) || (stringSize > 64) )
    {
        zklog.error("PrependZeros() called with a string with too large s.size=" + to_string(stringSize) + " n=" + to_string(n));
        exitProcess();
    }

    // Prepend zeros if needed
    if (stringSize < n) return sZeros[n-stringSize] + s;

    return s;
}

string Remove0xIfPresent(const string &s)
{
    if ( (s.size() >= 2) && (s.at(1) == 'x') && (s.at(0) == '0') ) return s.substr(2);
    return s;
}

char byte2char (uint8_t b)
{
    if (b < 10) return '0' + b;
    if (b < 16) return 'a' + b - 10;
    zklog.error("byte2char() called with an invalid byte: " + to_string(b));
    exitProcess();
    return 0;
}

string byte2string(uint8_t b)
{
    string result;
    result.push_back(byte2char(b >> 4));
    result.push_back(byte2char(b & 0x0F));
    return result;
}
mpz_class ScalarMask64  ("FFFFFFFFFFFFFFFF", 16);
mpz_class ScalarGoldilocksPrime = (uint64_t)GOLDILOCKS_PRIME;

inline void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element (&fea)[4])
{
    mpz_class aux = scalar & ScalarMask64;

    if (aux >= ScalarGoldilocksPrime)
    {
        zklog.error("scalar2fea() found value higher than prime: " + aux.get_str(16));
        exitProcess();
    }

    fea[0] = fr.fromU64(aux.get_ui());
    aux = scalar>>64 & ScalarMask64;

    if (aux >= ScalarGoldilocksPrime)
    {
        zklog.error("scalar2fea() found value higher than prime: " + aux.get_str(16));
        exitProcess();
    }

    fea[1] = fr.fromU64(aux.get_ui());
    aux = scalar>>128 & ScalarMask64;

    if (aux >= ScalarGoldilocksPrime)
    {
        zklog.error("scalar2fea() found value higher than prime: " + aux.get_str(16));
        exitProcess();
    }

    fea[2] = fr.fromU64(aux.get_ui());
    aux = scalar>>192 & ScalarMask64;

    if (aux >= ScalarGoldilocksPrime)
    {
        zklog.error("scalar2fea() found value higher than prime: " + aux.get_str(16));
        exitProcess();
    }

    fea[3] = fr.fromU64(aux.get_ui());
}

class WitnessContext
{
public:
    const string &witness;
    uint64_t p; // pointer to the first witness byte pending to be parsed
    uint64_t level; // SMT level, being level=0 the root, level>0 higher levels
    MTMap &db; // database to store all the hash-value
    ProgramMap &programs; // database to store all the programs (smart contracts)
#ifdef WITNESS_CHECK_BITS
    vector<uint8_t> bits; // key bits consumed while climbing the tree; used only for debugging
#endif
#ifdef WITNESS_CHECK_SMT
    Goldilocks::Element root[4]; // the root of the witness data SMT tree; used only for debugging
#endif
    WitnessContext(const string &witness, MTMap &db, ProgramMap &programs) : witness(witness), p(0), level(0), db(db), programs(programs)
    {
#ifdef WITNESS_CHECK_SMT
        root[0] = fr.zero();
        root[1] = fr.zero();
        root[2] = fr.zero();
        root[3] = fr.zero();
#endif
    }

};

zkresult calculateWitnessHash (WitnessContext &ctx, Goldilocks::Element (&hash)[4]);

class CborResult
{
public:
    enum ResultType
    {
        UNDEFINED = 0,
        U64 = 1,
        BA = 2,
        TEXT = 3,
        ARRAY = 4,
        TAG = 6
    };
    zkresult result;
    ResultType type;
    uint64_t u64;
    string ba;
    string text;
    vector<CborResult> array;
    uint64_t tagCount;
    vector<CborResult> tag;
    CborResult() : result(ZKR_UNSPECIFIED), type(UNDEFINED), u64(0), tagCount(0) {};
};

string cborType2string (CborResult::ResultType type);

// This function parses CBOR field and stores it in a CborResult
void cbor2result (const string &s, uint64_t &p, CborResult &result);

// This CBOR function expects a simple integer < 24; otherwise it fails
zkresult cbor2u64 (const string &s, uint64_t &p, uint64_t &value);

// This CBOR function expects a byte array; otherwise it fails
zkresult cbor2ba (const string &s, uint64_t &p, string &value);

// This CBOR function expects a text string; otherwise it fails
zkresult cbor2text (const string &s, uint64_t &p, string &value);

// This function expects an integer, which can be long, and returns a scalar
zkresult cbor2scalar (const string &s, uint64_t &p, mpz_class &value);

zkresult witness2db (const string &witness, MTMap &db, ProgramMap &programs, mpz_class &stateRoot)
{
    db.clear();
    programs.clear();
    
    zkresult zkr;

    // Check witness is not empty
    if (witness.empty())
    {
        zklog.error("witness2db() got an empty witness");
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }

    // Create witness context
    WitnessContext ctx(witness, db, programs);

    // Parse header version
    uint8_t headerVersion = ctx.witness[ctx.p];
    if (headerVersion != 1)
    {
        zklog.error("witness2db() expected headerVersion=1 but got value=" + to_string(headerVersion));
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }
    ctx.p++;

    // Calculate witness hash    
    Goldilocks::Element hash[4];
    zkr = calculateWitnessHash(ctx, hash);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("witness2db() failed calling calculateWitnessHash() result=" + zkresult2string(zkr));
        return zkr;
    }

    // Convert state root
    fea2scalar(fr, stateRoot, hash);

    zklog.info("witness2db() calculated stateRoot=" + stateRoot.get_str(16) + " from size=" + to_string(witness.size()));

#ifdef WITNESS_CHECK_SMT
    zklog.info("witness2db() calculated SMT root=" + fea2string(fr, ctx.root));
#endif

    return ZKR_SUCCESS;
}

zkresult calculateWitnessHash (WitnessContext &ctx, Goldilocks::Element (&hash)[4])
{
    zkresult zkr;

    // Check level range
    if (ctx.level > 255)
    {
        zklog.error("calculateWitnessHash() reached an invalid level=" + to_string(ctx.level));
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }  

#ifdef WITNESS_CHECK_BITS
    // Check level-bits consistency
    if (ctx.level != ctx.bits.size())
    {
        zklog.error("calculateWitnessHash() got level=" + to_string(ctx.level) + "different from bits.size()=" + to_string(ctx.bits.size()));
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }
#endif

    // Opcode counters to control that we parse CODE at most once, and if so parse another opcode
    uint64_t numberOfOpcodes = 0;
    uint64_t numberOfCodeOpcodes = 0;

    do // while (numberOfOpcodes==1 && numberOfCodeOpcodes==1), i.e. repeat to parse SMT_LEAF after CODE
    {
        // Get instruction opcode from witness
        if (ctx.p >= ctx.witness.size())
        {
            zklog.error("calculateWitnessHash() run out of witness data");
            return ZKR_SM_MAIN_INVALID_WITNESS;
        }
        uint8_t opcode = ctx.witness[ctx.p];
        ctx.p++;

        switch (opcode)
        {
            case 0x02: // BRANCH -> ( 0x02 CBOR(mask)... ); `mask` defines which children are present (e.g. `0000000000001011` means that children 0, 1 and 3 are present and the other ones are not)
            {
                // Get the mask
                uint64_t mask;
                zkr = cbor2u64(ctx.witness, ctx.p, mask);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("calculateWitnessHash() failed calling cbor2u64() result=" + zkresult2string(zkr));
                    return zkr;
                }
#ifdef LOG_WITNESS
                zklog.info("BRANCH level=" + to_string(ctx.level) + " mask=" + to_string(mask));
#endif

                // Get if there are children at the left and/or at the right, from the mask
                bool hasLeft;
                bool hasRight;
                switch (mask)
                {
                    case 1:
                    {
                        hasLeft = true;
                        hasRight = false;
                        break;
                    }
                    case 2:
                    {
                        hasLeft = false;
                        hasRight = true;
                        break;
                    }
                    case 3:
                    {
                        hasLeft = true;
                        hasRight = true;
                        break;
                    }
                    default:
                    {
                        zklog.error("calculateWitnessHash() found invalid mask=" + to_string(mask));
                        return ZKR_SM_MAIN_INVALID_WITNESS;
                    }
                }

                // Calculate the left hash
                Goldilocks::Element leftHash[4];
                if (hasLeft)
                {
#ifdef WITNESS_CHECK_BITS
                    ctx.bits.emplace_back(0);
#endif
                    ctx.level++;
                    zkr = calculateWitnessHash(ctx, leftHash);
                    ctx.level--;
#ifdef WITNESS_CHECK_BITS
                    ctx.bits.pop_back();
#endif
                    if (zkr != ZKR_SUCCESS)
                    {
                        return zkr;
                    }
                }
                else
                {
                    leftHash[0] = fr.zero();
                    leftHash[1] = fr.zero();
                    leftHash[2] = fr.zero();
                    leftHash[3] = fr.zero();
                }

                // Calculate the right hash
                Goldilocks::Element rightHash[4];
                if (hasRight)
                {
#ifdef WITNESS_CHECK_BITS
                    ctx.bits.emplace_back(1);
#endif
                    ctx.level++;
                    zkr = calculateWitnessHash(ctx, rightHash);
                    ctx.level--;
#ifdef WITNESS_CHECK_BITS
                    ctx.bits.pop_back();
#endif
                    if (zkr != ZKR_SUCCESS)
                    {
                        return zkr;
                    }
                }
                else
                {
                    rightHash[0] = fr.zero();
                    rightHash[1] = fr.zero();
                    rightHash[2] = fr.zero();
                    rightHash[3] = fr.zero();
                }

                // Calculate this intermediate node hash = poseidonHash(leftHash, rightHash, 0000)

                // Prepare input = [leftHash, rightHash, 0000]
                Goldilocks::Element input[12];
                input[0] = leftHash[0];
                input[1] = leftHash[1];
                input[2] = leftHash[2];
                input[3] = leftHash[3];
                input[4] = rightHash[0];
                input[5] = rightHash[1];
                input[6] = rightHash[2];
                input[7] = rightHash[3];
                input[8] = fr.zero();
                input[9] = fr.zero();
                input[10] = fr.zero();
                input[11] = fr.zero();

                // Calculate the poseidon hash
                poseidon.hash(hash, input);

                // Store the hash-value pair into db
                vector<Goldilocks::Element> valueData;
                valueData.reserve(12);
                for (uint64_t i=0; i<12; i++)
                {
                    valueData.emplace_back(input[i]);
                }
                ctx.db[fea2string(fr, hash)] = valueData;

#ifdef LOG_WITNESS
                zklog.info("BANCH level=" + to_string(ctx.level) + " leftHash=" + fea2string(fr, leftHash) + " rightHash=" + fea2string(fr, rightHash) + " hash=" + fea2string(fr, hash));
#endif

                break;
            }
            case 0x07: // SMT_LEAF -> ( 0x07 nodeType CBOR(address) /CBOR(storageKey).../ CBOR(value)...)
                // * if `nodeType` == `0x03`, then an extra field `storageKey` is read; otherwise it is skipped
            {
                // Read nodeType
                // 0 = BALANCE
                // 1 = NONCE
                // 2 = SC CODE
                // 3 = SC STORAGE
                // 4 = SC LENGTH
                // 5, 6 = touched addresses
                // < 11 (0xb) = info block tree of Etrog
                if (ctx.p >= ctx.witness.size())
                {
                    zklog.error("calculateWitnessHash() unexpected end of witness");
                    return ZKR_SM_MAIN_INVALID_WITNESS;
                }
                uint8_t nodeType = ctx.witness[ctx.p];
                ctx.p++;
                //zklog.info("SMT_LEAF nodeType=" + to_string(nodeType));

                // Read address
                mpz_class address;
                zkr = cbor2scalar(ctx.witness, ctx.p, address);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("calculateWitnessHash() failed calling cbor2scalar(address) result=" + zkresult2string(zkr));
                    return zkr;
                }
                //zklog.info("SMT_LEAF address=" + address.get_str(16));

                // Read storage key
                mpz_class storageKey;
                if (nodeType == 0x03) // SC STORAGE: an extra field storageKey is read
                {
                    zkr = cbor2scalar(ctx.witness, ctx.p, storageKey);
                    if (zkr != ZKR_SUCCESS)
                    {
                        zklog.error("calculateWitnessHash() failed calling cbor2scalar(storageKey) result=" + zkresult2string(zkr));
                        return zkr;
                    }
                    //zklog.info("SMT_LEAF storageKey=" + storageKey.get_str(16));
                }

                // Read value
                mpz_class value;
                zkr = cbor2scalar(ctx.witness, ctx.p, value);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("calculateWitnessHash() failed calling cbor2scalar(value) result=" + zkresult2string(zkr));
                    return zkr;
                }
                //zklog.info("SMT_LEAF value=" + value.get_str(16));

                // Calculate poseidonHash(storageKey)
                // TODO: skip if storageKey==0, use pre-calculated poseidon hash of zero
                Goldilocks::Element Kin0[12];
                scalar2fea(fr, storageKey, Kin0[0], Kin0[1], Kin0[2], Kin0[3], Kin0[4], Kin0[5], Kin0[6], Kin0[7]);
                Kin0[8] = fr.zero();
                Kin0[9] = fr.zero();
                Kin0[10] = fr.zero();
                Kin0[11] = fr.zero();
                Goldilocks::Element Kin0Hash[4];
                poseidon.hash(Kin0Hash, Kin0);

                // Calculate the key = poseidonHash(account, type, poseidonHash(storageKey))
                Goldilocks::Element Kin1[12];
                scalar2fea(fr, address, Kin1[0], Kin1[1], Kin1[2], Kin1[3], Kin1[4], Kin1[5], Kin1[6], Kin1[7]);
                if (!fr.isZero(Kin1[5]) || !fr.isZero(Kin1[6]) || !fr.isZero(Kin1[7]))
                {
                    zklog.error("calculateWitnessHash() found non-zero address field elements 5, 6 or 7");
                    return ZKR_SM_MAIN_INVALID_WITNESS;
                }

                switch(nodeType)
                {
                    case 0: // BALANCE
                    {
                        break;
                    }
                    case 1: // NONCE
                    {
                        Kin1[6] = fr.one();
                        break;
                    }
                    case 2: // SC CODE
                    {
                        Kin1[6] = fr.fromU64(2);
                        break;
                    }
                    case 3: // SC STORAGE
                    {
                        Kin1[6] = fr.fromU64(3);
                        break;
                    }
                    case 4: // SC LENGTH
                    {
                        Kin1[6] = fr.fromU64(4);
                        break;
                    }
                    default:
                    {
                        zklog.error("calculateWitnessHash() found invalid nodeType=" + to_string(nodeType));
                        return ZKR_SM_MAIN_INVALID_WITNESS;
                    }
                }

                // Reinject the first resulting hash as the capacity for the next poseidon hash
                Kin1[8] = Kin0Hash[0];
                Kin1[9] = Kin0Hash[1];
                Kin1[10] = Kin0Hash[2];
                Kin1[11] = Kin0Hash[3];

                // Call poseidon hash
                Goldilocks::Element key[4];
                poseidon.hash(key, Kin1);

                // Calculate this leaf node hash = poseidonHash(remainingKey, valueHash, 1000),
                // where valueHash = poseidonHash(value, 0000)

#ifdef WITNESS_CHECK_SMT
                HashDBInterface * pHashDB = HashDBClientFactory::createHashDBClient(fr,config);
                pHashDB->set("", 0, 0, ctx.root, key, value, PERSISTENCE_TEMPORARY, ctx.root, NULL, NULL);
#endif

                // Prepare input = [value8, 0000]
                Goldilocks::Element input[12];
                scalar2fea(fr, value, input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]);
                input[8] = fr.zero();
                input[9] = fr.zero();
                input[10] = fr.zero();
                input[11] = fr.zero();

                // Calculate the value hash
                Goldilocks::Element valueHash[4];
                poseidon.hash(valueHash, input);

                // Store the hash-value pair into db
                vector<Goldilocks::Element> valueData;
                valueData.reserve(12);
                for (uint64_t i=0; i<12; i++)
                {
                    valueData.emplace_back(input[i]);
                }
                ctx.db[fea2string(fr, valueHash)] = valueData;

#ifdef WITNESS_CHECK_BITS
                // Check key
                bool keyBits[256];
                splitKey(fr, key, keyBits);
                for (uint64_t i=0; i<ctx.level; i++)
                {
                    if (keyBits[i] != ctx.bits[i])
                    {
                        zklog.error("calculateWitnessHash() found different keyBits[i]=" + to_string(keyBits[i]) + " bits[i]=" + to_string(ctx.bits[i]) + " i=" + to_string(i));
                        zklog.error("bits=");
                        for (uint64_t b=0; b<ctx.bits.size(); b++)
                        {
                            zklog.error(" b=" + to_string(b) + " keyBits=" + to_string(keyBits[b]) + " bits=" + to_string(ctx.bits[b]));
                        }

                        return ZKR_SM_MAIN_INVALID_WITNESS;
                    }
                }
#endif

                // Calculate the remaining key
                Goldilocks::Element rkey[4];
                removeKeyBits(fr, key, ctx.level, rkey);

                // Prepare input = [rkey, valueHash, 1000]
                input[0] = rkey[0];
                input[1] = rkey[1];
                input[2] = rkey[2];
                input[3] = rkey[3];
                input[4] = valueHash[0];
                input[5] = valueHash[1];
                input[6] = valueHash[2];
                input[7] = valueHash[3];
                input[8] = fr.one();
                input[9] = fr.zero();
                input[10] = fr.zero();
                input[11] = fr.zero();

                // Calculate the leaf node hash
                poseidon.hash(hash, input);

                // Store the hash-value pair into db
                for (uint64_t i=0; i<12; i++)
                {
                    valueData[i] = input[i];
                }
                ctx.db[fea2string(fr, hash)] = valueData;

#ifdef LOG_WITNESS
                zklog.info("LEAF level=" + to_string(ctx.level) + " address=" + address.get_str(16) + " type=" + to_string(nodeType) + " storageKey=" + storageKey.get_str(16) + " value=" + value.get_str(16) + " key=" + fea2string(fr, key) + " rkey=" + fea2string(fr, rkey) + " valueHash=" + fea2string(fr, valueHash) + " hash=" + fea2string(fr, hash));
#endif

                break;
            }
            case 0x03: // HASH -> ( 0x03 hash_byte_1 ... hash_byte_32 )
            {
                // Read node hash
                mpz_class hashScalar;
                if (ctx.p + 32 > ctx.witness.size())
                {
                    zklog.error("calculateWitnessHash() run out of witness data");
                    return ZKR_SM_MAIN_INVALID_WITNESS;
                }
                ba2scalar((const uint8_t *)ctx.witness.c_str() + ctx.p, 32, hashScalar);
                ctx.p += 32;

#ifdef LOG_WITNESS
                zklog.info("HASH hash=" + hashScalar.get_str(16));
#endif

                // Convert to field elements
                scalar2fea(fr, hashScalar, hash); // TODO: return error if hashScalar is invalid, instead of killing the process

                break;
            }
            case 0x04: // CODE -> ( 0x04 CBOR(code)... )
            {
                // Check we parse CODE once, at most
                if (numberOfCodeOpcodes >= 1)
                {
                    zklog.error("calculateWitnessHash() found 2 consecutive CODE opcodes");
                    return ZKR_SM_MAIN_INVALID_WITNESS;
                }

                string program;

                // Parse CBOR data
                zkr = cbor2ba(ctx.witness, ctx.p, program);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("calculateWitnessHash() failed calling cbor2ba(program) result=" + zkresult2string(zkr));
                    return zkr;
                }
                if (program.empty())
                {
                    zklog.error("calculateWitnessHash() called cbor2ba(program) and got an empty byte array");
                    return ZKR_SM_MAIN_INVALID_WITNESS;
                }

                // Convert to vector
                vector<uint8_t> programVector;
                programVector.reserve(program.size());
                for (uint64_t i=0; i<program.size(); i++)
                {
                    programVector.emplace_back(program[i]);
                }

                // Calculate hash
                Goldilocks::Element linearHash[4];
                poseidonLinearHash(programVector, linearHash);

                // Save into programs
                string linearHashString = fea2string(fr, linearHash);
                ctx.programs[linearHashString] = programVector;

#ifdef LOG_WITNESS
                zklog.info("CODE size=" + to_string(program.size()) + " hash=" + linearHashString + " code=" + ba2string(program));
#endif

                numberOfCodeOpcodes++;

                break;
            }
            case 0x00: // LEAF -> ( 0x00 CBOR(ENCODE_KEY(key))... CBOR(value)... )
            case 0x01: // EXTENSION -> ( 0x01 CBOR(ENCODE_KEY(key))... )
            case 0x05: // ACCOUNT_LEAF -> ( 0x05 CBOR(ENCODE_KEY(key))... flags /CBOR(nonce).../ /CBOR(balance).../ )
                // `flags` is a bitset encoded in a single byte (bit endian):
                // * bit 0 defines if **code** is present; if set to 1, then `has_code=true`;
                // * bit 1 defines if **storage** is present; if set to 1, then `has_storage=true`;
                // * bit 2 defines if **nonce** is not 0; if set to 0, *nonce* field is not encoded;
                // * bit 3 defines if **balance** is not 0; if set to 0, *balance* field is not encoded;
            case 0xBB: // NEW_TRIE -> ( 0xBB )
            default:
            {
                zklog.error("calculateWitnessHash() got unsupported opcode=" + to_string(opcode));
                return ZKR_SM_MAIN_INVALID_WITNESS;
            }
        }

        // Increment number of parsed opcodes
        numberOfOpcodes++;

    } while ((numberOfOpcodes == 1) && (numberOfCodeOpcodes == 1));

#ifdef LOG_WITNESS
    zklog.info("calculateWitnessHash() returns hash=" + fea2string(fr, hash));
#endif

    return ZKR_SUCCESS;
}


string cborType2string (CborResult::ResultType type)
{
    switch (type)
    {
        case CborResult::UNDEFINED: return "UNDEFINED";
        case CborResult::U64: return "U64";
        case CborResult::BA: return "BA";
        case CborResult::TEXT: return "TEXT";
        case CborResult::ARRAY: return "ARRAY";
        case CborResult::TAG: return "TAG";
        default:
        {
            return "<UNRECOGNIZED TYPE=" + to_string(type) + ">"; 
        }
    }
}

// This function parses CBOR field and stores it in a CborResult
void cbor2result (const string &s, uint64_t &p, CborResult &cborResult)
{
    if (p >= s.size())
    {
        zklog.error("cbor2result() found too high p");
        cborResult.result= ZKR_CBOR_INVALID_DATA;
    }
    uint8_t firstByte = s[p];
    p++;
    if (firstByte < 24)
    {
        cborResult.type = CborResult::U64;
        cborResult.u64 = firstByte;
        cborResult.result = ZKR_SUCCESS;
        return;
    }
    uint8_t majorType = firstByte >> 5;
    uint8_t shortCount = firstByte & 0x1F;

    uint64_t longCount = 0;
    if (shortCount <= 23)
    {
        longCount = shortCount;
    }
    else if (shortCount == 24)
    {
        if (p >= s.size())
        {
            zklog.error("cbor2result() run out of bytes");
            cborResult.result = ZKR_CBOR_INVALID_DATA;
            return;
        }
        uint8_t secondByte = s[p];
        p++;
        longCount = secondByte;
    }
    else if (shortCount == 25)
    {
        if (p + 1 >= s.size())
        {
            zklog.error("cbor2result() run out of bytes");
            cborResult.result = ZKR_CBOR_INVALID_DATA;
            return;
        }
        uint8_t secondByte = s[p];
        p++;
        uint8_t thirdByte = s[p];
        p++;
        longCount = (uint64_t(secondByte)<<8) + uint64_t(thirdByte);
    }
    else if (shortCount == 26)
    {
        if (p + 3 >= s.size())
        {
            zklog.error("cbor2result() run out of bytes");
            cborResult.result = ZKR_CBOR_INVALID_DATA;
            return;
        }
        uint8_t secondByte = s[p];
        p++;
        uint8_t thirdByte = s[p];
        p++;
        uint8_t fourthByte = s[p];
        p++;
        uint8_t fifthByte = s[p];
        p++;
        longCount = (uint64_t(secondByte)<<24) + (uint64_t(thirdByte)<<16) + (uint64_t(fourthByte)<<8) + uint64_t(fifthByte);
    }
    else if (shortCount == 27)
    {
        if (p + 7 >= s.size())
        {
            zklog.error("cbor2result() run out of bytes");
            cborResult.result = ZKR_CBOR_INVALID_DATA;
            return;
        }
        uint8_t secondByte = s[p];
        p++;
        uint8_t thirdByte = s[p];
        p++;
        uint8_t fourthByte = s[p];
        p++;
        uint8_t fifthByte = s[p];
        p++;
        uint8_t sixthByte = s[p];
        p++;
        uint8_t seventhByte = s[p];
        p++;
        uint8_t eighthByte = s[p];
        p++;
        uint8_t ninethByte = s[p];
        p++;
        longCount = (uint64_t(secondByte)<<56) + (uint64_t(thirdByte)<<48) + (uint64_t(fourthByte)<<40) + (uint64_t(fifthByte)<<32) + (uint64_t(sixthByte)<<24) + (uint64_t(seventhByte)<<16) + (uint64_t(eighthByte)<<8) + uint64_t(ninethByte);
    }

    switch (majorType)
    {
        // Assuming CBOR short field encoding
        // For types 0, 1, and 7, there is no payload; the count is the value
        case 0:
        case 1:
        case 7:
        {
            cborResult.type = CborResult::U64;
            cborResult.u64 = shortCount;
            cborResult.result = ZKR_SUCCESS;
            break;
        }

        // For types 2 (byte string) and 3 (text string), the count is the length of the payload
        case 2: // byte string
        {
            if ((p + longCount) > s.size())
            {
                zklog.error("cbor2result() not enough space left for longCount=" + to_string(longCount));
                cborResult.result = ZKR_CBOR_INVALID_DATA;
                return;
            }
            cborResult.ba = s.substr(p, longCount);
            p += longCount;
            cborResult.type = CborResult::BA;
            cborResult.result = ZKR_SUCCESS;
            break;
        }
        case 3: // text string
        {
            if ((p + longCount) > s.size())
            {
                zklog.error("cbor2result() not enough space left for longCount=" + to_string(longCount));
                cborResult.result = ZKR_CBOR_INVALID_DATA;
                return;
            }
            cborResult.text = s.substr(p, longCount);
            p += longCount;
            cborResult.type = CborResult::TEXT;
            cborResult.result = ZKR_SUCCESS;
            break;
        }

        // For types 4 (array) and 5 (map), the count is the number of items (pairs) in the payload
        case 4: // array
        {
            //zklog.info("cbor2result() starting array of " + to_string(longCount) + " elements");
            //zklog.info(" data=" + ba2string(s.substr(p-1)));
            for (uint64_t a=0; a<longCount; a++)
            {
                CborResult result;
                cbor2result(s, p, result);
                if (result.result != ZKR_SUCCESS)
                {
                    zklog.error("cbor2result() found an array and failed calling itself a=" + to_string(a) + " result=" + zkresult2string(result.result));
                    cborResult.result = result.result;
                    return;
                }
                cborResult.array.emplace_back(result);
            }
            cborResult.type = CborResult::ARRAY;
            cborResult.result = ZKR_SUCCESS;
            //zklog.info("cbor2result() ending array of " + to_string(longCount) + " elements");
            break;
        }
        case 5: // map
        {
            zklog.error("cbor2result() majorType=5 (map) not supported longCount=" + to_string(longCount));
            cborResult.result = ZKR_CBOR_INVALID_DATA;
            return;
        }

        // For type 6 (tag), the payload is a single item and the count is a numeric tag number which describes the enclosed item
        case 6: // tag
        {
            //zklog.info("cbor2result() majorType=6 (tag)");
            CborResult result;
            cbor2result(s, p, result);
            if (result.result != ZKR_SUCCESS)
            {
                zklog.error("cbor2result() TAG failed calling itself result=" + zkresult2string(result.result));
                cborResult.result = result.result;
                return;
            }
            cborResult.tagCount = longCount;
            cborResult.tag.emplace_back(result);
            cborResult.type = CborResult::TAG;
            cborResult.result = ZKR_SUCCESS;
            break;
        }
    }
    
    //zklog.info("cbor2result() got result=" + zkresult2string(cborResult.result) + " type=" + cborType2string(cborResult.type));
}

// This CBOR function expects a simple integer < 24; otherwise it fails
zkresult cbor2u64 (const string &s, uint64_t &p, uint64_t &value)
{
    CborResult cborResult;
    cbor2result(s, p, cborResult);
    if (cborResult.result != ZKR_SUCCESS)
    {
        zklog.error("cbor2u64() failed calling cbor2result() result=" + zkresult2string(cborResult.result));
        return cborResult.result;
    }
    switch (cborResult.type)
    {
        case CborResult::U64:
        {
            value = cborResult.u64;
            return ZKR_SUCCESS;
        }
        default:
        {
            zklog.error("cbor2u64() called cbor2result() and got invalid result type=" + to_string(cborResult.type));
            return ZKR_CBOR_INVALID_DATA;
        }
    }
}

// This CBOR function expects a byte array; otherwise it fails
zkresult cbor2ba (const string &s, uint64_t &p, string &value)
{
    CborResult cborResult;
    cbor2result(s, p, cborResult);
    if (cborResult.result != ZKR_SUCCESS)
    {
        zklog.error("cbor2ba() failed calling cbor2result() result=" + zkresult2string(cborResult.result));
        return cborResult.result;
    }
    switch (cborResult.type)
    {
        case CborResult::BA:
        {
            value = cborResult.ba;
            return ZKR_SUCCESS;
        }
        default:
        {
            zklog.error("cbor2ba() called cbor2result() and got invalid result type=" + to_string(cborResult.type));
            return ZKR_CBOR_INVALID_DATA;
        }
    }
}

// This CBOR function expects a text string; otherwise it fails
zkresult cbor2text (const string &s, uint64_t &p, string &value)
{
    CborResult cborResult;
    cbor2result(s, p, cborResult);
    if (cborResult.result != ZKR_SUCCESS)
    {
        zklog.error("cbor2text() failed calling cbor2result() result=" + zkresult2string(cborResult.result));
        return cborResult.result;
    }
    switch (cborResult.type)
    {
        case CborResult::TEXT:
        {
            value = cborResult.text;
            return ZKR_SUCCESS;
        }
        default:
        {
            zklog.error("cbor2text() called cbor2result() and got invalid result type=" + to_string(cborResult.type));
            return ZKR_CBOR_INVALID_DATA;
        }
    }
}

// This function expects an integer, which can be long, and returns a scalar
zkresult cbor2scalar (const string &s, uint64_t &p, mpz_class &value)
{
    CborResult cborResult;
    cbor2result(s, p, cborResult);
    if (cborResult.result != ZKR_SUCCESS)
    {
        zklog.error("cbor2scalar() failed calling cbor2result() result=" + zkresult2string(cborResult.result));
        return cborResult.result;
    }
    switch (cborResult.type)
    {
        case CborResult::U64:
        {
            value = cborResult.u64;
            return ZKR_SUCCESS;
        }
        case CborResult::BA:
        {
            if (cborResult.ba.size() > 32)
            {
                zklog.error("cbor2scalar() got size too long size=" + to_string(cborResult.ba.size()));
                return ZKR_CBOR_INVALID_DATA;
            }
            ba2scalar(value, cborResult.ba);
            return ZKR_SUCCESS;
        }
        default:
        {
            zklog.error("cbor2scalar() called cbor2result() and got invalid result type=" + to_string(cborResult.type));
            return ZKR_CBOR_INVALID_DATA;
        }
    }
}

string fea2string (Goldilocks &fr, const Goldilocks::Element(&fea)[4])
{
    mpz_class auxScalar;
    fea2scalar(fr, auxScalar, fea);
    string s = auxScalar.get_str(16);
    s = PrependZeros(s, 64);
    return s;
}

inline void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element &fe0, Goldilocks::Element &fe1, Goldilocks::Element &fe2, Goldilocks::Element &fe3, Goldilocks::Element &fe4, Goldilocks::Element &fe5, Goldilocks::Element &fe6, Goldilocks::Element &fe7)
{
    mpz_class aux;
    aux = scalar & ScalarMask32;
    fe0 = fr.fromU64(aux.get_ui());
    aux = scalar>>32 & ScalarMask32;
    fe1 = fr.fromU64(aux.get_ui());
    aux = scalar>>64 & ScalarMask32;
    fe2 = fr.fromU64(aux.get_ui());
    aux = scalar>>96 & ScalarMask32;
    fe3 = fr.fromU64(aux.get_ui());
    aux = scalar>>128 & ScalarMask32;
    fe4 = fr.fromU64(aux.get_ui());
    aux = scalar>>160 & ScalarMask32;
    fe5 = fr.fromU64(aux.get_ui());
    aux = scalar>>192 & ScalarMask32;
    fe6 = fr.fromU64(aux.get_ui());
    aux = scalar>>224 & ScalarMask32;
    fe7 = fr.fromU64(aux.get_ui());
}

void removeKeyBits (Goldilocks &fr, const Goldilocks::Element (&key)[4], uint64_t nBits, Goldilocks::Element (&rkey)[4])
{
    uint64_t fullLevels = nBits / 4;
    uint64_t auxk[4];

    for (uint64_t i=0; i<4; i++)
    {
        auxk[i] = fr.toU64(key[i]);
    }

    for (uint64_t i = 0; i < 4; i++)
    {
        uint64_t n = fullLevels;
        if (fullLevels * 4 + i < nBits) n += 1;
        auxk[i] = auxk[i] >> n;
    }

    for (uint64_t i=0; i<4; i++)
    {
        rkey[i] = fr.fromU64(auxk[i]);
    }
}

void poseidonLinearHash (const vector<uint8_t> &_data, Goldilocks::Element (&result)[4])
{
    // Get a local copy of the bytes vector
    vector<uint8_t> data = _data;

    // Add padding = 0b1000...00001  up to a length of 56xN (7x8xN)
    data.push_back(0x01);
    while((data.size() % 56) != 0) data.push_back(0);
    data[data.size()-1] |= 0x80;

    // Create a FE buffer to store the transformed bytes into fe
    uint64_t bufferSize = data.size()/7;
    Goldilocks::Element * pBuffer = new Goldilocks::Element[bufferSize];
    if (pBuffer == NULL)
    {
        zklog.error("poseidonLinearHash() failed allocating memory of " + to_string(bufferSize) + " field elements");
        exitProcess();
    }

    // Init to zero
    for (uint64_t j=0; j<bufferSize; j++) pBuffer[j] = fr.zero();

    // Copy the bytes into the fe lower 7 sections
    for (uint64_t j=0; j<data.size(); j++)
    {
        uint64_t fePos = j/7;
        uint64_t shifted = uint64_t(data[j]) << ((j%7)*8);
        pBuffer[fePos] = fr.add(pBuffer[fePos], fr.fromU64(shifted));
    }

    // Call poseidon linear hash
    poseidon.linear_hash(result, pBuffer, bufferSize);

    // Free allocated memory
    delete[] pBuffer;
}

uint8_t char2byte (char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    zklog.error("char2byte() called with an invalid, non-hex char: " + to_string(c));
    exitProcess();
    return 0;
}

void string2ba (const string &textString, string &baString)
{
    baString.clear();

    string s = Remove0xIfPresent(textString);

    if (s.size()%2 != 0)
    {
        s = "0" + s;
    }

    uint64_t dsize = s.size()/2;

    const char *p = s.c_str();
    for (uint64_t i=0; i<dsize; i++)
    {
        uint8_t aux = char2byte(p[2*i])*16 + char2byte(p[2*i + 1]);
        baString.push_back(aux);
    }
}