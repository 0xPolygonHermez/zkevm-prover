#ifndef ROM_CONSTANTS_HPP_fork_6
#define ROM_CONSTANTS_HPP_fork_6

#include <unistd.h>
#include <gmpxx.h>

namespace fork_6
{

class RomConstants
{
public:
    /* Integer constants */
    uint64_t BATCH_DIFFICULTY;
    uint64_t TX_GAS_LIMIT;
    uint64_t GLOBAL_EXIT_ROOT_STORAGE_POS;
    uint64_t LOCAL_EXIT_ROOT_STORAGE_POS;
    uint64_t LAST_TX_STORAGE_POS;
    uint64_t STATE_ROOT_STORAGE_POS;
    uint64_t MAX_MEM_EXPANSION_BYTES;
    uint64_t FORK_ID;
    uint64_t MIN_VALUE_SHORT;
    uint64_t MIN_BYTES_LONG;
    uint64_t SMT_KEY_BALANCE;
    uint64_t SMT_KEY_NONCE;
    uint64_t SMT_KEY_SC_CODE;
    uint64_t SMT_KEY_SC_STORAGE;
    uint64_t SMT_KEY_SC_LENGTH;
    uint64_t SMT_KEY_TOUCHED_ADDR;
    uint64_t SMT_KEY_TOUCHED_SLOTS;
    uint64_t BASE_TX_GAS;
    uint64_t BASE_TX_DEPLOY_GAS;
    uint64_t SLOAD_GAS;
    uint64_t GAS_QUICK_STEP;
    uint64_t GAS_FASTEST_STEP;
    uint64_t GAS_FAST_STEP;
    uint64_t GAS_MID_STEP;
    uint64_t GAS_SLOW_STEP;
    uint64_t GAS_EXT_STEP;
    uint64_t CALL_VALUE_TRANSFER_GAS;
    uint64_t CALL_NEW_ACCOUNT_GAS;
    uint64_t CALL_STIPEND;
    uint64_t ECRECOVER_GAS;
    uint64_t IDENTITY_GAS;
    uint64_t IDENTITY_WORD_GAS;
    uint64_t KECCAK_GAS;
    uint64_t KECCAK_WORD_GAS;
    uint64_t LOG_GAS;
    uint64_t LOG_TOPIC_GAS;
    uint64_t JUMP_DEST_GAS;
    uint64_t WARM_STORGE_READ_GAS;
    uint64_t COLD_ACCOUNT_ACCESS_COST_REDUCED;
    uint64_t COLD_ACCOUNT_ACCESS_COST;
    uint64_t EXP_BYTE_GAS;
    uint64_t RETURN_GAS_COST;
    uint64_t CREATE_GAS;
    uint64_t CREATE_2_GAS;
    uint64_t SENDALL_GAS;
    uint64_t LOG_DATA_GAS;
    uint64_t SSTORE_ENTRY_EIP_2200_GAS;
    uint64_t SSTORE_SET_EIP_2200_GAS;
    uint64_t COLD_SLOAD_COST;
    uint64_t COLD_SLOAD_COST_REDUCED;
    uint64_t SSTORE_DYNAMIC_GAS;
    uint64_t SSTORE_SET_GAS;
    uint64_t SSTORE_SET_GAS_REDUCED;
    uint64_t SSTORE_RESET_GAS;
    uint64_t SSTORE_RESET_GAS_REDUCED;
    uint64_t SSTORE_CLEARS_SCHEDULE;
    uint64_t MIN_STEPS_FINISH_BATCH;
    uint64_t TOTAL_STEPS_LIMIT;
    uint64_t MAX_CNT_STEPS_LIMIT;
    uint64_t MAX_CNT_ARITH_LIMIT;
    uint64_t MAX_CNT_BINARY_LIMIT;
    uint64_t MAX_CNT_MEM_ALIGN_LIMIT;
    uint64_t MAX_CNT_KECCAK_F_LIMIT;
    uint64_t MAX_CNT_PADDING_PG_LIMIT;
    uint64_t MAX_CNT_POSEIDON_G_LIMIT;
    uint64_t SAFE_RANGE;
    uint64_t MAX_CNT_STEPS;
    uint64_t MAX_CNT_ARITH;
    uint64_t MAX_CNT_BINARY;
    uint64_t MAX_CNT_MEM_ALIGN;
    uint64_t MAX_CNT_KECCAK_F;
    uint64_t MAX_CNT_PADDING_PG;
    uint64_t MAX_CNT_POSEIDON_G;
    uint64_t MAX_CNT_POSEIDON_SLOAD_SSTORE;
    uint64_t MIN_CNT_KECCAK_BATCH;
    uint64_t CODE_SIZE_LIMIT;
    uint64_t BYTECODE_STARTS_EF;

    /* Scalar constants */
    mpz_class ADDRESS_GLOBAL_EXIT_ROOT_MANAGER_L2;
    mpz_class ADDRESS_SYSTEM;
    mpz_class MAX_NONCE;
    mpz_class MAX_UINT_256;
    mpz_class FPEC;
    mpz_class FPEC_MINUS_ONE;
    mpz_class FNEC_DIV_TWO;
    mpz_class FPEC_C2_256;
    mpz_class FPEC_NON_SQRT;
    mpz_class FNEC;
    mpz_class FNEC_MINUS_ONE;
    mpz_class ECGX;
    mpz_class ECGY;
    mpz_class P2_160;
    mpz_class P2_96;

    RomConstants() :
        BATCH_DIFFICULTY(0),
        TX_GAS_LIMIT(0),
        GLOBAL_EXIT_ROOT_STORAGE_POS(0),
        LOCAL_EXIT_ROOT_STORAGE_POS(0),
        LAST_TX_STORAGE_POS(0),
        STATE_ROOT_STORAGE_POS(0),
        MAX_MEM_EXPANSION_BYTES(0),
        FORK_ID(0),
        MIN_VALUE_SHORT(0),
        MIN_BYTES_LONG(0),
        SMT_KEY_BALANCE(0),
        SMT_KEY_NONCE(0),
        SMT_KEY_SC_CODE(0),
        SMT_KEY_SC_STORAGE(0),
        SMT_KEY_SC_LENGTH(0),
        SMT_KEY_TOUCHED_ADDR(0),
        SMT_KEY_TOUCHED_SLOTS(0),
        BASE_TX_GAS(0),
        BASE_TX_DEPLOY_GAS(0),
        SLOAD_GAS(0),
        GAS_QUICK_STEP(0),
        GAS_FASTEST_STEP(0),
        GAS_FAST_STEP(0),
        GAS_MID_STEP(0),
        GAS_SLOW_STEP(0),
        GAS_EXT_STEP(0),
        CALL_VALUE_TRANSFER_GAS(0),
        CALL_NEW_ACCOUNT_GAS(0),
        CALL_STIPEND(0),
        ECRECOVER_GAS(0),
        IDENTITY_GAS(0),
        IDENTITY_WORD_GAS(0),
        KECCAK_GAS(0),
        KECCAK_WORD_GAS(0),
        LOG_GAS(0),
        LOG_TOPIC_GAS(0),
        JUMP_DEST_GAS(0),
        WARM_STORGE_READ_GAS(0),
        COLD_ACCOUNT_ACCESS_COST_REDUCED(0),
        COLD_ACCOUNT_ACCESS_COST(0),
        EXP_BYTE_GAS(0),
        RETURN_GAS_COST(0),
        CREATE_GAS(0),
        CREATE_2_GAS(0),
        SENDALL_GAS(0),
        LOG_DATA_GAS(0),
        SSTORE_ENTRY_EIP_2200_GAS(0),
        SSTORE_SET_EIP_2200_GAS(0),
        COLD_SLOAD_COST(0),
        COLD_SLOAD_COST_REDUCED(0),
        SSTORE_DYNAMIC_GAS(0),
        SSTORE_SET_GAS(0),
        SSTORE_SET_GAS_REDUCED(0),
        SSTORE_RESET_GAS(0),
        SSTORE_RESET_GAS_REDUCED(0),
        SSTORE_CLEARS_SCHEDULE(0),
        MIN_STEPS_FINISH_BATCH(0),
        TOTAL_STEPS_LIMIT(0),
        MAX_CNT_STEPS_LIMIT(0),
        MAX_CNT_ARITH_LIMIT(0),
        MAX_CNT_BINARY_LIMIT(0),
        MAX_CNT_MEM_ALIGN_LIMIT(0),
        MAX_CNT_KECCAK_F_LIMIT(0),
        MAX_CNT_PADDING_PG_LIMIT(0),
        MAX_CNT_POSEIDON_G_LIMIT(0),
        SAFE_RANGE(0),
        MAX_CNT_STEPS(0),
        MAX_CNT_ARITH(0),
        MAX_CNT_BINARY(0),
        MAX_CNT_MEM_ALIGN(0),
        MAX_CNT_KECCAK_F(0),
        MAX_CNT_PADDING_PG(0),
        MAX_CNT_POSEIDON_G(0),
        MAX_CNT_POSEIDON_SLOAD_SSTORE(0),
        MIN_CNT_KECCAK_BATCH(0),
        CODE_SIZE_LIMIT(0),
        BYTECODE_STARTS_EF(0)
    {
        ;
    }

};

}

#endif