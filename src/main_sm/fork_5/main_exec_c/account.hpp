#ifndef ACCOUNT_HPP_fork_5
#define ACCOUNT_HPP_fork_5

#include <gmpxx.h>
#include <cstdint>
#include "zkresult.hpp"
#include "hashdb_interface.hpp"
#include "exit_process.hpp"
#include "poseidon_goldilocks.hpp"

using namespace std;

namespace fork_5
{

// Well-known addresses
#define ADDRESS_GLOBAL_EXIT_ROOT_MANAGER_L2 "0xa40D5f56745a118D0906a34E69aeC8C0Db1cB8fA"
#define ADDRESS_SYSTEM "0x000000000000000000000000000000005ca1ab1e"

class Account
{
private:
    Goldilocks &fr;
    PoseidonGoldilocks &poseidon;
    HashDBInterface &hashDB;
    mpz_class publicKey;
    bool bInitialized;
    static bool bZeroKeyGenerated;
    static Goldilocks::Element zeroKey[4];
    bool bBalanceKeyGenerated;
    Goldilocks::Element balanceKey[4];
    bool bNonceKeyGenerated;
    Goldilocks::Element nonceKey[4];
    bool bGlobalExitRootKeyGenerated;
    Goldilocks::Element globalExitRootKey[4];
    bool bLocalExitRootKeyGenerated;
    Goldilocks::Element localExitRootKey[4];
    bool bBatchNumberKeyGenerated;
    Goldilocks::Element batchNumberKey[4];
    bool bStateRootKeyGenerated;
    Goldilocks::Element stateRootKey[4];
public:
    
    // Constructor
    Account (Goldilocks &fr, PoseidonGoldilocks &poseidon, HashDBInterface &hashDB) :
        fr(fr), 
        poseidon(poseidon), 
        hashDB(hashDB), 
        bInitialized(false),
        bBalanceKeyGenerated(false),
        bNonceKeyGenerated(false),
        bGlobalExitRootKeyGenerated(false),
        bLocalExitRootKeyGenerated(false),
        bBatchNumberKeyGenerated(false),
        bStateRootKeyGenerated(false)
    {
        CheckZeroKey();
    };

    // Initialization
    zkresult Init (const mpz_class &publicKey_);

private:

    void GenerateZeroKey (Goldilocks::Element (&zeroKey)[4]);
    void GenerateBalanceKey (Goldilocks::Element (&balanceKey)[4]);
    void GenerateNonceKey (Goldilocks::Element (&nonceKey)[4]);
    void GenerateGlobalExitRootKey (const mpz_class &globalExitRoot, Goldilocks::Element (&globalExitRootKey)[4]);
    void GenerateLocalExitRootKey (Goldilocks::Element (&localExitRootKey)[4]);
    void GenerateBatchNumberKey (Goldilocks::Element (&batchNumberKey)[4]);
    void GenerateStateRootKey (uint64_t batchNumber, Goldilocks::Element (&stateRootKey)[4]);

    inline void CheckZeroKey (void)
    {
        if (!bZeroKeyGenerated)
        {
            GenerateZeroKey(zeroKey);
            bZeroKeyGenerated = true;
        }
    }
    inline void CheckBalanceKey (void)
    {
        if (!bBalanceKeyGenerated)
        {
            GenerateBalanceKey(balanceKey);
            bBalanceKeyGenerated = true;
        }
    }
    inline void CheckNonceKey (void)
    {
        if (!bNonceKeyGenerated)
        {
            GenerateNonceKey(nonceKey);
            bNonceKeyGenerated = true;
        }
    }
    inline void CheckGlobalExitRootKey (const mpz_class &globalExitRoot)
    {
        if (!bGlobalExitRootKeyGenerated)
        {
            GenerateGlobalExitRootKey(globalExitRoot, globalExitRootKey);
            bGlobalExitRootKeyGenerated = true;
        }
    }
    inline void CheckLocalExitRootKey (void)
    {
        if (!bLocalExitRootKeyGenerated)
        {
            GenerateLocalExitRootKey(localExitRootKey);
            bLocalExitRootKeyGenerated = true;
        }
    }
    inline void CheckBatchNumberKey (void)
    {
        if (!bBatchNumberKeyGenerated)
        {
            GenerateBatchNumberKey(batchNumberKey);
            bBatchNumberKeyGenerated = true;
        }
    }
    inline void CheckStateRootKey (uint64_t batchNumber)
    {
        if (!bStateRootKeyGenerated)
        {
            GenerateStateRootKey(batchNumber, stateRootKey);
            bStateRootKeyGenerated = true;
        }
    }
   
public:

    // Get account balance value
    zkresult GetBalance (const Goldilocks::Element (&root)[4], mpz_class &balance);

    // Set account balance value; root is updated with new root
    zkresult SetBalance (Goldilocks::Element (&root)[4], const mpz_class &balance);

    // Get account nonce value
    zkresult GetNonce (const Goldilocks::Element (&root)[4], uint64_t &nonce);

    // Set account nonce value; root is updated with new root
    zkresult SetNonce (Goldilocks::Element (&root)[4], const uint64_t &nonce);

    zkresult SetGlobalExitRoot (Goldilocks::Element (&root)[4], const mpz_class &globalExitRoot, const mpz_class &value);
    
    // Get account batch number value
    zkresult GetBatchNumber (const Goldilocks::Element (&root)[4], uint64_t &batchNumber); // TODO: oldBatchNumber and newBatchNumber are U32

    // Set account batch number value; root is updated with new root
    zkresult SetBatchNumber (Goldilocks::Element (&root)[4], const uint64_t &batchNumber);

    // Set account state root value; root is updated with new root
    zkresult SetStateRoot (Goldilocks::Element (&root)[4], const uint64_t &batchNumber, const mpz_class &stateRoot);
};

}

#endif