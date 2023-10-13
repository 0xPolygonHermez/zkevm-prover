#ifndef ACCOUNT_HPP_fork_6
#define ACCOUNT_HPP_fork_6

#include <gmpxx.h>
#include <cstdint>
#include "zkresult.hpp"
#include "hashdb_interface.hpp"
#include "exit_process.hpp"
#include "poseidon_goldilocks.hpp"

using namespace std;

namespace fork_6
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
    bool bTxCountKeyGenerated;
    Goldilocks::Element txCountKey[4];

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
        bTxCountKeyGenerated(false)
    {
        CheckZeroKey();
    };

    // Initialization
    zkresult Init (const mpz_class &publicKey_);

    // Generate the static zero key, common for all account instances
    static void GenerateZeroKey (Goldilocks &fr, PoseidonGoldilocks &poseidon);

private:

    void GenerateBalanceKey        (void);
    void GenerateNonceKey          (void);
    void GenerateGlobalExitRootKey (const mpz_class &globalExitRoot);
    void GenerateLocalExitRootKey  (void);
    void GenerateTxCountKey        (void);
    void GenerateStateRootKey      (const mpz_class &txCount, Goldilocks::Element (&stateRootKey)[4]);

    inline void CheckZeroKey (void)
    {
        zkassert(bZeroKeyGenerated);
    }
    inline void CheckBalanceKey (void)
    {
        if (!bBalanceKeyGenerated)
        {
            GenerateBalanceKey();
            bBalanceKeyGenerated = true;
        }
    }
    inline void CheckNonceKey (void)
    {
        if (!bNonceKeyGenerated)
        {
            GenerateNonceKey();
            bNonceKeyGenerated = true;
        }
    }
    inline void CheckGlobalExitRootKey (const mpz_class &globalExitRoot)
    {
        if (!bGlobalExitRootKeyGenerated)
        {
            GenerateGlobalExitRootKey(globalExitRoot);
            bGlobalExitRootKeyGenerated = true;
        }
    }
    inline void CheckLocalExitRootKey (void)
    {
        if (!bLocalExitRootKeyGenerated)
        {
            GenerateLocalExitRootKey();
            bLocalExitRootKeyGenerated = true;
        }
    }
    inline void CheckTxCountKey (void)
    {
        if (!bTxCountKeyGenerated)
        {
            GenerateTxCountKey();
            bTxCountKeyGenerated = true;
        }
    }
   
public:

    // Get account balance value
    zkresult GetBalance (const string &batchUUID, const Goldilocks::Element (&root)[4], mpz_class &balance);

    // Set account balance value; root is updated with state new root
    zkresult SetBalance (const string &batchUUID, uint64_t tx, Goldilocks::Element (&root)[4], const mpz_class &balance);

    // Get account nonce value
    zkresult GetNonce (const string &batchUUID, const Goldilocks::Element (&root)[4], uint64_t &nonce);

    // Set account nonce value; root is updated with new state root
    zkresult SetNonce (const string &batchUUID, uint64_t tx, Goldilocks::Element (&root)[4], const uint64_t &nonce);

    // Set account global exit root; root is updated with new state root
    zkresult SetGlobalExitRoot (const string &batchUUID, uint64_t tx, Goldilocks::Element (&root)[4], const mpz_class &globalExitRoot, const mpz_class &value);
    
    // Get account batch number value
    zkresult GetTxCount (const string &batchUUID, const Goldilocks::Element (&root)[4], mpz_class &txCount);

    // Set account TX count value; root is updated with new state root
    zkresult SetTxCount (const string &batchUUID, uint64_t tx, Goldilocks::Element (&root)[4], const mpz_class &txCount);

    // Set account state root value; root is updated with new state root
    zkresult SetStateRoot (const string &batchUUID, uint64_t tx, Goldilocks::Element (&root)[4], const mpz_class &txCount, const mpz_class &stateRoot);
};

}

#endif