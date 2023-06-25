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
public:
    
    // Constructor
    Account (Goldilocks &fr, PoseidonGoldilocks &poseidon, HashDBInterface &hashDB) :
        fr(fr), 
        poseidon(poseidon), 
        hashDB(hashDB), 
        bInitialized(false),
        bBalanceKeyGenerated(false),
        bNonceKeyGenerated(false)
    {
        CheckZeroKey();
    };

    // Initialization
    zkresult Init (const mpz_class &aux)
    {
        publicKey = aux;

        // Check range
        if (publicKey > mpz_class("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")) // TODO: make it a constant
        {
            zklog.error("Account::Init() received invalid publicKey, too big =" + publicKey.get_str(16));
            return ZKR_UNSPECIFIED;
        }

        bInitialized = true;

        return ZKR_SUCCESS;
    };

private:

    void GenerateZeroKey (Goldilocks::Element (&zeroKey)[4]);
    void GenerateBalanceKey (Goldilocks::Element (&balanceKey)[4]);
    void GenerateNonceKey (Goldilocks::Element (&nonceKey)[4]);

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
   
public:

    // Get account balance value
    zkresult GetBalance (const Goldilocks::Element (&root)[4], mpz_class &balance);

    // Set account balance value; root is updated with new root
    zkresult SetBalance (Goldilocks::Element (&root)[4], const mpz_class &balance);

    // Get account nonce value
    zkresult GetNonce (const Goldilocks::Element (&root)[4], uint64_t &nonce);

    // Set account nonce value; root is updated with new root
    zkresult SetNonce (Goldilocks::Element (&root)[4], const uint64_t &nonce);
};

}

#endif