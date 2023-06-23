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
    Goldilocks::Element zeroKey[4];
    Goldilocks::Element balanceKey[4];
    Goldilocks::Element nonceKey[4];
public:
    
    // Constructor
    Account (Goldilocks &fr, PoseidonGoldilocks &poseidon, HashDBInterface &hashDB) : fr(fr), poseidon(poseidon), hashDB(hashDB), bInitialized(false)
    {
        GetZeroKey(zeroKey);
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

        GetBalanceKey(balanceKey);
        GetNonceKey(nonceKey);

        bInitialized = true;

        return ZKR_SUCCESS;
    };

private:
    void GetZeroKey (Goldilocks::Element (&zeroKey)[4]);
    void GetBalanceKey (Goldilocks::Element (&balanceKey)[4]);
    void GetNonceKey (Goldilocks::Element (&nonceKey)[4]);
public:
    zkresult GetBalance (Goldilocks::Element (&oldRoot)[4], mpz_class &balance);
    zkresult SetBalance (mpz_class &balance);
    zkresult GetNonce (uint64_t &nonce);
    zkresult SetNonce (uint64_t &nonce);
};

}

#endif