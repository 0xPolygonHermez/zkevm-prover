#include "account.hpp"
#include "scalar.hpp"

namespace fork_6
{

// EVM constant
#define GLOBAL_EXIT_ROOT_STORAGE_POS 0
#define LOCAL_EXIT_ROOT_STORAGE_POS 1
#define BATCH_GAS_LIMIT 30000000
#define BATCH_DIFFICULTY 0
#define STATE_ROOT_STORAGE_POS 1

// SMT STATE-TREE CONSTANT KEYS
#define SMT_KEY_BALANCE 0
#define SMT_KEY_NONCE 1
#define SMT_KEY_SC_CODE 2
#define SMT_KEY_SC_STORAGE 3
#define SMT_KEY_SC_LENGTH 4

// TODO: init from main
bool Account::bZeroKeyGenerated = false;
Goldilocks::Element Account::zeroKey[4];

// Initialization
zkresult Account::Init (const mpz_class &publicKey_)
{
    publicKey = publicKey_;

    // Check range
    if (publicKey > ScalarMask160)
    {
        zklog.error("Account::Init() received invalid publicKey, too big =" + publicKey.get_str(16));
        return ZKR_UNSPECIFIED;
    }

    bInitialized = true;

    return ZKR_SUCCESS;
};

void Account::GenerateZeroKey (Goldilocks &fr, PoseidonGoldilocks &poseidon)
{
    zkassert(bZeroKeyGenerated == false);
    
    // TODO: Pre-calculate this value
    Goldilocks::Element Kin0[12];
    Kin0[0] = fr.zero();
    Kin0[1] = fr.zero();
    Kin0[2] = fr.zero();
    Kin0[3] = fr.zero();
    Kin0[4] = fr.zero();
    Kin0[5] = fr.zero();
    Kin0[6] = fr.zero();
    Kin0[7] = fr.zero();
    Kin0[8] = fr.zero();
    Kin0[9] = fr.zero();
    Kin0[10] = fr.zero();
    Kin0[11] = fr.zero();

    // Call poseidon and get the hash key
    poseidon.hash(zeroKey, Kin0);

#ifdef LOG_ACCOUNT
    zklog.info("Account::GenerateZeroKey() zeroKey=" + fea2string(fr, zeroKey));
#endif

    bZeroKeyGenerated = true;
}

void Account::GenerateBalanceKey (void)
{
    Goldilocks::Element Kin1[12];

    scalar2fea(fr, publicKey, Kin1[0], Kin1[1], Kin1[2], Kin1[3], Kin1[4], Kin1[5], Kin1[6], Kin1[7]); // TODO: Reuse this result
    if (!fr.isZero(Kin1[5]) || !fr.isZero(Kin1[6]) || !fr.isZero(Kin1[7]))
    {
        zklog.error("Accoung::GetBalanceKey() found non-zero field elements 5, 6 or 7");
        exitProcess();
    }

    Kin1[8] = zeroKey[0];
    Kin1[9] = zeroKey[1];
    Kin1[10] = zeroKey[2];
    Kin1[11] = zeroKey[3];

    // Call poseidon and get the hash key
    poseidon.hash(balanceKey, Kin1);

#ifdef LOG_ACCOUNT
    zklog.info("Account::GenerateBalanceKey() balanceKey=" + fea2string(fr, balanceKey));
#endif
}

void Account::GenerateNonceKey (void)
{
    Goldilocks::Element Kin1[12];

    scalar2fea(fr, publicKey, Kin1[0], Kin1[1], Kin1[2], Kin1[3], Kin1[4], Kin1[5], Kin1[6], Kin1[7]);
    if (!fr.isZero(Kin1[5]) || !fr.isZero(Kin1[6]) || !fr.isZero(Kin1[7]))
    {
        zklog.error("Accoung::GetNonceKey() found non-zero field elements 5, 6 or 7");
        exitProcess();
    }

    Kin1[6] = fr.one();

    Kin1[8] = zeroKey[0];
    Kin1[9] = zeroKey[1];
    Kin1[10] = zeroKey[2];
    Kin1[11] = zeroKey[3];

    // Call poseidon and get the hash key
    poseidon.hash(nonceKey, Kin1);

#ifdef LOG_ACCOUNT
    zklog.info("Account::GetNonceKey() nonceKey=" + fea2string(fr, nonceKey));
#endif
}

void Account::GenerateGlobalExitRootKey (const mpz_class &globalExitRoot)
{
    uint8_t data64[64];
    mpz_class auxScalar = globalExitRoot;
    scalar2bytesBE(auxScalar, &data64[0]);
    memset(&data64[32], 0, 32);

    mpz_class keccakScalar;
    keccak256(data64, 64, keccakScalar);
    Goldilocks::Element keccakFea[8];
    scalar2fea(fr, keccakScalar, keccakFea);

    // TODO: Pre-calculate this value
    Goldilocks::Element Kin0[12];
    for (uint64_t i = 0; i < 8; i++)
    {
        Kin0[i] = keccakFea[i];
    }
    Kin0[8] = fr.zero();
    Kin0[9] = fr.zero();
    Kin0[10] = fr.zero();
    Kin0[11] = fr.zero();

    // Call poseidon and get the hash key

    Goldilocks::Element Kin0Key[4];
    poseidon.hash(Kin0Key, Kin0);

    Goldilocks::Element Kin1[12];

    scalar2fea(fr, publicKey, Kin1[0], Kin1[1], Kin1[2], Kin1[3], Kin1[4], Kin1[5], Kin1[6], Kin1[7]);
    if (!fr.isZero(Kin1[5]) || !fr.isZero(Kin1[6]) || !fr.isZero(Kin1[7]))
    {
        zklog.error("Account::GenerateGlobalExitRootKey() found non-zero field elements 5, 6 or 7");
        exitProcess();
    }

    Kin1[6] = fr.fromU64(SMT_KEY_SC_STORAGE);

    Kin1[8] = Kin0Key[0];
    Kin1[9] = Kin0Key[1];
    Kin1[10] = Kin0Key[2];
    Kin1[11] = Kin0Key[3];

    // Call poseidon and get the hash key
    poseidon.hash(globalExitRootKey, Kin1);

#ifdef LOG_ACCOUNT
    zklog.info("Account::GenerateGlobalExitRootKey() globalExitRootKey=" + fea2string(fr, globalExitRootKey));
#endif
}

void Account::GenerateLocalExitRootKey (void)
{
    // TODO: Pre-calculate this value
    Goldilocks::Element Kin0[12];
    Kin0[0] = fr.one();
    for (uint64_t i = 1; i < 12; i++)
    {
        Kin0[i] = fr.zero();
    }

    // Call poseidon and get the hash key

    Goldilocks::Element Kin0Key[4];
    poseidon.hash(Kin0Key, Kin0);

    Goldilocks::Element Kin1[12];

    scalar2fea(fr, publicKey, Kin1[0], Kin1[1], Kin1[2], Kin1[3], Kin1[4], Kin1[5], Kin1[6], Kin1[7]);
    if (!fr.isZero(Kin1[5]) || !fr.isZero(Kin1[6]) || !fr.isZero(Kin1[7]))
    {
        zklog.error("Account::GenerateLocalExitRootKey() found non-zero field elements 5, 6 or 7");
        exitProcess();
    }

    Kin1[6] = fr.fromU64(SMT_KEY_SC_STORAGE);

    Kin1[8] = Kin0Key[0];
    Kin1[9] = Kin0Key[1];
    Kin1[10] = Kin0Key[2];
    Kin1[11] = Kin0Key[3];

    // Call poseidon and get the hash key
    poseidon.hash(localExitRootKey, Kin1);

#ifdef LOG_ACCOUNT
    zklog.info("Account::GenerateLocalExitRootKey() localExitRootKey=" + fea2string(fr, localExitRootKey));
#endif
}

void Account::GenerateTxCountKey (void)
{
    Goldilocks::Element Kin1[12];

    scalar2fea(fr, publicKey, Kin1[0], Kin1[1], Kin1[2], Kin1[3], Kin1[4], Kin1[5], Kin1[6], Kin1[7]);
    if (!fr.isZero(Kin1[5]) || !fr.isZero(Kin1[6]) || !fr.isZero(Kin1[7]))
    {
        zklog.error("Accoung::GenerateBatchNumberKey() found non-zero field elements 5, 6 or 7");
        exitProcess();
    }

    Kin1[6] = fr.fromU64(SMT_KEY_SC_STORAGE);

    Kin1[8] = zeroKey[0];
    Kin1[9] = zeroKey[1];
    Kin1[10] = zeroKey[2];
    Kin1[11] = zeroKey[3];

    // Call poseidon and get the hash key
    poseidon.hash(txCountKey, Kin1);

#ifdef LOG_ACCOUNT
    zklog.info("Accoung::GenerateBatchNumberKey() txCountKey=" + fea2string(fr, txCountKey));
#endif
}

void Account::GenerateStateRootKey (const mpz_class &batchNumber, Goldilocks::Element (&stateRootKey)[4])
{
    // 64B buffer = batchNumber (32B) + STATE_ROOT_STORAGE_POS (32B)
    uint8_t data64[64];
    mpz_class auxScalar = batchNumber;
    scalar2bytesBE(auxScalar, &data64[0]);
    auxScalar = STATE_ROOT_STORAGE_POS;
    scalar2bytesBE(auxScalar, &data64[32]);

    // Calculate Keccak-256 hash of 64B buffer
    mpz_class keccakScalar;
    keccak256(data64, 64, keccakScalar);
    Goldilocks::Element keccakFea[8];
    scalar2fea(fr, keccakScalar, keccakFea);

    // Prepare key hash input data
    Goldilocks::Element Kin0[12];
    for (uint64_t i = 0; i < 8; i++)
    {
        Kin0[i] = keccakFea[i];
    }
    Kin0[8] = fr.zero();
    Kin0[9] = fr.zero();
    Kin0[10] = fr.zero();
    Kin0[11] = fr.zero();

    // Call poseidon and get the poseidon hash
    Goldilocks::Element Kin0Key[4];
    poseidon.hash(Kin0Key, Kin0);

    // Prepare hash input data with public key + SMT_KEY_SC_STORAGE + previous hash
    Goldilocks::Element Kin1[12];
    scalar2fea(fr, publicKey, Kin1[0], Kin1[1], Kin1[2], Kin1[3], Kin1[4], Kin1[5], Kin1[6], Kin1[7]);
    if (!fr.isZero(Kin1[5]) || !fr.isZero(Kin1[6]) || !fr.isZero(Kin1[7]))
    {
        zklog.error("Account::GenerateStateRootKey() found non-zero field elements 5, 6 or 7");
        exitProcess();
    }

    Kin1[6] = fr.fromU64(SMT_KEY_SC_STORAGE);

    Kin1[8] = Kin0Key[0];
    Kin1[9] = Kin0Key[1];
    Kin1[10] = Kin0Key[2];
    Kin1[11] = Kin0Key[3];

    // Call poseidon and get the hash key
    poseidon.hash(stateRootKey, Kin1);

#ifdef LOG_ACCOUNT
    zklog.info("Account::GenerateStateRootKey() stateRootKey=" + fea2string(fr, stateRootKey));
#endif
}

zkresult Account::GetBalance (const string &batchUUID, const Goldilocks::Element (&root)[4], mpz_class &balance)
{
    // Check that balance key has been generated
    CheckBalanceKey();

    zkresult zkResult = hashDB.get(batchUUID, root, balanceKey, balance, /*&smtGetResult*/ NULL, NULL /*proverRequest.dbReadLog*/);
    if (zkResult != ZKR_SUCCESS)
    {
        zklog.error("Account::GetBalance() failed calling hashDB.get() result=" + zkresult2string(zkResult));
        return zkResult;
    }

#ifdef LOG_ACCOUNT
    zklog.info("Account::GetBalance() publicKey=" + publicKey.get_str(16) + " balanceKey=" + fea2string(fr, balanceKey) + " balance=" + balance.get_str(10) + " root=" + fea2string(fr, root));
#endif

    return ZKR_SUCCESS;
}

zkresult Account::SetBalance (const string &batchUUID, uint64_t tx, Goldilocks::Element (&root)[4], const mpz_class &balance)
{
    // Check that balance key has been generated
    CheckBalanceKey();

    zkresult zkResult = hashDB.set(batchUUID, tx, root, balanceKey, balance, /*proverRequest.input.bUpdateMerkleTree*/ PERSISTENCE_CACHE, root, /*&ctx.lastSWrite.res*/ NULL, /*proverRequest.dbReadLog*/ NULL);
    if (zkResult != ZKR_SUCCESS)
    {
        zklog.error("Account::SetBalance()) failed calling hashDB.set() result=" + zkresult2string(zkResult));
        return zkResult;
    }

#ifdef LOG_ACCOUNT
    zklog.info("Account::SetBalance() publicKey=" + publicKey.get_str(16) + " balanceKey=" + fea2string(fr, balanceKey) + " balance=" + balance.get_str(10) + " root=" + fea2string(fr, root));
#endif

    return ZKR_SUCCESS;
}

zkresult Account::GetNonce (const string &batchUUID, const Goldilocks::Element (&root)[4], uint64_t &nonce)
{
    // Check that nonce key has been generated
    CheckNonceKey();

    mpz_class value;
    zkresult zkResult = hashDB.get(batchUUID, root, nonceKey, value, /*&smtGetResult*/ NULL, NULL /*proverRequest.dbReadLog*/);
    if (zkResult != ZKR_SUCCESS)
    {
        zklog.error("Account::GetNonce() failed calling hashDB.get() result=" + zkresult2string(zkResult));
        return zkResult;
    }
    if (value > mpz_class("0xFFFFFFFFFFFFFFFF")) // TODO: Precalculat.  Do this check only in debug?
    {
        zklog.error("Account::GetNonce() failed called hashDB.get() but nonce is too big =" + value.get_str(10));
        return ZKR_UNSPECIFIED;
    }

    nonce = value.get_ui();

#ifdef LOG_ACCOUNT
    zklog.info("Account::GetNonce() publicKey=" + publicKey.get_str(16) + " nonceKey=" + fea2string(fr, nonceKey) + " nonce=" + to_string(nonce) + " root=" + fea2string(fr, root));
#endif

    return ZKR_SUCCESS;
}

zkresult Account::SetNonce (const string &batchUUID, uint64_t tx, Goldilocks::Element (&root)[4], const uint64_t &nonce)
{
    // Check that nonce key has been generated
    CheckNonceKey();

    mpz_class value = nonce;
    zkresult zkResult = hashDB.set(batchUUID, tx, root, nonceKey, value, /*proverRequest.input.bUpdateMerkleTree*/ PERSISTENCE_CACHE, root, /*&ctx.lastSWrite.res*/ NULL, /*proverRequest.dbReadLog*/ NULL);
    if (zkResult != ZKR_SUCCESS)
    {
        zklog.error("Account::SetNonce() failed calling hashDB.set() result=" + zkresult2string(zkResult));
        return zkResult;
    }

#ifdef LOG_ACCOUNT
    zklog.info("Account::SetNonce() publicKey=" + publicKey.get_str(16) + " nonceKey=" + fea2string(fr, nonceKey) + " nonce=" + to_string(nonce) + " root=" + fea2string(fr, root));
#endif

    return ZKR_SUCCESS;
}

zkresult Account::SetGlobalExitRoot (const string &batchUUID, uint64_t tx, Goldilocks::Element (&root)[4], const mpz_class &globalExitRoot, const mpz_class &value)
{
    // Check that nonce key has been generated
    CheckGlobalExitRootKey(globalExitRoot);

    zkresult zkResult = hashDB.set(batchUUID, tx, root, globalExitRootKey, value, /*proverRequest.input.bUpdateMerkleTree*/ PERSISTENCE_CACHE, root, /*&ctx.lastSWrite.res*/ NULL, /*proverRequest.dbReadLog*/ NULL);
    if (zkResult != ZKR_SUCCESS)
    {
        zklog.error("Account::SetGlobalExitRoot() failed calling hashDB.set() result=" + zkresult2string(zkResult));
        return zkResult;
    }

#ifdef LOG_ACCOUNT
    zklog.info("Account::SetGlobalExitRoot() publicKey=" + publicKey.get_str(16) + " globalExitRootKey=" + fea2string(fr, globalExitRootKey) + " value=" + value.get_str(10) + " root=" + fea2string(fr, root));
#endif

    return ZKR_SUCCESS;
}

zkresult Account::GetTxCount (const string &batchUUID, const Goldilocks::Element (&root)[4], mpz_class &txCount)
{
    // Check that batch number key has been generated
    CheckTxCountKey();

    zkresult zkResult = hashDB.get(batchUUID, root, txCountKey, txCount, /*&smtGetResult*/ NULL, NULL /*proverRequest.dbReadLog*/);
    if (zkResult != ZKR_SUCCESS)
    {
        zklog.error("Account::GetBatchNumber() failed calling hashDB.get() result=" + zkresult2string(zkResult));
        return zkResult;
    }

#ifdef LOG_ACCOUNT
    zklog.info("Account::GetBatchNumber() publicKey=" + publicKey.get_str(16) + " txCountKey=" + fea2string(fr, txCountKey) + " txCount=" + to_string(txCount) + " root=" + fea2string(fr, root));
#endif

    return ZKR_SUCCESS;
}

zkresult Account::SetTxCount (const string &batchUUID, uint64_t tx, Goldilocks::Element (&root)[4], const mpz_class &txCount)
{
    // Check that nonce key has been generated
    CheckTxCountKey();

    zkresult zkResult = hashDB.set(batchUUID, tx, root, txCountKey, txCount, /*proverRequest.input.bUpdateMerkleTree*/ PERSISTENCE_CACHE, root, /*&ctx.lastSWrite.res*/ NULL, /*proverRequest.dbReadLog*/ NULL);
    if (zkResult != ZKR_SUCCESS)
    {
        zklog.error("Account::SetBatchNumber() failed calling hashDB.set() result=" + zkresult2string(zkResult));
        return zkResult;
    }

#ifdef LOG_ACCOUNT
    zklog.info("Account::SetBatchNumber() publicKey=" + publicKey.get_str(16) + " batchNumberKey=" + fea2string(fr, batchNumberKey) + " txCount=" + txCount.get_str(10) + " root=" + fea2string(fr, root));
#endif

    return ZKR_SUCCESS;
}

zkresult Account::SetStateRoot (const string &batchUUID, uint64_t tx, Goldilocks::Element (&root)[4], const mpz_class &txCount, const mpz_class &stateRoot)
{
    // Check that nonce key has been generated
    Goldilocks::Element stateRootKey[4];
    GenerateStateRootKey(txCount, stateRootKey);

    zkresult zkResult = hashDB.set(batchUUID, tx, root, stateRootKey, stateRoot, /*proverRequest.input.bUpdateMerkleTree*/ PERSISTENCE_CACHE, root, /*&ctx.lastSWrite.res*/ NULL, /*proverRequest.dbReadLog*/ NULL);
    if (zkResult != ZKR_SUCCESS)
    {
        zklog.error("Account::SetStateRoot() failed calling hashDB.set() result=" + zkresult2string(zkResult));
        return zkResult;
    }

#ifdef LOG_ACCOUNT
    zklog.info("Account::SetStateRoot() publicKey=" + publicKey.get_str(16) + " stateRootKey=" + fea2string(fr, stateRootKey) + " stateRoot=" + stateRoot.get_str(10) + " root=" + fea2string(fr, root));
#endif

    return ZKR_SUCCESS;
}

}