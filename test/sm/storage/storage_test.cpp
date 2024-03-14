#include <iostream>
#include "storage_test.hpp"
#include "storage_executor.hpp"
#include "goldilocks_base_field.hpp"
#include "smt.hpp"
#include "smt_action.hpp"
#include "smt_action_list.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"

using namespace std;

void StorageSM_GetZeroTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config);
void StorageSM_UnitTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config);
void StorageSM_ZeroToZero2Test (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config);
void StorageSM_ZeroToZeroTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config);
void StorageSM_EmptyTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config);
void StorageSM_UseCaseTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config);

uint64_t StorageSMTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config)
{
    uint64_t numberOfErrors = 0;
    cout << "StorageSMTest starting..." << endl;

    StorageSM_GetZeroTest(fr, poseidon, config);
    StorageSM_UnitTest(fr, poseidon, config);
    StorageSM_ZeroToZeroTest(fr, poseidon, config);
    StorageSM_EmptyTest(fr, poseidon, config);
    StorageSM_UseCaseTest(fr, poseidon, config);

    cout << "StorageSMTest done" << endl;
    return numberOfErrors;
}

void StorageSM_UnitTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config)
{
    cout << "StorageSM_UnitTest starting..." << endl;

    string uuid = getUUID();
    uint64_t tx = 0;
    Smt smt(fr);
    Database db(fr, config);
    db.init();
    SmtActionList actionList;
    SmtGetResult getResult;
    SmtSetResult setResult;
    Goldilocks::Element root[4]={0,0,0,0};
    Goldilocks::Element key[4]={1,0,0,0};
    mpz_class value = 10;

    // Get zero
    smt.get(uuid, db, root, key, getResult);
    actionList.addGetAction(getResult);
    cout << "0: StorageSMTest Get zero value=" << getResult.value.get_str(16) << endl;

    // Set insertNotFound
    smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassertpermanent(setResult.mode=="insertNotFound");
    cout << "1: StorageSMTest Set insertNotFound root=" << fea2string(fr, root) << " mode=" << setResult.mode <<endl;

    // Get non zero
    smt.get(uuid, db, root, key, getResult);
    actionList.addGetAction(getResult);
    cout << "2: StorageSMTest Get nonZero value=" << getResult.value.get_str(16) << endl;

    // Set deleteLast
    value=0;
    smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassertpermanent(setResult.mode=="deleteLast");
    cout << "3: StorageSMTest Set deleteLast root=" << fea2string(fr, root) << " mode=" << setResult.mode <<endl;

    // Set insertNotFound
    value=10;
    smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    cout << "4: StorageSMTest Set insertNotFound root=" << fea2string(fr, root) << " mode=" << setResult.mode <<endl;

    // Set update
    value=20;
    smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassertpermanent(setResult.mode=="update");
    cout << "5: StorageSMTest Set update root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Get non zero
    smt.get(uuid, db, root, key, getResult);
    actionList.addGetAction(getResult);
    cout << "6: StorageSMTest Get nonZero value=" << getResult.value.get_str(16) << endl;

    // Set insertFound
    key[0] = fr.fromU64(3);
    value = 20;
    smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassertpermanent(setResult.mode=="insertFound");
    cout << "7: StorageSMTest Set insertFound root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Get non zero
    smt.get(uuid, db, root, key, getResult);
    actionList.addGetAction(getResult);
    cout << "8: StorageSMTest Get nonZero value=" << getResult.value.get_str(16) << endl;

    // Set deleteFound
    value=0;
    smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassertpermanent(setResult.mode=="deleteFound");
    cout << "9: StorageSMTest Set deleteFound root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Get zero
    smt.get(uuid, db, root, key, getResult);
    actionList.addGetAction(getResult);
    cout << "10: StorageSMTest Get zero value=" << getResult.value.get_str(16) << endl;

    // Set zeroToZzero
    value=0;
    smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassertpermanent(setResult.mode=="zeroToZero");
    cout << "11: StorageSMTest Set zeroToZero root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Set insertFound
    value=40;
    smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassertpermanent(setResult.mode=="insertFound");
    cout << "12: StorageSMTest Set insertFound root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Get non zero
    smt.get(uuid, db, root, key, getResult);
    actionList.addGetAction(getResult);
    cout << "13: StorageSMTest Get nonZero value=" << getResult.value.get_str(16) << endl;

    // Set insertNotFound
    key[0] = fr.zero();
    key[1] = fr.one();
    value=30;
    smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassertpermanent(setResult.mode=="insertNotFound");
    cout << "14: StorageSMTest Set insertNotFound root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Set deleteNotFound
    value=0;
    smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassertpermanent(setResult.mode=="deleteNotFound");
    cout << "15: StorageSMTest Set deleteNotFound root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Call storage state machine executor
    StorageExecutor storageExecutor(fr, poseidon, config);
    storageExecutor.execute(actionList.action);

    cout << "StorageSM_UnitTest done" << endl;
};

void StorageSM_ZeroToZeroTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config)
{
    cout << "StorageSM_ZeroToZeroTest() starting..." << endl;

    string uuid = getUUID();
    uint64_t tx = 0;

    // It should add a zero value. Test 1
    {
        cout << "StorageSM_ZeroToZeroTest() Add zero value test 1. Testing..." << endl;
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        Goldilocks::Element root[4]={fr.zero(), fr.zero(), fr.zero(), fr.zero()};
        Goldilocks::Element key[4]={fr.one(), fr.zero(), fr.zero(), fr.zero()};
        mpz_class value = 10;

        // Set insertNotFound
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(setResult.mode=="insertNotFound");

        // Set zeroToZzero
        key[0]=fr.zero();
        key[1]=fr.one();
        value=0;
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(setResult.mode=="zeroToZero");

        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_ZeroToZeroTest() Add zero value test 1. Done" << endl;
    }

    // It should add a zero value. Test 2
    {
        cout << "StorageSM_ZeroToZeroTest() Add zero value test 2. Testing..." << endl;
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element key[4]={0x23,0,0,0};
        mpz_class value = 10;

        // Set insertNotFound
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(setResult.mode=="insertNotFound");

        // Set insertNotFound
        key[0] = fr.fromU64(0x13);
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(setResult.mode=="insertFound");

        // Set zeroToZzero
        key[0] = fr.fromU64(0x73);
        value=0;
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(setResult.mode=="zeroToZero");

        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_ZeroToZeroTest() Add zero value test 2. Done" << endl;
    }

    // It should add a zero value in an empty tree
    {
        cout << "StorageSM_ZeroToZeroTest() Add zero value in an empty tree. Testing..." << endl;
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element key[4]={0,1,0,0};
        mpz_class value = 0;

        //Set zero in an empty tree;
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3]));

        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_ZeroToZeroTest() Add zero value in an empty tree. Done" << endl;
    }

    // It should add a zero value with an intermediate node as sibling
    {
        cout << "StorageSM_ZeroToZeroTest() Add a zero value with an intermediate node as sibling. Testing..." << endl;
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element key[4]={0,1,0,0};
        mpz_class value = 10;

        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];

        key[0] = fr.one();
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];

        key[2] = fr.one();
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];

        key[1] = fr.zero();
        key[2] = fr.zero();
        value=0;
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        zkassertpermanent(setResult.mode=="zeroToZero");
        zkassertpermanent(fr.equal(root[0],setResult.newRoot[0]) && fr.equal(root[1],setResult.newRoot[1]) &&
                 fr.equal(root[2],setResult.newRoot[2]) && fr.equal(root[3],setResult.newRoot[3]));

        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_ZeroToZeroTest() Add a zero value with an intermediate node as sibling. Done" << endl;
    }

    cout << "StorageSM_ZeroToZeroTest() done" << endl;
};

void StorageSM_EmptyTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config)
{
    cout << "StorageSM_EmptyTest starting..." << endl;

    SmtActionList actionList;

    // Call storage state machine executor
    StorageExecutor storageExecutor(fr, poseidon, config);
    storageExecutor.execute(actionList.action);

    cout << "StorageSM_EmptyTest done" << endl;
};

void StorageSM_UseCaseTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config)
{
    cout << "StorageSM_UseCaseTest starting..." << endl;

    string uuid = getUUID();
    uint64_t tx = 0;

    // It should add and remove an element
    {
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=1;
        scalar2key(fr, keyScalar, key);

        value=2;
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        smt.get(uuid, db, root, key, getResult);
        actionList.addGetAction(getResult);
        value = getResult.value;
        zkassertpermanent(value==2);

        value=0;
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3]));

        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_UseCaseTest test 1 done" << endl;
    }

    // It should update an element 1
    {
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element initialRoot[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=1;
        scalar2key(fr, keyScalar, key);

        value=2;
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        for (uint64_t i=0; i<4; i++) initialRoot[i] = root[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=3;
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=2;
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkassertpermanent(fr.equal(initialRoot[0], root[0]) && fr.equal(initialRoot[1], root[1]) && fr.equal(initialRoot[2], root[2]) && fr.equal(initialRoot[3], root[3]));

        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_UseCaseTest test 2 done" << endl;
    }

    // It should add a shared element 2
    {
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key1[4]={0,0,0,0};
        Goldilocks::Element key2[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=8;
        scalar2key(fr, keyScalar, key1);
        keyScalar=9;
        scalar2key(fr, keyScalar, key2);

        value=2;
        smt.set(uuid, tx, db, root, key1, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=3;
        smt.set(uuid, tx, db, root, key2, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=0;

        smt.set(uuid, tx, db, root, key1, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        smt.set(uuid, tx, db, root, key2, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3]));

        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_UseCaseTest test 3 done" << endl;
    }

    // It should add a shared element 3
    {
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key1[4]={0,0,0,0};
        Goldilocks::Element key2[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=7;
        scalar2key(fr, keyScalar, key1);
        keyScalar=15;
        scalar2key(fr, keyScalar, key2);

        value=2;
        smt.set(uuid, tx, db, root, key1, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=3;
        smt.set(uuid, tx, db, root, key2, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=0;

        smt.set(uuid, tx, db, root, key1, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        smt.set(uuid, tx, db, root, key2, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3]));

        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_UseCaseTest test 4 done" << endl;
    }

    // It should add a shared element
    {
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key1[4]={0,0,0,0};
        Goldilocks::Element key2[4]={0,0,0,0};
        Goldilocks::Element key3[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=7;
        scalar2key(fr, keyScalar, key1);
        keyScalar=15;
        scalar2key(fr, keyScalar, key2);
        keyScalar=3;
        scalar2key(fr, keyScalar, key3);

        value=107;
        smt.set(uuid, tx, db, root, key1, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=115;
        smt.set(uuid, tx, db, root, key2, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=103;
        smt.set(uuid, tx, db, root, key3, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=0;

        smt.set(uuid, tx, db, root, key1, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        smt.set(uuid, tx, db, root, key2, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        smt.set(uuid, tx, db, root, key3, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3]));


        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_UseCaseTest test 5 done" << endl;
    }

    // Add-Remove 128 elements
    {
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        for (uint64_t i = 0; i < 128; i++)
        {
            keyScalar=i;
            scalar2key(fr, keyScalar, key);
            value = i + 1000;
            smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
            actionList.addSetAction(setResult);
            for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
            zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));
        }

        value = 0;
        for (uint64_t i = 0; i < 128; i++)
        {
            keyScalar=i;
            scalar2key(fr, keyScalar, key);
            smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
            actionList.addSetAction(setResult);
            for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        }

        zkassertpermanent(fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3]));

        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_UseCaseTest test 6 done" << endl;
    }

    // Should read random
    {
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        for (uint64_t i = 0; i < 128; i++)
        {
            keyScalar = i;
            scalar2key(fr, keyScalar, key);
            value = i + 1000;
            smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
            actionList.addSetAction(setResult);
            for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
            zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));
        }

        for (uint64_t i = 0; i < 128; i++)
        {
            keyScalar = i;
            scalar2key(fr, keyScalar, key);
            smt.get(uuid, db, root, key, getResult);
            actionList.addGetAction(getResult);
            zkassertpermanent(getResult.value==(i+1000));
        }

        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_UseCaseTest test 7 done" << endl;
    }

    // It should add elements with similar keys
    {
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element expectedRoot[4]={  442750481621001142UL,
                                        12174547650106208885UL,
                                        10730437371575329832UL,
                                        4693848817100050981UL };
        mpz_class value;

        mpz_class keyScalar;

        keyScalar = 0; //0x00
        scalar2key(fr, keyScalar, key);
        value=2;
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar = 4369; //0x1111
        scalar2key(fr, keyScalar, key);
        value=2;
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar = 69905; //0x11111
        scalar2key(fr, keyScalar, key);
        value=3;
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkassertpermanent(fr.equal(root[0], expectedRoot[0]) && fr.equal(root[1], expectedRoot[1]) && fr.equal(root[2], expectedRoot[2]) && fr.equal(root[3], expectedRoot[3]));

        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_UseCaseTest test 8 done" << endl;
    }

    // It should update leaf with more than one level depth
    {
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element expectedRoot[4]={  13590506365193044307UL,
                                        13215874698458506886UL,
                                        4743455437729219665UL,
                                        1933616419393621600UL};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar.set_str("56714103185361745016746792718676985000067748055642999311525839752090945477479", 10);
        value.set_str("8163644824788514136399898658176031121905718480550577527648513153802600646339", 10);
        scalar2key(fr, keyScalar, key);
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar.set_str("980275562601266368747428591417466442501663392777380336768719359283138048405", 10);
        value.set_str("115792089237316195423570985008687907853269984665640564039457584007913129639934", 10);
        scalar2key(fr, keyScalar, key);
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));
;

        keyScalar.set_str("53001048207672216258532366725645107222481888169041567493527872624420899640125", 10);
        value.set_str("115792089237316195423570985008687907853269984665640564039457584007913129639935", 10);
        scalar2key(fr, keyScalar, key);
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar.set_str("60338373645545410525187552446039797737650319331856456703054942630761553352879", 10);
        value.set_str("7943875943875408", 10);
        scalar2key(fr, keyScalar, key);
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));


        keyScalar.set_str("56714103185361745016746792718676985000067748055642999311525839752090945477479", 10);
        value.set_str("35179347944617143021579132182092200136526168785636368258055676929581544372820", 10);
        scalar2key(fr, keyScalar, key);
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkassertpermanent(fr.equal(root[0], expectedRoot[0]) && fr.equal(root[1], expectedRoot[1]) && fr.equal(root[2], expectedRoot[2]) && fr.equal(root[3], expectedRoot[3]));

        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_UseCaseTest test 9 done" << endl;
    }

    // It should Zero to Zero with isOldZero=0
    {
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=1;
        value=2;
        scalar2key(fr, keyScalar, key);
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar=2;
        value=3;
        scalar2key(fr, keyScalar, key);
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar=0x10000;
        value=0;
        scalar2key(fr, keyScalar, key);
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];

        zkassertpermanent(setResult.mode=="zeroToZero");
        zkassertpermanent(!setResult.isOld0);

        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_UseCaseTest test 10 done" << endl;
    }

    // It should Zero to Zero with isOldZero=0
    {
        Smt smt(fr);
        Database db(fr, config);
        db.init();
        SmtActionList actionList;
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=1;
        value=2;
        scalar2key(fr, keyScalar, key);
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar=0x10000;
        value=0;
        scalar2key(fr, keyScalar, key);
        smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
        actionList.addSetAction(setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];

        zkassertpermanent(setResult.mode=="zeroToZero");
        zkassertpermanent(!setResult.isOld0);

        // Call storage state machine executor
        StorageExecutor storageExecutor(fr, poseidon, config);
        storageExecutor.execute(actionList.action);

        cout << "StorageSM_UseCaseTest test 11 done" << endl;
    }

    cout << "StorageSM_UseCaseTest done" << endl;
};

void StorageSM_GetZeroTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config)
{
    cout << "StorageSM_GetZeroTest starting..." << endl;

    string uuid = getUUID();
    uint64_t tx = 0;

    Smt smt(fr);
    Database db(fr, config);
    db.init();
    SmtActionList actionList;
    SmtSetResult setResult;
    SmtGetResult getResult;
    Goldilocks::Element root[4]={fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    Goldilocks::Element key[4]={fr.one(), fr.zero(), fr.zero(), fr.zero()};
    mpz_class value = 10;

    // Set insertNotFound
    smt.set(uuid, tx, db, root, key, value, PERSISTENCE_CACHE, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassertpermanent(setResult.mode=="insertNotFound");
    cout << "0: StorageSM_GetZeroTest Set insertNotFound root=" << fea2string(fr, root) << " mode=" << setResult.mode <<endl;

    // Get zero
    key[0]=fr.zero();
    key[1]=fr.one();
    smt.get(uuid, db, root, key, getResult);
    actionList.addGetAction(getResult);
    for (uint64_t i=0; i<4; i++) root[i] = getResult.root[i];
    cout << "1: StorageSM_GetZeroTest Get root=" << fea2string(fr, root) << " value=" << getResult.value.get_str(16) << endl;

    // Call storage state machine executor
    StorageExecutor storageExecutor(fr, poseidon, config);
    storageExecutor.execute(actionList.action);

    cout << "StorageSM_GetZeroTest done" << endl;
};