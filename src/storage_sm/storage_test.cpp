#include "storage_test.hpp"
#include "storage.hpp"
#include "ff/ff.hpp"
#include "smt.hpp"
#include "smt_action.hpp"
#include "smt_action_list.hpp"
#include "scalar.hpp"

void StorageSMTest (FiniteField &fr, Poseidon_goldilocks &poseidon, Config &config)
{
    cout << "StorageSMTest starting..." << endl;

    Smt smt(fr);
    Database db(fr);
    db.init(config);
    SmtActionList actionList;
    SmtGetResult getResult;
    SmtSetResult setResult;
    FieldElement root[4]={0,0,0,0};
    FieldElement key[4]={1,0,0,0};
    mpz_class value = 10;
    // Get zero
    smt.get(db, root, key, getResult);
    actionList.addGetAction(getResult);
    cout << "StorageSMTest Get zero value=" << getResult.value.get_str(16) << endl;

    // Set insertNotFound
    smt.set(db, root, key, value, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassert(setResult.mode=="insertNotFound");
    cout << "StorageSMTest Set insertNotFound root=" << fea2string(fr, root) << " mode=" << setResult.mode <<endl;

    // Get non zero
    smt.get(db, root, key, getResult);
    actionList.addGetAction(getResult);
    cout << "StorageSMTest Get nonZero value=" << getResult.value.get_str(16) << endl;

    // Set deleteLast
    value=0;
    smt.set(db, root, key, value, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassert(setResult.mode=="deleteLast");
    cout << "StorageSMTest Set deleteLast root=" << fea2string(fr, root) << " mode=" << setResult.mode <<endl;

    // Set insertNotFound
    value=10;
    smt.set(db, root, key, value, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    cout << "StorageSMTest Set insertNotFound root=" << fea2string(fr, root) << " mode=" << setResult.mode <<endl;

    // Set update
    value=20;
    smt.set(db, root, key, value, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassert(setResult.mode=="update");
    cout << "StorageSMTest Set update root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Get non zero
    smt.get(db, root, key, getResult);
    actionList.addGetAction(getResult);
    cout << "StorageSMTest Get nonZero value=" << getResult.value.get_str(16) << endl;

    // Set insertFound
    key[0]=3;
    value=20;
    smt.set(db, root, key, value, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassert(setResult.mode=="insertFound");
    cout << "StorageSMTest Set insertFound root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Get non zero
    smt.get(db, root, key, getResult);
    actionList.addGetAction(getResult);
    cout << "StorageSMTest Get nonZero value=" << getResult.value.get_str(16) << endl;

    // Set deleteFound
    value=0;
    smt.set(db, root, key, value, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassert(setResult.mode=="deleteFound");
    cout << "StorageSMTest Set deleteFound root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Get zero
    smt.get(db, root, key, getResult);
    actionList.addGetAction(getResult);
    cout << "StorageSMTest Get zero value=" << getResult.value.get_str(16) << endl;

    // Set zeroToZzero
    value=0;
    smt.set(db, root, key, value, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassert(setResult.mode=="zeroToZero");
    cout << "StorageSMTest Set zeroToZero root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Set insertFound
    value=40;
    smt.set(db, root, key, value, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassert(setResult.mode=="insertFound");
    cout << "StorageSMTest Set insertFound root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Set insertNotFound
    key[0]=0;
    key[1]=1;
    value=30;
    smt.set(db, root, key, value, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassert(setResult.mode=="insertNotFound");
    cout << "StorageSMTest Set insertNotFound root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Set deleteNotFound
    value=0;
    smt.set(db, root, key, value, setResult);
    actionList.addSetAction(setResult);
    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
    zkassert(setResult.mode=="deleteNotFound");
    cout << "StorageSMTest Set deleteNotFound root=" << fea2string(fr, root) << " mode=" << setResult.mode << endl;

    // Call storage state machine executor
    StorageExecutor storageExecutor(fr, poseidon, config);
    storageExecutor.execute(actionList.action);

    cout << "StorageSMTest done" << endl;
};