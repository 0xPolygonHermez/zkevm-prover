#include "hashdb_test.hpp"
#include <nlohmann/json.hpp>
#include <unistd.h>
#include "hashdb_interface.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include "database.hpp"
#include <thread>
#include "timer.hpp"
#include "goldilocks_base_field.hpp"
#include "hashdb_test_client.hpp"
#include "hashdb_factory.hpp"
#include "utils.hpp"

void runHashDBTestClient (const Config& config)
{
    thread* t = new thread {hashDBTestClientThread, config};
    t->join();
}

void* hashDBTestClientThread (const Config& config)
{
    TimerStart(HASHDB_TEST_CLIENT);

    cout << "HashDB test client started" << endl;
    Goldilocks fr;
    string uuid = getUUID();
    uint64_t tx = 0;
    zkresult zkr;
    Persistence persistence = PERSISTENCE_DATABASE;
    HashDBInterface* client = HashDBClientFactory::createHashDBClient(fr, config);
    uint64_t flushId, storedFlushId;

    // It should add and remove an element
    {
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element newRoot[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=1;
        scalar2key(fr, keyScalar, key);
        value=2;
        
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        zkassertpermanent(zkr==ZKR_SUCCESS);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkr = client->get(uuid, root, key, value, &getResult, NULL);
        cout << "GET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << endl;
        zkassertpermanent(zkr==ZKR_SUCCESS);
        value = getResult.value;
        zkassertpermanent(value==2);

        value=0;

        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        zkassertpermanent(zkr==ZKR_SUCCESS);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(config.hashDB64 || (fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3])));

        zkr = client->purge(uuid, root, persistence);
        zkassertpermanent(zkr==ZKR_SUCCESS);

        Goldilocks::Element consolidatedStateRoot[4];
        zkr = client->consolidateState(root, persistence, consolidatedStateRoot, flushId, storedFlushId);
        zkassertpermanent(zkr==ZKR_SUCCESS);
        
        cout << "HashDB client test 1 done" << endl;
    }

    uuid = getUUID();
    tx = 0;

    // It should update an element 1
    {
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element newRoot[4]={0,0,0,0};
        Goldilocks::Element initialRoot[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=1;
        scalar2key(fr, keyScalar, key);

        value=2;
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        for (uint64_t i=0; i<4; i++) initialRoot[i] = root[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkr = client->get(uuid, root, key, value, &getResult, NULL);
        cout << "GET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << endl;
        zkassertpermanent(zkr==ZKR_SUCCESS);
        value = getResult.value;
        zkassertpermanent(value==2);

        value=3;
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkr = client->get(uuid, root, key, value, &getResult, NULL);
        cout << "GET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << endl;
        zkassertpermanent(zkr==ZKR_SUCCESS);
        value = getResult.value;
        zkassertpermanent(value==3);

        value=2;
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));
        zkassertpermanent(config.hashDB64 || (fr.equal(initialRoot[0], root[0]) && fr.equal(initialRoot[1], root[1]) && fr.equal(initialRoot[2], root[2]) && fr.equal(initialRoot[3], root[3])));

        zkr = client->get(uuid, root, key, value, &getResult, NULL);
        cout << "GET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << endl;
        zkassertpermanent(zkr==ZKR_SUCCESS);
        value = getResult.value;
        zkassertpermanent(value==2);
        
        zkr = client->flush(uuid, fea2string(fr, root), persistence, flushId, storedFlushId);
        zkassertpermanent(zkr==ZKR_SUCCESS);

        cout << "HashDB client test 2 done" << endl;
    }

    uuid = getUUID();
    tx = 0;

    // It should add a shared element 2
    {
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key1[4]={0,0,0,0};
        Goldilocks::Element key2[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element newRoot[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=8;
        scalar2key(fr, keyScalar, key1);
        keyScalar=9;
        scalar2key(fr, keyScalar, key2);

        value=2;
        zkr = client->set(uuid, tx, root, key1, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key1=" << fea2string(fr, key1) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=3;
        zkr = client->set(uuid, tx, root, key2, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key2=" << fea2string(fr, key2) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=0;

        zkr = client->set(uuid, tx, root, key1, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key1=" << fea2string(fr, key1) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkr = client->set(uuid, tx, root, key2, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key2=" << fea2string(fr, key2) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(config.hashDB64 || (fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3])));
        
        zkr = client->flush(uuid, fea2string(fr, root), persistence, flushId, storedFlushId);
        zkassertpermanent(zkr==ZKR_SUCCESS);

        cout << "HashDB client test 3 done" << endl;
    }

    uuid = getUUID();
    tx = 0;

    // It should add a shared element 3
    {
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key1[4]={0,0,0,0};
        Goldilocks::Element key2[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element newRoot[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=7;
        scalar2key(fr, keyScalar, key1);
        keyScalar=15;
        scalar2key(fr, keyScalar, key2);

        value=2;
        zkr = client->set(uuid, tx, root, key1, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key1=" << fea2string(fr, key1) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=3;
        zkr = client->set(uuid, tx, root, key2, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key2=" << fea2string(fr, key2) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=0;

        zkr = client->set(uuid, tx, root, key1, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key1=" << fea2string(fr, key1) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkr = client->set(uuid, tx, root, key2, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key2=" << fea2string(fr, key2) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(config.hashDB64 || (fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3])));
        
        zkr = client->flush(uuid, fea2string(fr, root), persistence, flushId, storedFlushId);
        zkassertpermanent(zkr==ZKR_SUCCESS);

        cout << "HashDB client test 4 done" << endl;
    }

    uuid = getUUID();
    tx = 0;

    // It should add a shared element
    {

        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key1[4]={0,0,0,0};
        Goldilocks::Element key2[4]={0,0,0,0};
        Goldilocks::Element key3[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element newRoot[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=7;
        scalar2key(fr, keyScalar, key1);
        keyScalar=15;
        scalar2key(fr, keyScalar, key2);
        keyScalar=3;
        scalar2key(fr, keyScalar, key3);

        value=107;
        zkr = client->set(uuid, tx, root, key1, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key1=" << fea2string(fr, key1) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;

        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=115;
        zkr = client->set(uuid, tx, root, key2, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key2=" << fea2string(fr, key2) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=103;
        zkr = client->set(uuid, tx, root, key3, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key3=" << fea2string(fr, key3) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=0;

        zkr = client->set(uuid, tx, root, key1, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key1=" << fea2string(fr, key1) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkr = client->set(uuid, tx, root, key2, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key2=" << fea2string(fr, key2) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkr = client->set(uuid, tx, root, key3, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key3=" << fea2string(fr, key3) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;

        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(config.hashDB64 || (fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3])));
        
        zkr = client->flush(uuid, fea2string(fr, root), persistence, flushId, storedFlushId);
        zkassertpermanent(zkr==ZKR_SUCCESS);

        cout << "HashDB client test 5 done" << endl;
    }

    uuid = getUUID();
    tx = 0;

    // Add-Remove 128 elements
    {
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element newRoot[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        for (uint64_t i = 0; i < 128; i++)
        {
            keyScalar=i;
            scalar2key(fr, keyScalar, key);
            value = i + 1000;
            zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
            cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;

            for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
            zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));
        }

        value = 0;
        for (uint64_t i = 0; i < 128; i++)
        {
            keyScalar=i;
            scalar2key(fr, keyScalar, key);
            zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
            cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;

            for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        }

        zkassertpermanent(config.hashDB64 || (fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3])));

        zkr = client->flush(uuid, fea2string(fr, root), persistence, flushId, storedFlushId);
        zkassertpermanent(zkr==ZKR_SUCCESS);

        cout << "HashDB client test 6 done" << endl;
    }

    uuid = getUUID();
    tx = 0;

    // Should read random
    {
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element newRoot[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        for (uint64_t i = 0; i < 128; i++)
        {
            keyScalar = i;
            scalar2key(fr, keyScalar, key);
            value = i + 1000;
            zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
            cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
            for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
            zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));
        }

        for (uint64_t i = 0; i < 128; i++)
        {
            keyScalar = i;
            scalar2key(fr, keyScalar, key);
            zkr = client->get(uuid, root, key, value, &getResult, NULL);
            cout << "GET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << endl;
            zkassertpermanent(getResult.value==(i+1000));
        }
        
        zkr = client->flush(uuid, fea2string(fr, root), persistence, flushId, storedFlushId);
        zkassertpermanent(zkr==ZKR_SUCCESS);

        cout << "HashDB client test 7 done" << endl;
    }

    uuid = getUUID();
    tx = 0;

    // It should add elements with similar keys
    {
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element newRoot[4]={0,0,0,0};
        Goldilocks::Element expectedRoot[4]={  442750481621001142UL,
                                        12174547650106208885UL,
                                        10730437371575329832UL,
                                        4693848817100050981UL };
        mpz_class value;

        mpz_class keyScalar;

        keyScalar = 0; //0x00
        scalar2key(fr, keyScalar, key);
        value=2;
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar = 4369; //0x1111
        scalar2key(fr, keyScalar, key);
        value=2;
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar = 69905; //0x11111
        scalar2key(fr, keyScalar, key);
        value=3;
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkassertpermanent(config.hashDB64 || (fr.equal(expectedRoot[0], root[0]) && fr.equal(expectedRoot[1], root[1]) && fr.equal(expectedRoot[2], root[2]) && fr.equal(expectedRoot[3], root[3])));
        
        zkr = client->flush(uuid, fea2string(fr, root), persistence, flushId, storedFlushId);
        zkassertpermanent(zkr==ZKR_SUCCESS);

        cout << "HashDB client test 8 done" << endl;
    }

    uuid = getUUID();
    tx = 0;

    // It should update leaf with more than one level depth
    {

        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element newRoot[4]={0,0,0,0};
        Goldilocks::Element expectedRoot[4]={  13590506365193044307UL,
                                        13215874698458506886UL,
                                        4743455437729219665UL,
                                        1933616419393621600UL};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar.set_str("56714103185361745016746792718676985000067748055642999311525839752090945477479", 10);
        value.set_str("8163644824788514136399898658176031121905718480550577527648513153802600646339", 10);
        scalar2key(fr, keyScalar, key);
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar.set_str("980275562601266368747428591417466442501663392777380336768719359283138048405", 10);
        value.set_str("115792089237316195423570985008687907853269984665640564039457584007913129639934", 10);
        scalar2key(fr, keyScalar, key);
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar.set_str("53001048207672216258532366725645107222481888169041567493527872624420899640125", 10);
        value.set_str("115792089237316195423570985008687907853269984665640564039457584007913129639935", 10);
        scalar2key(fr, keyScalar, key);
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar.set_str("60338373645545410525187552446039797737650319331856456703054942630761553352879", 10);
        value.set_str("7943875943875408", 10);
        scalar2key(fr, keyScalar, key);
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar.set_str("56714103185361745016746792718676985000067748055642999311525839752090945477479", 10);
        value.set_str("35179347944617143021579132182092200136526168785636368258055676929581544372820", 10);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        scalar2key(fr, keyScalar, key);
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkassertpermanent(config.hashDB64 || (fr.equal(expectedRoot[0], root[0]) && fr.equal(expectedRoot[1], root[1]) && fr.equal(expectedRoot[2], root[2]) && fr.equal(expectedRoot[3], root[3])));
        
        zkr = client->flush(uuid, fea2string(fr, root), persistence, flushId, storedFlushId);
        zkassertpermanent(zkr==ZKR_SUCCESS);

        cout << "HashDB client test 9 done" << endl;
    }

    uuid = getUUID();
    tx = 0;

    // It should Zero to Zero with isOldZero=0
    {
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element newRoot[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=1;
        value=2;
        scalar2key(fr, keyScalar, key);
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar=2;
        value=3;
        scalar2key(fr, keyScalar, key);
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar=0x10000;
        value=0;
        scalar2key(fr, keyScalar, key);
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];

        zkassertpermanent(config.hashDB64 || (setResult.mode=="zeroToZero"));
        zkassertpermanent(config.hashDB64 || !setResult.isOld0);
        
        zkr = client->flush(uuid, fea2string(fr, root), persistence, flushId, storedFlushId);
        zkassertpermanent(zkr==ZKR_SUCCESS);

        cout << "HashDB client test 10 done" << endl;
    }

    uuid = getUUID();
    tx = 0;

    // It should Zero to Zero with isOldZero=0
    {

        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element newRoot[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=1;
        value=2;
        scalar2key(fr, keyScalar, key);
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));


        keyScalar=0x10000;
        value=0;
        scalar2key(fr, keyScalar, key);
        zkr = client->set(uuid, tx, root, key, value, persistence, newRoot, &setResult, NULL);
        cout << "SET zkr=" << zkresult2string(zkr) << " root=" << fea2string(fr, root) << " key=" << fea2string(fr, key) << " value=" << value.get_str() << " newRoot=" << fea2string(fr, newRoot) << endl;
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];

        zkassertpermanent(config.hashDB64 || (setResult.mode=="zeroToZero"));
        zkassertpermanent(config.hashDB64 || !setResult.isOld0);
        
        zkr = client->flush(uuid, fea2string(fr, root), persistence, flushId, storedFlushId);
        zkassertpermanent(zkr==ZKR_SUCCESS);

        cout << "HashDB client test 11 done" << endl;
    }

    uuid = getUUID();
    tx = 0;

    // It should add program data (setProgram) and retrieve it (getProgram)
    {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<unsigned long long> distrib(0, std::llround(std::pow(2,64)));

        Goldilocks::Element key[4]={0,0,0,0};

        for (int k=0; k<4; k++) {
            fr.fromU64(key[k], distrib(gen));
        }

        std::vector<uint8_t> in, out;
        for (uint8_t i=0; i<128; i++) {
            in.push_back(i);
        }

        zkr = client->setProgram(key, in, true);
        zkr = client->getProgram(key, out, NULL);

        for (uint8_t i=0; i<128; i++) {
            zkassertpermanent(in[i]==out[i]);
        }
        
        //zkr = client->flush(uuid, fea2string(fr, root), persistence, flushId, storedFlushId);
        //zkassertpermanent(zkr==ZKR_SUCCESS);

        cout << "HashDB client test 12 done" << endl;
    }

    sleep(1);

    delete client;

    cout << "HashDB test client done" << endl;

    TimerStopAndLog(HASHDB_TEST_CLIENT);

    return NULL;
}
