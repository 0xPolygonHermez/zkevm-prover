#include "statedb_test.hpp"
#include <nlohmann/json.hpp>
#include "statedb_client.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include "database.hpp"
#include <thread>
#include "timer.hpp"
#include "poseidon_opt/poseidon_goldilocks.hpp"
#include "statedb.hpp"

using namespace std;

/***************************************

CREATE TABLE STATE.ROOT (INT ID, R0 NUMERIC, R1 NUMERIC, R2 NUMERIC, R3 NUMERIC, HASH BYTEA);
CREATE TABLE STATE.TOTALSET (TOTAL NUMERIC);
CREATE TABLE STATE.KEY (KEY BYTEA, VALUE BYTEA); 
 
****************************************/

void runStateDBLoad (const Config& config)
{
    const int rounds = 0;
    const bool calculateRoot = true;
    const bool basicTest = true;
    const bool perfTest = false;

    Goldilocks fr;
    StateDB stateDB (fr, config, true, false);    

    TimerStart(STATE_DB_LOAD);
    vector<thread*> threadList;
    thread* loadThread;
    for (int c=1; c<=rounds; c++) {
        cout << ">>>>>>>>>> DB LOAD ROUND " << c << "/" << rounds << "<<<<<<<<<<" << endl;
        for (int i=0; i<4; i++) {
            loadThread = new thread {stateDBLoadThread, config, i};
            threadList.push_back(loadThread);
        }
        for (long unsigned int i=0; i<threadList.size(); i++) {
            loadThread = threadList.at(i);
            loadThread->join();
            delete loadThread;
        }
        threadList.clear();
    }
    TimerStopAndLog(STATE_DB_LOAD);

    if (calculateRoot) {   
        // Calculate root of the tree
        Goldilocks::Element roots[4][4];
        pqxx::connection * pConnection;
        try
        {
            string uri = "postgresql://zkstatedb:zkstatedb@127.0.0.1:5532/perf_db";
            pConnection = new pqxx::connection{uri};

            for (int r=0; r<4; r++) {
                if (!loadRoot(fr, pConnection, r, roots[r])) return;
            }

            // Capacity = 0, 0, 0, 0
            Goldilocks::Element c[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};

            //Store in h the hashes for branch 00 and 10
            Goldilocks::Element h[8];
            for (int i=0; i<4; i++) h[i] = roots[0][i]; // hash0 = hash branch 00
            for (int i=4; i<8; i++) h[i] = roots[2][i-4]; // hash1 = hash branch 10

            // Save and get the new hash for branch 0
            Goldilocks::Element hash0[4]; 
            stateDB.hashSave(h,c,true,hash0);

            //Store in h the hashes for branch 01 and 11
            for (int i=0; i<4; i++) h[i] = roots[1][i]; // hash0 = hash branch 01
            for (int i=4; i<8; i++) h[i] = roots[3][i-4]; // hash1 = hash branch 11

            // Save and get the new hash for branch 1
            Goldilocks::Element hash1[4]; 
            stateDB.hashSave(h,c,true,hash1);

            //Store in h the hashes for branch 0 and 1
            for (int i=0; i<4; i++) h[i] = hash0[i]; // hash0 = hash branch 0
            for (int i=4; i<8; i++) h[i] = hash1[i-4]; // hash1 = hash branch 1

            // Save and get the root of the tree
            Goldilocks::Element newRoot[4]; 
            stateDB.hashSave(h,c,true, newRoot);

            saveRoot (fr, pConnection, 100, newRoot);

            string hashString = NormalizeToNFormat(fea2string(fr, newRoot), 64);
            cout << "Calculate Root:: NewRoot=[" << fr.toString(newRoot[0]) << "," << fr.toString(newRoot[1]) << "," << fr.toString(newRoot[2]) << "," << fr.toString(newRoot[3]) << "]" << endl;
            cout << "Calculate Root:: NewRoot hash=" << hashString << endl;

            delete pConnection;
        }
        catch (const std::exception &e)
        {
            cerr << "Calculate Root:: Error: stateDBLoadThread:database:: exception: " << e.what() << endl;
            delete pConnection;
            return;
        }  
    }
    
    if (basicTest) {
        cout << "Basic test running..." << endl;
        std::this_thread::sleep_for(5000ms);

        std::random_device rd;  
        std::mt19937_64 gen(rd()); 
        std::uniform_int_distribution<unsigned long long> distrib(0, std::llround(std::pow(2,64)));

        Goldilocks::Element oldRoot[4]={0,0,0,0};

        pqxx::connection *pConnection;
        try
        {
            string uri = "postgresql://zkstatedb:zkstatedb@127.0.0.1:5532/perf_db";
            pConnection = new pqxx::connection{uri};
            if (!loadRoot(fr, pConnection, 100, oldRoot)) return;
            cout << "stateDBPerfTestThread:: Root=[" << fr.toString(oldRoot[0]) << "," << fr.toString(oldRoot[1]) << "," << fr.toString(oldRoot[2]) << "," << fr.toString(oldRoot[3]) << "]" << endl;
            string hashString = NormalizeToNFormat(fea2string(fr, oldRoot), 64);
            cout << "stateDBPerfTestThread:: Root hash=" << hashString << endl;
        }
        catch (const std::exception &e)
        {
            cerr << "stateDBPerfTestThread:: Error: stateDBLoadThread:database:: exception: " << e.what() << endl;
            delete pConnection;
            return;
        } 

        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;
        uint64_t r;

        for (int k=0; k<4; k++) {
            r = distrib(gen); 
            keyScalar = (keyScalar << 64) + r;
        }

        scalar2key(fr, keyScalar, key);
        //·stateDB.setDBDebug(true);
        value=2;        
        stateDB.set(oldRoot, key, value, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        stateDB.get(root, key, getResult);
        value = getResult.value;
        zkassert(value==2);

        value=0;
        stateDB.set(root, key, value, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(fr.equal(oldRoot[0],root[0]) && fr.equal(oldRoot[1],root[1]) && fr.equal(oldRoot[2],root[2]) && fr.equal(oldRoot[3],root[3]));

        cout << "Basic test done" << endl;
        delete pConnection;
    }

    if (perfTest) {
        StateDBClient stateDBClient(fr, config);
        runStateDBPerfTest(config, &stateDBClient);
    }
}

bool saveRoot (Goldilocks &fr, pqxx::connection *pConnection, int id, Goldilocks::Element (&root)[4])
{
    string sid = std::to_string(id);

    try
    {
        pqxx::work w(*pConnection);
        string query = "DELETE FROM STATE.ROOT WHERE id=" + sid + ";";
        pqxx::result res = w.exec(query);

        string hashString = NormalizeToNFormat(fea2string(fr, root), 64);

        query = "INSERT INTO STATE.ROOT ( id, r0, r1, r2, r3, hash ) VALUES (" + sid + ", " + 
            fr.toString(root[0]) + "," + fr.toString(root[1]) + "," + fr.toString(root[2]) + "," + fr.toString(root[3]) + "," +
            "E\'\\\\x" + hashString + "');";
        res = w.exec(query);

        w.commit();

        return true;
    }
    catch (const std::exception &e)
    {
        cerr << "saveRoot " << sid << ":: Error: stateDBLoadThread:database:: exception: " << e.what() << endl;
        return false;
    }           
}

bool loadRoot (Goldilocks &fr, pqxx::connection *pConnection, int id, Goldilocks::Element (&root)[4])
{
    string sid = std::to_string(id);
    try
    {
        pqxx::nontransaction n(*pConnection);

        string query = "SELECT r0, r1, r2, r3, hash FROM state.root WHERE id="+ sid +";";
        pqxx::result res = n.exec(query);
        if (res.size()==1) {
            pqxx::row row = res[0];
            for (int i=0; i<4; i++) {
                root[i] = fr.fromString(row[i].c_str(),10);         
            }
            string hash = row[4].c_str();
            if (NormalizeToNFormat(fea2string(fr,root),64)!=hash.substr(2,64)) {
                cerr << "loadRoot " << sid << ":: Error: loaded root!=hash " << fea2string(fr,root) << "!=" <<hash.substr(2,64) << endl;
                return false;
            }
            n.commit();
            return true;
        } else {
            cerr << "loadRoot " << sid << ":: root not found" << endl;
            return false;
        }
    }
    catch (const std::exception &e)
    {
        cerr << "loadRoot " << sid << ":: Error: stateDBLoadThread:database:: exception: " << e.what() << endl;
        return false;
    }           
}

void* stateDBLoadThread (const Config& config, uint8_t idBranch)
{
    const uint64_t testItems = 10000;
    const string stestItems = std::to_string(testItems);

    Goldilocks fr;

    SmtSetResult setResult;
    SmtGetResult getResult;

    Goldilocks::Element key[4]={0,0,0,0};
    Goldilocks::Element root[4]={0,0,0,0};
    mpz_class value;
    mpz_class keyScalar, rScalarOld, rScalarNew;
    string sid = std::to_string(idBranch);
    uint64_t r;
    int id = idBranch;
    struct timeval t, tc;

    std::random_device rd;  
    std::mt19937_64 gen(rd()); 
    std::uniform_int_distribution<unsigned long long> distrib(0, std::llround(std::pow(2,64)));
 
    StateDB stateDB (fr, config, false, false);

    cout << id << ":: START DB thread (" << testItems << ")..." << endl;

    pqxx::connection *pConnection;
    try
    {
        string uri = "postgresql://zkstatedb:zkstatedb@127.0.0.1:5532/perf_db";
        pConnection = new pqxx::connection{uri};
        loadRoot(fr, pConnection, id, root);
        cout << id <<":: Root=[" << fr.toString(root[0]) << "," << fr.toString(root[1]) << "," << fr.toString(root[2]) << "," << fr.toString(root[3]) << "]" << endl;
    }
    catch (const std::exception &e)
    {
        cerr << id << ":: Error: stateDBLoadThread:database:: exception: " << e.what() << endl;
        delete pConnection;
        return NULL;
    }        

    gettimeofday(&tc, NULL);
    gettimeofday(&t, NULL);
    for (mpz_class i=100; i<=testItems+100; i++) {
        keyScalar = 0;
        for (int k=0; k<4; k++) {
            r = distrib(gen); 
            if (k==0) r = (r & 0xFFFFFFFFFFFFFFFE) | (idBranch & 0x01);
            if (k==1) r = (r & 0xFFFFFFFFFFFFFFFE) | ((idBranch >> 1) & 0x01);
            keyScalar = (keyScalar << 64) + r;
        }
        scalar2key(fr, keyScalar, key);

        stateDB.set(root, key, i, true, setResult);

        for (int j=0; j<4; j++) root[j] = setResult.newRoot[j];
        if ((i)%(testItems*0.05)==0) {
            std::this_thread::sleep_for(1ms);
            cout << id <<":: " << i << "/" << testItems << " " << (float)TimeDiff(t)/1000000 << endl;
            gettimeofday(&t, NULL);
        }
    }
    stateDB.commit();
    cout << id << ":: Commit..." << endl;

    string hashString = NormalizeToNFormat(fea2string(fr, root), 64);
    cout << id << ":: NewRoot=[" << fr.toString(root[0]) << "," << fr.toString(root[1]) << "," << fr.toString(root[2]) << "," << fr.toString(root[3]) << "]" << endl;
    cout << id << ":: NewRoot hash=" << hashString << endl;
    
    try
    {
        saveRoot(fr, pConnection, id, root);

        pqxx::work w(*pConnection);
        string query = "UPDATE STATE.TOTALSET SET total = total + " + stestItems + ";";
        pqxx::result res = w.exec(query);

        // Commit your transaction
        w.commit();

        delete pConnection;
    }
    catch (const std::exception &e)
    {
        cerr << id << ":: Error: stateDBLoadThread:database:: exception: " << e.what() << endl;
        delete pConnection;
        return NULL;
    }

    cout << id << ":: END DB thread " << id << " -> " << (float)TimeDiff(tc)/1000000 << "s" << endl;

    return NULL;
}

void runStateDBPerfTest (const Config& config, StateDBClient* client)
{
    new thread {stateDBPerfTestThread, config, client};
}

void* stateDBPerfTestThread (const Config& config, StateDBClient* pClient)
{
    //#define useGRPC 
    //#define perfSET
    #define perfGET

    const uint64_t setCount = 50000;

    cout << "StateDB performance test started" << endl;
    Goldilocks fr;

    bool persistent = true;

    std::random_device rd;  
    std::mt19937_64 gen(rd()); 
    std::uniform_int_distribution<unsigned long long> distrib(0, std::llround(std::pow(2,64)));

    Goldilocks::Element root[4]={0,0,0,0};

    pqxx::connection *pConnection;
    try
    {
        string uri = "postgresql://zkstatedb:zkstatedb@127.0.0.1:5532/perf_db";
        pConnection = new pqxx::connection{uri};
        if (!loadRoot(fr, pConnection, 100, root)) return NULL;
        cout << "stateDBPerfTestThread:: Root=[" << fr.toString(root[0]) << "," << fr.toString(root[1]) << "," << fr.toString(root[2]) << "," << fr.toString(root[3]) << "]" << endl;
    }
    catch (const std::exception &e)
    {
        cerr << "stateDBPerfTestThread:: Error: stateDBLoadThread:database:: exception: " << e.what() << endl;
        delete pConnection;
        return NULL;
    } 

    StateDB stateDB (fr, config, true, false);

    SmtSetResult setResult;
    SmtGetResult getResult;

    Goldilocks::Element key[4]={0,0,0,0};
    mpz_class value;
    mpz_class keyScalar;
    uint64_t r;

    #if defined(perfSET)
        #if defined(useGRPC)
        cout << "Executing " << setCount << " SET operations using GRPC client..." << endl;
        #else
        cout << "Executing " << setCount << " SET operations using direct client..." << endl;
        #endif
    #elif defined(perfGET)
        #if defined(useGRPC)
        cout << "Executing " << setCount << " GET operations using GRPC client..." << endl;
        #else
        cout << "Executing " << setCount << " GET operations using direct client..." << endl;
        #endif
    #endif

    struct timeval tset;
    gettimeofday(&tset, NULL);
    for (uint64_t i=1; i<=setCount; i++) {
        keyScalar = 0;
        for (int k=0; k<4; k++) {
            r = distrib(gen); 
            keyScalar = (keyScalar << 64) + r;
        }
    
        scalar2key(fr, keyScalar, key);
        value=i;

        #if defined(perfSET)
            #if defined(useGRPC)
            pClient->set(root, key, value, persistent, true, setResult);
            #else
            stateDB.set(root, key, value, persistent, setResult);
            #endif    
            for (int j=0; j<4; j++) root[j] = setResult.newRoot[j];
        #elif defined(perfGET)
            #if defined(useGRPC)
            pClient->get(root, key, true, getResult);
            #else
            stateDB.get(root, key, getResult);
            #endif    
        #endif
    }
    uint64_t totalTimeUS = TimeDiff(tset);
    
    cout << "Total Execution time (us): " << totalTimeUS << endl;
    #if defined(perfSET)    
    cout << "Time per SET: " << totalTimeUS/setCount << "us" << endl;
    cout << "SETs per second: " << (float)1000000/(totalTimeUS/setCount) << endl;
    cout << "Saving new root..." << endl;
    saveRoot (fr, pConnection, 100, root);
    #elif defined(perfGET)
    cout << "Time per GET: " << totalTimeUS/setCount << "us" << endl;
    cout << "GETs per second: " << (float)1000000/(totalTimeUS/setCount) << endl;
    #endif

    cout << "StateDB performance test done" << endl;
    delete pConnection;
    return NULL;
}

void runStateDBTest (StateDBClient* client)
{
    new thread {stateDBTestThread, client};
}

void* stateDBTestThread (StateDBClient* pClient)
{
    cout << "StateDB test client started" << endl;
    Goldilocks fr;
    string uuid;

    bool persistent = true;

    //filldbMT(pClient);
    //filldbRandom();
    /*vector<thread*> threadList;
    thread* fillThread;
    const int loadCount = 100/5; //Space needed = 0.5GB per 1M hashes. 50GB per 100M hashes
    TimerStart(FILL_DATABASE_RANDOM);
    for (int c=0; c<loadCount; c++) { 
        cout << "Database fill count process " << c << endl;
        TimerStart(FILL_DATABASE_RANDOM_5M);
        for (int i=1; i<=10; i++) { // 5M
            fillThread = new thread {filldbRandom, i, 500000};
            threadList.push_back(fillThread);
        }
        for (long unsigned int i=0; i<threadList.size(); i++) {
            fillThread = threadList.at(i);
            fillThread->join();
            delete fillThread;
        }
        threadList.clear();
        TimerStopAndLog(FILL_DATABASE_RANDOM_5M);
    }
    TimerStopAndLog(FILL_DATABASE_RANDOM);

    return NULL;*/

    // It should add and remove an element
    {
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=1;
        scalar2key(fr, keyScalar, key);

        value=2;
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        pClient->get(root, key, true, getResult);
        value = getResult.value;
        zkassert(value==2);

        value=0;
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3]));

        cout << "StateDB client test 1 done" << endl;
    }

    // It should update an element 1
    {
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
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        for (uint64_t i=0; i<4; i++) initialRoot[i] = root[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=3;
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=2;
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkassert(fr.equal(initialRoot[0], root[0]) && fr.equal(initialRoot[1], root[1]) && fr.equal(initialRoot[2], root[2]) && fr.equal(initialRoot[3], root[3]));

        cout << "StateDB client test 2 done" << endl;
    }

    // It should add a shared element 2
    {
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
        pClient->set(root, key1, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=3;
        pClient->set(root, key2, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=0;

        pClient->set(root, key1, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        pClient->set(root, key2, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3]));

        cout << "StateDB client test 3 done" << endl;
    }

    // It should add a shared element 3
    {
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
        pClient->set(root, key1, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=3;
        pClient->set(root, key2, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=0;

        pClient->set(root, key1, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        pClient->set(root, key2, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3]));

        cout << "StateDB client test 4 done" << endl;
    }

    // It should add a shared element
    {

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
        pClient->set(root, key1, value, persistent, true, setResult);
        
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=115;
        pClient->set(root, key2, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=103;
        pClient->set(root, key3, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        value=0;

        pClient->set(root, key1, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        pClient->set(root, key2, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        pClient->set(root, key3, value, persistent, true, setResult);
        
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3]));

        cout << "StateDB client test 5 done" << endl;
    }

    // Add-Remove 128 elements
    {
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
            pClient->set(root, key, value, persistent, true, setResult);
            
            for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
            zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));
        }

        value = 0;
        for (uint64_t i = 0; i < 128; i++)
        {
            keyScalar=i;
            scalar2key(fr, keyScalar, key);
            pClient->set(root, key, value, persistent, true, setResult);
            
            for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        }

        zkassert(fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3]));

        cout << "StateDB client test 6 done" << endl;
    }

    // Should read random
    {
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
            pClient->set(root, key, value, persistent, true, setResult);
            for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
            zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));
        }

        for (uint64_t i = 0; i < 128; i++)
        {
            keyScalar = i;
            scalar2key(fr, keyScalar, key);
            pClient->get(root, key, true, getResult);
            zkassert(getResult.value==(i+1000));
        }

        cout << "StateDB client test 7 done" << endl;
    }

    // It should add elements with similar keys
    {
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
        pClient->set(root, key, value, persistent, true, setResult);  
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar = 4369; //0x1111
        scalar2key(fr, keyScalar, key);
        value=2;
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar = 69905; //0x11111
        scalar2key(fr, keyScalar, key);
        value=3;
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkassert(fr.equal(expectedRoot[0], root[0]) && fr.equal(expectedRoot[1], root[1]) && fr.equal(expectedRoot[2], root[2]) && fr.equal(expectedRoot[3], root[3]));

        cout << "StateDB client test 8 done" << endl;
    }

    // It should update leaf with more than one level depth
    {

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
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar.set_str("980275562601266368747428591417466442501663392777380336768719359283138048405", 10);
        value.set_str("115792089237316195423570985008687907853269984665640564039457584007913129639934", 10);
        scalar2key(fr, keyScalar, key);
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar.set_str("53001048207672216258532366725645107222481888169041567493527872624420899640125", 10);
        value.set_str("115792089237316195423570985008687907853269984665640564039457584007913129639935", 10);
        scalar2key(fr, keyScalar, key);
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar.set_str("60338373645545410525187552446039797737650319331856456703054942630761553352879", 10);
        value.set_str("7943875943875408", 10);
        scalar2key(fr, keyScalar, key);
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar.set_str("56714103185361745016746792718676985000067748055642999311525839752090945477479", 10);
        value.set_str("35179347944617143021579132182092200136526168785636368258055676929581544372820", 10);
        scalar2key(fr, keyScalar, key);
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        zkassert(fr.equal(expectedRoot[0], root[0]) && fr.equal(expectedRoot[1], root[1]) && fr.equal(expectedRoot[2], root[2]) && fr.equal(expectedRoot[3], root[3]));

        cout << "StateDB client test 9 done" << endl;
    }

    // It should Zero to Zero with isOldZero=0
    {
        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=1;
        value=2;
        scalar2key(fr, keyScalar, key);
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar=2;
        value=3;
        scalar2key(fr, keyScalar, key);
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        keyScalar=0x10000;
        value=0;
        scalar2key(fr, keyScalar, key);
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];

        zkassert(setResult.mode=="zeroToZero");
        zkassert(!setResult.isOld0);

        cout << "StateDB client test 10 done" << endl;
    }

    // It should Zero to Zero with isOldZero=0
    {

        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;

        keyScalar=1;
        value=2;
        scalar2key(fr, keyScalar, key);
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassert(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));


        keyScalar=0x10000;
        value=0;
        scalar2key(fr, keyScalar, key);
        pClient->set(root, key, value, persistent, true, setResult);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];

        zkassert(setResult.mode=="zeroToZero");
        zkassert(!setResult.isOld0);

        cout << "StateDB client test 11 done" << endl;
    }

    cout << "StateDB client done" << endl;

    return NULL;
}

void* filldbRandom (int id, uint64_t testItems) {
    struct timeval t, tc;

    mpz_class value;
    mpz_class keyScalar;
    uint64_t r;
    string keyString;
    string valueString;

    std::random_device rd;  
    std::mt19937_64 gen(rd()); 
    std::uniform_int_distribution<unsigned long long> distrib(0, std::llround(std::pow(2,64)));

    pqxx::connection * pConnection;

    try
    {
        // Build the remote database URI
        string uri = "postgresql://zkstatedb:zkstatedb@127.0.0.1:5532/perf_db";

        // Create the connection
        pConnection = new pqxx::connection{uri};

        cout << id << ":: Filling database with random hashes (" << testItems << ")..." << endl;
        gettimeofday(&t, NULL);
        gettimeofday(&tc, NULL);
        pqxx::work w(*pConnection);
        for (mpz_class i=1; i<=testItems; i++) {
            keyScalar = 0;
            for (int k=0; k<4; k++) {
                r = distrib(gen);
                keyScalar = (keyScalar << 64) + r;
            }
            keyString = NormalizeToNFormat(keyScalar.get_str(16),64);
            valueString = "";
            for (int j=0; j<12; j++) valueString += keyString;

            string query = "INSERT INTO state.merkletree ( hash, data ) VALUES ( E\'\\\\x" + keyString + "\', E\'\\\\x" + valueString + "\' ) "+
                        "ON CONFLICT (hash) DO NOTHING;";

            w.exec(query);

            if ((i)%(testItems*0.05)==0) {
                cout << id <<":: " << i << "/" << testItems << " " << (float)TimeDiff(t)/1000000 << endl;
                gettimeofday(&t, NULL);
            }
        }
        w.commit();
        cout << id << ":: Fill database with random hashes completed " << (float)TimeDiff(tc)/1000000 << "s" << endl;
        delete pConnection;
    }
    catch (const std::exception &e)
    {
        cerr << id << ":: Error: filldb:database:: exception: " << e.what() << endl;
        delete pConnection;
        return NULL;
    }
    return NULL;
}
