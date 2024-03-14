#include "hashdb_test.hpp"
#include <nlohmann/json.hpp>
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
#include "hashdb.hpp"
#include "hashdb_test_load.hpp"
#include "utils.hpp"

using namespace std;

/***************************************

CREATE TABLE STATE.ROOT (INT ID, R0 NUMERIC, R1 NUMERIC, R2 NUMERIC, R3 NUMERIC, HASH BYTEA);
CREATE TABLE STATE.TOTALSET (TOTAL NUMERIC);
CREATE TABLE STATE.KEY (KEY BYTEA, VALUE BYTEA);

****************************************/

const int ROUNDS = 0;
const bool CALCULATE_ROOT = true;
const bool BASIC_TEST = true;

void runHashDBTestLoad (const Config& config)
{
    string uuid = getUUID();
    uint64_t tx = 0;

    Goldilocks fr;
    HashDB client (fr, config);

    TimerStart(STATE_DB_LOAD);
    vector<thread*> threadList;
    thread* loadThread;
    for (int c=1; c<=ROUNDS; c++) {
        cout << ">>>>>>>>>> DB LOAD ROUND " << c << "/" << ROUNDS << "<<<<<<<<<<" << endl;
        for (int i=0; i<4; i++) {
            loadThread = new thread {hashDBTestLoadThread, config, i};
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

    if (CALCULATE_ROOT) {
        // Calculate root of the tree
        cout << "Calculating root of the tree..." << endl;

        Goldilocks::Element roots[4][4];
        pqxx::connection* pConnection = NULL;
        try
        {
            string uri = "postgresql://zkhashdb:zkhashdb@127.0.0.1:5532/perf_db";
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
            client.hashSave(h,c,PERSISTENCE_DATABASE,hash0);

            //Store in h the hashes for branch 01 and 11
            for (int i=0; i<4; i++) h[i] = roots[1][i]; // hash0 = hash branch 01
            for (int i=4; i<8; i++) h[i] = roots[3][i-4]; // hash1 = hash branch 11

            // Save and get the new hash for branch 1
            Goldilocks::Element hash1[4];
            client.hashSave(h,c,PERSISTENCE_DATABASE,hash1);

            //Store in h the hashes for branch 0 and 1
            for (int i=0; i<4; i++) h[i] = hash0[i]; // hash0 = hash branch 0
            for (int i=4; i<8; i++) h[i] = hash1[i-4]; // hash1 = hash branch 1

            // Save and get the root of the tree
            Goldilocks::Element newRoot[4];
            client.hashSave(h,c,PERSISTENCE_DATABASE, newRoot);

            saveRoot (fr, pConnection, 100, newRoot);

            string hashString = NormalizeToNFormat(fea2string(fr, newRoot), 64);
            cout << "NewRoot=[" << fr.toString(newRoot[0]) << "," << fr.toString(newRoot[1]) << "," << fr.toString(newRoot[2]) << "," << fr.toString(newRoot[3]) << "]" << endl;
            cout << "NewRoot hash=" << hashString << endl;

            delete pConnection;
        }
        catch (const std::exception &e)
        {
            cerr << "runHashDBTestLoad: database.exception: " << e.what() << endl;
            if (pConnection!=NULL) delete pConnection;
            return;
        }
    }

    if (BASIC_TEST) {
        cout << "Basic test running..." << endl;
        std::this_thread::sleep_for(5000ms);

        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<unsigned long long> distrib(0, std::llround(std::pow(2,64)));

        Goldilocks::Element oldRoot[4]={0,0,0,0};

        pqxx::connection* pConnection = NULL;
        try
        {
            string uri = "postgresql://zkhashdb:zkhashdb@127.0.0.1:5532/perf_db";
            pConnection = new pqxx::connection{uri};
            if (!loadRoot(fr, pConnection, 100, oldRoot)) return;
            cout << "Root=[" << fr.toString(oldRoot[0]) << "," << fr.toString(oldRoot[1]) << "," << fr.toString(oldRoot[2]) << "," << fr.toString(oldRoot[3]) << "]" << endl;
            string hashString = NormalizeToNFormat(fea2string(fr, oldRoot), 64);
            cout << "Root hash=" << hashString << endl;
        }
        catch (const std::exception &e)
        {
            cerr << "runHashDBTestLoad: database.exception: " << e.what() << endl;
            if (pConnection!=NULL) delete pConnection;
            return;
        }

        SmtSetResult setResult;
        SmtGetResult getResult;

        Goldilocks::Element key[4]={0,0,0,0};
        Goldilocks::Element root[4]={0,0,0,0};
        Goldilocks::Element newRoot[4]={0,0,0,0};
        mpz_class value;
        mpz_class keyScalar;
        uint64_t r;

        for (int k=0; k<4; k++) {
            r = distrib(gen);
            keyScalar = (keyScalar << 64) + r;
        }

        scalar2key(fr, keyScalar, key);
        //hashDB.setDBDebug(true);
        value=2;
        client.set(uuid, tx, oldRoot, key, value, PERSISTENCE_DATABASE, newRoot, &setResult, NULL);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]));

        client.get(uuid, root, key, value, &getResult, NULL);
        value = getResult.value;
        zkassertpermanent(value==2);

        value=0;
        client.set(uuid, tx, root, key, value, PERSISTENCE_DATABASE, newRoot, &setResult, NULL);
        for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
        zkassertpermanent(fr.equal(oldRoot[0],root[0]) && fr.equal(oldRoot[1],root[1]) && fr.equal(oldRoot[2],root[2]) && fr.equal(oldRoot[3],root[3]));

        cout << "Basic test done" << endl;
        delete pConnection;
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
        cerr << "saveRoot(" << sid << "): database.exception: " << e.what() << endl;
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
                cerr << "loadRoot(" << sid << "): root!=hash " << fea2string(fr,root) << "!=" <<hash.substr(2,64) << endl;
                return false;
            }
            n.commit();
            return true;
        } else {
            cerr << "loadRoot(" << sid << ": root not found" << endl;
            return false;
        }
    }
    catch (const std::exception &e)
    {
        cerr << "loadRoot(" << sid << "): database.exception: " << e.what() << endl;
        return false;
    }
}

void* hashDBTestLoadThread (const Config& config, uint8_t idBranch)
{
    const uint64_t testItems = 10000;
    const string stestItems = std::to_string(testItems);

    string uuid = getUUID();
    uint64_t tx = 0;

    Goldilocks fr;

    SmtSetResult setResult;
    SmtGetResult getResult;

    Goldilocks::Element key[4]={0,0,0,0};
    Goldilocks::Element root[4]={0,0,0,0};
    Goldilocks::Element newRoot[4]={0,0,0,0};
    mpz_class value;
    mpz_class keyScalar, rScalarOld, rScalarNew;
    string sid = std::to_string(idBranch);
    uint64_t r;
    int id = idBranch;
    struct timeval t, tc;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> distrib(0, std::llround(std::pow(2,64)));

    HashDB client (fr, config);
    client.setAutoCommit (false);

    cout << id << ": Start DB load thread (" << testItems << ")..." << endl;

    pqxx::connection* pConnection = NULL;
    try
    {
        string uri = "postgresql://zkhashdb:zkhashdb@127.0.0.1:5532/perf_db";
        pConnection = new pqxx::connection{uri};
        loadRoot(fr, pConnection, id, root);
        cout << id <<": Root=[" << fr.toString(root[0]) << "," << fr.toString(root[1]) << "," << fr.toString(root[2]) << "," << fr.toString(root[3]) << "]" << endl;
    }
    catch (const std::exception &e)
    {
        cerr << id << ": database.exception: " << e.what() << endl;
        if (pConnection!=NULL) delete pConnection;
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

        client.set(uuid, tx, root, key, i, PERSISTENCE_DATABASE, newRoot, &setResult, NULL);

        for (int j=0; j<4; j++) root[j] = setResult.newRoot[j];
        if ((i)%(testItems*0.05)==0) {
            std::this_thread::sleep_for(1ms);
            cout << id <<": " << i << "/" << testItems << " " << (float)TimeDiff(t)/1000000 << endl;
            gettimeofday(&t, NULL);
        }
    }
    client.commit(); //TODO: Poner el autocommit = false
    cout << id << ": Commit..." << endl;

    string hashString = NormalizeToNFormat(fea2string(fr, root), 64);
    cout << id << ": NewRoot=[" << fr.toString(root[0]) << "," << fr.toString(root[1]) << "," << fr.toString(root[2]) << "," << fr.toString(root[3]) << "]" << endl;
    cout << id << ": NewRoot hash=" << hashString << endl;

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
        cerr << id << ": database.exception: " << e.what() << endl;
        if (pConnection!=NULL) delete pConnection;
        return NULL;
    }

    cout << id << ": end DB load thread " << id << " -> " << (float)TimeDiff(tc)/1000000 << "s" << endl;

    return NULL;
}
