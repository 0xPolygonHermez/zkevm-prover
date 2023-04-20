#include <iostream>
#include <thread>
#include "database.hpp"
#include "config.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include "definitions.hpp"
#include "zkresult.hpp"
#include "utils.hpp"
#include <unistd.h>
#include "timer.hpp"
#include "statedb_singleton.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

#ifdef DATABASE_USE_CACHE

// Create static Database::dbMTCache and DatabaseCacheProgram objects
// This will be used to store DB records in memory and it will be shared for all the instances of Database class
// DatabaseCacheMT and DatabaseCacheProgram classes are thread-safe
DatabaseMTCache Database::dbMTCache;
DatabaseProgramCache Database::dbProgramCache;

string Database::dbStateRootKey("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"); // 64 f's

#endif

// Helper functions
string removeBSXIfExists(string s) {return ((s.at(0) == '\\') && (s.at(1) == 'x')) ? s.substr(2) : s;}

// Database class implementation
void Database::init(void)
{
    // Check that it has not been initialized before
    if (bInitialized)
    {
        zklog.error("Database::init() called when already initialized");
        exitProcess();
    }

#ifdef DATABASE_USE_CACHE
    useDBMTCache = dbMTCache.enabled();
    useDBProgramCache = dbProgramCache.enabled();
#endif

    // Configure the server, if configuration is provided
    if (config.databaseURL != "local")
    {
        initRemote();
        useRemoteDB = true;
    } else useRemoteDB = false;

    // Mark the database as initialized
    bInitialized = true;
}

zkresult Database::read(const string &_key, vector<Goldilocks::Element> &value, DatabaseMap *dbReadLog, const bool update, const vector<uint64_t> *keys, uint64_t level)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database::read() called uninitialized");
        exitProcess();
    }

    struct timeval t;
    if (dbReadLog != NULL) gettimeofday(&t, NULL);

    zkresult r = ZKR_UNSPECIFIED;

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

#ifdef DATABASE_USE_CACHE
    // If the key is found in local database (cached) simply return it
    if (useDBMTCache)
    {
        // If the key is present in the cache, get its value from there
        if (Database::dbMTCache.find(key, value))
        {
            // Add to the read log
            if (dbReadLog != NULL) dbReadLog->add(key, value, true, TimeDiff(t));

            r = ZKR_SUCCESS;
        }
        // If get tree is configured, read the tree from the branch (key hash) to the leaf (keys since level)
        else if (config.dbGetTree && (keys != NULL))
        {
            // Get the tree
            uint64_t numberOfFields;
            r = readTreeRemote(key, keys, level, numberOfFields);

            // Add to the read log, and restart the timer
            if (dbReadLog != NULL)
            {
                dbReadLog->addGetTree(TimeDiff(t), numberOfFields);
                gettimeofday(&t, NULL);
            }

            // If succeeded, now the value should be present in the cache
            if ( r == ZKR_SUCCESS)
            {
                if (Database::dbMTCache.find(key, value))
                {
                    // Add to the read log
                    if (dbReadLog != NULL) dbReadLog->add(key, value, true, TimeDiff(t));

                    r = ZKR_SUCCESS;
                }
                else
                {
                    zklog.warning("Database::read() called readTreeRemote() but key=" + key + " is not present");
                    r = ZKR_UNSPECIFIED;
                }
            }
        }
    }
#endif
    if (useRemoteDB && (r == ZKR_UNSPECIFIED))
    {
        // If multi write is enabled, flush pending data, since some previously written keys
        // could be in the multi write string but flushed from the cache
        if (config.dbMultiWrite)
        {
            flush();
        }

        // Otherwise, read it remotelly
        string sData;
        r = readRemote(false, key, sData);
        if (r == ZKR_SUCCESS)
        {
            string2fea(fr, sData, value);

#ifdef DATABASE_USE_CACHE
            // Store it locally to avoid any future remote access for this key
            if (useDBMTCache) Database::dbMTCache.add(key, value, update);
#endif

            // Add to the read log
            if (dbReadLog != NULL) dbReadLog->add(key, value, false, TimeDiff(t));
        }
    }

    // If we could not find the value, report the error
    if (r == ZKR_UNSPECIFIED)
    {
        zklog.error("Database::read() requested a key that does not exist: " + key);
        r = ZKR_DB_KEY_NOT_FOUND;
    }

#ifdef LOG_DB_READ
    {
        string s = "Database::read()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + string(zkresult2string(r));
        s += " key=" + key;
        s += " value=";
        for (uint64_t i = 0; i < value.size(); i++)
            s += fr.toString(value[i], 16) + ":";
        zklog.info(s);
    }
#endif

    return r;
}

zkresult Database::write(const string &_key, const vector<Goldilocks::Element> &value, const bool persistent, const bool update)
{
    // Check that it has  been initialized before
    if (!bInitialized)
    {
        zklog.error("Database::write() called uninitialized");
        exitProcess();
    }

    if (config.dbMultiWrite && !useDBMTCache && !persistent)
    {
        zklog.error("Database::write() called with multi-write active, cache disabled and no persistance in database, so there is no place to store the date");
        return ZKR_DB_ERROR;
    }

    zkresult r;

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

    if ( useRemoteDB
#ifdef DATABASE_USE_CACHE
         && persistent
#endif
         )
    {
        // Prepare the query
        string valueString = "";
        string aux;
        for (uint64_t i = 0; i < value.size(); i++)
        {
            valueString += PrependZeros(fr.toString(value[i], 16), 16);
        }

        r = writeRemote(false, key, valueString, update);
    }
    else
    {
        r = ZKR_SUCCESS;
    }

#ifdef DATABASE_USE_CACHE
    if ((r == ZKR_SUCCESS) && (useDBMTCache))
    {
        // Create in memory cache
        Database::dbMTCache.add(key, value, update);
    }
#endif

#ifdef LOG_DB_WRITE
    {
        string s = "Database::write()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + sring(zkresult2string(r));
        s += " key=" + key;
        s += " value=";
        for (uint64_t i = 0; i < value.size(); i++)
            s += fr.toString(value[i], 16) + ":";
        s += " persistent=" + to_string(persistent) + " update=" + to_string(update);
        zklog.info(s);
    }
#endif

    return r;
}

void Database::initRemote(void)
{
    TimerStart(DB_INIT_REMOTE);

    try
    {
        // Build the remote database URI
        string uri = config.databaseURL;
        //zklog.info("Database URI: " + uri);

        // Create the database connections
        connLock();

        if (config.dbConnectionsPool)
        {
            // Check that we don't support more threads than available connections
            if ( config.runStateDBServer && (config.maxStateDBThreads > config.dbNumberOfPoolConnections) )
            {
                zklog.error("Database::initRemote() found config.maxStateDBThreads=" + to_string(config.maxStateDBThreads) + " > config.dbNumberOfPoolConnections=" + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }
            if ( config.runExecutorServer && (config.maxExecutorThreads > config.dbNumberOfPoolConnections) )
            {
                zklog.error("Database::initRemote() found config.maxExecutorThreads=" + to_string(config.maxExecutorThreads) + " > config.dbNumberOfPoolConnections=" + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }
            if ( config.runStateDBServer && config.runExecutorServer && ((config.maxStateDBThreads+config.maxExecutorThreads) > config.dbNumberOfPoolConnections) )
            {
                zklog.error("Database::initRemote() found config.maxStateDBThreads+config.maxExecutorThreads=" + to_string(config.maxStateDBThreads+config.maxExecutorThreads) + " > config.dbNumberOfPoolConnections=" + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }

            // Allocate write connections pool
            connectionsPool = new DatabaseConnection[config.dbNumberOfPoolConnections];
            if (connectionsPool == NULL)
            {
                zklog.error("Database::initRemote() failed creating write connection pool of size " + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }

            // Create write connections
            for (uint64_t i=0; i<config.dbNumberOfPoolConnections; i++)
            {
                connectionsPool[i].pConnection = new pqxx::connection{uri};
                if (connectionsPool[i].pConnection == NULL)
                {
                    zklog.error("Database::initRemote() failed creating write connection " + to_string(i));
                    exitProcess();
                }
                connectionsPool[i].bInUse = false;
                //zklog.info("Database::initRemote() created write connection i=" + to_string(i) + " connectionsPool[i]=" + to_string((uint64_t)connectionsPool[i].pConnection));
            }

            // Reset counters
            nextConnection = 0;
            usedConnections = 0;
        }
        else
        {
            connection.pConnection = new pqxx::connection{uri};
            if (connection.pConnection == NULL)
            {
                zklog.error("Database::initRemote() failed creating unique connection");
                exitProcess();
            }
            connection.bInUse = false;
        }
        
        connUnlock();
    }
    catch (const std::exception &e)
    {
        zklog.error("Database::initRemote() exception: " + string(e.what()));
        exitProcess();
    }

    // If configured to use the get tree function, we must install it in the database before using it
    if (config.dbGetTree && !config.dbReadOnly)
    {
        writeGetTreeFunction();
    }

    TimerStopAndLog(DB_INIT_REMOTE);
}

DatabaseConnection * Database::getConnection (void)
{
    if (config.dbConnectionsPool)
    {
        connLock();
        DatabaseConnection * pConnection = NULL;
        uint64_t i=0;
        for (i=0; i<config.dbNumberOfPoolConnections; i++)
        {
            if (!connectionsPool[nextConnection].bInUse) break;
            nextConnection++;
            if (nextConnection == config.dbNumberOfPoolConnections)
            {
                nextConnection = 0;
            }
        }
        if (i==config.dbNumberOfPoolConnections)
        {
            zklog.error("Database::getWriteConnection() run out of free connections");
            exitProcess();
        }

        pConnection = &connectionsPool[nextConnection];
        zkassert(pConnection->bInUse == false);
        pConnection->bInUse = true;
        nextConnection++;
        if (nextConnection == config.dbNumberOfPoolConnections)
        {
            nextConnection = 0;
        }
        usedConnections++;
        //zklog.info("Database::getWriteConnection() pConnection=" + to_string((uint64_t)pConnection) + " nextConnection=" + to_string(nextConnection) + " usedConnections=" + to_string(usedConnections));
        connUnlock();
        return pConnection;
    }
    else
    {
        connLock();
        zkassert(connection.bInUse == false);
#ifdef DEBUG
        connection.bInUse = true;
#endif
        return &connection;
    }
}

void Database::disposeConnection (DatabaseConnection * pConnection)
{
    if (config.dbConnectionsPool)
    {
        connLock();
        zkassert(pConnection->bInUse == true);
        pConnection->bInUse = false;
        zkassert(usedConnections > 0);
        usedConnections--;
        //zklog.info("Database::disposeWriteConnection() pConnection=" + to_string((uint64_t)pConnection) + " nextConnection=" + to_string(nextConnection) + " usedConnections=" + to_string(usedConnections));
        connUnlock();
    }
    else
    {
        zkassert(pConnection == &connection);
        zkassert(pConnection->bInUse == true);
#ifdef DEBUG
        pConnection->bInUse = false;
#endif
        connUnlock();
    }
}

zkresult Database::readRemote(bool bProgram, const string &key, string &value)
{
    const string &tableName = (bProgram ? config.dbProgramTableName : config.dbNodesTableName);

    if (config.logRemoteDbReads)
    {
        zklog.info("Database::readRemote() table=" + tableName + " key=" + key);
    }

    // Get a free read db connection
    DatabaseConnection * pDatabaseConnection = getConnection();

    try
    {
        // Prepare the query
        string query = "SELECT * FROM " + tableName + " WHERE hash = E\'\\\\x" + key + "\';";

        pqxx::result rows;

        // Start a transaction.
        pqxx::nontransaction n(*(pDatabaseConnection->pConnection));

        // Execute the query
        rows = n.exec(query);

        // Commit your transaction
        n.commit();

        // Process the result
        if (rows.size() == 0)
        {
            disposeConnection(pDatabaseConnection);
            return ZKR_DB_KEY_NOT_FOUND;
        }
        else if (rows.size() > 1)
        {
            zklog.error("Database::readRemote() table=" + tableName + " got more than one row for the same key: " + to_string(rows.size()));
            exitProcess();
        }

        pqxx::row const row = rows[0];
        if (row.size() != 2)
        {
            zklog.error("Database::readRemote() table=" + tableName + " got an invalid number of colums for the row: " + to_string(row.size()));
            exitProcess();
        }
        pqxx::field const fieldData = row[1];
        value = removeBSXIfExists(fieldData.c_str());
    }
    catch (const std::exception &e)
    {
        zklog.error("Database::readRemote() table=" + tableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
        exitProcess();
    }
    
    // Dispose the read db conneciton
    disposeConnection(pDatabaseConnection);

    return ZKR_SUCCESS;
}

zkresult Database::readTreeRemote(const string &key, const vector<uint64_t> *keys, uint64_t level, uint64_t &numberOfFields)
{
    zkassert(keys != NULL);

    if (config.logRemoteDbReads)
    {
        zklog.info("Database::readTreeRemote() key=" + key);
    }
    string rkey;
    for (uint64_t i=level; i<keys->size(); i++)
    {
        uint8_t auxByte = (*keys)[i];
        if (auxByte > 1)
        {
            zklog.error("Database::readTreeRemote() found invalid keys value=" + to_string(auxByte) + " at position " + to_string(i));
            return ZKR_DB_ERROR;
        }
        rkey.append(1, byte2char(auxByte >> 4));
        rkey.append(1, byte2char(auxByte & 0x0F));
    }

    // Get a free read db connection
    DatabaseConnection * pDatabaseConnection = getConnection();

    numberOfFields = 0;

    try
    {
        // Prepare the query
        string query = "SELECT get_tree (E\'\\\\x" + key + "\', E\'\\\\x" + rkey + "\');";

        pqxx::result rows;

        // Start a transaction.
        pqxx::nontransaction n(*(pDatabaseConnection->pConnection));

        // Execute the query
        rows = n.exec(query);

        // Commit your transaction
        n.commit();

        // Process the result
        numberOfFields = rows.size();
        for (uint64_t i=0; i<numberOfFields; i++)
        {
            pqxx::row const row = rows[i];
            if (row.size() != 1)
            {
                zklog.error("Database::readTreeRemote() got an invalid number of colums for the row: " + to_string(row.size()));
                disposeConnection(pDatabaseConnection);
                return ZKR_UNSPECIFIED;
            }
            pqxx::field const fieldData = row[0];
            string fieldDataString = fieldData.c_str();
            //zklog.info("got value=" + fieldDataString);
            string hash, data;

            string first = "(\"\\\\x";
            string second = "\",\"\\\\x";
            string third = "\")";

            size_t firstPosition = fieldDataString.find(first);
            size_t secondPosition = fieldDataString.find(second);
            size_t thirdPosition = fieldDataString.find(third);

            if ( (firstPosition != 0) ||
                 (firstPosition + first.size() + 32*2 != secondPosition ) ||
                 (secondPosition <= first.size()) ||
                 (thirdPosition == 0) ||
                 ( (secondPosition + second.size() + 12*8*2 != thirdPosition) &&
                   (secondPosition + second.size() + 8*8*2 != thirdPosition) ))
            {
                zklog.error("Database::readTreeRemote() got an invalid field=" + fieldDataString);
                disposeConnection(pDatabaseConnection);
                return ZKR_UNSPECIFIED;
            }

            hash = fieldDataString.substr(firstPosition + first.size(), 32*2);
            data = fieldDataString.substr(secondPosition + second.size(), thirdPosition - secondPosition - second.size());
            vector<Goldilocks::Element> value;
            string2fea(fr, data, value);

#ifdef DATABASE_USE_CACHE
            // Store it locally to avoid any future remote access for this key
            if (useDBMTCache)
            {
                //zklog.info("Database::readTreeRemote() adding hash=" + hash + " to dbMTCache");
                Database::dbMTCache.add(hash, value, false);
            }
#endif
        }
    }
    catch (const std::exception &e)
    {
        zklog.warning("Database::readTreeRemote() exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
        return ZKR_DB_ERROR;
    }
    
    // Dispose the read db conneciton
    disposeConnection(pDatabaseConnection);

    if (config.logRemoteDbReads)
    {
        zklog.info("Database::readTreeRemote() key=" + key + " read " + to_string(numberOfFields));
    }

    return ZKR_SUCCESS;
    
}

zkresult Database::writeRemote(bool bProgram, const string &key, const string &value, const bool update)
{
    zkresult result = ZKR_SUCCESS;

    const string &tableName = (bProgram ? config.dbProgramTableName : config.dbNodesTableName);
    
    if (config.dbMultiWrite)
    {
        if (update && (key==dbStateRootKey))
        {
            multiWriteLock();
            multiWriteNodesStateRoot = "INSERT INTO " + tableName + " ( hash, data ) VALUES ( E\'\\\\x" + key + "\', E\'\\\\x" + value + "\' ) ";
            multiWriteUnlock();       
        }
        else
        {
            string &multiWrite = bProgram ? (update ? multiWriteProgramUpdate : multiWriteProgram) : (update ? multiWriteNodesUpdate : multiWriteNodes);
            multiWriteLock();
            if (multiWrite.size() == 0)
            {
                multiWrite = "INSERT INTO " + tableName + " ( hash, data ) VALUES ( E\'\\\\x" + key + "\', E\'\\\\x" + value + "\' ) ";
            }
            else
            {
                multiWrite += ", ( E\'\\\\x" + key + "\', E\'\\\\x" + value + "\' )";
            }
            multiWriteUnlock();       
        }
    }
    else
    {
        string query = "INSERT INTO " + tableName + " ( hash, data ) VALUES ( E\'\\\\x" + key + "\', E\'\\\\x" + value + "\' ) " +
                    (update ? "ON CONFLICT (hash) DO UPDATE SET data = EXCLUDED.data;" : "ON CONFLICT (hash) DO NOTHING;");
            
        DatabaseConnection * pDatabaseConnection = getConnection();

        try
        {        

#ifdef DATABASE_COMMIT
            if (autoCommit)
#endif
            {
                pqxx::work w(*(pDatabaseConnection->pConnection));
                pqxx::result res = w.exec(query);
                w.commit();
                disposeConnection(pDatabaseConnection);
            }
#ifdef DATABASE_COMMIT
            else
            {
                if (transaction == NULL)
                    transaction = new pqxx::work{*pConnectionWrite};
                pqxx::result res = transaction->exec(query);
            }
#endif
        }
        catch (const std::exception &e)
        {
            zklog.error("Database::writeRemote() table=" + tableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
            result = ZKR_DB_ERROR;
        }
    }

    return result;
}

zkresult Database::writeGetTreeFunction(void)
{
    if (!config.dbGetTree)
    {
        zklog.error("Database::writeGetTreeFunction() dalled with config.dbGetTree=false");
        return ZKR_DB_ERROR;
    }
    
    if (config.databaseURL == "local")
    {
        zklog.error("Database::writeGetTreeFunction() dalled with config.databaseURL=local");
        return ZKR_DB_ERROR;
    }

    zkresult result = ZKR_SUCCESS;

    string query = string("") +
    "create or replace function get_tree (root_hash bytea, remaining_key bytea)\n" +
	"   returns setof state.nodes\n" +
	"   language plpgsql\n" +
    "as $$\n" +
    "declare\n" +
    "	current_hash bytea;\n" +
    "	current_row " + config.dbNodesTableName + "%rowtype;\n" +
    "	remaining_key_length integer;\n" +
    "	remaining_key_bit integer;\n" +
    "	byte_71 integer;\n" +
    "	aux_integer integer;\n" +
    "begin\n" +
    "	remaining_key_length = octet_length(remaining_key);\n" +
    "	current_hash = root_hash;\n" +

    "	-- For every bit (0 or 1) in remaining key\n" +
    "	for counter in 0..(remaining_key_length-1) loop\n" +

    "		-- Get the current_hash row and store it into current_row\n" +
    "		select * into current_row from " + config.dbNodesTableName + " where hash = current_hash;\n" +
    "		if not found then\n" +
    "			raise EXCEPTION 'Hash % not found', current_hash;\n" +
    "		end if;\n" +

    "		-- Return it as a result\n" +
    "		return next current_row;\n" +

    "		-- Data should be a byte array of 12x8 bytes (12 field elements)\n" +
    "		-- Check data length is exactly 12 field elements\n" +
    "		if (octet_length(current_row.data) != 12*8) then\n" +
    "			raise EXCEPTION 'Hash % got invalid data size %', current_hash, octet_length(current_row.data);\n" +
    "		end if;\n" +
	//	-- Check that last 3 field elements are zero
	//	--if (substring(current_row.data from 89 for 8) != E'\\x0000000000000000') then
	//	--	RAISE EXCEPTION 'Hash % got non-null 12th field element data=%', current_hash, current_row.data;
	//	--end if;
	//	--if (substring(current_row.data from 81 for 8) != E'\\x0000000000000000') then
	//	--	RAISE EXCEPTION 'Hash % got non-null 11th field element data=%', current_hash, current_row.data;
	//	--end if;
	//	--if (substring(current_row.data from 73 for 8) != E'\\x0000000000000000') then
	//	--	RAISE EXCEPTION 'Hash % got non-null 10th field element data=%', current_hash, current_row.data;
	//	--end if;
    "		-- If last 4 field elements are 0000, this is an intermediate node\n" +
    "		byte_71 = get_byte(current_row.data, 71);\n" +
    "		case byte_71\n" +
    "		when 0 then\n" +

    "			-- If the next remaining key is a 0, take the left sibling way, if it is a 1, take the right one\n" +
    "			remaining_key_bit = get_byte(remaining_key, counter);\n" +
    "			case remaining_key_bit\n" +
    "			when 0 then\n" +
    "				current_hash =\n" +
    "					substring(current_row.data from 25 for 8) ||\n" +
    "					substring(current_row.data from 17 for 8) ||\n" +
    "					substring(current_row.data from 9 for 8) ||\n" +
    "					substring(current_row.data from 1 for 8);\n" +
    "			when 1 then\n" +
    "				current_hash =\n" +
    "					substring(current_row.data from 57 for 8) ||\n" +
    "					substring(current_row.data from 49 for 8) ||\n" +
    "					substring(current_row.data from 41 for 8) ||\n" +
    "					substring(current_row.data from 33 for 8);\n" +
    "			else\n" +
    "				raise EXCEPTION 'Invalid remaining key bit at position % with value %', counter, remaining_key_bit ;\n" +
    "			end case;\n" +
    
    "			-- If the hash is a 0, we reached the end of the branch\n" +
    "			if (get_byte(current_hash, 0) = 0) and\n" +
    "			   (get_byte(current_hash, 1) = 0) and\n" +
    "			   (get_byte(current_hash, 2) = 0) and\n" +
    "			   (get_byte(current_hash, 3) = 0) and\n" +
    "			   (get_byte(current_hash, 4) = 0) and\n" +
    "			   (get_byte(current_hash, 5) = 0) and\n" +
    "			   (get_byte(current_hash, 6) = 0) and\n" +
    "			   (get_byte(current_hash, 7) = 0) and\n" +
    "			   (get_byte(current_hash, 8) = 0) and\n" +
    "			   (get_byte(current_hash, 9) = 0) and\n" +
    "			   (get_byte(current_hash, 10) = 0) and\n" +
    "			   (get_byte(current_hash, 11) = 0) and\n" +
    "			   (get_byte(current_hash, 12) = 0) and\n" +
    "			   (get_byte(current_hash, 13) = 0) and\n" +
    "			   (get_byte(current_hash, 14) = 0) and\n" +
    "			   (get_byte(current_hash, 15) = 0) and\n" +
    "			   (get_byte(current_hash, 16) = 0) and\n" +
    "			   (get_byte(current_hash, 17) = 0) and\n" +
    "			   (get_byte(current_hash, 18) = 0) and\n" +
    "			   (get_byte(current_hash, 19) = 0) and\n" +
    "			   (get_byte(current_hash, 20) = 0) and\n" +
    "			   (get_byte(current_hash, 21) = 0) and\n" +
    "			   (get_byte(current_hash, 22) = 0) and\n" +
    "			   (get_byte(current_hash, 23) = 0) and\n" +
    "			   (get_byte(current_hash, 24) = 0) and\n" +
    "			   (get_byte(current_hash, 25) = 0) and\n" +
    "			   (get_byte(current_hash, 26) = 0) and\n" +
    "			   (get_byte(current_hash, 27) = 0) and\n" +
    "			   (get_byte(current_hash, 28) = 0) and\n" +
    "			   (get_byte(current_hash, 29) = 0) and\n" +
    "			   (get_byte(current_hash, 30) = 0) and\n" +
    "			   (get_byte(current_hash, 31) = 0) then\n" +
    "			   return;\n" +
    "			end if;\n" +

    "		-- If last 4 field elements are 1000, this is a leaf node\n" +
    "		when 1 then	\n" +

    "			current_hash =\n" +
    "				substring(current_row.data from 57 for 8) ||\n" +
    "				substring(current_row.data from 49 for 8) ||\n" +
    "				substring(current_row.data from 41 for 8) ||\n" +
    "				substring(current_row.data from 33 for 8);\n" +
    "			select * into current_row from " + config.dbNodesTableName + " where hash = current_hash;\n" +
    "			if not found then\n" +
    "				raise EXCEPTION 'Hash % not found', current_hash;\n" +
    "			end if;\n" +
    "			return next current_row;\n" +
    "			return;\n" +

    "		else\n" +
    "			raise EXCEPTION 'Hash % got invalid 9th field element data=%', current_hash, current_row.data;\n" +
    "		end case;\n" +
			
    "	end loop;\n" +

    "	return;\n" +
    "end;$$\n";
        
    DatabaseConnection * pDatabaseConnection = getConnection();
    
    try
    {
#ifdef DATABASE_COMMIT
        if (autoCommit)
#endif
        {
            pqxx::work w(*(pDatabaseConnection->pConnection));
            pqxx::result res = w.exec(query);
            w.commit();
            disposeConnection(pDatabaseConnection);
        }
#ifdef DATABASE_COMMIT
        else
        {
            if (transaction == NULL)
                transaction = new pqxx::work{*pConnectionWrite};
            pqxx::result res = transaction->exec(query);
        }
#endif
    }
    catch (const std::exception &e)
    {
        zklog.error("Database::writeGetTreeFunction() exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
        result = ZKR_DB_ERROR;
    }

    zklog.info("Database::writeGetTreeFunction() returns " + string(zkresult2string(result)));
        
    return result;
}

zkresult Database::setProgram(const string &_key, const vector<uint8_t> &data, const bool persistent, const bool update)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database::setProgram() called uninitialized");
        exitProcess();
    }

    zkresult r;

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

    if ( useRemoteDB
#ifdef DATABASE_USE_CACHE
         && persistent
#endif
         )
    {
        string sData = "";
        for (uint64_t i=0; i<data.size(); i++)
        {
            sData += byte2string(data[i]);
        }

        r = writeRemote(true, key, sData, update);
    }
    else
    {
        r = ZKR_SUCCESS;
    }

#ifdef DATABASE_USE_CACHE
    if ((r == ZKR_SUCCESS) && (useDBProgramCache))
    {
        // Create in memory cache
        Database::dbProgramCache.add(key, data, update);
    }
#endif

#ifdef LOG_DB_WRITE
    {
        string s = "Database::setProgram()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + string(zkresult2string(r));
        s += " key=" + key;
        s += " data=";
        for (uint64_t i = 0; (i < (data.size()) && (i < 100)); i++)
            s += byte2string(data[i]);
        if (data.size() > 100) s += "...";
        s += " persistent=" + to_string(persistent) + " update=" + to_string(update);
        zklog.info(s);
    }
#endif

    return r;
}

zkresult Database::getProgram(const string &_key, vector<uint8_t> &data, DatabaseMap *dbReadLog, const bool update)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database::getProgram() called uninitialized");
        exitProcess();
    }

    zkresult r;

    struct timeval t;
    if (dbReadLog != NULL) gettimeofday(&t, NULL);

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

#ifdef DATABASE_USE_CACHE
    // If the key is found in local database (cached) simply return it
    if (useDBProgramCache && !update && Database::dbProgramCache.find(key, data))
    {
        // Add to the read log
        if (dbReadLog != NULL) dbReadLog->add(key, data, true, TimeDiff(t));

        r = ZKR_SUCCESS;
    }
    else
#endif
    if (useRemoteDB)
    {
        // Otherwise, read it remotelly
        string sData;
        r = readRemote(true, key, sData);
        if (r == ZKR_SUCCESS)
        {
            //String to byte/uint8_t vector
            string2ba(sData, data);

#ifdef DATABASE_USE_CACHE
            // Store it locally to avoid any future remote access for this key
            if (useDBProgramCache) Database::dbProgramCache.add(key, data, update);
#endif

            // Add to the read log
            if (dbReadLog != NULL) dbReadLog->add(key, data, false, TimeDiff(t));
        }
    }
    else
    {
        zklog.error("Database::getProgram() requested a key that does not exist: " + key);
        r = ZKR_DB_KEY_NOT_FOUND;
    }

#ifdef LOG_DB_READ
    {
        string s = "Database::getProgram()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + string(zkresult2string(r));
        s += " key=" + key;
        s += " data=";
        for (uint64_t i = 0; (i < (data.size()) && (i < 100)); i++)
            s += byte2string(data[i]);
        if (data.size() > 100) s += "...";
        s += " update=" + to_string(update);
        zklog.info(s);
    }
#endif

    return r;
}

zkresult Database::flush()
{
    if (!config.dbMultiWrite)
    {
        return ZKR_SUCCESS;
    }

    // If we are connected to a read-only database, just pretend to have sent all the data
    if (config.dbReadOnly)
    {
        multiWriteLock();
        multiWriteProgram.clear();
        multiWriteProgramUpdate.clear();
        multiWriteNodes.clear();
        multiWriteNodesUpdate.clear();
        multiWriteNodesStateRoot.clear();
        multiWriteUnlock();

        return ZKR_SUCCESS;
    }

    //TimerStart(DATABASE_FLUSH);

    zkresult zkr = ZKR_SUCCESS;

    multiWriteLock();

    if ( (multiWriteNodes.size() > 0) || (multiWriteNodesStateRoot.size() > 0) || (multiWriteNodesUpdate.size() > 0) || (multiWriteProgram.size() > 0) || (multiWriteProgramUpdate.size() > 0) )
    {

        // Get a free write db connection
        DatabaseConnection * pDatabaseConnection = getConnection();

        try
        {
            string query;
            if (multiWriteProgram.size() > 0)
            {
                query = multiWriteProgram + " ON CONFLICT (hash) DO NOTHING;";
                
                // Start a transaction
                pqxx::work w(*(pDatabaseConnection->pConnection));

                // Execute the query
                pqxx::result res = w.exec(query);

                // Commit your transaction
                w.commit();

                //zklog.info("Database::flush() sent query=" + query);

                // Delete the accumulated query data only if the query succeeded
                multiWriteProgram.clear();
            }
            if (multiWriteProgramUpdate.size() > 0)
            {
                query = multiWriteProgramUpdate + " ON CONFLICT (hash) DO UPDATE SET data = EXCLUDED.data;";
                
                // Start a transaction
                pqxx::work w(*(pDatabaseConnection->pConnection));

                // Execute the query
                pqxx::result res = w.exec(query);

                // Commit your transaction
                w.commit();

                //zklog.info("Database::flush() sent query=" + query);

                // Delete the accumulated query data only if the query succeeded
                multiWriteProgramUpdate.clear();
            }
            if (multiWriteNodes.size() > 0)
            {
                query = multiWriteNodes + " ON CONFLICT (hash) DO NOTHING;";
                
                // Start a transaction
                pqxx::work w(*(pDatabaseConnection->pConnection));

                // Execute the query
                pqxx::result res = w.exec(query);

                // Commit your transaction
                w.commit();

                //zklog.info("Database::flush() sent query=" + query);

                // Delete the accumulated query data only if the query succeeded
                multiWriteNodes.clear();
            }
            if (multiWriteNodesUpdate.size() > 0)
            {
                query = multiWriteNodesUpdate + " ON CONFLICT (hash) DO UPDATE SET data = EXCLUDED.data;";
                
                // Start a transaction
                pqxx::work w(*(pDatabaseConnection->pConnection));

                // Execute the query
                pqxx::result res = w.exec(query);

                // Commit your transaction
                w.commit();

                //zklog.info("Database::flush() sent query=" + query);

                // Delete the accumulated query data only if the query succeeded
                multiWriteNodesUpdate.clear();
            }
            if (multiWriteNodesStateRoot.size() > 0)
            {
                query = multiWriteNodesStateRoot + " ON CONFLICT (hash) DO UPDATE SET data = EXCLUDED.data;";
                
                // Start a transaction
                pqxx::work w(*(pDatabaseConnection->pConnection));

                // Execute the query
                pqxx::result res = w.exec(query);

                // Commit your transaction
                w.commit();

                //zklog.info("Database::flush() sent query=" + query);

                // Delete the accumulated query data only if the query succeeded
                multiWriteNodesStateRoot.clear();
            }
        }
        catch (const std::exception &e)
        {
            zklog.error("Database::flush() execute query exception: " + string(e.what()));
            zkr = ZKR_DB_ERROR;
        }

        // Dispose the write db connection
        disposeConnection(pDatabaseConnection);
    }
    multiWriteUnlock();

    //TimerStopAndLog(DATABASE_FLUSH);
    
    return zkr;
}

#ifdef DATABASE_COMMIT

void Database::setAutoCommit(const bool ac)
{
    if (ac && !autoCommit)
        commit();
    autoCommit = ac;
}

void Database::commit()
{
    if ((!autoCommit) && (transaction != NULL))
    {
        transaction->commit();
        delete transaction;
        transaction = NULL;
    }
}

#endif

void Database::printTree(const string &root, string prefix)
{
    if (prefix == "")
    {
        zklog.info("Printint tree of root=" + root);
    }
    string key = root;
    vector<Goldilocks::Element> value;
    read(key, value, NULL);
    if (value.size() != 12)
    {
        zklog.error("Database::printTree() found value.size()=" + to_string(value.size()));
        return;
    }
    if (!fr.equal(value[11], fr.zero()))
    {
        zklog.error("Database::printTree() found value[11]=" + fr.toString(value[11], 16));
        return;
    }
    if (!fr.equal(value[10], fr.zero()))
    {
        zklog.error("Database::printTree() found value[10]=" + fr.toString(value[10], 16));
        return;
    }
    if (!fr.equal(value[9], fr.zero()))
    {
        zklog.error("Database::printTree() found value[9]=" + fr.toString(value[9], 16));
        return;
    }
    if (fr.equal(value[8], fr.zero())) // Intermediate node
    {
        string leftKey = fea2string(fr, value[0], value[1], value[2], value[3]);
        zklog.info(prefix + "Intermediate node - left hash=" + leftKey);
        if (leftKey != "0")
            printTree(leftKey, prefix + "  ");
        string rightKey = fea2string(fr, value[4], value[5], value[6], value[7]);
        zklog.info(prefix + "Intermediate node - right hash=" + rightKey);
        if (rightKey != "0")
            printTree(rightKey, prefix + "  ");
    }
    else if (fr.equal(value[8], fr.one())) // Leaf node
    {
        string rKey = fea2string(fr, value[0], value[1], value[2], value[3]);
        zklog.info(prefix + "rKey=" + rKey);
        string hashValue = fea2string(fr, value[4], value[5], value[6], value[7]);
        zklog.info(prefix + "hashValue=" + hashValue);
        vector<Goldilocks::Element> leafValue;
        read(hashValue, leafValue, NULL);
        if (leafValue.size() == 12)
        {
            if (!fr.equal(leafValue[8], fr.zero()))
            {
                zklog.error("Database::printTree() found leafValue[8]=" + fr.toString(leafValue[8], 16));
                return;
            }
            if (!fr.equal(leafValue[9], fr.zero()))
            {
                zklog.error("Database::printTree() found leafValue[9]=" + fr.toString(leafValue[9], 16));
                return;
            }
            if (!fr.equal(leafValue[10], fr.zero()))
            {
                zklog.error("Database::printTree() found leafValue[10]=" + fr.toString(leafValue[10], 16));
                return;
            }
            if (!fr.equal(leafValue[11], fr.zero()))
            {
                zklog.error("Database::printTree() found leafValue[11]=" + fr.toString(leafValue[11], 16));
                return;
            }
        }
        else if (leafValue.size() == 8)
        {
            zklog.info(prefix + "leafValue.size()=" + to_string(leafValue.size()));
        }
        else
        {
            zklog.error("Database::printTree() found lleafValue.size()=" + to_string(leafValue.size()));
            return;
        }
        mpz_class scalarValue;
        fea2scalar(fr, scalarValue, leafValue[0], leafValue[1], leafValue[2], leafValue[3], leafValue[4], leafValue[5], leafValue[6], leafValue[7]);
        zklog.info(prefix + "leafValue=" + PrependZeros(scalarValue.get_str(16), 64));
    }
    else
    {
        zklog.error("Database::printTree() found value[8]=" + fr.toString(value[8], 16));
        return;
    }
    if (prefix == "") zklog.info("");
}

Database::~Database()
{
    if (config.dbConnectionsPool)
    {
        if (connectionsPool != NULL)
        {
            for (uint64_t i=0; i<config.dbNumberOfPoolConnections; i++)
            {
                if (connectionsPool[i].pConnection != NULL)
                {
                    //zklog.info("Database::~Database() deleting writeConnectionsPool[" + to_string(i) + "].pConnection=" + to_string((uint64_t)writeConnectionsPool[i].pConnection));
                    delete[] connectionsPool[i].pConnection;
                }
            }
            delete connectionsPool;
        }
    }
    else
    {
        if (connection.pConnection != NULL)
        {
            delete connection.pConnection;
        }
    }
}

void Database::clearCache (void)
{
    dbMTCache.clear();
    dbProgramCache.clear();
}

void loadDb2MemCache(const Config config)
{
    if (config.databaseURL == "local")
    {
        zklog.error("loadDb2MemCache() called with config.stateDBURL==local");
        exitProcess();
    }

#ifdef DATABASE_USE_CACHE

    TimerStart(LOAD_DB_TO_CACHE);

    Goldilocks fr;
    StateDB * pStateDB = (StateDB *)stateDBSingleton.get(fr, config);

    vector<Goldilocks::Element> dbValue;

    zkresult zkr = pStateDB->db.read(Database::dbStateRootKey, dbValue, NULL, true);
    if (zkr == ZKR_DB_KEY_NOT_FOUND)
    {
        zklog.warning("loadDb2MemCache() dbStateRootKey=" +  Database::dbStateRootKey + " not found in database; normal only if database is empty");
        TimerStopAndLog(LOAD_DB_TO_CACHE);
        return;
    }
    else if (zkr != ZKR_SUCCESS)
    {
        zklog.error("loadDb2MemCache() failed calling db.read result=" + string(zkresult2string(zkr)));
        TimerStopAndLog(LOAD_DB_TO_CACHE);
        return;
    }
    
    string stateRootKey = fea2string(fr, dbValue[0], dbValue[1], dbValue[2], dbValue[3]);
    zklog.info("loadDb2MemCache() found state root=" + stateRootKey);

    if (stateRootKey == "0")
    {
        zklog.warning("loadDb2MemCache() found an empty tree");
        TimerStopAndLog(LOAD_DB_TO_CACHE);
        return;
    }

    struct timeval loadCacheStartTime;
    gettimeofday(&loadCacheStartTime, NULL);

    unordered_map<uint64_t, vector<string>> treeMap;
    vector<string> emptyVector;
    string hash, leftHash, rightHash;
    uint64_t counter = 0;

    treeMap[0] = emptyVector;
    treeMap[0].push_back(stateRootKey);
    unordered_map<uint64_t, std::vector<std::string>>::iterator treeMapIterator;
    for (uint64_t level=0; level<256; level++)
    {
        // Spend only 10 seconds
        if (TimeDiff(loadCacheStartTime) > config.loadDBToMemTimeout)
        {
            break;
        }

        treeMapIterator = treeMap.find(level);
        if (treeMapIterator == treeMap.end())
        {
            break;
        }

        if (treeMapIterator->second.size()==0)
        {
            break;
        }

        treeMap[level+1] = emptyVector;

        //zklog.info("loadDb2MemCache() searching at level=" + to_string(level) + " for elements=" + to_string(treeMapIterator->second.size()));
        
        for (uint64_t i=0; i<treeMapIterator->second.size(); i++)
        {
            // Spend only 10 seconds
            if (TimeDiff(loadCacheStartTime) > config.loadDBToMemTimeout)
            {
                break;
            }

            hash = treeMapIterator->second[i];
            dbValue.clear();
            zkresult zkr = pStateDB->db.read(hash, dbValue, NULL, true);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("loadDb2MemCache() failed calling db.read(" + hash + ") result=" + string(zkresult2string(zkr)));
                TimerStopAndLog(LOAD_DB_TO_CACHE);
                return;
            }
            if (dbValue.size() != 12)
            {
                zklog.error("loadDb2MemCache() failed calling db.read(" + hash + ") dbValue.size()=" + to_string(dbValue.size()));
                TimerStopAndLog(LOAD_DB_TO_CACHE);
                return;
            }
            counter++;
            double sizePercentage = double(Database::dbMTCache.getCurrentSize())*100.0/double(Database::dbMTCache.getMaxSize());
            if ( sizePercentage > 90 )
            {
                zklog.info("loadDb2MemCache() stopping since size percentage=" + to_string(sizePercentage));
                break;
            }

            // If capaxity is X000
            if (fr.isZero(dbValue[9]) && fr.isZero(dbValue[10]) && fr.isZero(dbValue[11]))
            {
                // If capacity is 0000, this is an intermediate node that contains left and right hashes of its children
                if (fr.isZero(dbValue[8]))
                {
                    leftHash = fea2string(fr, dbValue[0], dbValue[1], dbValue[2], dbValue[3]);
                    if (leftHash != "0")
                    {
                        treeMap[level+1].push_back(leftHash);
                        //zklog.info("loadDb2MemCache() level=" + to_string(level) + " found leftHash=" + leftHash);
                    }
                    rightHash = fea2string(fr, dbValue[4], dbValue[5], dbValue[6], dbValue[7]);
                    if (rightHash != "0")
                    {
                        treeMap[level+1].push_back(rightHash);
                        //zklog.info("loadDb2MemCache() level=" + to_string(level) + " found rightHash=" + rightHash);
                    }
                }
                // If capacity is 1000, this is a leaf node that contains right hash of the value node
                else if (fr.isOne(dbValue[8]))
                {
                    rightHash = fea2string(fr, dbValue[4], dbValue[5], dbValue[6], dbValue[7]);
                    if (rightHash != "0")
                    {
                        //zklog.info("loadDb2MemCache() level=" + to_string(level) + " found value rightHash=" + rightHash);
                        dbValue.clear();
                        zkresult zkr = pStateDB->db.read(rightHash, dbValue, NULL, true);
                        if (zkr != ZKR_SUCCESS)
                        {
                            zklog.error("loadDb2MemCache() failed calling db.read(" + rightHash + ") result=" + string(zkresult2string(zkr)));
                            TimerStopAndLog(LOAD_DB_TO_CACHE);
                            return;
                        }
                        counter++;
                    }
                }
            }
        }
    }

    zklog.info("loadDb2MemCache() done counter=" + to_string(counter) + " cache at " + to_string((double(Database::dbMTCache.getCurrentSize())/double(Database::dbMTCache.getMaxSize()))*100) + "%");

    TimerStopAndLog(LOAD_DB_TO_CACHE);

#endif
}