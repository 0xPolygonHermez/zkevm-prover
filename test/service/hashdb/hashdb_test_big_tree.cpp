#include "hashdb_test_big_tree.hpp"
#include "goldilocks_base_field.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "hashdb_factory.hpp"

void runHashDBTestBigTree (const Config& config)
{
    string uuid = getUUID();
    uint64_t tx = 0;
    Goldilocks fr;
    HashDBInterface * pHashDB = HashDBClientFactory::createHashDBClient(fr,config);
    uint64_t flushId, lastSentFlushId;

    Goldilocks::Element root[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    for (uint64_t i=0; i<10000000; i++)
    {
        
        mpz_class value = 1;
        value = uint64_t(random())*uint64_t(random());
        value = ScalarTwoTo64*value + uint64_t(random())*uint64_t(random());
        value = ScalarTwoTo64*value + uint64_t(random())*uint64_t(random());
        value = ScalarTwoTo64*value + uint64_t(random())*uint64_t(random());
        Goldilocks::Element key[4];
        key[0] = fr.fromU64(i);
        key[1] = fr.zero();
        key[2] = fr.zero();
        key[3] = fr.zero();
        //key[0] = fr.fromU64(uint64_t(random())*uint64_t(random()));
        //key[1] = fr.fromU64(uint64_t(random())*uint64_t(random()));
        //key[2] = fr.fromU64(uint64_t(random())*uint64_t(random()));
        //key[3] = fr.fromU64(uint64_t(random())*uint64_t(random()));

        zkresult zkr = pHashDB->set(uuid, tx, root, key, value, PERSISTENCE_DATABASE, root, NULL, NULL );
        if (zkr != ZKR_SUCCESS)
        {
            cerr << "Error: i=" << i << " zkr=" << zkr << "=" << zkresult2string(zkr) << endl;
            exitProcess();
        }
        if (i%10000 == 0)
        {
            cout << getTimestamp() << " i=" << i << endl;
            printMemoryInfo(true);
        }
        if (i%100==0)
        {
            pHashDB->flush(uuid, fea2string(fr, root), PERSISTENCE_DATABASE, flushId, lastSentFlushId);
        }
    }
}