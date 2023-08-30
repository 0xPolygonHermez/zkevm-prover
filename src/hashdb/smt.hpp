#ifndef SMT_HPP
#define SMT_HPP

#include <vector>
#include <map>
#include <gmpxx.h>

#include "poseidon_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#include "compare_fe.hpp"
#include "database.hpp"
#include "database_map.hpp"
#include "zkresult.hpp"
#include "persistence.hpp"
#include "smt_set_result.hpp"
#include "smt_get_result.hpp"

class SmtContext
{
public:
    Database &db;
    bool bUseStateManager;
    const string &batchUUID;
    uint64_t tx;
    const Persistence persistence;
    SmtContext(Database &db, bool bUseStateManager, const string &batchUUID, uint64_t tx, const Persistence persistence) :
        db(db),
        bUseStateManager(bUseStateManager),
        batchUUID(batchUUID),
        tx(tx),
        persistence(persistence) {};
};

// SMT class
class Smt
{
private:
    Goldilocks  &fr;
    PoseidonGoldilocks poseidon;
    Goldilocks::Element capacityZero[4];
    Goldilocks::Element capacityOne[4];
public:
    Smt(Goldilocks &fr) : fr(fr)
    {
        capacityZero[0] = fr.zero();
        capacityZero[1] = fr.zero();
        capacityZero[2] = fr.zero();
        capacityZero[3] = fr.zero();
        capacityOne[0] = fr.one();
        capacityOne[1] = fr.zero();
        capacityOne[2] = fr.zero();
        capacityOne[3] = fr.zero();
    }
    zkresult set(const string &batchUUID, uint64_t tx, Database &db, const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const Persistence persistence, SmtSetResult &result, DatabaseMap *dbReadLog = NULL);
    zkresult get(const string &batchUUID, Database &db, const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], SmtGetResult &result, DatabaseMap *dbReadLog = NULL);
    zkresult hashSave(const SmtContext &ctx, const Goldilocks::Element (&v)[12], Goldilocks::Element (&hash)[4]);

    // Consolidate value and capacity
    zkresult hashSave(const SmtContext &ctx, const Goldilocks::Element (&a)[8], const Goldilocks::Element (&c)[4], Goldilocks::Element (&hash)[4])
    {
        // Calculate the poseidon hash of the vector of field elements: v = a | c
        Goldilocks::Element v[12];
        for (uint64_t i=0; i<8; i++) v[i] = a[i];
        for (uint64_t i=0; i<4; i++) v[8+i] = c[i];
        return hashSave(ctx, v, hash);
    }
    
    // Use capacity zero for intermediate nodes and value hashes
    zkresult hashSaveZero(const SmtContext &ctx, const Goldilocks::Element (&a)[8], Goldilocks::Element (&hash)[4])
    {
        return hashSave(ctx, a, capacityZero, hash);
    }
    
    // Use capacity one for leaf nodes
    zkresult hashSaveOne(const SmtContext &ctx, const Goldilocks::Element (&a)[8], Goldilocks::Element (&hash)[4])
    {
        return hashSave(ctx, a, capacityOne, hash);
    }

    zkresult updateStateRoot(Database &db, const Goldilocks::Element (&stateRoot)[4]);
    int64_t getUniqueSibling(vector<Goldilocks::Element> &a);
};

#endif