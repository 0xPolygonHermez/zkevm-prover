#include "keccak_helper.hpp"
#include "zklog.hpp"
#include "definitions.hpp"

//#define LOG_HASHK

namespace fork_13
{

zkresult HashK_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t hashAddr)
{
    mpz_class result;

    zkassert(ctx.pZKPC != NULL);
    uint64_t zkPC = *ctx.pZKPC;
    zkassert((ctx.rom.line[zkPC].hashK == 1)  || (ctx.rom.line[zkPC].hashK1 == 1));
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    unordered_map< uint64_t, HashValue >::iterator hashKIterator;

    // If there is no entry in the hash database for this address, then create a new one
    hashKIterator = ctx.hashK.find(hashAddr);
    if (hashKIterator == ctx.hashK.end())
    {
        HashValue hashValue;
        ctx.hashK[hashAddr] = hashValue;
        hashKIterator = ctx.hashK.find(hashAddr);
        zkassert(hashKIterator != ctx.hashK.end());
    }

    // Get the size of the hash from D0
    uint64_t size = 1;
    if (ctx.rom.line[zkPC].hashK == 1)
    {
        size = fr.toU64(ctx.pols.D0[i]);
        if (size > 32)
        {
            zklog.error("HashK_calculate() Invalid size>32 for hashK 1: pols.D0[i]=" + fr.toString(ctx.pols.D0[i], 16) + " size=" + to_string(size));
            return ZKR_SM_MAIN_HASHK_SIZE_OUT_OF_RANGE;
        }
    }

    // Get the positon of the hash from HASHPOS
    int64_t iPos;
    fr.toS64(iPos, ctx.pols.HASHPOS[i]);
    if (iPos < 0)
    {
        zklog.error("HashK_calculate() Invalid pos<0 for HashK 1: pols.HASHPOS[i]=" + fr.toString(ctx.pols.HASHPOS[i], 16) + " pos=" + to_string(iPos));
        return ZKR_SM_MAIN_HASHK_POSITION_NEGATIVE;
    }
    uint64_t pos = iPos;

    // Check that pos+size do not exceed data size
    if ( (pos+size) > hashKIterator->second.data.size())
    {
        zklog.error("HashK_calculate() HashK 1 invalid size of hash: pos=" + to_string(pos) + " + size=" + to_string(size) + " > data.size=" + to_string(hashKIterator->second.data.size()));
        return ZKR_SM_MAIN_HASHK_POSITION_PLUS_SIZE_OUT_OF_RANGE;
    }

    // Copy data into fi
    mpz_class s;
    for (uint64_t j=0; j<size; j++)
    {
        uint8_t data = hashKIterator->second.data[pos+j];
        s = (s<<uint64_t(8)) + mpz_class(data);
    }
    scalar2fea(fr, s, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

#ifdef LOG_HASHK
    zklog.info("HashK_calculate() hashK 1 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " hashAddr=" + to_string(hashAddr) + " pos=" + to_string(pos) + " size=" + to_string(size) + " data=" + s.get_str(16));
#endif

    return ZKR_SUCCESS;
}

zkresult HashK_verify ( Context &ctx,
                        Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                        MainExecRequired *required,
                        int32_t hashAddr)
{
    zkassert(ctx.pZKPC != NULL);
    uint64_t zkPC = *ctx.pZKPC;
    zkassert((ctx.rom.line[zkPC].hashK == 1)  || (ctx.rom.line[zkPC].hashK1 == 1));
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    if (!ctx.bProcessBatch)
    {
        if (ctx.rom.line[zkPC].hashK == 1)
        {
            ctx.pols.hashK[i] = fr.one();
        }
        else
        {
            ctx.pols.hashK1[i] = fr.one();
        }
    }

    unordered_map< uint64_t, HashValue >::iterator hashKIterator;

    // If there is no entry in the hash database for this address, then create a new one
    hashKIterator = ctx.hashK.find(hashAddr);
    if (hashKIterator == ctx.hashK.end())
    {
        HashValue hashValue;
        ctx.hashK[hashAddr] = hashValue;
        hashKIterator = ctx.hashK.find(hashAddr);
        zkassert(hashKIterator != ctx.hashK.end());
    }

    // Get the size of the hash from D0
    uint64_t size = 1;
    if (ctx.rom.line[zkPC].hashK == 1)
    {
        size = fr.toU64(ctx.pols.D0[i]);
        if (size > 32)
        {
            zklog.error("HashK_verify() Invalid size>32 for hashK 2: pols.D0[i]=" + fr.toString(ctx.pols.D0[i], 16) + " size=" + to_string(size));
            return ZKR_SM_MAIN_HASHK_SIZE_OUT_OF_RANGE;
        }
    }

    // Get the position of the hash from HASHPOS
    int64_t iPos;
    fr.toS64(iPos, ctx.pols.HASHPOS[i]);
    if (iPos < 0)
    {
        zklog.error("HashK_verify() Invalid pos<0 for HashK 2: pols.HASHPOS[i]=" + fr.toString(ctx.pols.HASHPOS[i], 16) + " pos=" + to_string(iPos));
        return ZKR_SM_MAIN_HASHK_POSITION_NEGATIVE;
    }
    uint64_t pos = iPos;

    // Get contents of opN into a
    mpz_class a;
    if (!fea2scalar(fr, a, op0, op1, op2, op3, op4, op5, op6, op7))
    {
        zklog.error("HashK_verify() Failed calling fea2scalar(op)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }

    // Fill the hash data vector with chunks of the scalar value
    mpz_class result;
    for (uint64_t j=0; j<size; j++)
    {
        result = (a >> ((size-j-1)*8)) & ScalarMask8;
        uint8_t bm = result.get_ui();
        if (hashKIterator->second.data.size() == (pos+j))
        {
            hashKIterator->second.data.push_back(bm);
        }
        else if (hashKIterator->second.data.size() < (pos+j))
        {
            zklog.error("HashK_verify() HashK 2: trying to insert data in a position:" + to_string(pos+j) + " higher than current data size:" + to_string(ctx.hashK[hashAddr].data.size()));
            return ZKR_SM_MAIN_HASHK_POSITION_PLUS_SIZE_OUT_OF_RANGE;
        }
        else
        {
            uint8_t bh;
            bh = hashKIterator->second.data[pos+j];
            if (bm != bh)
            {
                zklog.error("HashK_verify() HashK 2 bytes do not match: hashAddr=" + to_string(hashAddr) + " pos+j=" + to_string(pos+j) + " is bm=" + to_string(bm) + " and it should be bh=" + to_string(bh));
                return ZKR_SM_MAIN_HASHK_VALUE_MISMATCH;
            }
        }
    }

    // Check that the remaining of a (op) is zero, i.e. no more data exists beyond size
    mpz_class paddingA = a >> (size*8);
    if (paddingA != 0)
    {
        zklog.error("HashK_verify() HashK 2 incoherent size=" + to_string(size) + " a=" + a.get_str(16) + " paddingA=" + paddingA.get_str(16));
        return ZKR_SM_MAIN_HASHK_PADDING_MISMATCH;
    }

    // Record the read operation
    unordered_map<uint64_t, uint64_t>::iterator readsIterator;
    readsIterator = hashKIterator->second.reads.find(pos);
    if ( readsIterator != hashKIterator->second.reads.end() )
    {
        if (readsIterator->second != size)
        {
            zklog.error("HashK_verify() HashK 2 different read sizes in the same position hashAddr=" + to_string(hashAddr) + " pos=" + to_string(pos) + " ctx.hashK[addr].reads[pos]=" + to_string(ctx.hashK[hashAddr].reads[pos]) + " size=" + to_string(size));
            return ZKR_SM_MAIN_HASHK_SIZE_MISMATCH;
        }
    }
    else
    {
        hashKIterator->second.reads[pos] = size;
    }

    // Store the size
    ctx.incHashPos = size;

#ifdef LOG_HASHK
    zklog.info("HashK_verify() hashK 2 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " hashAddr=" + to_string(hashAddr) + " pos=" + to_string(pos) + " size=" + to_string(size) + " data=" + a.get_str(16));
#endif

    return ZKR_SUCCESS;
}

void HashKLen_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t hashAddr)
{
    mpz_class result;

    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].hashKLen == 1);
    zkassert(ctx.pStep != NULL);

    unordered_map<uint64_t, HashValue>::const_iterator it;
    it = ctx.hashK.find(hashAddr);
    if (it == ctx.hashK.end())
    {
        fi0 = fr.zero();
    }
    else
    {
        fi0 = fr.fromU64(it->second.data.size());
    }
    fi1 = fr.zero();
    fi2 = fr.zero();
    fi3 = fr.zero();
    fi4 = fr.zero();
    fi5 = fr.zero();
    fi6 = fr.zero();
    fi7 = fr.zero();
}

zkresult HashKLen_verify ( Context &ctx,
                           Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                           MainExecRequired *required,
                           int32_t hashAddr)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].hashKLen == 1);
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    if (!ctx.bProcessBatch)
    {
        ctx.pols.hashKLen[i] = fr.one();
    }

    unordered_map< uint64_t, HashValue >::iterator hashKIterator;

    // Get the length
    uint64_t lm = fr.toU64(op0);

    // Find the entry in the hash database for this address
    hashKIterator = ctx.hashK.find(hashAddr);

    // If it's undefined, compute a hash of 0 bytes
    if (hashKIterator == ctx.hashK.end())
    {
        // Check that length = 0
        if (lm != 0)
        {
            zklog.error("HashKLen_verify() HashKLen 2 hashK[hashAddr] is empty but lm is not 0 hashAddr=" + to_string(hashAddr) + " lm=" + to_string(lm));
            return ZKR_SM_MAIN_HASHKLEN_LENGTH_MISMATCH;
        }

        // Create an empty entry in this address slot
        HashValue hashValue;
        ctx.hashK[hashAddr] = hashValue;
        hashKIterator = ctx.hashK.find(hashAddr);
        zkassert(hashKIterator != ctx.hashK.end());
    }

    if (hashKIterator->second.lenCalled)
    {
        zklog.error("HashKLen_verify() HashKLen 2 called more than once hashAddr=" + to_string(hashAddr));
        return ZKR_SM_MAIN_HASHKLEN_CALLED_TWICE;
    }
    hashKIterator->second.lenCalled = true;

    uint64_t lh = hashKIterator->second.data.size();
    if (lm != lh)
    {
        zklog.error("HashKLen_verify() HashKLen 2 length does not match hashAddr=" + to_string(hashAddr) + " is lm=" + to_string(lm) + " and it should be lh=" + to_string(lh));
        return ZKR_SM_MAIN_HASHKLEN_LENGTH_MISMATCH;
    }
    if (!hashKIterator->second.digestCalled)
    {
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&ctx.t, NULL);
#endif
        keccak256(hashKIterator->second.data.data(), hashKIterator->second.data.size(), hashKIterator->second.digest);
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        ctx.mainMetrics.add("Keccak", TimeDiff(ctx.t));
#endif

#ifdef LOG_HASHK
        {
            string s = "HashKLen_verify() hashKLen 2 calculate hashKLen: hashAddr:" + to_string(hashAddr) + " hash:" + ctx.hashK[hashAddr].digest.get_str(16) + " size:" + to_string(ctx.hashK[hashAddr].data.size()) + " data:";
            for (uint64_t k=0; k<ctx.hashK[hashAddr].data.size(); k++) s += byte2string(ctx.hashK[hashAddr].data[k]) + ":";
            zklog.info(s);
        }
#endif
    }

#ifdef LOG_HASHK
    zklog.info("HashKLen_verify() hashKLen 2 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " hashAddr=" + to_string(hashAddr));
#endif

    return ZKR_SUCCESS;
}

zkresult HashKDigest_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t hashAddr)
{
    mpz_class result;

    zkassert(ctx.pZKPC != NULL);
#ifdef LOG_HASHK
    uint64_t zkPC = *ctx.pZKPC;
#endif
    zkassert(ctx.rom.line[*ctx.pZKPC].hashKDigest == 1);
    zkassert(ctx.pStep != NULL);
#ifdef LOG_HASHK
    uint64_t i = *ctx.pStep;
#endif

    unordered_map< uint64_t, HashValue >::iterator hashKIterator;

    // If there is no entry in the hash database for this address, this is an error
    hashKIterator = ctx.hashK.find(hashAddr);
    if (hashKIterator == ctx.hashK.end())
    {
        zklog.error("HashKDigest_calculate() HashKDigest 1: digest not defined for hashAddr=" + to_string(hashAddr));
        return ZKR_SM_MAIN_HASHKDIGEST_ADDRESS_NOT_FOUND;
    }

    // If digest was not calculated, this is an error
    if (!hashKIterator->second.lenCalled)
    {
        zklog.error("HashKDigest_calculate() HashKDigest 1: digest not calculated for hashAddr=" + to_string(hashAddr) + ".  Call hashKLen to finish digest.");
        return ZKR_SM_MAIN_HASHKDIGEST_NOT_COMPLETED;
    }

    // Copy digest into fi
    scalar2fea(fr, hashKIterator->second.digest, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

#ifdef LOG_HASHK
    zklog.info("HashKDigest_calculate() hashKDigest 1 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " hashAddr=" + to_string(hashAddr) + " digest=" + ctx.hashK[hashAddr].digest.get_str(16));
#endif

    return ZKR_SUCCESS;
}

zkresult HashKDigest_verify ( Context &ctx,
                           Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                           MainExecRequired *required,
                           int32_t hashAddr)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].hashKDigest == 1);
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    if (!ctx.bProcessBatch)
    {
        ctx.pols.hashKDigest[i] = fr.one();
    }

    // Get contents of op into dg
    mpz_class dg;
    if (!fea2scalar(fr, dg, op0, op1, op2, op3, op4, op5, op6, op7))
    {
        zklog.error("HashKDigest_verify() Failed calling fea2scalar(op)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }

    // Find the entry in the hash database for this address
    unordered_map< uint64_t, HashValue >::iterator hashKIterator;
    hashKIterator = ctx.hashK.find(hashAddr);
    if (hashKIterator == ctx.hashK.end())
    {
#ifdef BLOB_INNER
        HashValue hashValue;
        Goldilocks::Element keyFea[4];
        scalar2fea(fr, dg, keyFea);
        zkresult zkr = pHashDB->getProgram(emptyString, keyFea, hashValue.data, proverRequest.dbReadLog);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("HashKDigest_verify() HashKDigest 2: blob inner data not found in DB dg=" + dg.get_str(16));
            return zkr;
        }

        hashValue.digest = dg;
        hashValue.lenCalled = false;
        ctx.hashK[hashAddr] = hashValue;
        hashKIterator = ctx.hashK.find(hashAddr);
        zkassertpermanent(hashKIterator != ctx.hashK.end());
#else
        zklog.error("HashKDigest_verify() HashKDigest 2 could not find entry for hashAddr=" + to_string(hashAddr));
        return ZKR_SM_MAIN_HASHKDIGEST_NOT_FOUND;
#endif
    }

    if (dg != hashKIterator->second.digest)
    {
        zklog.error("HashKDigest_verify() HashKDigest 2: Digest does not match op");
        return ZKR_SM_MAIN_HASHKDIGEST_DIGEST_MISMATCH;
    }

    if (hashKIterator->second.digestCalled)
    {
        zklog.error("HashKDigest_verify() HashKDigest 2 called more than once hashAddr=" + to_string(hashAddr));
        return ZKR_SM_MAIN_HASHKDIGEST_CALLED_TWICE;
    }
    hashKIterator->second.digestCalled = true;

    ctx.incCounter = ceil((double(hashKIterator->second.data.size()) + double(1)) / double(136));

#ifdef LOG_HASHK
    zklog.info("HashKDigest_verify() hashKDigest 2 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " hashAddr=" + to_string(hashAddr) + " digest=" + ctx.hashK[hashAddr].digest.get_str(16));
#endif

    return ZKR_SUCCESS;
}

} // namespace