#include "poseidon_helper.hpp"
#include "zklog.hpp"
#include "definitions.hpp"
#include "utils.hpp"

//#define LOG_HASHP

namespace fork_12
{

zkresult HashP_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t hashAddr)
{
    mpz_class result;

    zkassert(ctx.pZKPC != NULL);
    uint64_t zkPC = *ctx.pZKPC;
    zkassert((ctx.rom.line[zkPC].hashP == 1) || (ctx.rom.line[zkPC].hashP1 == 1));
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    unordered_map< uint64_t, HashValue >::iterator hashPIterator;

    // If there is no entry in the hash database for this address, then create a new one
    hashPIterator = ctx.hashP.find(hashAddr);
    if (hashPIterator == ctx.hashP.end())
    {
        HashValue hashValue;
        ctx.hashP[hashAddr] = hashValue;
        hashPIterator = ctx.hashP.find(hashAddr);
        zkassert(hashPIterator != ctx.hashP.end());
    }

    // Get the size of the hash from D0
    uint64_t size = 1;
    if (ctx.rom.line[zkPC].hashP == 1)
    {
        size = fr.toU64(ctx.pols.D0[i]);
        if (size > 32)
        {
            zklog.error("HashP_calculate() Invalid size>32 for hashP 1: pols.D0[i]=" + fr.toString(ctx.pols.D0[i], 16) + " size=" + to_string(size));
            return ZKR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE;
        }
    }

    // Get the positon of the hash from HASHPOS
    int64_t iPos;
    fr.toS64(iPos, ctx.pols.HASHPOS[i]);
    if (iPos < 0)
    {
        zklog.error("HashP_calculate() Invalid pos<0 for HashP 1: pols.HASHPOS[i]=" + fr.toString(ctx.pols.HASHPOS[i], 16) + " pos=" + to_string(iPos));
        return ZKR_SM_MAIN_HASHP_POSITION_NEGATIVE;
    }
    uint64_t pos = iPos;

    // Check that pos+size do not exceed data size
    if ( (pos+size) > hashPIterator->second.data.size())
    {
        zklog.error("HashP_calculate() HashP 1 invalid size of hash: pos=" + to_string(pos) + " size=" + to_string(size) + " data.size=" + to_string(ctx.hashP[hashAddr].data.size()));
        return ZKR_SM_MAIN_HASHP_POSITION_PLUS_SIZE_OUT_OF_RANGE;
    }

    // Copy data into fi
    mpz_class s;
    for (uint64_t j=0; j<size; j++)
    {
        uint8_t data = hashPIterator->second.data[pos+j];
        s = (s<<uint64_t(8)) + mpz_class(data);
    }
    scalar2fea(fr, s, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

    return ZKR_SUCCESS;
}

zkresult HashP_verify ( Context &ctx,
                        Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                        MainExecRequired *required,
                        int32_t hashAddr)
{
    zkassert(ctx.pZKPC != NULL);
    uint64_t zkPC = *ctx.pZKPC;
    zkassert((ctx.rom.line[zkPC].hashP == 1) || (ctx.rom.line[zkPC].hashP1 == 1));
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    if (!ctx.bProcessBatch)
    {
        if (ctx.rom.line[zkPC].hashP == 1)
        {
            ctx.pols.hashP[i] = fr.one();
        }
        else
        {
            ctx.pols.hashP1[i] = fr.one();
        }
    }

    unordered_map< uint64_t, HashValue >::iterator hashPIterator;

    // If there is no entry in the hash database for this address, then create a new one
    hashPIterator = ctx.hashP.find(hashAddr);
    if (hashPIterator == ctx.hashP.end())
    {
        HashValue hashValue;
        ctx.hashP[hashAddr] = hashValue;
        hashPIterator = ctx.hashP.find(hashAddr);
        zkassert(hashPIterator != ctx.hashP.end());
    }

    // Get the size of the hash from D0
    uint64_t size = 1;
    if (ctx.rom.line[zkPC].hashP == 1)
    {
        size = fr.toU64(ctx.pols.D0[i]);
        if (size > 32)
        {
            zklog.error("HashP_verify() Invalid size>32 for hashP 2: pols.D0[i]=" + fr.toString(ctx.pols.D0[i], 16) + " size=" + to_string(size));
            return ZKR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE;
        }
    }

    // Get the positon of the hash from HASHPOS
    int64_t iPos;
    fr.toS64(iPos, ctx.pols.HASHPOS[i]);
    if (iPos < 0)
    {
        zklog.error("HashP_verify() Invalid pos<0 for HashP 2: pols.HASHPOS[i]=" + fr.toString(ctx.pols.HASHPOS[i], 16) + " pos=" + to_string(iPos));
        return ZKR_SM_MAIN_HASHP_POSITION_NEGATIVE;
    }
    uint64_t pos = iPos;

    // Get contents of opN into a
    mpz_class a;
    if (!fea2scalar(fr, a, op0, op1, op2, op3, op4, op5, op6, op7))
    {
        zklog.error("HashP_verify() Failed calling fea2scalar(op)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }

    // Fill the hash data vector with chunks of the scalar value
    mpz_class result;
    for (uint64_t j=0; j<size; j++)
    {
        result = (a >> (size-j-1)*8) & ScalarMask8;
        uint8_t bm = result.get_ui();

        // Allow to fill the first byte with a zero
        if (((pos+j) == 1) && hashPIterator->second.data.empty() && !hashPIterator->second.firstByteWritten)
        {
            // Fill a zero
            hashPIterator->second.data.push_back(0);
            
            // Record the read operation
            unordered_map<uint64_t, uint64_t>::iterator readsIterator;
            readsIterator = hashPIterator->second.reads.find(0);
            if ( readsIterator != hashPIterator->second.reads.end() )
            {
                zklog.error("HashP_verify() HashP 2 zero position already existed hashAddr=" + to_string(hashAddr) + " pos=" + to_string(pos));
                return ZKR_SM_MAIN_HASHP_SIZE_MISMATCH;
            }
            else
            {
                hashPIterator->second.reads[0] = 1;
            }
        }

        // Allow to overwrite the first byte
        if (((pos+j) == 0) && (size==1) && !hashPIterator->second.data.empty() && !hashPIterator->second.firstByteWritten)
        {
            hashPIterator->second.data[0] = bm;
            hashPIterator->second.firstByteWritten = true;
        }
        else if (hashPIterator->second.data.size() == (pos+j))
        {
            hashPIterator->second.data.push_back(bm);
        }
        else if (hashPIterator->second.data.size() < (pos+j))
        {
            zklog.error("HashP_verify() HashP 2: trying to insert data in a position:" + to_string(pos+j) + " higher than current data size:" + to_string(ctx.hashP[hashAddr].data.size()));
            return ZKR_SM_MAIN_HASHP_POSITION_PLUS_SIZE_OUT_OF_RANGE;
        }
        else
        {
            uint8_t bh;
            bh = hashPIterator->second.data[pos+j];
            if (bm != bh)
            {
                zklog.error("HashP_verify() HashP 2 bytes do not match: hashAddr=" + to_string(hashAddr) + " pos+j=" + to_string(pos+j) + " is bm=" + to_string(bm) + " and it should be bh=" + to_string(bh));
                return ZKR_SM_MAIN_HASHP_VALUE_MISMATCH;
            }
        }
    }

    // Check that the remaining of a (op) is zero, i.e. no more data exists beyond size
    mpz_class paddingA = a >> (size*8);
    if (paddingA != 0)
    {
        zklog.error("HashP_verify() HashP2 incoherent size=" + to_string(size) + " a=" + a.get_str(16) + " paddingA=" + paddingA.get_str(16));
        return ZKR_SM_MAIN_HASHP_PADDING_MISMATCH;
    }

    // Record the read operation
    unordered_map<uint64_t, uint64_t>::iterator readsIterator;
    readsIterator = hashPIterator->second.reads.find(pos);
    if ( readsIterator != hashPIterator->second.reads.end() )
    {
        if (readsIterator->second != size)
        {
            zklog.error("HashP_verify() HashP 2 diferent read sizes in the same position hashAddr=" + to_string(hashAddr) + " pos=" + to_string(pos));
            return ZKR_SM_MAIN_HASHP_SIZE_MISMATCH;
        }
    }
    else
    {
        hashPIterator->second.reads[pos] = size;
    }

    // Store the size
    ctx.incHashPos = size;

    return ZKR_SUCCESS;
}

void HashPLen_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t hashAddr)
{
    mpz_class result;

    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].hashPLen == 1);
    zkassert(ctx.pStep != NULL);

    unordered_map<uint64_t, HashValue>::const_iterator it;
    it = ctx.hashP.find(hashAddr);
    mpz_class auxScalar;
    if (it == ctx.hashP.end())
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

zkresult HashPLen_verify ( Context &ctx,
                           Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                           MainExecRequired *required,
                           int32_t hashAddr)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].hashPLen == 1);
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    if (!ctx.bProcessBatch)
    {
        ctx.pols.hashPLen[i] = fr.one();
    }

    unordered_map< uint64_t, HashValue >::iterator hashPIterator;

    // Get the length
    uint64_t lm = fr.toU64(op0);

    // Find the entry in the hash database for this address
    hashPIterator = ctx.hashP.find(hashAddr);

    // If it's undefined, compute a hash of 0 bytes
    if (hashPIterator == ctx.hashP.end())
    {
        // Check that length = 0
        if (lm != 0)
        {
            zklog.error("HashPLen 2 hashP[hashAddr] is empty but lm is not 0 hashAddr=" + to_string(hashAddr) + " lm=" + to_string(lm));
            return ZKR_SM_MAIN_HASHPLEN_LENGTH_MISMATCH;
        }

        // Create an empty entry in this address slot
        HashValue hashValue;
        ctx.hashP[hashAddr] = hashValue;
        hashPIterator = ctx.hashP.find(hashAddr);
        zkassert(hashPIterator != ctx.hashP.end());
    }

    if (hashPIterator->second.lenCalled)
    {
        zklog.error("HashPLen 2 called more than once hashAddr=" + to_string(hashAddr));
        return ZKR_SM_MAIN_HASHPLEN_CALLED_TWICE;
    }
    hashPIterator->second.lenCalled = true;

    uint64_t lh = hashPIterator->second.data.size();
    if (lm != lh)
    {
        zklog.error("HashPLen 2 does not match match hashAddr=" + to_string(hashAddr) + " is lm=" + to_string(lm) + " and it should be lh=" + to_string(lh));
        return ZKR_SM_MAIN_HASHPLEN_LENGTH_MISMATCH;
    }
    if (!hashPIterator->second.digestCalled)
    {
        // Calculate the linear poseidon hash
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&ctx.t, NULL);
#endif
        Goldilocks::Element result[4];
        poseidonLinearHash(hashPIterator->second.data, result);
        fea2scalar(fr, hashPIterator->second.digest, result);
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        ctx.mainMetrics.add("Poseidon", TimeDiff(ctx.t));
#endif

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&ctx.t, NULL);
#endif
        // Collect the keys used to read or write program data
        if (ctx.proverRequest.input.bGetKeys)
        {
            ctx.proverRequest.programKeys.insert(fea2string(fr, result));
        }

        zkresult zkResult = ctx.pHashDB->setProgram(ctx.proverRequest.uuid, ctx.proverRequest.pFullTracer->get_block_number(), ctx.proverRequest.pFullTracer->get_tx_number(), result, hashPIterator->second.data, ctx.proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE);
        if (zkResult != ZKR_SUCCESS)
        {
            zklog.error(string("Failed calling pHashDB->setProgram() result=") + zkresult2string(zkResult) + " key=" + fea2string(fr, result));
            return zkResult;
        }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        ctx.mainMetrics.add("Set program", TimeDiff(ctx.t));
#endif

#ifdef LOG_HASH
        {
            string s = "Hash calculate hashPLen 2: hashAddr:" + to_string(hashAddr) + " hash:" + ctx.hashP[hashAddr].digest.get_str(16) + " size:" + to_string(ctx.hashP[hashAddr].data.size()) + " data:";
            for (uint64_t k=0; k<ctx.hashP[hashAddr].data.size(); k++) s += byte2string(ctx.hashP[hashAddr].data[k]) + ":";
            zklog.info(s);
        }
#endif
    }

    return ZKR_SUCCESS;
}

zkresult HashPDigest_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t hashAddr)
{
    mpz_class result;

    zkassert(ctx.pZKPC != NULL);
#ifdef LOG_HASHP
    uint64_t zkPC = *ctx.pZKPC;
#endif
    zkassert(ctx.rom.line[*ctx.pZKPC].hashPDigest == 1);
    zkassert(ctx.pStep != NULL);
#ifdef LOG_HASHP
    uint64_t i = *ctx.pStep;
#endif

    unordered_map< uint64_t, HashValue >::iterator hashPIterator;

    // If there is no entry in the hash database for this address, this is an error
    hashPIterator = ctx.hashP.find(hashAddr);
    if (hashPIterator == ctx.hashP.end())
    {
        zklog.error("HashPDigest_calculate() HashPDigest 1: digest not defined hashAddr=" + to_string(hashAddr));
        return ZKR_SM_MAIN_HASHPDIGEST_ADDRESS_NOT_FOUND;
    }

    // If digest was not calculated, this is an error
    if (!hashPIterator->second.lenCalled)
    {
        zklog.error("HashPDigest_calculate() HashPDigest 1: digest not calculated.  Call hashPLen to finish digest.");
        return ZKR_SM_MAIN_HASHPDIGEST_NOT_COMPLETED;
    }

    // Copy digest into fi
    scalar2fea(fr, hashPIterator->second.digest, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

    return ZKR_SUCCESS;
}

zkresult HashPDigest_verify ( Context &ctx,
                              Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                              MainExecRequired *required,
                              int32_t hashAddr)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].hashPDigest == 1);
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    if (!ctx.bProcessBatch)
    {
        ctx.pols.hashPDigest[i] = fr.one();
    }

    // Get contents of op into dg
    mpz_class dg;
    if (!fea2scalar(fr, dg, op0, op1, op2, op3, op4, op5, op6, op7))
    {
        zklog.error("Failed calling fea2scalar(op)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }

    unordered_map< uint64_t, HashValue >::iterator hashPIterator;
    hashPIterator = ctx.hashP.find(hashAddr);
    if (hashPIterator == ctx.hashP.end())
    {
        HashValue hashValue;
        hashValue.digest = dg;
        Goldilocks::Element aux[4];
        scalar2fea(fr, dg, aux);
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&ctx.t, NULL);
#endif
        // Collect the keys used to read or write store data
        if (ctx.proverRequest.input.bGetKeys)
        {
            ctx.proverRequest.programKeys.insert(fea2string(fr, aux));
        }

        zkresult zkResult = ctx.pHashDB->getProgram(ctx.proverRequest.uuid, aux, hashValue.data, ctx.proverRequest.dbReadLog);
        if (zkResult != ZKR_SUCCESS)
        {
            zklog.error("Failed calling pHashDB->getProgram() result=" + zkresult2string(zkResult) + " key=" + fea2string(fr, aux));
            return zkResult;
        }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        ctx.mainMetrics.add("Get program", TimeDiff(ctx.t));
#endif
        ctx.hashP[hashAddr] = hashValue;
        hashPIterator = ctx.hashP.find(hashAddr);
        zkassert(hashPIterator != ctx.hashP.end());
    }

    if (hashPIterator->second.digestCalled)
    {
        zklog.error("HashPDigest 2 called more than once hashAddr=" + to_string(hashAddr));
        return ZKR_SM_MAIN_HASHPDIGEST_CALLED_TWICE;
    }
    hashPIterator->second.digestCalled = true;

    ctx.incCounter = ceil((double(hashPIterator->second.data.size()) + double(1)) / double(56));

    // Check that digest equals op
    if (dg != hashPIterator->second.digest)
    {
        zklog.error("HashPDigest 2: ctx.hashP[hashAddr].digest=" + ctx.hashP[hashAddr].digest.get_str(16) + " does not match op=" + dg.get_str(16));
        return ZKR_SM_MAIN_HASHPDIGEST_DIGEST_MISMATCH;
    }

    return ZKR_SUCCESS;
}

} // namespace