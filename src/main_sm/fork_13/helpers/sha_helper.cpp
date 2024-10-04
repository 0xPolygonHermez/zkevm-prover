#include "sha_helper.hpp"
#include "zklog.hpp"
#include "definitions.hpp"
#include "utils.hpp"

//#define LOG_HASHS

namespace fork_13
{

zkresult HashS_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t hashAddr)
{
    mpz_class result;

    zkassert(ctx.pZKPC != NULL);
    uint64_t zkPC = *ctx.pZKPC;
    zkassert((ctx.rom.line[zkPC].hashS == 1) || (ctx.rom.line[zkPC].hashS1 == 1));
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    unordered_map< uint64_t, HashValue >::iterator hashSIterator;

    // If there is no entry in the hash database for this address, then create a new one
    hashSIterator = ctx.hashS.find(hashAddr);
    if (hashSIterator == ctx.hashS.end())
    {
        HashValue hashValue;
        ctx.hashS[hashAddr] = hashValue;
        hashSIterator = ctx.hashS.find(hashAddr);
        zkassert(hashSIterator != ctx.hashS.end());
    }

    // Get the size of the hash from D0
    uint64_t size = 1;
    if (ctx.rom.line[zkPC].hashS == 1)
    {
        size = fr.toU64(ctx.pols.D0[i]);
        if (size > 32)
        {
            zklog.error("HashS_calculate() Invalid size>32 for hashS 1: pols.D0[i]=" + fr.toString(ctx.pols.D0[i], 16) + " size=" + to_string(size));
            return ZKR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE;
        }
    }

    // Get the positon of the hash from HASHPOS
    int64_t iPos;
    fr.toS64(iPos, ctx.pols.HASHPOS[i]);
    if (iPos < 0)
    {
        zklog.error("HashS_calculate() Invalid pos<0 for HashS 1: pols.HASHPOS[i]=" + fr.toString(ctx.pols.HASHPOS[i], 16) + " pos=" + to_string(iPos));
        return ZKR_SM_MAIN_HASHS_POSITION_NEGATIVE;
    }
    uint64_t pos = iPos;

    // Check that pos+size do not exceed data size
    if ( (pos+size) > hashSIterator->second.data.size())
    {
        zklog.error("HashS_calculate() HashS 1 invalid size of hash: pos=" + to_string(pos) + " + size=" + to_string(size) + " > data.size=" + to_string(hashSIterator->second.data.size()));
        return ZKR_SM_MAIN_HASHS_POSITION_PLUS_SIZE_OUT_OF_RANGE;
    }

    // Copy data into fi
    mpz_class s;
    for (uint64_t j=0; j<size; j++)
    {
        uint8_t data = hashSIterator->second.data[pos+j];
        s = (s<<uint64_t(8)) + mpz_class(data);
    }
    scalar2fea(fr, s, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

#ifdef LOG_HASHS
    zklog.info("HashS_calculate() hashS 1 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " hashAddr=" + to_string(hashAddr) + " pos=" + to_string(pos) + " size=" + to_string(size) + " data=" + s.get_str(16));
#endif

    return ZKR_SUCCESS;
}

zkresult HashS_verify ( Context &ctx,
                        Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                        MainExecRequired *required,
                        int32_t hashAddr)
{
    zkassert(ctx.pZKPC != NULL);
    uint64_t zkPC = *ctx.pZKPC;
    zkassert((ctx.rom.line[zkPC].hashS == 1) || (ctx.rom.line[zkPC].hashS1 == 1));
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    if (!ctx.bProcessBatch)
    {
        if (ctx.rom.line[zkPC].hashS == 1)
        {
            ctx.pols.hashS[i] = fr.one();
        }
        else
        {
            ctx.pols.hashS1[i] = fr.one();
        }
    }

    unordered_map< uint64_t, HashValue >::iterator hashSIterator;

    // If there is no entry in the hash database for this address, then create a new one
    hashSIterator = ctx.hashS.find(hashAddr);
    if (hashSIterator == ctx.hashS.end())
    {
        HashValue hashValue;
        ctx.hashS[hashAddr] = hashValue;
        hashSIterator = ctx.hashS.find(hashAddr);
        zkassert(hashSIterator != ctx.hashS.end());
    }

    // Get the size of the hash from D0
    uint64_t size = 1;
    if (ctx.rom.line[zkPC].hashS == 1)
    {
        size = fr.toU64(ctx.pols.D0[i]);
        if (size > 32)
        {
            zklog.error("HashS_verify() Invalid size>32 for hashS 2: pols.D0[i]=" + fr.toString(ctx.pols.D0[i], 16) + " size=" + to_string(size));
            return ZKR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE;
        }
    }

    // Get the position of the hash from HASHPOS
    int64_t iPos;
    fr.toS64(iPos, ctx.pols.HASHPOS[i]);
    if (iPos < 0)
    {
        zklog.error("HashS_verify() Invalid pos<0 for HashS 2: pols.HASHPOS[i]=" + fr.toString(ctx.pols.HASHPOS[i], 16) + " pos=" + to_string(iPos));
        return ZKR_SM_MAIN_HASHS_POSITION_NEGATIVE;
    }
    uint64_t pos = iPos;

    // Get contents of opN into a
    mpz_class a;
    if (!fea2scalar(fr, a, op0, op1, op2, op3, op4, op5, op6, op7))
    {
        zklog.error("HashS_verify() Failed calling fea2scalar(op)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }

    // Fill the hash data vector with chunks of the scalar value
    mpz_class result;
    for (uint64_t j=0; j<size; j++)
    {
        result = (a >> ((size-j-1)*8)) & ScalarMask8;
        uint8_t bm = result.get_ui();
        if (hashSIterator->second.data.size() == (pos+j))
        {
            hashSIterator->second.data.push_back(bm);
        }
        else if (hashSIterator->second.data.size() < (pos+j))
        {
            zklog.error("HashS_verify() HashS 2: trying to insert data in a position:" + to_string(pos+j) + " higher than current data size:" + to_string(ctx.hashS[hashAddr].data.size()));
            return ZKR_SM_MAIN_HASHS_POSITION_PLUS_SIZE_OUT_OF_RANGE;
        }
        else
        {
            uint8_t bh;
            bh = hashSIterator->second.data[pos+j];
            if (bm != bh)
            {
                zklog.error("HashS_verify() HashS 2 bytes do not match: hashAddr=" + to_string(hashAddr) + " pos+j=" + to_string(pos+j) + " is bm=" + to_string(bm) + " and it should be bh=" + to_string(bh));
                return ZKR_SM_MAIN_HASHS_VALUE_MISMATCH;
            }
        }
    }

    // Check that the remaining of a (op) is zero, i.e. no more data exists beyond size
    mpz_class paddingA = a >> (size*8);
    if (paddingA != 0)
    {
        zklog.error("HashS_verify() HashS 2 incoherent size=" + to_string(size) + " a=" + a.get_str(16) + " paddingA=" + paddingA.get_str(16));
        return ZKR_SM_MAIN_HASHS_PADDING_MISMATCH;
    }

    // Record the read operation
    unordered_map<uint64_t, uint64_t>::iterator readsIterator;
    readsIterator = hashSIterator->second.reads.find(pos);
    if ( readsIterator != hashSIterator->second.reads.end() )
    {
        if (readsIterator->second != size)
        {
            zklog.error("HashS_verify() HashS 2 different read sizes in the same position hashAddr=" + to_string(hashAddr) + " pos=" + to_string(pos) + " ctx.hashS[hashAddr].reads[pos]=" + to_string(ctx.hashS[hashAddr].reads[pos]) + " size=" + to_string(size));
            return ZKR_SM_MAIN_HASHS_SIZE_MISMATCH;
        }
    }
    else
    {
        hashSIterator->second.reads[pos] = size;
    }

    // Store the size
    ctx.incHashPos = size;

#ifdef LOG_HASHS
    zklog.info("HashS_verify() hashS 2 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " hashAddr=" + to_string(hashAddr) + " pos=" + to_string(pos) + " size=" + to_string(size) + " data=" + a.get_str(16));
#endif

    return ZKR_SUCCESS;
}

void HashSLen_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t hashAddr)
{
    mpz_class result;

    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].hashSLen == 1);
    zkassert(ctx.pStep != NULL);

    unordered_map<uint64_t, HashValue>::const_iterator it;
    it = ctx.hashS.find(hashAddr);
    mpz_class auxScalar;
    if (it == ctx.hashS.end())
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

zkresult HashSLen_verify ( Context &ctx,
                           Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                           MainExecRequired *required,
                           int32_t hashAddr)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].hashSLen == 1);
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    if (!ctx.bProcessBatch)
    {
        ctx.pols.hashSLen[i] = fr.one();
    }

    unordered_map< uint64_t, HashValue >::iterator hashSIterator;

    // Get the length
    uint64_t lm = fr.toU64(op0);

    // Find the entry in the hash database for this address
    hashSIterator = ctx.hashS.find(hashAddr);

    // If it's undefined, compute a hash of 0 bytes
    if (hashSIterator == ctx.hashS.end())
    {
        // Check that length = 0
        if (lm != 0)
        {
            zklog.error("HashSLen_verify() HashSLen 2 hashS[hashAddr] is empty but lm is not 0 hashAddr=" + to_string(hashAddr) + " lm=" + to_string(lm));
            return ZKR_SM_MAIN_HASHSLEN_LENGTH_MISMATCH;
        }

        // Create an empty entry in this address slot
        HashValue hashValue;
        ctx.hashS[hashAddr] = hashValue;
        hashSIterator = ctx.hashS.find(hashAddr);
        zkassert(hashSIterator != ctx.hashS.end());
    }

    if (hashSIterator->second.lenCalled)
    {
        zklog.error("HashSLen_verify() HashSLen 2 called more than once hashAddr=" + to_string(hashAddr));
        return ZKR_SM_MAIN_HASHSLEN_CALLED_TWICE;
    }
    hashSIterator->second.lenCalled = true;

    uint64_t lh = hashSIterator->second.data.size();
    if (lm != lh)
    {
        zklog.error("HashSLen_verify() HashSLen 2 length does not match hashAddr=" + to_string(hashAddr) + " is lm=" + to_string(lm) + " and it should be lh=" + to_string(lh));
        return ZKR_SM_MAIN_HASHSLEN_LENGTH_MISMATCH;
    }
    if (!hashSIterator->second.digestCalled)
    {
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&ctx.t, NULL);
#endif
        SHA256(hashSIterator->second.data.data(), hashSIterator->second.data.size(), hashSIterator->second.digest);
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        ctx.mainMetrics.add("SHA256", TimeDiff(ctx.t));
#endif

#ifdef LOG_HASHS
        {
            string s = "HashSLen_verify() hashSLen 2 calculate hashSLen: hashAddr:" + to_string(hashAddr) + " hash:" + ctx.hashS[hashAddr].digest.get_str(16) + " size:" + to_string(ctx.hashS[hashAddr].data.size()) + " data:";
            for (uint64_t k=0; k<ctx.hashS[hashAddr].data.size(); k++) s += byte2string(ctx.hashS[hashAddr].data[k]) + ":";
            zklog.info(s);
        }
#endif
    }

#ifdef LOG_HASHS
    zklog.info("HashSLen_verify() hashSLen 2 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " hashAddr=" + to_string(hashAddr));
#endif

    return ZKR_SUCCESS;
}

zkresult HashSDigest_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t hashAddr)
{
    mpz_class result;

    zkassert(ctx.pZKPC != NULL);
#ifdef LOG_HASHP
    uint64_t zkPC = *ctx.pZKPC;
#endif
    zkassert(ctx.rom.line[*ctx.pZKPC].hashSDigest == 1);
    zkassert(ctx.pStep != NULL);
#ifdef LOG_HASHP
    uint64_t i = *ctx.pStep;
#endif

    unordered_map< uint64_t, HashValue >::iterator hashSIterator;

    // If there is no entry in the hash database for this address, this is an error
    hashSIterator = ctx.hashS.find(hashAddr);
    if (hashSIterator == ctx.hashS.end())
    {
        zklog.error("HashSDigest_calculate() HashSDigest 1: digest not defined for hashAddr=" + to_string(hashAddr));
        return ZKR_SM_MAIN_HASHSDIGEST_ADDRESS_NOT_FOUND;
    }

    // If digest was not calculated, this is an error
    if (!hashSIterator->second.lenCalled)
    {
        zklog.error("HashSDigest_calculate() HashSDigest 1: digest not calculated for hashAddr=" + to_string(hashAddr) + ".  Call hashSLen to finish digest.");
        return ZKR_SM_MAIN_HASHSDIGEST_NOT_COMPLETED;
    }

    // Copy digest into fi
    scalar2fea(fr, hashSIterator->second.digest, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

#ifdef LOG_HASHS
    zklog.info("HashSDigest_calculate() hashSDigest 1 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " hashAddr=" + to_string(hashAddr) + " digest=" + ctx.hashS[hashAddr].digest.get_str(16));
#endif

    return ZKR_SUCCESS;
}

zkresult HashSDigest_verify ( Context &ctx,
                              Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                              MainExecRequired *required,
                              int32_t hashAddr)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].hashSDigest == 1);
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    if (!ctx.bProcessBatch)
    {
        ctx.pols.hashSDigest[i] = fr.one();
    }

    // Get contents of op into dg
    mpz_class dg;
    if (!fea2scalar(fr, dg, op0, op1, op2, op3, op4, op5, op6, op7))
    {
        zklog.error("HashSDigest_verify() Failed calling fea2scalar(op)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }

    unordered_map< uint64_t, HashValue >::iterator hashSIterator;

    // Find the entry in the hash database for this address
    hashSIterator = ctx.hashS.find(hashAddr);
    if (hashSIterator == ctx.hashS.end())
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
            zklog.error("HashSDigest_verify() Failed calling pHashDB->getProgram() result=" + zkresult2string(zkResult) + " key=" + fea2string(fr, aux));
            return zkResult;
        }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        ctx.mainMetrics.add("Get program", TimeDiff(ctx.t));
#endif
        ctx.hashS[hashAddr] = hashValue;
        hashSIterator = ctx.hashS.find(hashAddr);
        zkassert(hashSIterator != ctx.hashS.end());
    }

    if (dg != hashSIterator->second.digest)
    {
        zklog.error("HashSDigest_verify() HashSDigest 2: Digest does not match op");
        return ZKR_SM_MAIN_HASHSDIGEST_DIGEST_MISMATCH;
    }

    if (hashSIterator->second.digestCalled)
    {
        zklog.error("HashSDigest_verify() HashSDigest 2 called more than once hashAddr=" + to_string(hashAddr));
        return ZKR_SM_MAIN_HASHSDIGEST_CALLED_TWICE;
    }
    hashSIterator->second.digestCalled = true;

    ctx.incCounter = ceil((double(hashSIterator->second.data.size()) + double(1+8)) / double(64));

#ifdef LOG_HASHS
    zklog.info("HashSDigest_verify() hashSDigest 2 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " hashAddr=" + to_string(hashAddr) + " digest=" + ctx.hashS[hashAddr].digest.get_str(16) + " data.size=" + to_string(hashSIterator->second.data.size()) + " incCounter=" + to_string(ctx.incCounter));
#endif

    return ZKR_SUCCESS;
}

} // namespace