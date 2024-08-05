#include "storage_helper.hpp"
#include "zklog.hpp"
#include "definitions.hpp"
#include "utils.hpp"
#include "main_sm/fork_12/main/eval_command.hpp"
#include "poseidon_g_permutation.hpp"

namespace fork_12
{

zkresult Storage_read_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].sRD == 1);
    zkassert(ctx.rom.line[*ctx.pZKPC].sWR == 0);
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    zkresult zkResult;

    // Check the range of the registers A and B
    if  ( !fr.isZero(ctx.pols.A5[i]) || !fr.isZero(ctx.pols.A6[i]) || !fr.isZero(ctx.pols.A7[i]) || !fr.isZero(ctx.pols.B2[i]) || !fr.isZero(ctx.pols.B3[i]) || !fr.isZero(ctx.pols.B4[i]) || !fr.isZero(ctx.pols.B5[i])|| !fr.isZero(ctx.pols.B6[i])|| !fr.isZero(ctx.pols.B7[i]) )
    {
        zklog.error("Storage_read_calculate() Storage read free in found non-zero A-B storage registers");
        return ZKR_SM_MAIN_STORAGE_INVALID_KEY;
    }

    // Get old state root
    Goldilocks::Element oldRoot[4];
    sr8to4(fr, ctx.pols.SR0[i], ctx.pols.SR1[i], ctx.pols.SR2[i], ctx.pols.SR3[i], ctx.pols.SR4[i], ctx.pols.SR5[i], ctx.pols.SR6[i], ctx.pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

    // Track if we used the state override or not
    bool bStateOverride = false;
    mpz_class value;
    SmtGetResult smtGetResult;

    // If the input contains a state override section, then use it
    if (!ctx.proverRequest.input.stateOverride.empty())
    {
        // Get the key address
        mpz_class auxScalar;
        if (!fea2scalar(fr, auxScalar, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
        {
            zklog.error("Storage_read_calculate() Failed calling fea2scalar(pols.A)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        string keyAddress = NormalizeTo0xNFormat(auxScalar.get_str(16), 40);

        // Get the key type
        if (!fea2scalar(fr, auxScalar, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
        {
            zklog.error("Storage_read_calculate() Failed calling fea2scalar(pols.B)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        uint64_t keyType = auxScalar.get_ui();

        // Get the key storage
        if (!fea2scalar(fr, auxScalar, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]))
        {
            zklog.error("Storage_read_calculate() Failed calling fea2scalar(pols.C)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        string keyStorage = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

        unordered_map<string, OverrideEntry>::const_iterator it;
        it = ctx.proverRequest.input.stateOverride.find(keyAddress);
        if (it != ctx.proverRequest.input.stateOverride.end())
        {
            if ((keyType == ctx.rom.constants.SMT_KEY_BALANCE) && it->second.bBalance)
            {
                value = it->second.balance;
                bStateOverride = true;
            }
            else if ((keyType == ctx.rom.constants.SMT_KEY_NONCE) && (it->second.nonce > 0))
            {
                value = it->second.nonce;
                bStateOverride = true;
            }
            else if ((keyType == ctx.rom.constants.SMT_KEY_SC_CODE) && (it->second.code.size() > 0))
            {
                // Calculate the linear poseidon hash
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                gettimeofday(&ctx.t, NULL);
#endif
                Goldilocks::Element result[4];
                poseidonLinearHash(it->second.code, result);
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                ctx.mainMetrics.add("Poseidon", TimeDiff(ctx.t));
#endif
                // Convert to scalar
                fea2scalar(fr, value, result);

                bStateOverride = true;
            }
            else if ((keyType == ctx.rom.constants.SMT_KEY_SC_LENGTH) && (it->second.code.size() > 0))
            {
                value = it->second.code.size();
                bStateOverride = true;
            }
            else if ((keyType == ctx.rom.constants.SMT_KEY_SC_STORAGE) && (it->second.state.size() > 0))
            {
                unordered_map<string, mpz_class>::const_iterator itState;
                itState = it->second.state.find(keyStorage);
                if (itState != it->second.state.end())
                {
                    value = itState->second;
                }
                else
                {
                    value = 0;
                }
                bStateOverride = true;
            }
            else if ((keyType == ctx.rom.constants.SMT_KEY_SC_STORAGE) && (it->second.stateDiff.size() > 0))
            {
                unordered_map<string, mpz_class>::const_iterator itState;
                itState = it->second.stateDiff.find(keyStorage);
                if (itState != it->second.stateDiff.end())
                {
                    value = itState->second;
                }
                else
                {
                    value = 0;
                }
                bStateOverride = true;
            }
        }
    }

    if (bStateOverride)
    {
        smtGetResult.value = value;

#ifdef LOG_SMT_KEY_DETAILS
    zklog.info("Storage_read_calculate() SMT get state override C=" + fea2stringchain(fr, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]) +
        " A=" + fea2stringchain(fr, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]) +
        " B=" + fea2stringchain(fr, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]) +
        " oldRoot=" + fea2string(fr, oldRoot) +
        " value=" + value.get_str(10));
#endif
    }
    else
    {
        Goldilocks::Element Kin0[12];
        Kin0[0] = ctx.pols.C0[i];
        Kin0[1] = ctx.pols.C1[i];
        Kin0[2] = ctx.pols.C2[i];
        Kin0[3] = ctx.pols.C3[i];
        Kin0[4] = ctx.pols.C4[i];
        Kin0[5] = ctx.pols.C5[i];
        Kin0[6] = ctx.pols.C6[i];
        Kin0[7] = ctx.pols.C7[i];
        Kin0[8] = fr.zero();
        Kin0[9] = fr.zero();
        Kin0[10] = fr.zero();
        Kin0[11] = fr.zero();

        Goldilocks::Element Kin1[12];
        Kin1[0] = ctx.pols.A0[i];
        Kin1[1] = ctx.pols.A1[i];
        Kin1[2] = ctx.pols.A2[i];
        Kin1[3] = ctx.pols.A3[i];
        Kin1[4] = ctx.pols.A4[i];
        Kin1[5] = ctx.pols.A5[i];
        Kin1[6] = ctx.pols.B0[i];
        Kin1[7] = ctx.pols.B1[i];

        uint64_t b0 = fr.toU64(ctx.pols.B0[i]);
        bool bIsTouchedAddressTree = (b0 == 5) || (b0 == 6);

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&ctx.t, NULL);
#endif
        // Call poseidon and get the hash key
        Goldilocks::Element Kin0Hash[4];
        poseidon.hash(Kin0Hash, Kin0);

        // Reinject the first resulting hash as the capacity for the next poseidon hash
        Kin1[8] = Kin0Hash[0];
        Kin1[9] = Kin0Hash[1];
        Kin1[10] = Kin0Hash[2];
        Kin1[11] = Kin0Hash[3];

        // Call poseidon hash
        Goldilocks::Element Kin1Hash[4];
        poseidon.hash(Kin1Hash, Kin1);

        Goldilocks::Element key[4];
        key[0] = Kin1Hash[0];
        key[1] = Kin1Hash[1];
        key[2] = Kin1Hash[2];
        key[3] = Kin1Hash[3];
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        ctx.mainMetrics.add("Poseidon", TimeDiff(ctx.t), 3);
#endif

#ifdef LOG_STORAGE
        zklog.info("Storage_read_calculate() Storage read sRD got poseidon key: " + ctx.fr.toString(ctx.lastSWrite.key, 16));
#endif

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&ctx.t, NULL);
#endif
        // Collect the keys used to read or write store data
        if (ctx.proverRequest.input.bGetKeys && !bIsTouchedAddressTree)
        {
            ctx.proverRequest.nodesKeys.insert(fea2string(fr, key));
        }

        zkResult = ctx.pHashDB->get(ctx.proverRequest.uuid, oldRoot, key, value, &smtGetResult, ctx.proverRequest.dbReadLog);
        if (zkResult != ZKR_SUCCESS)
        {
            zklog.error("Storage_read_calculate() Failed calling pHashDB->get() result=" + zkresult2string(zkResult) + " key=" + fea2string(fr, key));
            return zkResult;
        }
        ctx.incCounter = smtGetResult.proofHashCounter + 2;

#ifdef LOG_SMT_KEY_DETAILS
        zklog.info("SMT get C=" + fea2stringchain(fr, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]) +
            " A=" + fea2stringchain(fr, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]) +
            " B=" + fea2stringchain(fr, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]) +
            " Kin0Hash=" + fea2string(fr, Kin0Hash) +
            " Kin1Hash=" + fea2string(fr, Kin1Hash) +
            " oldRoot=" + fea2string(fr, oldRoot) +
            " value=" + value.get_str(10));
#endif

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        ctx.mainMetrics.add("SMT Get", TimeDiff(ctx.t));
#endif

        if (ctx.bProcessBatch)
        {
            zkResult = eval_addReadWriteAddress(ctx, smtGetResult.value, key);
            if (zkResult != ZKR_SUCCESS)
            {
                zklog.error("Storage_read_calculate() Failed calling eval_addReadWriteAddress() 1 result=" + zkresult2string(zkResult));
                return zkResult;
            }
        }
    }

    scalar2fea(fr, smtGetResult.value, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);

#ifdef LOG_STORAGE
    zklog.info("Storage_read_calculate() Storage read sRD read from key: " + ctx.fr.toString(ctx.lastSWrite.key, 16) + " value:" + fr.toString(fi3, 16) + ":" + fr.toString(fi2, 16) + ":" + fr.toString(fi1, 16) + ":" + fr.toString(fi0, 16));
#endif

    return ZKR_SUCCESS;
}

zkresult Storage_read_verify ( Context &ctx,
                               Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                               MainExecRequired *required)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].sRD == 1);
    zkassert(ctx.rom.line[*ctx.pZKPC].sWR == 0);
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    if (!ctx.bProcessBatch)
    {
        ctx.pols.sRD[i] = fr.one();
    }

    Goldilocks::Element Kin0[12];
    Kin0[0] = ctx.pols.C0[i];
    Kin0[1] = ctx.pols.C1[i];
    Kin0[2] = ctx.pols.C2[i];
    Kin0[3] = ctx.pols.C3[i];
    Kin0[4] = ctx.pols.C4[i];
    Kin0[5] = ctx.pols.C5[i];
    Kin0[6] = ctx.pols.C6[i];
    Kin0[7] = ctx.pols.C7[i];
    Kin0[8] = fr.zero();
    Kin0[9] = fr.zero();
    Kin0[10] = fr.zero();
    Kin0[11] = fr.zero();

    Goldilocks::Element Kin1[12];
    Kin1[0] = ctx.pols.A0[i];
    Kin1[1] = ctx.pols.A1[i];
    Kin1[2] = ctx.pols.A2[i];
    Kin1[3] = ctx.pols.A3[i];
    Kin1[4] = ctx.pols.A4[i];
    Kin1[5] = ctx.pols.A5[i];
    Kin1[6] = ctx.pols.B0[i];
    Kin1[7] = ctx.pols.B1[i];

    uint64_t b0 = fr.toU64(ctx.pols.B0[i]);
    bool bIsTouchedAddressTree = (b0 == 5) || (b0 == 6);

    if  ( !fr.isZero(ctx.pols.A5[i]) || !fr.isZero(ctx.pols.A6[i]) || !fr.isZero(ctx.pols.A7[i]) || !fr.isZero(ctx.pols.B2[i]) || !fr.isZero(ctx.pols.B3[i]) || !fr.isZero(ctx.pols.B4[i]) || !fr.isZero(ctx.pols.B5[i])|| !fr.isZero(ctx.pols.B6[i])|| !fr.isZero(ctx.pols.B7[i]) )
    {
        zklog.error("Storage_read_verify() Storage read instruction found non-zero A-B registers");
        return ZKR_SM_MAIN_STORAGE_INVALID_KEY;
    }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    gettimeofday(&ctx.t, NULL);
#endif
    // Call poseidon and get the hash key
    Goldilocks::Element Kin0Hash[4];
    poseidon.hash(Kin0Hash, Kin0);

    Goldilocks::Element keyI[4];
    keyI[0] = Kin0Hash[0];
    keyI[1] = Kin0Hash[1];
    keyI[2] = Kin0Hash[2];
    keyI[3] = Kin0Hash[3];

    Kin1[8] = Kin0Hash[0];
    Kin1[9] = Kin0Hash[1];
    Kin1[10] = Kin0Hash[2];
    Kin1[11] = Kin0Hash[3];

    Goldilocks::Element Kin1Hash[4];
    poseidon.hash(Kin1Hash, Kin1);

    // Store PoseidonG required data
    if (required != NULL)
    {
        // Declare PoseidonG required data
        array<Goldilocks::Element,17> pg;

        // Store PoseidonG required data
        for (uint64_t j=0; j<12; j++)
        {
            pg[j] = Kin0[j];
        }
        for (uint64_t j=0; j<4; j++)
        {
            pg[12+j] = Kin0Hash[j];
        }
        pg[16] = fr.fromU64(POSEIDONG_PERMUTATION1_ID);
        required->PoseidonG.push_back(pg);

        // Store PoseidonG required data
        for (uint64_t j=0; j<12; j++)
        {
            pg[j] = Kin1[j];
        }
        for (uint64_t j=0; j<4; j++)
        {
            pg[12+j] = Kin1Hash[j];
        }
        pg[16] = fr.fromU64(POSEIDONG_PERMUTATION2_ID);
        required->PoseidonG.push_back(pg);
    }

    Goldilocks::Element key[4];
    key[0] = Kin1Hash[0];
    key[1] = Kin1Hash[1];
    key[2] = Kin1Hash[2];
    key[3] = Kin1Hash[3];

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    ctx.mainMetrics.add("Poseidon", TimeDiff(ctx.t), 3);
#endif

#ifdef LOG_STORAGE
    zklog.info("Storage_read_verify() Storage read sRD got poseidon key: " + ctx.fr.toString(ctx.lastSWrite.key, 16));
#endif
    Goldilocks::Element oldRoot[4];
    sr8to4(fr, ctx.pols.SR0[i], ctx.pols.SR1[i], ctx.pols.SR2[i], ctx.pols.SR3[i], ctx.pols.SR4[i], ctx.pols.SR5[i], ctx.pols.SR6[i], ctx.pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    gettimeofday(&ctx.t, NULL);
#endif

    // Collect the keys used to read or write store data
    if (ctx.proverRequest.input.bGetKeys && !bIsTouchedAddressTree)
    {
        ctx.proverRequest.nodesKeys.insert(fea2string(fr, key));
    }

    SmtGetResult smtGetResult;
    mpz_class value;
    zkresult zkResult = ctx.pHashDB->get(ctx.proverRequest.uuid, oldRoot, key, value, &smtGetResult, ctx.proverRequest.dbReadLog);
    if (zkResult != ZKR_SUCCESS)
    {
        zklog.error("Storage_read_verify() Failed calling pHashDB->get() result=" + zkresult2string(zkResult) + " key=" + fea2string(fr, key));
        return zkResult;
    }
    ctx.incCounter = smtGetResult.proofHashCounter + 2;
    //cout << "smt.get() returns value=" << smtGetResult.value.get_str(16) << endl;

    if (ctx.bProcessBatch)
    {
        zkResult = eval_addReadWriteAddress(ctx, smtGetResult.value, key);
        if (zkResult != ZKR_SUCCESS)
        {
            zklog.error("Storage_read_verify() Failed calling eval_addReadWriteAddress() 3 result=" + zkresult2string(zkResult));
            return zkResult;
        }
    }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    ctx.mainMetrics.add("SMT Get", TimeDiff(ctx.t));
#endif 
    if (required != NULL)
    {
        SmtAction smtAction;
        smtAction.bIsSet = false;
        smtAction.getResult = smtGetResult;
        required->Storage.push_back(smtAction);
    }
#ifdef LOG_STORAGE
            zklog.info("Storage_read_verify() Storage read sRD read from key: " + ctx.fr.toString(ctx.lastSWrite.key, 16) + " value:" + value.get_str(16));
#endif
    mpz_class opScalar;
    if (!fea2scalar(fr, opScalar, op0, op1, op2, op3, op4, op5, op6, op7))
    {
        zklog.error("Storage_read_verify() Failed calling fea2scalar(op)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    if (smtGetResult.value != opScalar)
    {
        zklog.error("Storage_read_verify() Storage read does not match: smtGetResult.value=" + smtGetResult.value.get_str() + " opScalar=" + opScalar.get_str());
        return ZKR_SM_MAIN_STORAGE_READ_MISMATCH;
    }

    for (uint64_t k=0; k<4; k++)
    {
        ctx.pols.sKeyI[k][i] = keyI[k];
        ctx.pols.sKey[k][i] = key[k];
    }

    return ZKR_SUCCESS;
}

zkresult Storage_write_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].sRD == 0);
    zkassert(ctx.rom.line[*ctx.pZKPC].sWR == 1);
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;
    zkresult zkResult;

    // Check the range of the registers A and B
    if  ( !fr.isZero(ctx.pols.A5[i]) || !fr.isZero(ctx.pols.A6[i]) || !fr.isZero(ctx.pols.A7[i]) || !fr.isZero(ctx.pols.B2[i]) || !fr.isZero(ctx.pols.B3[i]) || !fr.isZero(ctx.pols.B4[i]) || !fr.isZero(ctx.pols.B5[i])|| !fr.isZero(ctx.pols.B6[i])|| !fr.isZero(ctx.pols.B7[i]) )
    {
        zklog.error("Storage_write_calculate() Storage write free in found non-zero A-B registers");
        return ZKR_SM_MAIN_STORAGE_INVALID_KEY;
    }
                    
    // Get old state root
    Goldilocks::Element oldRoot[4];
    sr8to4(fr, ctx.pols.SR0[i], ctx.pols.SR1[i], ctx.pols.SR2[i], ctx.pols.SR3[i], ctx.pols.SR4[i], ctx.pols.SR5[i], ctx.pols.SR6[i], ctx.pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

    // Get the value to write
    mpz_class value;
    if (!fea2scalar(fr, value, ctx.pols.D0[i], ctx.pols.D1[i], ctx.pols.D2[i], ctx.pols.D3[i], ctx.pols.D4[i], ctx.pols.D5[i], ctx.pols.D6[i], ctx.pols.D7[i]))
    {
        zklog.error("Storage_write_calculate() Failed calling fea2scalar(pols.D)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }

    // Track if we used the state override or not
    bool bStatusOverride = false;

    // If the input contains a state override section, then use it
    if (!ctx.proverRequest.input.stateOverride.empty())
    {
        // Get the key address
        mpz_class auxScalar;
        if (!fea2scalar(fr, auxScalar, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
        {
            zklog.error("Storage_write_calculate() Failed calling fea2scalar(pols.A)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        string keyAddress = NormalizeTo0xNFormat(auxScalar.get_str(16), 40);

        // Get the key type
        if (!fea2scalar(fr, auxScalar, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
        {
            zklog.error("Storage_write_calculate() Failed calling fea2scalar(pols.B)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        uint64_t keyType = auxScalar.get_ui();

        // Get the key storage
        if (!fea2scalar(fr, auxScalar, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]))
        {
            zklog.error("Storage_write_calculate() Failed calling fea2scalar(pols.C)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        string keyStorage = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

        unordered_map<string, OverrideEntry>::iterator it;
        it = ctx.proverRequest.input.stateOverride.find(keyAddress);
        if (it != ctx.proverRequest.input.stateOverride.end())
        {
            if ((keyType == ctx.rom.constants.SMT_KEY_BALANCE) && it->second.bBalance)
            {
                it->second.balance = value;
                it->second.bBalance = true;
                bStatusOverride = true;
            }
            else if ((keyType == ctx.rom.constants.SMT_KEY_NONCE) && (it->second.nonce > 0))
            {
                it->second.nonce = value.get_ui();
                bStatusOverride = true;
            }
            else if ((keyType == ctx.rom.constants.SMT_KEY_SC_CODE) && (it->second.code.size() > 0))
            {
                it->second.code = ctx.proverRequest.input.contractsBytecode[NormalizeTo0xNFormat(value.get_str(16), 64)];
                bStatusOverride = true;
            }
            else if ((keyType == ctx.rom.constants.SMT_KEY_SC_STORAGE) && (it->second.state.size() > 0))
            {
                it->second.state[keyStorage] = value;
                bStatusOverride = true;
            }
            else if ((keyType == ctx.rom.constants.SMT_KEY_SC_STORAGE) && (it->second.stateDiff.size() > 0))
            {
                it->second.stateDiff[keyStorage] = value;
                bStatusOverride = true;
            }
        }
    }

    if (bStatusOverride)
    {

#ifdef LOG_SMT_KEY_DETAILS
    zklog.info("Storage_write_calculate() SMT set state override C=" + fea2stringchain(fr, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]) +
        " A=" + fea2stringchain(fr, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]) +
        " B=" + fea2stringchain(fr, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]) +
        " oldRoot=" + fea2string(fr, oldRoot) +
        " value=" + value.get_str(10));
#endif
    }
    else
    {
        // reset lastSWrite
        ctx.lastSWrite.reset();
        Goldilocks::Element Kin0[12];
        Kin0[0] = ctx.pols.C0[i];
        Kin0[1] = ctx.pols.C1[i];
        Kin0[2] = ctx.pols.C2[i];
        Kin0[3] = ctx.pols.C3[i];
        Kin0[4] = ctx.pols.C4[i];
        Kin0[5] = ctx.pols.C5[i];
        Kin0[6] = ctx.pols.C6[i];
        Kin0[7] = ctx.pols.C7[i];
        Kin0[8] = fr.zero();
        Kin0[9] = fr.zero();
        Kin0[10] = fr.zero();
        Kin0[11] = fr.zero();

        Goldilocks::Element Kin1[12];
        Kin1[0] = ctx.pols.A0[i];
        Kin1[1] = ctx.pols.A1[i];
        Kin1[2] = ctx.pols.A2[i];
        Kin1[3] = ctx.pols.A3[i];
        Kin1[4] = ctx.pols.A4[i];
        Kin1[5] = ctx.pols.A5[i];
        Kin1[6] = ctx.pols.B0[i];
        Kin1[7] = ctx.pols.B1[i];

        uint64_t b0 = fr.toU64(ctx.pols.B0[i]);
        bool bIsTouchedAddressTree = (b0 == 5) || (b0 == 6);
        bool bIsBlockL2Hash = (b0 > 6);

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&ctx.t, NULL);
#endif
        // Call poseidon and get the hash key
        Goldilocks::Element Kin0Hash[4];
        poseidon.hash(Kin0Hash, Kin0);

        Kin1[8] = Kin0Hash[0];
        Kin1[9] = Kin0Hash[1];
        Kin1[10] = Kin0Hash[2];
        Kin1[11] = Kin0Hash[3];

        // Call poseidon hash
        Goldilocks::Element Kin1Hash[4];
        poseidon.hash(Kin1Hash, Kin1);

        // Store a copy of the data in ctx.lastSWrite
        if (!ctx.bProcessBatch)
        {
            for (uint64_t j=0; j<12; j++)
            {
                ctx.lastSWrite.Kin0[j] = Kin0[j];
            }
            for (uint64_t j=0; j<12; j++)
            {
                ctx.lastSWrite.Kin1[j] = Kin1[j];
            }
        }
        for (uint64_t j=0; j<4; j++)
        {
            ctx.lastSWrite.keyI[j] = Kin0Hash[j];
        }
        for (uint64_t j=0; j<4; j++)
        {
            ctx.lastSWrite.key[j] = Kin1Hash[j];
        }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        ctx.mainMetrics.add("Poseidon", TimeDiff(ctx.t));
#endif

#ifdef LOG_STORAGE
        zklog.info("Storage write sWR got poseidon key: " + ctx.fr.toString(ctx.lastSWrite.key, 16));
#endif
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&ctx.t, NULL);
#endif
        // Collect the keys used to read or write store data
        if (ctx.proverRequest.input.bGetKeys && !bIsTouchedAddressTree)
        {
            ctx.proverRequest.nodesKeys.insert(fea2string(fr, ctx.lastSWrite.key));
        }

        zkResult = ctx.pHashDB->set(ctx.proverRequest.uuid, ctx.proverRequest.pFullTracer->get_block_number(), ctx.proverRequest.pFullTracer->get_tx_number(), oldRoot, ctx.lastSWrite.key, value, bIsTouchedAddressTree ? PERSISTENCE_TEMPORARY : bIsBlockL2Hash ? PERSISTENCE_TEMPORARY_HASH : ctx.proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, ctx.lastSWrite.newRoot, &ctx.lastSWrite.res, ctx.proverRequest.dbReadLog);
        if (zkResult != ZKR_SUCCESS)
        {
            zklog.error("Storage_write_calculate() Failed calling pHashDB->set() result=" + zkresult2string(zkResult) + " key=" + fea2string(fr, ctx.lastSWrite.key) + " value=" + value.get_str(16));
            return zkResult;
        }
        ctx.incCounter = ctx.lastSWrite.res.proofHashCounter + 2;

#ifdef LOG_SMT_KEY_DETAILS
        zklog.info("SMT set C=" + fea2stringchain(fr, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]) +
            " A=" + fea2stringchain(fr, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]) +
            " B=" + fea2stringchain(fr, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]) +
            " Kin0Hash=" + fea2string(fr, Kin0Hash) +
            " Kin1Hash=" + fea2string(fr, Kin1Hash) +
            " oldRoot=" + fea2string(fr, oldRoot) +
            " value=" + value.get_str(10) +
            " newRoot=" + fea2string(fr, ctx.lastSWrite.newRoot) +
            " siblingLeftChild=" + fea2string(fr, ctx.lastSWrite.res.siblingLeftChild) +
            " siblingRightChild=" + fea2string(fr, ctx.lastSWrite.res.siblingRightChild));
#endif
    }
    if (ctx.bProcessBatch)
    {
        zkResult = eval_addReadWriteAddress(ctx, value, ctx.lastSWrite.key);
        if (zkResult != ZKR_SUCCESS)
        {
            zklog.error("Storage_write_calculate() Failed calling eval_addReadWriteAddress() 2 result=" + zkresult2string(zkResult));
            return zkResult;
        }
    }

    // If we just modified a balance
    if ( fr.isZero(ctx.pols.B0[i]) && fr.isZero(ctx.pols.B1[i]) )
    {
        mpz_class balanceDifference = ctx.lastSWrite.res.newValue - ctx.lastSWrite.res.oldValue;
        ctx.totalTransferredBalance += balanceDifference;
        //cout << "Set balance: oldValue=" << ctx.lastSWrite.res.oldValue.get_str(10) <<
        //        " newValue=" << ctx.lastSWrite.res.newValue.get_str(10) <<
        //        " difference=" << balanceDifference.get_str(10) <<
        //        " total=" << ctx.totalTransferredBalance.get_str(10) << endl;
    }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    ctx.mainMetrics.add("SMT Set", TimeDiff(ctx.t));
#endif
    ctx.lastSWrite.step = i;

    sr4to8(fr, ctx.lastSWrite.newRoot[0], ctx.lastSWrite.newRoot[1], ctx.lastSWrite.newRoot[2], ctx.lastSWrite.newRoot[3], fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);

#ifdef LOG_STORAGE
    zklog.info("Storage_write_calculate() Storage write sWR stored at key: " + ctx.fr.toString(ctx.lastSWrite.key, 16) + " newRoot: " + fr.toString(ctx.lastSWrite.res.newRoot, 16));
#endif

    return ZKR_SUCCESS;
}

zkresult Storage_write_verify ( Context &ctx,
                                Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                                MainExecRequired *required)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].sRD == 0);
    zkassert(ctx.rom.line[*ctx.pZKPC].sWR == 1);
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    // Copy ROM flags into the polynomials
    if (!ctx.bProcessBatch)
    {
        ctx.pols.sWR[i] = fr.one();
    }

    if ( (!ctx.bProcessBatch && (ctx.lastSWrite.step == 0)) || (ctx.lastSWrite.step != i) )
    {
        // Reset lastSWrite
        ctx.lastSWrite.reset();

        Goldilocks::Element Kin0[12];
        Kin0[0] = ctx.pols.C0[i];
        Kin0[1] = ctx.pols.C1[i];
        Kin0[2] = ctx.pols.C2[i];
        Kin0[3] = ctx.pols.C3[i];
        Kin0[4] = ctx.pols.C4[i];
        Kin0[5] = ctx.pols.C5[i];
        Kin0[6] = ctx.pols.C6[i];
        Kin0[7] = ctx.pols.C7[i];
        Kin0[8] = fr.zero();
        Kin0[9] = fr.zero();
        Kin0[10] = fr.zero();
        Kin0[11] = fr.zero();

        Goldilocks::Element Kin1[12];
        Kin1[0] = ctx.pols.A0[i];
        Kin1[1] = ctx.pols.A1[i];
        Kin1[2] = ctx.pols.A2[i];
        Kin1[3] = ctx.pols.A3[i];
        Kin1[4] = ctx.pols.A4[i];
        Kin1[5] = ctx.pols.A5[i];
        Kin1[6] = ctx.pols.B0[i];
        Kin1[7] = ctx.pols.B1[i];

        uint64_t b0 = fr.toU64(ctx.pols.B0[i]);
        bool bIsTouchedAddressTree = (b0 == 5) || (b0 == 6);
        bool bIsBlockL2Hash = (b0 > 6);

        if  ( !fr.isZero(ctx.pols.A5[i]) || !fr.isZero(ctx.pols.A6[i]) || !fr.isZero(ctx.pols.A7[i]) || !fr.isZero(ctx.pols.B2[i]) || !fr.isZero(ctx.pols.B3[i]) || !fr.isZero(ctx.pols.B4[i]) || !fr.isZero(ctx.pols.B5[i])|| !fr.isZero(ctx.pols.B6[i])|| !fr.isZero(ctx.pols.B7[i]) )
        {
            zklog.error("Storage_write_verify() Storage write instruction found non-zero A-B registers");
            return ZKR_SM_MAIN_STORAGE_INVALID_KEY;
        }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&ctx.t, NULL);
#endif
        // Call poseidon and get the hash key
        Goldilocks::Element Kin0Hash[4];
        poseidon.hash(Kin0Hash, Kin0);

        Kin1[8] = Kin0Hash[0];
        Kin1[9] = Kin0Hash[1];
        Kin1[10] = Kin0Hash[2];
        Kin1[11] = Kin0Hash[3];

        Goldilocks::Element Kin1Hash[4];
        poseidon.hash(Kin1Hash, Kin1);

        // Store a copy of the data in ctx.lastSWrite
        if (!ctx.bProcessBatch)
        {
            for (uint64_t j=0; j<12; j++)
            {
                ctx.lastSWrite.Kin0[j] = Kin0[j];
            }
            for (uint64_t j=0; j<12; j++)
            {
                ctx.lastSWrite.Kin1[j] = Kin1[j];
            }
        }
        for (uint64_t j=0; j<4; j++)
        {
            ctx.lastSWrite.keyI[j] = Kin0Hash[j];
        }
        for (uint64_t j=0; j<4; j++)
        {
            ctx.lastSWrite.key[j] = Kin1Hash[j];
        }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        ctx.mainMetrics.add("Poseidon", TimeDiff(ctx.t));
#endif
        // Call SMT to get the new Merkel Tree root hash
        mpz_class scalarD;
        if (!fea2scalar(fr, scalarD, ctx.pols.D0[i], ctx.pols.D1[i], ctx.pols.D2[i], ctx.pols.D3[i], ctx.pols.D4[i], ctx.pols.D5[i], ctx.pols.D6[i], ctx.pols.D7[i]))
        {
            zklog.error("Storage_write_verify() Failed calling fea2scalar(pols.D)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&ctx.t, NULL);
#endif
        Goldilocks::Element oldRoot[4];
        sr8to4(fr, ctx.pols.SR0[i], ctx.pols.SR1[i], ctx.pols.SR2[i], ctx.pols.SR3[i], ctx.pols.SR4[i], ctx.pols.SR5[i], ctx.pols.SR6[i], ctx.pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

        // Collect the keys used to read or write store data
        if (ctx.proverRequest.input.bGetKeys && !bIsTouchedAddressTree)
        {
            ctx.proverRequest.nodesKeys.insert(fea2string(fr, ctx.lastSWrite.key));
        }

        zkresult zkResult = ctx.pHashDB->set(ctx.proverRequest.uuid, ctx.proverRequest.pFullTracer->get_block_number(), ctx.proverRequest.pFullTracer->get_tx_number(), oldRoot, ctx.lastSWrite.key, scalarD, bIsTouchedAddressTree ? PERSISTENCE_TEMPORARY : bIsBlockL2Hash ? PERSISTENCE_TEMPORARY_HASH : ctx.proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, ctx.lastSWrite.newRoot, &ctx.lastSWrite.res, ctx.proverRequest.dbReadLog);
        if (zkResult != ZKR_SUCCESS)
        {
            zklog.error("Storage_write_verify() Failed calling pHashDB->set() result=" + zkresult2string(zkResult) + " key=" + fea2string(fr, ctx.lastSWrite.key) + " value=" + scalarD.get_str(16));
            return zkResult;
        }

        ctx.incCounter = ctx.lastSWrite.res.proofHashCounter + 2;

        if (ctx.bProcessBatch)
        {
            zkResult = eval_addReadWriteAddress(ctx, scalarD, ctx.lastSWrite.key);
            if (zkResult != ZKR_SUCCESS)
            {
                zklog.error("Storage_write_verify() Failed calling eval_addReadWriteAddress() 4 result=" + zkresult2string(zkResult));
                return zkResult;
            }
        }

        // If we just modified a balance
        if ( fr.isZero(ctx.pols.B0[i]) && fr.isZero(ctx.pols.B1[i]) )
        {
            mpz_class balanceDifference = ctx.lastSWrite.res.newValue - ctx.lastSWrite.res.oldValue;
            ctx.totalTransferredBalance += balanceDifference;
        }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        ctx.mainMetrics.add("SMT Set", TimeDiff(ctx.t));
#endif
        ctx.lastSWrite.step = i;
    }

    // Store PoseidonG required data
    if (required != NULL)
    {
        // Declare PoseidonG required data
        array<Goldilocks::Element,17> pg;

        // Store PoseidonG required data
        for (uint64_t j=0; j<12; j++)
        {
            pg[j] = ctx.lastSWrite.Kin0[j];
        }
        for (uint64_t j=0; j<4; j++)
        {
            pg[12+j] = ctx.lastSWrite.keyI[j];
        }
        pg[16] = fr.fromU64(POSEIDONG_PERMUTATION1_ID);
        required->PoseidonG.push_back(pg);

        // Store PoseidonG required data
        for (uint64_t j=0; j<12; j++)
        {
            pg[j] = ctx.lastSWrite.Kin1[j];
        }
        for (uint64_t j=0; j<4; j++)
        {
            pg[12+j] = ctx.lastSWrite.key[j];
        }
        pg[16] = fr.fromU64(POSEIDONG_PERMUTATION2_ID);
        required->PoseidonG.push_back(pg);
    }

    if (required != NULL)
    {
        SmtAction smtAction;
        smtAction.bIsSet = true;
        smtAction.setResult = ctx.lastSWrite.res;
        required->Storage.push_back(smtAction);
    }

    // Check that the new root hash equals op0
    Goldilocks::Element oldRoot[4];
    sr8to4(fr, op0, op1, op2, op3, op4, op5, op6, op7, oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

    if ( !fr.equal(ctx.lastSWrite.newRoot[0], oldRoot[0]) ||
         !fr.equal(ctx.lastSWrite.newRoot[1], oldRoot[1]) ||
         !fr.equal(ctx.lastSWrite.newRoot[2], oldRoot[2]) ||
         !fr.equal(ctx.lastSWrite.newRoot[3], oldRoot[3]) )
    {
        zklog.error("Storage_write_verify() Storage write does not match: ctx.lastSWrite.newRoot: " + fr.toString(ctx.lastSWrite.newRoot[3], 16) + ":" + fr.toString(ctx.lastSWrite.newRoot[2], 16) + ":" + fr.toString(ctx.lastSWrite.newRoot[1], 16) + ":" + fr.toString(ctx.lastSWrite.newRoot[0], 16) +
            " oldRoot: " + fr.toString(oldRoot[3], 16) + ":" + fr.toString(oldRoot[2], 16) + ":" + fr.toString(oldRoot[1], 16) + ":" + fr.toString(oldRoot[0], 16));
        return ZKR_SM_MAIN_STORAGE_WRITE_MISMATCH;
    }

    Goldilocks::Element fea[4];
    sr8to4(fr, op0, op1, op2, op3, op4, op5, op6, op7, fea[0], fea[1], fea[2], fea[3]);
    if ( !fr.equal(ctx.lastSWrite.newRoot[0], fea[0]) ||
            !fr.equal(ctx.lastSWrite.newRoot[1], fea[1]) ||
            !fr.equal(ctx.lastSWrite.newRoot[2], fea[2]) ||
            !fr.equal(ctx.lastSWrite.newRoot[3], fea[3]) )
    {
        zklog.error("Storage_write_verify() Storage write does not match: ctx.lastSWrite.newRoot=" + fea2string(fr, ctx.lastSWrite.newRoot) + " op=" + fea2string(fr, fea));
        return ZKR_SM_MAIN_STORAGE_WRITE_MISMATCH;
    }

    for (uint64_t k=0; k<4; k++)
    {
        ctx.pols.sKeyI[k][i] =  ctx.lastSWrite.keyI[k];
        ctx.pols.sKey[k][i] = ctx.lastSWrite.key[k];
    }
    
    return ZKR_SUCCESS;
}

} // namespace