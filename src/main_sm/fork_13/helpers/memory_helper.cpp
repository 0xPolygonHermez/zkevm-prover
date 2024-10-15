#include "memory_helper.hpp"
#include "zklog.hpp"
#include "definitions.hpp"

namespace fork_13
{

void Memory_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t memAddr)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].mOp == 1);
    zkassert(ctx.rom.line[*ctx.pZKPC].mWR == 0);

    std::unordered_map<uint64_t, Fea>::iterator memIterator;
    memIterator = ctx.mem.find(memAddr);
    if (memIterator != ctx.mem.end()) {
#ifdef LOG_MEMORY
        zklog.info("Memory_calculate() read mRD: memAddr:" + to_string(memAddr) + " " + fea2stringchain(fr, ctx.mem[memAddr].fe0, ctx.mem[memAddr].fe1, ctx.mem[memAddr].fe2, ctx.mem[memAddr].fe3, ctx.mem[memAddr].fe4, ctx.mem[memAddr].fe5, ctx.mem[memAddr].fe6, ctx.mem[memAddr].fe7));
#endif
        fi0 = memIterator->second.fe0;
        fi1 = memIterator->second.fe1;
        fi2 = memIterator->second.fe2;
        fi3 = memIterator->second.fe3;
        fi4 = memIterator->second.fe4;
        fi5 = memIterator->second.fe5;
        fi6 = memIterator->second.fe6;
        fi7 = memIterator->second.fe7;

    } else {
        fi0 = fr.zero();
        fi1 = fr.zero();
        fi2 = fr.zero();
        fi3 = fr.zero();
        fi4 = fr.zero();
        fi5 = fr.zero();
        fi6 = fr.zero();
        fi7 = fr.zero();
    }
}

zkresult Memory_verify ( Context &ctx,
                         Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                         MainExecRequired *required,
                         int32_t memAddr)
{
    zkassert(ctx.pZKPC != NULL);
    uint64_t zkPC = *ctx.pZKPC;
    zkassert(ctx.rom.line[*ctx.pZKPC].mOp == 1);
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    // If MEMORY WRITE, store op in memory
    if (ctx.rom.line[zkPC].mWR == 1)
    {
        ctx.mem[memAddr].fe0 = op0;
        ctx.mem[memAddr].fe1 = op1;
        ctx.mem[memAddr].fe2 = op2;
        ctx.mem[memAddr].fe3 = op3;
        ctx.mem[memAddr].fe4 = op4;
        ctx.mem[memAddr].fe5 = op5;
        ctx.mem[memAddr].fe6 = op6;
        ctx.mem[memAddr].fe7 = op7;

        if (required != NULL)
        {
            MemoryAccess memoryAccess;
            memoryAccess.bIsWrite = true;
            memoryAccess.address = memAddr;
            memoryAccess.pc = i;
            memoryAccess.fe0 = op0;
            memoryAccess.fe1 = op1;
            memoryAccess.fe2 = op2;
            memoryAccess.fe3 = op3;
            memoryAccess.fe4 = op4;
            memoryAccess.fe5 = op5;
            memoryAccess.fe6 = op6;
            memoryAccess.fe7 = op7;
            required->Memory.push_back(memoryAccess);
        }
        if (!ctx.bProcessBatch)
        {
            ctx.pols.mOp[i] = fr.one();
            ctx.pols.mWR[i] = fr.one();
        }
#ifdef LOG_MEMORY
        zklog.info("Memory write mWR: memAddr:" + to_string(memAddr) + " " + fea2stringchain(fr, ctx.mem[memAddr].fe0, ctx.mem[memAddr].fe1, ctx.mem[memAddr].fe2, ctx.mem[memAddr].fe3, ctx.mem[memAddr].fe4, ctx.mem[memAddr].fe5, ctx.mem[memAddr].fe6, ctx.mem[memAddr].fe7));
#endif
    }
    // If MEMORY READ, check that OP equals memory read value
    else
    {
        Goldilocks::Element value[8];
        if (ctx.rom.line[zkPC].assumeFree == 1)
        {
            value[0] = ctx.pols.FREE0[i];
            value[1] = ctx.pols.FREE1[i];
            value[2] = ctx.pols.FREE2[i];
            value[3] = ctx.pols.FREE3[i];
            value[4] = ctx.pols.FREE4[i];
            value[5] = ctx.pols.FREE5[i];
            value[6] = ctx.pols.FREE6[i];
            value[7] = ctx.pols.FREE7[i];
        }
        else
        {
            value[0] = op0;
            value[1] = op1;
            value[2] = op2;
            value[3] = op3;
            value[4] = op4;
            value[5] = op5;
            value[6] = op6;
            value[7] = op7;
        }
        if (required != NULL)
        {
            MemoryAccess memoryAccess;
            memoryAccess.bIsWrite = false;
            memoryAccess.address = memAddr;
            memoryAccess.pc = i;
            memoryAccess.fe0 = value[0];
            memoryAccess.fe1 = value[1];
            memoryAccess.fe2 = value[2];
            memoryAccess.fe3 = value[3];
            memoryAccess.fe4 = value[4];
            memoryAccess.fe5 = value[5];
            memoryAccess.fe6 = value[6];
            memoryAccess.fe7 = value[7];
            required->Memory.push_back(memoryAccess);
        }
        if (!ctx.bProcessBatch)
        {
            ctx.pols.mOp[i] = fr.one();
        }
        if (ctx.mem.find(memAddr) != ctx.mem.end())
        {
            if ( (!fr.equal(ctx.mem[memAddr].fe0, value[0])) ||
                 (!fr.equal(ctx.mem[memAddr].fe1, value[1])) ||
                 (!fr.equal(ctx.mem[memAddr].fe2, value[2])) ||
                 (!fr.equal(ctx.mem[memAddr].fe3, value[3])) ||
                 (!fr.equal(ctx.mem[memAddr].fe4, value[4])) ||
                 (!fr.equal(ctx.mem[memAddr].fe5, value[5])) ||
                 (!fr.equal(ctx.mem[memAddr].fe6, value[6])) ||
                 (!fr.equal(ctx.mem[memAddr].fe7, value[7])) )
            {
                zklog.error("Memory_verify() Memory Read does not match value=" + fea2stringchain(fr, value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7]) +
                    " mem=" + fea2stringchain(fr, ctx.mem[memAddr].fe0, ctx.mem[memAddr].fe1, ctx.mem[memAddr].fe2, ctx.mem[memAddr].fe3, ctx.mem[memAddr].fe4, ctx.mem[memAddr].fe5, ctx.mem[memAddr].fe6, ctx.mem[memAddr].fe7));
                return ZKR_SM_MAIN_MEMORY;
            }
        }
        else
        {
            if ( (!fr.isZero(value[0])) ||
                 (!fr.isZero(value[1])) ||
                 (!fr.isZero(value[2])) ||
                 (!fr.isZero(value[3])) ||
                 (!fr.isZero(value[4])) ||
                 (!fr.isZero(value[5])) ||
                 (!fr.isZero(value[6])) ||
                 (!fr.isZero(value[7])) )
            {
                zklog.error("Memory_verify() Memory Read does not match (value!=0) value=" + fea2stringchain(fr, value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7]));
                return ZKR_SM_MAIN_MEMORY;
            }
        }
    }

    return ZKR_SUCCESS;
}

} // namespace