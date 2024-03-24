#include <nlohmann/json.hpp>
#include "storage_executor.hpp"
#include "storage_rom.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "poseidon_g_permutation.hpp"
#include "goldilocks_precomputed.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "climb_key_executor.hpp"

using json = nlohmann::json;
using namespace std;

#define PRE_CLIMB_UP_LIMIT 0x7FFFFFFF80000000ULL

//#define LOG_STORAGE_EXECUTOR
//#define LOG_STORAGE_EXECUTOR_ROM_LINE

void StorageExecutor::execute (vector<SmtAction> &action, StorageCommitPols &pols, vector<array<Goldilocks::Element, 17>> &poseidonRequired, vector<ClimbKeyAction> &climbKeyRequired)
{
    uint64_t l=0; // rom line number, so current line is rom.line[l]
    uint64_t a=0; // action number, so current action is action[a]
    bool actionListEmpty = (action.size()==0); // becomes true when we run out of actions
    uint64_t lastStep = 0; // Set to the first evaluation that calls isAlmostEndPolynomial

    // Init the context if the list is not empty
    SmtActionContext ctx;
    if (!actionListEmpty)
    {
        ctx.init(fr, action[a]);
    }

    // For all polynomial evaluations
    for (uint64_t i=0; i<N; i++) // TODO: How do we control if we run out of space?  exit(-1)?
    {
        // op is the internal register, reset to 0 at every evaluation
        Goldilocks::Element op[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};

        // Current rom line is set by the program counter of this evaluation
        l = fr.toU64(pols.pc[i]);

        // Set the next evaluation index, which will be 0 when we reach the last evaluation
        uint64_t nexti = (i+1)%N;

#ifdef LOG_STORAGE_EXECUTOR_ROM_LINE
        string source = "";
        if (rom.line[l].funcName!="isAlmostEndPolynomial")
        {
            source = rom.line[l].fileName.substr(8, rom.line[l].fileName.length() - 14) + ":" + to_string(rom.line[l].line);
            printf("[SR%04d I%03d %-28s] %s\n", (int)l, (int)a, source.c_str(), rom.line[l].lineStr.c_str());
            // rom.line[l].print(l); // Print the rom line content
        }
#endif
        /*************/
        /* Selectors */
        /*************/

        // When the rom assembler code calls inFREE, it specifies the requested input data
        // using an operation + function name string couple

        if (rom.line[l].inFREE)
        {
            const int64_t currentLevel = fr.toU64(pols.level[i]);

            if (rom.line[l].op == "functionCall")
            {
                /* Possible values of mode when action is SMT Set:
                    - update -> update existing value
                    - insertFound -> insert with found key; found a leaf node with a common set of key bits
                    - insertNotFound -> insert with no found key
                    - deleteFound -> delete with found key
                    - deleteNotFound -> delete with no found key
                    - deleteLast -> delete the last node, so root becomes 0
                    - zeroToZero -> value was zero and remains zero
                */
                if (rom.line[l].funcName == "isSetUpdate")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "update")
                    {
                        op[0] = fr.one();

#ifdef LOG_STORAGE_EXECUTOR
                        zklog.info("StorageExecutor isUpdate returns " + fea2string(fr, op));
#endif
                    }
                }
                else if (rom.line[l].funcName == "isSetInsertFound")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "insertFound")
                    {
                        op[0] = fr.one();

#ifdef LOG_STORAGE_EXECUTOR
                        zklog.info("StorageExecutor isInsertFound returns " + fea2string(fr, op));
#endif
                    }
                }
                else if (rom.line[l].funcName == "isSetInsertNotFound")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "insertNotFound")
                    {
                        op[0] = fr.one();

#ifdef LOG_STORAGE_EXECUTOR
                        zklog.info("StorageExecutor isInsertNotFound returns " + fea2string(fr, op));
#endif
                    }
                }
                else if (rom.line[l].funcName == "isSetDeleteLast")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "deleteLast")
                    {
                        op[0] = fr.one();

#ifdef LOG_STORAGE_EXECUTOR
                        zklog.info("StorageExecutor isDeleteLast returns " + fea2string(fr, op));
#endif
                    }
                }
                else if (rom.line[l].funcName == "isSetDeleteFound")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "deleteFound")
                    {
                        op[0] = fr.one();

#ifdef LOG_STORAGE_EXECUTOR
                        zklog.info("StorageExecutor isSetDeleteFound returns " + fea2string(fr, op));
#endif
                    }
                }
                else if (rom.line[l].funcName == "isSetDeleteNotFound")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "deleteNotFound")
                    {
                        op[0] = fr.one();

#ifdef LOG_STORAGE_EXECUTOR
                        zklog.info("StorageExecutor isSetDeleteNotFound returns " + fea2string(fr, op));
#endif
                    }
                }
                else if (rom.line[l].funcName == "isSetZeroToZero")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "zeroToZero")
                    {
                        op[0] = fr.one();

#ifdef LOG_STORAGE_EXECUTOR
                        zklog.info("StorageExecutor isZeroToZero returns " + fea2string(fr, op));
#endif
                    }
                }

                // The SMT action can be a final leaf (isOld0 = true)
                else if (rom.line[l].funcName == "GetIsOld0")
                {
                    if (!actionListEmpty && (action[a].bIsSet ? action[a].setResult.isOld0 : action[a].getResult.isOld0))
                    {
                        op[0] = fr.one();
#ifdef LOG_STORAGE_EXECUTOR
                        zklog.info("StorageExecutor isOld0 returns " + fea2string(fr, op));
#endif
                    }
                }

                // The SMT action can be a get, which can return a zero value (key not found) or a non-zero value
                else if (rom.line[l].funcName=="isGet")
                {
                    if (!actionListEmpty &&
                        !action[a].bIsSet)
                    {
                        op[0] = fr.one();

#ifdef LOG_STORAGE_EXECUTOR
                        zklog.info("StorageExecutor isGet returns " + fea2string(fr, op));
#endif
                    }
                }

                // Get the remaining key, i.e. the key after removing the bits used in the tree node navigation
                else if (rom.line[l].funcName=="GetRkey")
                {
                    op[0] = ctx.rKey[0];
                    op[1] = ctx.rKey[1];
                    op[2] = ctx.rKey[2];
                    op[3] = ctx.rKey[3];

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetRkey returns " + fea2string(fr, op));
#endif
                }

                // Get the sibling remaining key, i.e. the part that is not common to the value key
                else if (rom.line[l].funcName=="GetSiblingRkey")
                {
                    op[0] = ctx.siblingRKey[0];
                    op[1] = ctx.siblingRKey[1];
                    op[2] = ctx.siblingRKey[2];
                    op[3] = ctx.siblingRKey[3];

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetSiblingRkey returns " + fea2string(fr, op));
#endif
                }

                // Get the sibling hash, obtained from the siblings array of the current level,
                // taking into account that the sibling bit is the opposite (1-x) of the value bit
                else if (rom.line[l].funcName=="GetSiblingHash")
                {
                    if (action[a].bIsSet)
                    {
                        op[0] = action[a].setResult.siblings[currentLevel][(1-ctx.bits[currentLevel])*4];
                        op[1] = action[a].setResult.siblings[currentLevel][(1-ctx.bits[currentLevel])*4+1];
                        op[2] = action[a].setResult.siblings[currentLevel][(1-ctx.bits[currentLevel])*4+2];
                        op[3] = action[a].setResult.siblings[currentLevel][(1-ctx.bits[currentLevel])*4+3];
                    }
                    else
                    {
                        op[0] = action[a].getResult.siblings[currentLevel][(1-ctx.bits[currentLevel])*4];
                        op[1] = action[a].getResult.siblings[currentLevel][(1-ctx.bits[currentLevel])*4+1];
                        op[2] = action[a].getResult.siblings[currentLevel][(1-ctx.bits[currentLevel])*4+2];
                        op[3] = action[a].getResult.siblings[currentLevel][(1-ctx.bits[currentLevel])*4+3];
                    }

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetSiblingHash returns " + fea2string(fr, op));
#endif
                }
                else if (rom.line[l].funcName=="GetSiblingLeftChildHash")
                {
                    if (action[a].bIsSet)
                    {
                        op[0] = action[a].setResult.siblingLeftChild[0];
                        op[1] = action[a].setResult.siblingLeftChild[1];
                        op[2] = action[a].setResult.siblingLeftChild[2];
                        op[3] = action[a].setResult.siblingLeftChild[3];
                    }
                    else
                    {
                        zklog.error("StorageExecutor.execute() called GetSiblingLeftChildHash() on GET operation input = " + to_string(a));
                        exitProcess();
                    }

// #ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetSiblingLeftChildHash returns " + fea2string(fr, op) + " input=" + to_string(a));
// #endif
                }
                else if (rom.line[l].funcName=="GetSiblingRightChildHash")
                {
                    if (action[a].bIsSet)
                    {
                        op[0] = action[a].setResult.siblingRightChild[0];
                        op[1] = action[a].setResult.siblingRightChild[1];
                        op[2] = action[a].setResult.siblingRightChild[2];
                        op[3] = action[a].setResult.siblingRightChild[3];
                    }
                    else
                    {
                        zklog.error("StorageExecutor.execute() called GetSiblingRightChildHash() on GET operation input = " + to_string(a));
                        exitProcess();
                    }

// #ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetSiblingRightChildHash returns " + fea2string(fr, op) + " input=" + to_string(a));
// #endif
                }

                // Return if value is zero

                else if (rom.line[l].funcName=="isValueZero")
                {
                    // if ctionList is empty => finish, value is zero
                    if (actionListEmpty || (action[a].bIsSet ? action[a].setResult.newValue : action[a].getResult.value) == 0) {
                        op[0] = fr.one();
                    }

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor isValueZero returns " + fea2string(fr, op));
#endif
                }

                // Value is an u256 split in 8 u32 chuncks, each one stored in the lower 32 bits of an u63 field element
                // u63 means that it is not an u64, since some of the possible values are lost due to the prime effect

                // Get the lower 4 field elements of the value
                else if (rom.line[l].funcName=="GetValueLow")
                {
                    Goldilocks::Element fea[8];
                    scalar2fea(fr, action[a].bIsSet ? action[a].setResult.newValue : action[a].getResult.value, fea);
                    op[0] = fea[0];
                    op[1] = fea[1];
                    op[2] = fea[2];
                    op[3] = fea[3];

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetValueLow returns " + fea2string(fr, op));
#endif
                }

                // Get the higher 4 field elements of the value
                else if (rom.line[l].funcName=="GetValueHigh")
                {
                    Goldilocks::Element fea[8];
                    scalar2fea(fr, action[a].bIsSet ? action[a].setResult.newValue : action[a].getResult.value, fea);
                    op[0] = fea[4];
                    op[1] = fea[5];
                    op[2] = fea[6];
                    op[3] = fea[7];

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetValueHigh returns " + fea2string(fr, op));
#endif
                }

                // Get the lower 4 field elements of the sibling value
                else if (rom.line[l].funcName=="GetSiblingValueLow")
                {
                    Goldilocks::Element fea[8];
                    scalar2fea(fr, action[a].bIsSet ? action[a].setResult.insValue : action[a].getResult.insValue, fea);
                    op[0] = fea[0];
                    op[1] = fea[1];
                    op[2] = fea[2];
                    op[3] = fea[3];

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetSiblingValueLow returns " + fea2string(fr, op));
#endif
                }

                // Get the higher 4 field elements of the sibling value
                else if (rom.line[l].funcName=="GetSiblingValueHigh")
                {
                    Goldilocks::Element fea[8];
                    scalar2fea(fr, action[a].bIsSet ? action[a].setResult.insValue : action[a].getResult.insValue, fea);
                    op[0] = fea[4];
                    op[1] = fea[5];
                    op[2] = fea[6];
                    op[3] = fea[7];

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetSiblingValueHigh returns " + fea2string(fr, op));
#endif
                }

                // Get the lower 4 field elements of the old value
                else if (rom.line[l].funcName=="GetOldValueLow")
                {
                    // This call only makes sense then this is an SMT set
                    if (!action[a].bIsSet)
                    {
                        zklog.error("StorageExecutor() GetOldValueLow called in an SMT get action");
                        exitProcess();
                    }

                    // Convert the oldValue scalar to an 8 field elements array
                    Goldilocks::Element fea[8];
                    scalar2fea(fr, action[a].setResult.oldValue, fea);

                    // Take the lower 4 field elements
                    op[0] = fea[0];
                    op[1] = fea[1];
                    op[2] = fea[2];
                    op[3] = fea[3];

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetOldValueLow returns " + fea2string(fr, op));
#endif
                }

                // Get the higher 4 field elements of the old value
                else if (rom.line[l].funcName=="GetOldValueHigh")
                {
                    // This call only makes sense then this is an SMT set
                    if (!action[a].bIsSet)
                    {
                        zklog.error("StorageExecutor() GetOldValueLow called in an SMT get action");
                        exitProcess();
                    }

                    // Convert the oldValue scalar to an 8 field elements array
                    Goldilocks::Element fea[8];
                    scalar2fea(fr, action[a].setResult.oldValue, fea);

                    // Take the higher 4 field elements
                    op[0] = fea[4];
                    op[1] = fea[5];
                    op[2] = fea[6];
                    op[3] = fea[7];

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetOldValueHigh returns " + fea2string(fr, op));
#endif
                }

                // Get the level number
                else if (rom.line[l].funcName=="GetLevel")
                {
                    // Check that we have the no parameters
                    if (rom.line[l].params.size()!=0)
                    {
                        zklog.error("StorageExecutor() called with GetBit but wrong number of parameters=" + to_string(rom.line[l].params.size()));
                        exitProcess();
                    }

                    // Set the bit in op[0]
                    if (ctx.level)
                    {
                        op[0] = fr.fromU64(ctx.level);
                    }

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetLevel() returns " + fea2string(fr, op));
#endif
                }

                // Returns 0 if we reached the top of the tree, i.e. if the current level is 0
                else if (rom.line[l].funcName=="GetTopTree")
                {
                    // Return 0 only if we reached the end of the tree, i.e. if the current level is 0
                    if (currentLevel > 0)
                    {
                        op[0] = fr.one();
                    }

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetTopTree returns " + fea2string(fr, op));
#endif
                }

                // Returns 0 if we reached the top of the branch, i.e. if the level matches the siblings size
                else if (rom.line[l].funcName=="GetTopOfBranch")
                {
                    // If we have consumed enough key bits to reach the deepest level of the siblings array, then we are at the top of the branch and we can start climing the tree
                    int64_t siblingsSize = action[a].bIsSet ? action[a].setResult.siblings.size() : action[a].getResult.siblings.size();
                    if (currentLevel > siblingsSize )
                    {
                        op[0] = fr.one();
                    }

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetTopOfBranch returns " + fea2string(fr, op));
#endif
                }

                // Get the next key bit
                // This call decrements automatically the current level
                else if (rom.line[l].funcName=="GetNextKeyBit")
                {
                    // Decrease current level
                    ctx.currentLevel--;
                    if (ctx.currentLevel<0)
                    {
                        zklog.error("StorageExecutor.execute() GetNextKeyBit() found ctx.currentLevel<0 =" + to_string(ctx.currentLevel));
                        exitProcess();
                    }

                    // Get the key bit corresponding to the current level
                    op[0] = fr.fromU64(ctx.bits[ctx.currentLevel]);

#ifdef LOG_STORAGE_EXECUTOR
                    zklog.info("StorageExecutor GetNextKeyBit returns " + fea2string(fr, op));
#endif
                }

                // Return 1 if we completed all evaluations, except one
                else if (rom.line[l].funcName=="isAlmostEndPolynomial")
                {
                    // Return one if this is the one before the last evaluation of the polynomials
                    if (i == (N-2))
                    {
                        op[0] = fr.one();
#ifdef LOG_STORAGE_EXECUTOR
                        zklog.info("StorageExecutor isEndPolynomial returns " + fea2string(fr,op));
#endif
                    }

                    // Record the first time isAlmostEndPolynomial is called
                    if (lastStep == 0) lastStep = i;
                }
                else
                {
                    zklog.error("StorageExecutor() unknown funcName:" + rom.line[l].funcName);
                    exitProcess();
                }
            }
            else if (rom.line[l].climbRkey) {
                const int bit = rom.line[l].climbBitN? 1 - fr.toU64(pols.rkeyBit[i]) : fr.toU64(pols.rkeyBit[i]);
                const int level = fr.toU64(pols.level[i]);
                const int zlevel = level % 4;
                Goldilocks::Element rkeys[4] = {pols.rkey0[i], pols.rkey1[i], pols.rkey2[i], pols.rkey3[i]};
                Goldilocks::Element rkeyClimbed;

                if (!ClimbKeyHelper::calculate(fr, rkeys[zlevel], bit, rkeyClimbed)) {
                    zklog.error("StorageExecutor() ClimbRkey fails because rkey["+to_string(zlevel)+"] has an invalid value ("+fr.toString(rkeys[zlevel])+") before climb with bit="+to_string(bit));
                    exitProcess();
                }
                rkeys[zlevel] = rkeyClimbed;
                op[0] = rkeys[0];
                op[1] = rkeys[1];
                op[2] = rkeys[2];
                op[3] = rkeys[3];
            }
            else if (rom.line[l].climbSiblingRkey) {
                const int bit = rom.line[l].climbBitN? 1 - fr.toU64(pols.rkeyBit[i]) : fr.toU64(pols.rkeyBit[i]);
                const int level = fr.toU64(pols.level[i]);
                const int zlevel = level % 4;
                Goldilocks::Element rkeys[4] = {pols.siblingRkey0[i], pols.siblingRkey1[i], pols.siblingRkey2[i], pols.siblingRkey3[i]};
                Goldilocks::Element rkeyClimbed;

                if (!ClimbKeyHelper::calculate(fr, rkeys[zlevel], bit, rkeyClimbed)) {
                    zklog.error("StorageExecutor() climbSiblingRkey fails because siblingRkey["+to_string(zlevel)+"] has an invalid value ("+fr.toString(rkeys[zlevel])+") before climb with bit="+to_string(bit));
                    exitProcess();
                }
                rkeys[zlevel] = rkeyClimbed;
                op[0] = rkeys[0];
                op[1] = rkeys[1];
                op[2] = rkeys[2];
                op[3] = rkeys[3];
            }

            // Ignore; this is just to report a list of setters
            else if (rom.line[l].op=="")
            {
            }

            // Any other value is an unexpected value
            else
            {
                zklog.error("StorageExecutor() unknown op:" + rom.line[l].op);
                exitProcess();
            }

            // free[] = op[]
            if (!fr.isZero(op[0])) pols.free0[i] = op[0];
            if (!fr.isZero(op[1])) pols.free1[i] = op[1];
            if (!fr.isZero(op[2])) pols.free2[i] = op[2];
            if (!fr.isZero(op[3])) pols.free3[i] = op[3];

            // Mark the selFree register as 1
            pols.inFree[i] = fr.one();
        }

        // If a constant is provided, add constant to op0
        if (rom.line[l].CONST!="")
        {
            // Convert constant to scalar
            mpz_class constScalar;
            constScalar.set_str(rom.line[l].CONST, 10);
            Goldilocks::Element const0 = fr.fromS64(constScalar.get_si());
            op[0] = fr.add(op[0], const0);

            // Store constant field elements in their registers
            pols.const0[i] = const0;
        }

        // If inOLD_ROOT then op=OLD_ROOT
        if (rom.line[l].inOLD_ROOT)
        {
            op[0] = fr.add(op[0], pols.oldRoot0[i]);
            op[1] = fr.add(op[1], pols.oldRoot1[i]);
            op[2] = fr.add(op[2], pols.oldRoot2[i]);
            op[3] = fr.add(op[3], pols.oldRoot3[i]);
            pols.inOldRoot[i] = fr.one();
        }

        // If inNEW_ROOT then op=NEW_ROOT
        if (rom.line[l].inNEW_ROOT)
        {
            op[0] = fr.add(op[0], pols.newRoot0[i]);
            op[1] = fr.add(op[1], pols.newRoot1[i]);
            op[2] = fr.add(op[2], pols.newRoot2[i]);
            op[3] = fr.add(op[3], pols.newRoot3[i]);
            pols.inNewRoot[i] = fr.one();
        }

        // If inRKEY_BIT then op=RKEY_BIT
        if (rom.line[l].inRKEY_BIT)
        {
            op[0] = fr.add(op[0], pols.rkeyBit[i]);
            op[1] = fr.add(op[1], fr.zero());
            op[2] = fr.add(op[2], fr.zero());
            op[3] = fr.add(op[3], fr.zero());
            pols.inRkeyBit[i] = fr.one();
        }

        // If inVALUE_LOW then op=VALUE_LOW
        if (rom.line[l].inVALUE_LOW)
        {
            op[0] = fr.add(op[0], pols.valueLow0[i]);
            op[1] = fr.add(op[1], pols.valueLow1[i]);
            op[2] = fr.add(op[2], pols.valueLow2[i]);
            op[3] = fr.add(op[3], pols.valueLow3[i]);
            pols.inValueLow[i] = fr.one();
        }

        // If inVALUE_HIGH then op=VALUE_HIGH
        if (rom.line[l].inVALUE_HIGH)
        {
            op[0] = fr.add(op[0], pols.valueHigh0[i]);
            op[1] = fr.add(op[1], pols.valueHigh1[i]);
            op[2] = fr.add(op[2], pols.valueHigh2[i]);
            op[3] = fr.add(op[3], pols.valueHigh3[i]);
            pols.inValueHigh[i] = fr.one();
        }

        // If inRKEY then op=RKEY
        if (rom.line[l].inRKEY)
        {
            op[0] = fr.add(op[0], pols.rkey0[i]);
            op[1] = fr.add(op[1], pols.rkey1[i]);
            op[2] = fr.add(op[2], pols.rkey2[i]);
            op[3] = fr.add(op[3], pols.rkey3[i]);
            pols.inRkey[i] = fr.one();
        }

        // If inSIBLING_RKEY then op=SIBLING_RKEY
        if (rom.line[l].inSIBLING_RKEY)
        {
            pols.inSiblingRkey[i] = fr.fromS64(rom.line[l].inSIBLING_RKEY);
            op[0] = fr.add(op[0], fr.mul(pols.inSiblingRkey[i], pols.siblingRkey0[i]));
            op[1] = fr.add(op[1], fr.mul(pols.inSiblingRkey[i], pols.siblingRkey1[i]));
            op[2] = fr.add(op[2], fr.mul(pols.inSiblingRkey[i], pols.siblingRkey2[i]));
            op[3] = fr.add(op[3], fr.mul(pols.inSiblingRkey[i], pols.siblingRkey3[i]));
        }

        // If inSIBLING_VALUE_HASH then op=SIBLING_VALUE_HASH
        if (rom.line[l].inSIBLING_VALUE_HASH)
        {
            op[0] = fr.add(op[0], pols.siblingValueHash0[i]);
            op[1] = fr.add(op[1], pols.siblingValueHash1[i]);
            op[2] = fr.add(op[2], pols.siblingValueHash2[i]);
            op[3] = fr.add(op[3], pols.siblingValueHash3[i]);
            pols.inSiblingValueHash[i] = fr.one();
        }

        // If inROTL_VH then op=rotate_left(VALUE_HIGH)
        if (rom.line[l].inROTL_VH)
        {
            op[0] = fr.add(op[0], pols.valueHigh3[i]);
            op[1] = fr.add(op[1], pols.valueHigh0[i]);
            op[2] = fr.add(op[2], pols.valueHigh1[i]);
            op[3] = fr.add(op[3], pols.valueHigh2[i]);
            pols.inRotlVh[i] = fr.one();
        }

        // If inROTL_VH then op=rotate_left(VALUE_HIGH)
        if (rom.line[l].inLEVEL)
        {
            pols.inLevel[i] = fr.one();
            op[0] = fr.add(op[0], pols.level[i]);
        }

        /****************/
        /* Instructions */
        /****************/

        // JMPZ: Jump if OP==0
        if (rom.line[l].jmpz)
        {
            if (fr.isZero(op[0]))
            {
                pols.pc[nexti] = fr.fromU64(rom.line[l].jmpAddress);
                //zklog.info("StorageExecutor jmpz jmpAddress=" + to_string(rom.line[l].jmpAddress));
            }
            else
            {
                pols.pc[nexti] = fr.add(pols.pc[i], fr.one());
            }
            pols.jmpAddress[i] = fr.fromU64(rom.line[l].jmpAddress);
            pols.jmpz[i] = fr.one();
        }

        // JMPNZ: Jump if OP!=0
        else if (rom.line[l].jmpnz)
        {
            if (fr.isZero(op[0]))
            {
                pols.pc[nexti] = fr.add(pols.pc[i], fr.one());
            }
            else
            {
                pols.pc[nexti] = fr.fromU64(rom.line[l].jmpAddress);
                //zklog.info("StorageExecutor jmpz jmpAddress=" + to_string(rom.line[l].jmpAddress));
            }
            pols.jmpAddress[i] = fr.fromU64(rom.line[l].jmpAddress);
            pols.jmpnz[i] = fr.one();
        }

        // JMP: Jump always
        else if (rom.line[l].jmp)
        {
            pols.pc[nexti] = fr.fromU64(rom.line[l].jmpAddress);
            pols.jmpAddress[i] = fr.fromU64(rom.line[l].jmpAddress);
            //zklog.info("StorageExecutor iJmp jmpAddress=" + to_string(rom.line[l].jmpAddress));
            pols.jmp[i] = fr.one();
        }

        // If not any jump, then simply increment program counter
        else
        {
            pols.pc[nexti] = fr.add(pols.pc[i], fr.one());
        }

        // Hash: op = poseidon.hash(HASH_LEFT + HASH_RIGHT + (0 or 1, depending on iHashType))
        if (rom.line[l].hash)
        {
            // Prepare the data to hash: HASH_LEFT + HASH_RIGHT + 0 or 1, depending on iHashType
            Goldilocks::Element fea[12];
            fea[0] = pols.hashLeft0[i];
            fea[1] = pols.hashLeft1[i];
            fea[2] = pols.hashLeft2[i];
            fea[3] = pols.hashLeft3[i];
            fea[4] = pols.hashRight0[i];
            fea[5] = pols.hashRight1[i];
            fea[6] = pols.hashRight2[i];
            fea[7] = pols.hashRight3[i];
            if (rom.line[l].hashType==0)
            {
                fea[8] = fr.zero();
            }
            else if (rom.line[l].hashType==1)
            {
                fea[8] = fr.one();
                pols.hashType[i] = fr.one();
            }
            else
            {
                zklog.error("StorageExecutor:execute() found invalid iHashType=" + to_string(rom.line[l].hashType));
                exitProcess();
            }
            fea[9] = fr.zero();
            fea[10] = fr.zero();
            fea[11] = fr.zero();

#ifdef LOG_STORAGE_EXECUTOR
            Goldilocks::Element auxFea[12];
            for (uint64_t i=0; i<12; i++) auxFea[i] = fea[i];
#endif
            // To be used to load required poseidon data
            array<Goldilocks::Element,17> req;
            for (uint64_t j=0; j<12; j++)
            {
                req[j] = fea[j];
            }

            // Call poseidon
            Goldilocks::Element feaHash[4];
            poseidon.hash(feaHash, fea);

            // Get the calculated hash from the first 4 elements
            pols.free0[i] = feaHash[0];
            pols.free1[i] = feaHash[1];
            pols.free2[i] = feaHash[2];
            pols.free3[i] = feaHash[3];

            op[0] = fr.add(op[0], fr.mul(fr.fromU64(rom.line[l].inFREE), feaHash[0]));
            op[1] = fr.add(op[1], fr.mul(fr.fromU64(rom.line[l].inFREE), feaHash[1]));
            op[2] = fr.add(op[2], fr.mul(fr.fromU64(rom.line[l].inFREE), feaHash[2]));
            op[3] = fr.add(op[3], fr.mul(fr.fromU64(rom.line[l].inFREE), feaHash[3]));

            pols.hash[i] = fr.one();

            req[12] = feaHash[0];
            req[13] = feaHash[1];
            req[14] = feaHash[2];
            req[15] = feaHash[3];
            req[16] = fr.fromU64(POSEIDONG_PERMUTATION3_ID);
            poseidonRequired.push_back(req);

#ifdef LOG_STORAGE_EXECUTOR
            {
                string s = "StorageExecutor iHash hashType=" + to_string(rom.line[l].hashType) + " hash=" + fea2string(fr, op) + " value=";
                for (uint64_t i=0; i<12; i++) s += fr.toString(auxFea[i],16) + ":";
                zklog.info(s);
            }
#endif
        }

        if (rom.line[l].climbBitN) {
            pols.climbBitN[i] = fr.one();
#ifdef LOG_STORAGE_EXECUTOR
            zklog.info("StorageExecutor climbBitN = 1");
#endif
        }

        // Climb the remaining key, by injecting the RKEY_BIT in the register specified by LEVEL
        if (rom.line[l].climbRkey)
        {
            const int bit = rom.line[l].climbBitN? 1 - fr.toU64(pols.rkeyBit[i]) : fr.toU64(pols.rkeyBit[i]);
            const int level = fr.toU64(pols.level[i]);
            const int zlevel = level % 4;
            Goldilocks::Element rkeys[4] = {pols.rkey0[i], pols.rkey1[i], pols.rkey2[i], pols.rkey3[i]};
            Goldilocks::Element rkeyClimbed;

            if (!ClimbKeyHelper::calculate(fr, rkeys[zlevel], bit, rkeyClimbed)) {
                zklog.error("StorageExecutor() ClimbRkey fails because rkey["+to_string(zlevel)+"] has an invalid value ("+fr.toString(rkeys[zlevel])+") before climb with bit="+to_string(bit));
                exitProcess();
            }
            else if (!fr.equal(rkeyClimbed, op[zlevel])) {
                zklog.error("StorageExecutor() ClimbRkey fails because rkey["+to_string(zlevel)+"] not match ("+fr.toString(op[zlevel])+" vs "+fr.toString(rkeyClimbed)+") after climb with bit="+to_string(bit));
                exitProcess();
            }
            pols.climbRkey[i] = fr.one();

            ClimbKeyAction climbKeyAction;
            climbKeyAction.key[0] = rkeys[0];
            climbKeyAction.key[1] = rkeys[1];
            climbKeyAction.key[2] = rkeys[2];
            climbKeyAction.key[3] = rkeys[3];
            climbKeyAction.level = level;
            climbKeyAction.bit = bit;
            climbKeyRequired.push_back(climbKeyAction);

#ifdef LOG_STORAGE_EXECUTOR
            zklog.info("StorageExecutor iClimbRkey bit=" + to_string(bit) + " rkey=" + fea2string(fr,rkeys)+ " op=" + fea2string(fr, op));
#endif
        }

        // Climb the sibling remaining key, by injecting the sibling bit in the register specified by LEVEL
        if (rom.line[l].climbSiblingRkey)
        {
            const int bit = rom.line[l].climbBitN? 1 - fr.toU64(pols.rkeyBit[i]) : fr.toU64(pols.rkeyBit[i]);
            const int level = fr.toU64(pols.level[i]);
            const int zlevel = level % 4;
            Goldilocks::Element rkeys[4] = {pols.siblingRkey0[i], pols.siblingRkey1[i], pols.siblingRkey2[i], pols.siblingRkey3[i]};
            Goldilocks::Element rkeyClimbed;

            if (!ClimbKeyHelper::calculate(fr, rkeys[zlevel], bit, rkeyClimbed)) {
                zklog.error("StorageExecutor() climbSiblingRkey fails because siblingRkey["+to_string(zlevel)+"] has an invalid value ("+fr.toString(rkeys[zlevel])+") before climb with bit="+to_string(bit));
                exitProcess();
            }
            else if (!fr.equal(rkeyClimbed, op[zlevel])) {
                zklog.error("StorageExecutor() climbSiblingRkey fails because siblingRkey["+to_string(zlevel)+"] not match ("+fr.toString(op[zlevel])+" vs "+fr.toString(rkeyClimbed)+") after climb with bit="+to_string(bit));
                exitProcess();
            }
            pols.climbSiblingRkey[i] = fr.one();

            ClimbKeyAction climbKeyAction;
            climbKeyAction.key[0] = rkeys[0];
            climbKeyAction.key[1] = rkeys[1];
            climbKeyAction.key[2] = rkeys[2];
            climbKeyAction.key[3] = rkeys[3];
            climbKeyAction.level = level;
            climbKeyAction.bit = bit;
            climbKeyRequired.push_back(climbKeyAction);

#ifdef LOG_STORAGE_EXECUTOR
            zklog.info("StorageExecutor ClimbSiblingRkey bit=" + to_string(bit) + " rkey=" + fea2string(fr,rkeys)+ " op=" + fea2string(fr, op));
#endif
        }
        // Latch get: at this point consistency is granted: OLD_ROOT, RKEY (complete key), VALUE_LOW, VALUE_HIGH, LEVEL
        if (rom.line[l].latchGet)
        {
            // Check that the current action is an SMT get
            if (action[a].bIsSet)
            {
                zklog.error("StorageExecutor() LATCH GET found action " + to_string(a) + " bIsSet=true");
                exitProcess();
            }

            // Check that the calculated old root is the same as the provided action root
            if ( !fr.equal(pols.oldRoot0[i], action[a].getResult.root[0]) ||
                 !fr.equal(pols.oldRoot1[i], action[a].getResult.root[1]) ||
                 !fr.equal(pols.oldRoot2[i], action[a].getResult.root[2]) ||
                 !fr.equal(pols.oldRoot3[i], action[a].getResult.root[3]) )
            {
                zklog.error("StorageExecutor() LATCH GET found action " + to_string(a) + " pols.oldRoot=" + fea2string(fr, pols.oldRoot0[i], pols.oldRoot1[i], pols.oldRoot2[i], pols.oldRoot3[i]) + " different from action.getResult.oldRoot=" + fea2string(fr, action[a].getResult.root[0], action[a].getResult.root[1], action[a].getResult.root[2], action[a].getResult.root[3]));
                exitProcess();
            }

            // Check that the calculated complete key is the same as the provided action key
            if ( !fr.equal(pols.rkey0[i], action[a].getResult.key[0]) ||
                 !fr.equal(pols.rkey1[i], action[a].getResult.key[1]) ||
                 !fr.equal(pols.rkey2[i], action[a].getResult.key[2]) ||
                 !fr.equal(pols.rkey3[i], action[a].getResult.key[3]) )
            {
                zklog.error("StorageExecutor() LATCH GET found action " + to_string(a) + " pols.rkey=" + fea2string(fr, pols.rkey0[i], pols.rkey1[i], pols.rkey2[i], pols.rkey3[i]) + " different from action.getResult.key=" + fea2string(fr, action[a].getResult.key[0], action[a].getResult.key[1], action[a].getResult.key[2], action[a].getResult.key[3]));
                exitProcess();
            }

            // Check that final level state is consistent
            if ( !fr.isZero(pols.level[i]) )
            {
                zklog.error("StorageExecutor() LATCH GET found action " + to_string(a) + " wrong level=" + fr.toString(pols.level[i], 10));
                exitProcess();
            }

            // Check that the calculated value key is the same as the provided action value
            Goldilocks::Element valueFea[8];
            valueFea[0] = pols.valueLow0[i];
            valueFea[1] = pols.valueLow1[i];
            valueFea[2] = pols.valueLow2[i];
            valueFea[3] = pols.valueLow3[i];
            valueFea[4] = pols.valueHigh0[i];
            valueFea[5] = pols.valueHigh1[i];
            valueFea[6] = pols.valueHigh2[i];
            valueFea[7] = pols.valueHigh3[i];
            mpz_class valueScalar;
            fea2scalar(fr, valueScalar, valueFea);
            if ( valueScalar != action[a].getResult.value )
            {
                zklog.error("StorageExecutor() LATCH GET found action " + to_string(a) + " pols.value=" + valueScalar.get_str(16) + " != action.getResult.value=" + action[a].getResult.value.get_str(16));
                exitProcess();
            }

            if ( fr.toU64(pols.incCounter[i]) != action[a].getResult.proofHashCounter )
            {
                zklog.error("StorageExecutor() LATCH SET found action " + to_string(a) + " wrong incCounter=" + fr.toString(pols.incCounter[i], 10) + " mode=" + to_string(action[a].getResult.proofHashCounter));
                exitProcess();
            }


#ifdef LOG_STORAGE_EXECUTOR
            zklog.info("StorageExecutor LATCH GET");
#endif

            // Increase action
            a++;

            // In case we run out of actions, report the empty list to consume the rest of evaluations
            if (a>=action.size())
            {
                actionListEmpty = true;

#ifdef LOG_STORAGE_EXECUTOR
                zklog.info("StorageExecutor LATCH GET detected the end of the action list a=" + to_string(a) + " i=" + to_string(i));
#endif
            }
            // Initialize the context for the new action
            else
            {
                ctx.init(fr, action[a]);
            }

            pols.latchGet[i] = fr.one();
        }

        // Latch set: at this point consistency is granted: OLD_ROOT, NEW_ROOT, RKEY (complete key), VALUE_LOW, VALUE_HIGH, LEVEL
        if (rom.line[l].latchSet)
        {
            // Check that the current action is an SMT set
            if (!action[a].bIsSet)
            {
                zklog.error("StorageExecutor() LATCH SET found action " + to_string(a) + " bIsSet=false");
                exitProcess();
            }

            // Check that the calculated old root is the same as the provided action root
            if ( !fr.equal(pols.oldRoot0[i], action[a].setResult.oldRoot[0]) ||
                 !fr.equal(pols.oldRoot1[i], action[a].setResult.oldRoot[1]) ||
                 !fr.equal(pols.oldRoot2[i], action[a].setResult.oldRoot[2]) ||
                 !fr.equal(pols.oldRoot3[i], action[a].setResult.oldRoot[3]) )
            {
                zklog.error("StorageExecutor() LATCH SET found action " + to_string(a) + " pols.oldRoot=" + fea2string(fr, pols.oldRoot0[i], pols.oldRoot1[i], pols.oldRoot2[i], pols.oldRoot3[i]) + " different from action.setResult.oldRoot=" + fea2string(fr, action[a].setResult.oldRoot[0], action[a].setResult.oldRoot[1], action[a].setResult.oldRoot[2], action[a].setResult.oldRoot[3]) + " mode=" + action[a].setResult.mode);
                exitProcess();
            }

            // Check that the calculated old root is the same as the provided action root
            if ( !fr.equal(pols.newRoot0[i], action[a].setResult.newRoot[0]) ||
                 !fr.equal(pols.newRoot1[i], action[a].setResult.newRoot[1]) ||
                 !fr.equal(pols.newRoot2[i], action[a].setResult.newRoot[2]) ||
                 !fr.equal(pols.newRoot3[i], action[a].setResult.newRoot[3]) )
            {
                zklog.error("StorageExecutor() LATCH SET found action " + to_string(a) + " pols.newRoot=" + fea2string(fr, pols.newRoot0[i], pols.newRoot1[i], pols.newRoot2[i], pols.newRoot3[i]) + " different from action.setResult.newRoot=" + fea2string(fr, action[a].setResult.newRoot[0], action[a].setResult.newRoot[1], action[a].setResult.newRoot[2], action[a].setResult.newRoot[3]) + " mode=" + action[a].setResult.mode);
                exitProcess();
            }

            // Check that the calculated complete key is the same as the provided action key
            if ( !fr.equal(pols.rkey0[i], action[a].setResult.key[0]) ||
                 !fr.equal(pols.rkey1[i], action[a].setResult.key[1]) ||
                 !fr.equal(pols.rkey2[i], action[a].setResult.key[2]) ||
                 !fr.equal(pols.rkey3[i], action[a].setResult.key[3]) )
            {
                zklog.error("StorageExecutor() LATCH SET found action " + to_string(a) + " pols.rkey=" + fea2string(fr, pols.rkey0[i], pols.rkey1[i], pols.rkey2[i], pols.rkey3[i]) + " different from action.setResult.key=" + fea2string(fr, action[a].setResult.key[0], action[a].setResult.key[1], action[a].setResult.key[2], action[a].setResult.key[3]) + " mode=" + action[a].setResult.mode);
                exitProcess();
            }

            // Check that final level state is consistent
            if ( !fr.isZero(pols.level[i]) )
            {
                zklog.error("StorageExecutor() LATCH SET found action " + to_string(a) + " wrong level=" + fr.toString(pols.level[i], 10) + " mode=" + action[a].setResult.mode);
                exitProcess();
            }

            // Check that the calculated value key is the same as the provided action value
            Goldilocks::Element valueFea[8];
            valueFea[0] = pols.valueLow0[i];
            valueFea[1] = pols.valueLow1[i];
            valueFea[2] = pols.valueLow2[i];
            valueFea[3] = pols.valueLow3[i];
            valueFea[4] = pols.valueHigh0[i];
            valueFea[5] = pols.valueHigh1[i];
            valueFea[6] = pols.valueHigh2[i];
            valueFea[7] = pols.valueHigh3[i];
            mpz_class valueScalar;
            fea2scalar(fr, valueScalar, valueFea);
            if ( valueScalar != action[a].setResult.newValue )
            {
                zklog.error("StorageExecutor() LATCH SET found action " + to_string(a) + " pols.value=" + valueScalar.get_str(16) + " != action.setResult.newValue=" + action[a].setResult.newValue.get_str(16) + " mode=" + action[a].setResult.mode);
                exitProcess();
            }

            // Check that final level state is consistent
            if ( fr.toU64(pols.incCounter[i]) != action[a].setResult.proofHashCounter )
            {
                zklog.error("StorageExecutor() LATCH SET found action " + to_string(a) + " wrong incCounter=" + fr.toString(pols.incCounter[i], 10) + " mode=" + to_string(action[a].setResult.proofHashCounter));
                exitProcess();
            }

#ifdef LOG_STORAGE_EXECUTOR
            zklog.info("StorageExecutor LATCH SET");
#endif

            // Increase action
            a++;

            // In case we run out of actions, report the empty list to consume the rest of evaluations
            if (a>=action.size())
            {
                actionListEmpty = true;

#ifdef LOG_STORAGE_EXECUTOR
                zklog.info("StorageExecutor() LATCH SET detected the end of the action list a=" + to_string(a) + " i=" + to_string(i));
#endif
            }
            // Initialize the context for the new action
            else
            {
                ctx.init(fr, action[a]);
            }

            pols.latchSet[i] = fr.one();
        }

#ifdef LOG_STORAGE_EXECUTOR_ROM_LINE
        if (rom.line[l].funcName!="isAlmostEndPolynomial")
        {
            printf("[SR%04d I%03d %-28s] OP=[\x1B[35m%s\x1B[0m]\n", (int)l, (int)a, source.c_str(), fea2string(fr, op).c_str());
        }
#endif


        /***********/
        /* Setters */
        /***********/

        // If setRKEY then RKEY=op
        if (rom.line[l].setRKEY)
        {
            pols.rkey0[nexti] = op[0];
            pols.rkey1[nexti] = op[1];
            pols.rkey2[nexti] = op[2];
            pols.rkey3[nexti] = op[3];
            pols.setRkey[i] = fr.one();
        }
        else if (fr.isZero(pols.climbRkey[i]))
        {
            pols.rkey0[nexti] = pols.rkey0[i];
            pols.rkey1[nexti] = pols.rkey1[i];
            pols.rkey2[nexti] = pols.rkey2[i];
            pols.rkey3[nexti] = pols.rkey3[i];
        }

        // If setRKEY_BIT then RKEY_BIT=op
        if (rom.line[l].setRKEY_BIT)
        {
            pols.rkeyBit[nexti] = op[0];
            pols.setRkeyBit[i] = fr.one();
        }
        else
        {
            pols.rkeyBit[nexti] = pols.rkeyBit[i];
        }

        // If setVALUE_LOW then VALUE_LOW=op
        if (rom.line[l].setVALUE_LOW)
        {
            pols.valueLow0[nexti] = op[0];
            pols.valueLow1[nexti] = op[1];
            pols.valueLow2[nexti] = op[2];
            pols.valueLow3[nexti] = op[3];
            pols.setValueLow[i] = fr.one();
        }
        else
        {
            pols.valueLow0[nexti] = pols.valueLow0[i];
            pols.valueLow1[nexti] = pols.valueLow1[i];
            pols.valueLow2[nexti] = pols.valueLow2[i];
            pols.valueLow3[nexti] = pols.valueLow3[i];
        }

        // If setVALUE_HIGH then VALUE_HIGH=op
        if (rom.line[l].setVALUE_HIGH)
        {
            pols.valueHigh0[nexti] = op[0];
            pols.valueHigh1[nexti] = op[1];
            pols.valueHigh2[nexti] = op[2];
            pols.valueHigh3[nexti] = op[3];
            pols.setValueHigh[i] = fr.one();
        }
        else
        {
            pols.valueHigh0[nexti] = pols.valueHigh0[i];
            pols.valueHigh1[nexti] = pols.valueHigh1[i];
            pols.valueHigh2[nexti] = pols.valueHigh2[i];
            pols.valueHigh3[nexti] = pols.valueHigh3[i];
        }

        // If setLEVEL then LEVEL=op
        if (rom.line[l].setLEVEL)
        {
            pols.level[nexti] = op[0];
            pols.setLevel[i] = fr.one();
        }
        else
        {
            pols.level[nexti] = pols.level[i];
        }

        // If setOLD_ROOT then OLD_ROOT=op
        if (rom.line[l].setOLD_ROOT)
        {
            pols.oldRoot0[nexti] = op[0];
            pols.oldRoot1[nexti] = op[1];
            pols.oldRoot2[nexti] = op[2];
            pols.oldRoot3[nexti] = op[3];
            pols.setOldRoot[i] = fr.one();
        }
        else
        {
            pols.oldRoot0[nexti] = pols.oldRoot0[i];
            pols.oldRoot1[nexti] = pols.oldRoot1[i];
            pols.oldRoot2[nexti] = pols.oldRoot2[i];
            pols.oldRoot3[nexti] = pols.oldRoot3[i];
        }

        // If setNEW_ROOT then NEW_ROOT=op
        if (rom.line[l].setNEW_ROOT)
        {
            pols.newRoot0[nexti] = op[0];
            pols.newRoot1[nexti] = op[1];
            pols.newRoot2[nexti] = op[2];
            pols.newRoot3[nexti] = op[3];
            pols.setNewRoot[i] = fr.one();
        }
        else
        {
            pols.newRoot0[nexti] = pols.newRoot0[i];
            pols.newRoot1[nexti] = pols.newRoot1[i];
            pols.newRoot2[nexti] = pols.newRoot2[i];
            pols.newRoot3[nexti] = pols.newRoot3[i];
        }

        // If setHASH_LEFT then HASH_LEFT=op
        if (rom.line[l].setHASH_LEFT)
        {
            pols.hashLeft0[nexti] = op[0];
            pols.hashLeft1[nexti] = op[1];
            pols.hashLeft2[nexti] = op[2];
            pols.hashLeft3[nexti] = op[3];
            pols.setHashLeft[i] = fr.one();
        }
        else
        {
            pols.hashLeft0[nexti] = pols.hashLeft0[i];
            pols.hashLeft1[nexti] = pols.hashLeft1[i];
            pols.hashLeft2[nexti] = pols.hashLeft2[i];
            pols.hashLeft3[nexti] = pols.hashLeft3[i];
        }

        // If setHASH_RIGHT then HASH_RIGHT=op
        if (rom.line[l].setHASH_RIGHT)
        {
            pols.hashRight0[nexti] = op[0];
            pols.hashRight1[nexti] = op[1];
            pols.hashRight2[nexti] = op[2];
            pols.hashRight3[nexti] = op[3];
            pols.setHashRight[i] = fr.one();
        }
        else
        {
            pols.hashRight0[nexti] = pols.hashRight0[i];
            pols.hashRight1[nexti] = pols.hashRight1[i];
            pols.hashRight2[nexti] = pols.hashRight2[i];
            pols.hashRight3[nexti] = pols.hashRight3[i];
        }

        // If setSIBLING_RKEY then SIBLING_RKEY=op
        if (rom.line[l].setSIBLING_RKEY)
        {
            pols.siblingRkey0[nexti] = op[0];
            pols.siblingRkey1[nexti] = op[1];
            pols.siblingRkey2[nexti] = op[2];
            pols.siblingRkey3[nexti] = op[3];
            pols.setSiblingRkey[i] = fr.one();
        }
        else
        {
            pols.siblingRkey0[nexti] = pols.siblingRkey0[i];
            pols.siblingRkey1[nexti] = pols.siblingRkey1[i];
            pols.siblingRkey2[nexti] = pols.siblingRkey2[i];
            pols.siblingRkey3[nexti] = pols.siblingRkey3[i];
        }

        // If setSIBLING_VALUE_HASH then SIBLING_VALUE_HASH=op
        if (rom.line[l].setSIBLING_VALUE_HASH)
        {
            pols.siblingValueHash0[nexti] = op[0];
            pols.siblingValueHash1[nexti] = op[1];
            pols.siblingValueHash2[nexti] = op[2];
            pols.siblingValueHash3[nexti] = op[3];
            pols.setSiblingValueHash[i] = fr.one();
        }
        else
        {
            pols.siblingValueHash0[nexti] = pols.siblingValueHash0[i];
            pols.siblingValueHash1[nexti] = pols.siblingValueHash1[i];
            pols.siblingValueHash2[nexti] = pols.siblingValueHash2[i];
            pols.siblingValueHash3[nexti] = pols.siblingValueHash3[i];
        }

        // Calculate op0 inverse
        if (!fr.isZero(op[0]))
        {
            pols.op0inv[i] = glp.inv(op[0]);
        }

        // Increment counter at every hash, and reset it at every latch
        if (rom.line[l].hash)
        {
            pols.incCounter[nexti] = fr.add(pols.incCounter[i], fr.one());
        }
        else if (rom.line[l].latchGet || rom.line[l].latchSet)
        {
            pols.incCounter[nexti] = fr.zero();
        }
        else
        {
            pols.incCounter[nexti] = pols.incCounter[i];
        }

#ifdef LOG_STORAGE_EXECUTOR
        if ((i%1000) == 0) zklog.info("StorageExecutor step " + to_string(i) + " done");
#endif
    }

    // Check that ROM has done all its work
    if (lastStep == 0)
    {
        zklog.error("StorageExecutor::execute() finished execution but ROM did not call isAlmostEndPolynomial");
        exitProcess();
    }

    zklog.info("StorageExecutor successfully processed " + to_string(action.size()) + " SMT actions (" + to_string((double(lastStep)*100)/N) + "%)");
}

// To be used only for testing, since it allocates a lot of memory
void StorageExecutor::execute (vector<SmtAction> &action)
{
    void * pAddress = malloc(CommitPols::pilSize());
    if (pAddress == NULL)
    {
        zklog.error("StorageExecutor::execute() failed calling malloc() of size=" + to_string(CommitPols::pilSize()));
        exitProcess();
    }
    CommitPols cmPols(pAddress, CommitPols::pilDegree());
    vector<array<Goldilocks::Element, 17>> poseidonRequired;
    vector<ClimbKeyAction> climbKeyRequired;
    execute(action, cmPols.Storage, poseidonRequired, climbKeyRequired);
    free(pAddress);
}
