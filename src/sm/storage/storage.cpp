
#include <nlohmann/json.hpp>
#include "storage.hpp"
#include "storage_rom.hpp"
#include "storage_pols.hpp"
#include "utils.hpp"
#include "scalar.hpp"

using json = nlohmann::json;
using namespace std;

void StorageExecutor::execute (vector<SmtAction> &action, StorageCommitPols &pols)
{
//#if 0
    uint64_t l=0; // rom line number, so current line is rom.line[l]
    uint64_t a=0; // action number, so current action is action[a]
    bool actionListEmpty = (action.size()==0); // becomes true when we run out of actions

    // Init the context if the list is not empty
    SmtActionContext ctx;
    if (!actionListEmpty)
    {
        ctx.init(fr, action[a]);
    }

    // For all polynomial evaluations
    uint64_t degree = pols.degree();
    for (uint64_t i=0; i<degree; i++)
    {
        // op is the internal register, reset to 0 at every evaluation
        uint64_t op[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};

        // Current rom line is set by the program counter of this evaluation
        l = pols.pc[i];

        // Set the next evaluation index, which will be 0 when we reach the last evaluation
        uint64_t nexti = (i+1)%degree;

#ifdef LOG_STORAGE_EXECUTOR_ROM_LINE
        if (rom.line[l].funcName!="isAlmostEndPolynomial")
        {
            rom.line[l].print(l); // Print the rom line content 
        }
#endif

        /*************/
        /* Selectors */
        /*************/

        // When the rom assembler code calls inFREE, it specifies the requested input data
        // using an operation + function name string couple

        if (rom.line[l].inFREE)
        {
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
                        cout << "StorageExecutor isUpdate returns " << fea2string(fr, op) << endl;
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
                        cout << "StorageExecutor isInsertFound returns " << fea2string(fr, op) << endl;
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
                        cout << "StorageExecutor isInsertNotFound returns " << fea2string(fr, op) << endl;
#endif
                    }
                }
                else if (rom.line[l].funcName == "isSetReplacingZero")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "insertNotFound")
                    {
                        op[0] = fr.one();

#ifdef LOG_STORAGE_EXECUTOR
                        cout << "StorageExecutor isSetReplacingZero returns " << fea2string(fr, op) << endl;
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
                        cout << "StorageExecutor isDeleteLast returns " << fea2string(fr, op) << endl;
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
                        cout << "StorageExecutor isSetDeleteFound returns " << fea2string(fr, op) << endl;
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
                        cout << "StorageExecutor isSetDeleteNotFound returns " << fea2string(fr, op) << endl;
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
                        cout << "StorageExecutor isZeroToZero returns " << fea2string(fr, op) << endl;
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
                        cout << "StorageExecutor isGet returns " << fea2string(fr, op) << endl;
#endif
                    }
                }

                // Get the remaining key, i.e. the key after removing the bits used in the tree node navigation
                else if (rom.line[l].funcName=="GetRKey")
                {
                    op[0] = ctx.rKey[0];
                    op[1] = ctx.rKey[1];
                    op[2] = ctx.rKey[2];
                    op[3] = ctx.rKey[3];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetRKey returns " << fea2string(fr, op) << endl;
#endif
                }

                // Get the sibling remaining key, i.e. the part that is not common to the value key
                else if (rom.line[l].funcName=="GetSiblingRKey")
                {
                    op[0] = ctx.siblingRKey[0];
                    op[1] = ctx.siblingRKey[1];
                    op[2] = ctx.siblingRKey[2];
                    op[3] = ctx.siblingRKey[3];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetSiblingRKey returns " << fea2string(fr, op) << endl;
#endif
                }

                // Get the sibling hash, obtained from the siblings array of the current level,
                // taking into account that the sibling bit is the opposite (1-x) of the value bit
                else if (rom.line[l].funcName=="GetSiblingHash")
                {
                    if (action[a].bIsSet)
                    {
                        op[0] = action[a].setResult.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4];
                        op[1] = action[a].setResult.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4+1];
                        op[2] = action[a].setResult.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4+2];
                        op[3] = action[a].setResult.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4+3];
                    }
                    else
                    {
                        op[0] = action[a].getResult.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4];
                        op[1] = action[a].getResult.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4+1];
                        op[2] = action[a].getResult.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4+2];
                        op[3] = action[a].getResult.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4+3];
                    }

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetSiblingHash returns " << fea2string(fr, op) << endl;
#endif
                }

                // Value is an u256 split in 8 u32 chuncks, each one stored in the lower 32 bits of an u63 field element
                // u63 means that it is not an u64, since some of the possible values are lost due to the prime effect 

                // Get the lower 4 field elements of the value
                else if (rom.line[l].funcName=="GetValueLow")
                {
                    FieldElement fea[8];
                    scalar2fea(fr, action[a].bIsSet ? action[a].setResult.newValue : action[a].getResult.value, fea);
                    op[0] = fea[0];
                    op[1] = fea[1];
                    op[2] = fea[2];
                    op[3] = fea[3];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetValueLow returns " << fea2string(fr, op) << endl;
#endif
                }

                // Get the higher 4 field elements of the value
                else if (rom.line[l].funcName=="GetValueHigh")
                {
                    FieldElement fea[8];
                    scalar2fea(fr, action[a].bIsSet ? action[a].setResult.newValue : action[a].getResult.value, fea);
                    op[0] = fea[4];
                    op[1] = fea[5];
                    op[2] = fea[6];
                    op[3] = fea[7];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetValueHigh returns " << fea2string(fr, op) << endl;
#endif
                }

                // Get the lower 4 field elements of the sibling value
                else if (rom.line[l].funcName=="GetSiblingValueLow")
                {
                    FieldElement fea[8];
                    scalar2fea(fr, action[a].bIsSet ? action[a].setResult.insValue : action[a].getResult.insValue, fea);
                    op[0] = fea[0];
                    op[1] = fea[1];
                    op[2] = fea[2];
                    op[3] = fea[3];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetSiblingValueLow returns " << fea2string(fr, op) << endl;
#endif
                }

                // Get the higher 4 field elements of the sibling value
                else if (rom.line[l].funcName=="GetSiblingValueHigh")
                {
                    FieldElement fea[8];
                    scalar2fea(fr, action[a].bIsSet ? action[a].setResult.insValue : action[a].getResult.insValue, fea);
                    op[0] = fea[4];
                    op[1] = fea[5];
                    op[2] = fea[6];
                    op[3] = fea[7];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetSiblingValueHigh returns " << fea2string(fr, op) << endl;
#endif
                }

                // Get the lower 4 field elements of the old value
                else if (rom.line[l].funcName=="GetOldValueLow")
                {
                    // This call only makes sense then this is an SMT set
                    if (!action[a].bIsSet)
                    {
                        cerr << "Error: StorageExecutor() GetOldValueLow called in an SMT get action" << endl;
                        exit(-1);
                    }

                    // Convert the oldValue scalar to an 8 field elements array
                    FieldElement fea[8];
                    scalar2fea(fr, action[a].setResult.oldValue, fea);

                    // Take the lower 4 field elements
                    op[0] = fea[0];
                    op[1] = fea[1];
                    op[2] = fea[2];
                    op[3] = fea[3];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetOldValueLow returns " << fea2string(fr, op) << endl;
#endif
                }

                // Get the higher 4 field elements of the old value
                else if (rom.line[l].funcName=="GetOldValueHigh")
                {
                    // This call only makes sense then this is an SMT set
                    if (!action[a].bIsSet)
                    {
                        cerr << "Error: StorageExecutor() GetOldValueLow called in an SMT get action" << endl;
                        exit(-1);
                    }

                    // Convert the oldValue scalar to an 8 field elements array
                    FieldElement fea[8];
                    scalar2fea(fr, action[a].setResult.oldValue, fea);

                    // Take the higher 4 field elements
                    op[0] = fea[4];
                    op[1] = fea[5];
                    op[2] = fea[6];
                    op[3] = fea[7];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetOldValueHigh returns " << fea2string(fr, op) << endl;
#endif
                }

                // Get the level bit, i.e. the bit x (specified by the parameter) of the level number
                else if (rom.line[l].funcName=="GetLevelBit")
                {
                    // Check that we have the one single parameter: the bit number
                    if (rom.line[l].params.size()!=1)
                    {
                        cerr << "Error: StorageExecutor() called with GetLevelBit but wrong number of parameters=" << rom.line[l].params.size() << endl;
                        exit(-1);
                    }

                    // Get the bit parameter
                    uint64_t bit = rom.line[l].params[0];

                    // Check that the bit is either 0 or 1
                    if (bit!=0 && bit!=1)
                    {
                        cerr << "Error: StorageExecutor() called with GetLevelBit but wrong bit=" << bit << endl;
                        exit(-1);
                    }

                    // Set the bit in op[0]
                    if ( ( ctx.level & (1<<bit) ) != 0)
                    {
                        op[0] = fr.one();
                    }

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetLevelBit(" << bit << ") returns " << fea2string(fr, op) << endl;
#endif
                }

                // Returns 0 if we reached the top of the tree, i.e. if the current level is 0
                else if (rom.line[l].funcName=="GetTopTree")
                {
                    // Return 0 only if we reached the end of the tree, i.e. if the current level is 0
                    if (ctx.currentLevel > 0)
                    {
                        op[0] = fr.one();
                    }

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetTopTree returns " << fea2string(fr, op) << endl;
#endif
                }

                // Returns 0 if we reached the top of the branch, i.e. if the level matches the siblings size
                else if (rom.line[l].funcName=="GetTopOfBranch")
                {
                    // If we have consumed enough key bits to reach the deepest level of the siblings array, then we are at the top of the branch and we can start climing the tree
                    int64_t siblingsSize = action[a].bIsSet ? action[a].setResult.siblings.size() : action[a].getResult.siblings.size();
                    if (ctx.currentLevel > siblingsSize )
                    {
                        op[0] = fr.one();
                    }

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetTopOfBranch returns " << fea2string(fr, op) << endl;
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
                        cerr << "Error: StorageExecutor.execute() GetNextKeyBit() found ctx.currentLevel<0" << endl;
                        exit(-1);
                    }

                    // Get the key bit corresponding to the current level
                    op[0] = ctx.bits[ctx.currentLevel];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetNextKeyBit returns " << fea2string(fr, op) << endl;
#endif
                }

                // Return 1 if we completed all evaluations, except one
                else if (rom.line[l].funcName=="isAlmostEndPolynomial")
                {
                    // Return one if this is the one before the last evaluation of the polynomials
                    if (i == (degree-2))
                    {
                        op[0] = fr.one();
#ifdef LOG_STORAGE_EXECUTOR
                        cout << "StorageExecutor isEndPolynomial returns " << fea2string(fr,op) << endl;
#endif
                    }
                }
                else
                {
                    cerr << "Error: StorageExecutor() unknown funcName:" << rom.line[l].funcName << endl;
                    exit(-1);
                }                
            }

            // Ignore; this is just to report a list of setters 
            else if (rom.line[l].op=="")
            {                
            }

            // Any other value is an unexpected value
            else
            {
                cerr << "Error: StorageExecutor() unknown op:" << rom.line[l].op << endl;
                exit(-1);
            }

            // free[] = op[]
            if (op[0] != 0) pols.free0[i] = op[0];
            if (op[1] != 0) pols.free1[i] = op[1];
            if (op[2] != 0) pols.free2[i] = op[2];
            if (op[3] != 0) pols.free3[i] = op[3];            

            // Mark the selFree register as 1
            pols.selFree[i] = 1;
        }

        // If a constant is provided, set op to the constant
        if (rom.line[l].CONST!="")
        {
            // Convert constant to scalar
            mpz_class constScalar;
            constScalar.set_str(rom.line[l].CONST, 10);

            // Convert scalar to field element array
            scalar2fea(fr, constScalar, op);
            
            // Store constant field elements in their registers
            pols.iConst0[i] = op[0];
            pols.iConst1[i] = op[1];
            pols.iConst2[i] = op[2];
            pols.iConst3[i] = op[3];
        }

        // If inOLD_ROOT then op=OLD_ROOT
        if (rom.line[l].inOLD_ROOT)
        {
            op[0] = pols.oldRoot0[i];
            op[1] = pols.oldRoot1[i];
            op[2] = pols.oldRoot2[i];
            op[3] = pols.oldRoot3[i];
            pols.setOldRoot[i] = 1;
        }

        // If inNEW_ROOT then op=NEW_ROOT
        if (rom.line[l].inNEW_ROOT)
        {
            op[0] = pols.newRoot0[i];
            op[1] = pols.newRoot1[i];
            op[2] = pols.newRoot2[i];
            op[3] = pols.newRoot3[i];
            pols.setNewRoot[i] = 1;
        }

        // If inRKEY_BIT then op=RKEY_BIT
        if (rom.line[l].inRKEY_BIT)
        {
            op[0] = pols.rkeyBit[i];
            op[1] = 0;
            op[2] = 0;
            op[3] = 0;
            pols.setRkeyBit[i] = 1;
        }

        // If inVALUE_LOW then op=VALUE_LOW
        if (rom.line[l].inVALUE_LOW)
        {
            op[0] = pols.valueLow0[i];
            op[1] = pols.valueLow1[i];
            op[2] = pols.valueLow2[i];
            op[3] = pols.valueLow3[i];
            pols.setValueLow[i] = 1;
        }

        // If inVALUE_HIGH then op=VALUE_HIGH
        if (rom.line[l].inVALUE_HIGH)
        {
            op[0] = pols.valueHigh0[i];
            op[1] = pols.valueHigh1[i];
            op[2] = pols.valueHigh2[i];
            op[3] = pols.valueHigh3[i];
            pols.setValueHigh[i] = 1;
        }

        // If inRKEY then op=RKEY
        if (rom.line[l].inRKEY)
        {
            op[0] = pols.rkey0[i];
            op[1] = pols.rkey1[i];
            op[2] = pols.rkey2[i];
            op[3] = pols.rkey3[i];
            pols.setRkey[i] = 1;
        }

        // If inSIBLING_RKEY then op=SIBLING_RKEY
        if (rom.line[l].inSIBLING_RKEY)
        {
            op[0] = pols.siblingRkey0[i];
            op[1] = pols.siblingRkey1[i];
            op[2] = pols.siblingRkey2[i];
            op[3] = pols.siblingRkey3[i];
            pols.setSiblingRkey[i] = 1;
        }

        // If inSIBLING_VALUE_HASH then op=SIBLING_VALUE_HASH
        if (rom.line[l].inSIBLING_VALUE_HASH)
        {
            op[0] = pols.siblingValueHash0[i];
            op[1] = pols.siblingValueHash1[i];
            op[2] = pols.siblingValueHash2[i];
            op[3] = pols.siblingValueHash3[i];
            pols.setSiblingValueHash[i] = 1;
        }

        /****************/
        /* Instructions */
        /****************/

        // JMPZ: Jump if OP==0
        if (rom.line[l].iJmpz)
        {
            if (op[0]==0 && op[1]==0 && op[2]==0 && op[3]==0)
            {
                pols.pc[nexti] = rom.line[l].address;
                pols.iAddress[i] = rom.line[l].address;
                //cout << "StorageExecutor iJmpz address=" << rom.line[l].address << endl;
            }
            else
            {
                pols.pc[nexti] = pols.pc[i] + 1;
            }
            pols.iJmpz[i] = 1;
        }

        // JMP: Jump always
        else if (rom.line[l].iJmp)
        {
            pols.pc[nexti] = rom.line[l].address;
            pols.iAddress[i] = rom.line[l].address;
            //cout << "StorageExecutor iJmp address=" << rom.line[l].address << endl;
            pols.iJmp[i] = 1;
        }

        // If not any jump, then simply increment program counter
        else
        {
            pols.pc[nexti] = pols.pc[i] + 1;
        }

        // Rotate level registers values, from higher to lower
        if (rom.line[l].iRotateLevel)
        {
            uint64_t aux;
            aux = pols.level0[i];
            pols.level0[i] = pols.level1[i];
            pols.level1[i] = pols.level2[i];
            pols.level2[i] = pols.level3[i];
            pols.level3[i] = aux;
            pols.iRotateLevel[i] = 1;

#ifdef LOG_STORAGE_EXECUTOR
            cout << "StorageExecutor iRotateLevel level[3:2:1:0]=" << pols.level3[i] << ":" << pols.level2[i] << ":" << pols.level1[i] << ":" << pols.level0[i] << endl;
#endif
        }

        // Hash: op = poseidon.hash(HASH_LEFT + HASH_RIGHT + (0 or 1, depending on iHashType))
        if (rom.line[l].iHash)
        {
            // Prepare the data to hash: HASH_LEFT + HASH_RIGHT + 0 or 1, depending on iHashType
            FieldElement fea[12];
            fea[0] = pols.hashLeft0[i];
            fea[1] = pols.hashLeft1[i];
            fea[2] = pols.hashLeft2[i];
            fea[3] = pols.hashLeft3[i];
            fea[4] = pols.hashRight0[i];
            fea[5] = pols.hashRight1[i];
            fea[6] = pols.hashRight2[i];
            fea[7] = pols.hashRight3[i];
            if (rom.line[l].iHashType==0)
            {
                fea[8] = fr.zero();
            }
            else if (rom.line[l].iHashType==1)
            {
                fea[8] = fr.one();
                pols.iHashType[i] = 1;
            }
            else
            {
                cerr << "Error: StorageExecutor:execute() found invalid iHashType=" << rom.line[l].iHashType << endl;
                exit(-1);
            }
            fea[9] = fr.zero();
            fea[10] = fr.zero();
            fea[11] = fr.zero();

#ifdef LOG_STORAGE_EXECUTOR
            FieldElement auxFea[12];
            for (uint64_t i=0; i<12; i++) auxFea[i] = fea[i];
#endif

            // Call poseidon
            poseidon.hash(fea);

            // Get the calculated hash from the first 4 elements
            pols.free0[i] = fea[0];
            pols.free1[i] = fea[1];
            pols.free2[i] = fea[2];
            pols.free3[i] = fea[3];

            op[0] = fr.add(op[0], fr.mul(rom.line[l].inFREE, pols.free0[i]));
            op[1] = fr.add(op[1], fr.mul(rom.line[l].inFREE, pols.free1[i]));
            op[2] = fr.add(op[2], fr.mul(rom.line[l].inFREE, pols.free2[i]));
            op[3] = fr.add(op[3], fr.mul(rom.line[l].inFREE, pols.free3[i]));

            pols.iHash[i] = 1;

#ifdef LOG_STORAGE_EXECUTOR
            cout << "StorageExecutor iHash" << rom.line[l].iHashType << " hash=" << fea2string(fr, op) << " value=";
            for (uint64_t i=0; i<12; i++) cout << fr.toString(auxFea[i],16) << ":";
            cout << endl;
#endif
        }

        // Climb the remaining key, by injecting the RKEY_BIT in the register specified by LEVEL
        if (rom.line[l].iClimbRkey)
        {
            uint64_t bit = pols.rkeyBit[i];
            if (pols.level0[i] == 1)
            {
                pols.rkey0[i] = (pols.rkey0[i]<<1) + bit;
            }
            if (pols.level1[i] == 1)
            {
                pols.rkey1[i] = (pols.rkey1[i]<<1) + bit;
            }
            if (pols.level2[i] == 1)
            {
                pols.rkey2[i] = (pols.rkey2[i]<<1) + bit;
            }
            if (pols.level3[i] == 1)
            {
                pols.rkey3[i] = (pols.rkey3[i]<<1) + bit;
            }
            pols.iClimbRkey[i] = 1;

#ifdef LOG_STORAGE_EXECUTOR
            FieldElement fea[4] = {pols.rkey0[i], pols.rkey1[i], pols.rkey2[i], pols.rkey3[i]};
            cout << "StorageExecutor iClimbRkey sibling bit=" << bit << " rkey=" << fea2string(fr,fea) << endl;
#endif
        }

        // Climb the sibling remaining key, by injecting the sibling bit in the register specified by LEVEL
        if (rom.line[l].iClimbSiblingRkey)
        {
#ifdef LOG_STORAGE_EXECUTOR
            FieldElement fea1[4] = {pols.siblingRkey0[i], pols.siblingRkey1[i], pols.siblingRkey2[i], pols.siblingRkey3[i]};
            cout << "StorageExecutor iClimbSiblingRkey before rkey=" << fea2string(fr,fea1) << endl;
#endif
            uint64_t bit = ctx.siblingBits[ctx.currentLevel];
            if (pols.level0[i] == 1)
            {
                pols.siblingRkey0[i] = (pols.siblingRkey0[i]<<1) + bit;
            }
            if (pols.level1[i] == 1)
            {
                pols.siblingRkey1[i] = (pols.siblingRkey1[i]<<1) + bit;
            }
            if (pols.level2[i] == 1)
            {
                pols.siblingRkey2[i] = (pols.siblingRkey2[i]<<1) + bit;
            }
            if (pols.level3[i] == 1)
            {
                pols.siblingRkey3[i] = (pols.siblingRkey3[i]<<1) + bit;
            }
            pols.iClimbSiblingRkey[i] = 1;

#ifdef LOG_STORAGE_EXECUTOR
            FieldElement fea[4] = {pols.siblingRkey0[i], pols.siblingRkey1[i], pols.siblingRkey2[i], pols.siblingRkey3[i]};
            cout << "StorageExecutor iClimbSiblingRkey after sibling bit=" << bit << " rkey=" << fea2string(fr,fea) << endl;
#endif
        }

        // Latch get: at this point consistency is granted: OLD_ROOT, RKEY (complete key), VALUE_LOW, VALUE_HIGH, LEVEL
        if (rom.line[l].iLatchGet)
        {            
            // Check that the current action is an SMT get
            if (action[a].bIsSet)
            {
                cerr << "Error: StorageExecutor() LATCH GET found action " << a << " bIsSet=true" << endl;
                exit(-1);
            }

            // Check only if key was found
            if (action[a].getResult.isOld0)
            {
                // Check that the calculated old root is the same as the provided action root
                FieldElement oldRoot[4] = {pols.oldRoot0[i], pols.oldRoot1[i], pols.oldRoot2[i], pols.oldRoot3[i]};
                if ( !fr.eq(oldRoot, action[a].getResult.root) )
                {
                    cerr << "Error: StorageExecutor() LATCH GET found action " << a << " pols.oldRoot=" << fea2string(fr,oldRoot) << " different from action.getResult.root=" << fea2string(fr,action[a].getResult.root) << endl;
                    exit(-1);
                }

                // Check that the calculated complete key is the same as the provided action key
                if ( pols.rkey0[i] != action[a].getResult.key[0] ||
                     pols.rkey1[i] != action[a].getResult.key[1] ||
                     pols.rkey2[i] != action[a].getResult.key[2] ||
                     pols.rkey3[i] != action[a].getResult.key[3] )
                {
                    cerr << "Error: StorageExecutor() LATCH GET found action " << a << " pols.rkey!=action.getResult.key" << endl;
                    exit(-1);                
                }

                // Check that final level state is consistent
                if ( pols.level0[i] != 1 ||
                     pols.level1[i] != 0 ||
                     pols.level2[i] != 0 ||
                     pols.level3[i] != 0 )
                {
                    cerr << "Error: StorageExecutor() LATCH GET found action " << a << " wrong level=" << pols.level3[i] << ":" << pols.level2[i] << ":" << pols.level1[i] << ":" << pols.level0[i] << endl;
                    exit(-1);                
                }
            }

#ifdef LOG_STORAGE_EXECUTOR
            cout << "StorageExecutor LATCH GET" << endl;
#endif

            // Increase action
            a++;

            // In case we run out of actions, report the empty list to consume the rest of evaluations
            if (a>=action.size())
            {
                actionListEmpty = true;

#ifdef LOG_STORAGE_EXECUTOR
                cout << "StorageExecutor LATCH GET detected the end of the action list a=" << a << " i=" << i << endl;
#endif
            }
            // Initialize the context for the new action
            else
            {
                ctx.init(fr, action[a]);
            }

            pols.iLatchGet[i] = 1;
        }

        // Latch set: at this point consistency is granted: OLD_ROOT, NEW_ROOT, RKEY (complete key), VALUE_LOW, VALUE_HIGH, LEVEL
        if (rom.line[l].iLatchSet)
        {
            // Check that the current action is an SMT set
            if (!action[a].bIsSet)
            {
                cerr << "Error: StorageExecutor() LATCH SET found action " << a << " bIsSet=false" << endl;
                exit(-1);
            }

            // Check that the calculated old root is the same as the provided action root
            FieldElement oldRoot[4] = {pols.oldRoot0[i], pols.oldRoot1[i], pols.oldRoot2[i], pols.oldRoot3[i]};
            if ( !fr.eq(oldRoot, action[a].setResult.oldRoot) )
            {
                cerr << "Error: StorageExecutor() LATCH SET found action " << a << " pols.oldRoot=" << fea2string(fr,oldRoot) << " different from action.setResult.oldRoot=" << fea2string(fr,action[a].setResult.oldRoot) << endl;
                exit(-1);
            }

            // Check that the calculated old root is the same as the provided action root
            FieldElement newRoot[4] = {pols.newRoot0[i], pols.newRoot1[i], pols.newRoot2[i], pols.newRoot3[i]};
            if ( !fr.eq(newRoot, action[a].setResult.newRoot) )
            {
                cerr << "Error: StorageExecutor() LATCH SET found action " << a << " pols.newRoot=" << fea2string(fr,newRoot) << " different from action.setResult.newRoot=" << fea2string(fr,action[a].setResult.newRoot) << endl;
                exit(-1);
            }

            // Check that the calculated complete key is the same as the provided action key
            if ( pols.rkey0[i] != action[a].setResult.key[0] ||
                 pols.rkey1[i] != action[a].setResult.key[1] ||
                 pols.rkey2[i] != action[a].setResult.key[2] ||
                 pols.rkey3[i] != action[a].setResult.key[3] )
            {
                cerr << "Error: StorageExecutor() LATCH SET found action " << a << " pols.rkey!=action.setResult.key" << endl;
                exit(-1);                
            }

            // Check that final level state is consistent
            if ( pols.level0[i] != 1 ||
                 pols.level1[i] != 0 ||
                 pols.level2[i] != 0 ||
                 pols.level3[i] != 0 )
            {
                cerr << "Error: StorageExecutor() LATCH SET found action " << a << " wrong level=" << pols.level3[i] << ":" << pols.level2[i] << ":" << pols.level1[i] << ":" << pols.level0[i] << endl;
                exit(-1);                
            }

#ifdef LOG_STORAGE_EXECUTOR
            cout << "StorageExecutor LATCH SET" << endl;
#endif

            // Increase action
            a++;

            // In case we run out of actions, report the empty list to consume the rest of evaluations
            if (a>=action.size())
            {
                actionListEmpty = true;

#ifdef LOG_STORAGE_EXECUTOR
                cout << "StorageExecutor() LATCH SET detected the end of the action list a=" << a << " i=" << i << endl;
#endif
            }
            // Initialize the context for the new action
            else
            {
                ctx.init(fr, action[a]);
            }

            pols.iLatchSet[i] = 1;
        }

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
            pols.setRkey[i] = 1;
        }
        else
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
            pols.setRkeyBit[i] = 1;
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
            pols.setValueLow[i] = 1;
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
            pols.setValueHigh[i] = 1;
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
            pols.level0[nexti] = op[0];
            pols.level1[nexti] = op[1];
            pols.level2[nexti] = op[2];
            pols.level3[nexti] = op[3];
            pols.setLevel[i] = 1;
        }
        else
        {
            pols.level0[nexti] = pols.level0[i];
            pols.level1[nexti] = pols.level1[i];
            pols.level2[nexti] = pols.level2[i];
            pols.level3[nexti] = pols.level3[i];
        }
        
        // If setOLD_ROOT then OLD_ROOT=op
        if (rom.line[l].setOLD_ROOT)
        {
            pols.oldRoot0[nexti] = op[0];
            pols.oldRoot1[nexti] = op[1];
            pols.oldRoot2[nexti] = op[2];
            pols.oldRoot3[nexti] = op[3];
            pols.setOldRoot[i] = 1;
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
            pols.setNewRoot[i] = 1;
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
            pols.setHashLeft[i] = 1;
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
            pols.setHashRight[i] = 1;
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
            pols.setSiblingRkey[i] = 1;
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
            pols.setSiblingValueHash[i] = 1;
        }
        else
        {
            pols.siblingValueHash0[nexti] = pols.siblingValueHash0[i];
            pols.siblingValueHash1[nexti] = pols.siblingValueHash1[i];
            pols.siblingValueHash2[nexti] = pols.siblingValueHash2[i];
            pols.siblingValueHash3[nexti] = pols.siblingValueHash3[i];
        }

        // Calculate op0 inverse
        if (op[0] != 0 )
        {
            pols.op0Inv[i] = fr.inv(op[0]);
        }

#ifdef LOG_STORAGE_EXECUTOR
        if ((i%1000) == 0) cout << "StorageExecutor step " << i << " done" << endl;
#endif
    }

    cout << "StorageExecutor successfully processed " << action.size() << " SMT actions" << endl;

    // TODO: Pending to integrate zkevm.pil.json when storage committed polynomials are listed there
//#endif
}

// To be used only for testing, since it allocates a lot of memory
void StorageExecutor::execute (vector<SmtAction> &action)
{
    void * pAddress = mapFile(config.cmPolsFile, CommitPols::size(), true);
    CommitPols cmPols(pAddress);
    execute(action, cmPols.Storage);
    unmapFile(pAddress, CommitPols::size());
}