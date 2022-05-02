
#include <nlohmann/json.hpp>
#include "storage.hpp"
#include "storage_rom.hpp"
#include "storage_pols.hpp"
#include "utils.hpp"
#include "scalar.hpp"

using json = nlohmann::json;

void StorageExecutor::execute (vector<SmtAction> &action)
{
    // Allocate polynomials
    StoragePols pols(config);
    uint64_t polSize = 1<<16;
    pols.alloc(polSize, pilJson);

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
    for (uint64_t i=0; i<polSize; i++)
    {
        // op is the internal register, reset to 0 at every evaluation
        uint64_t op[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};

        // Current rom line is set by the program counter of this evaluation
        l = pols.PC[i];

        // Set the next evaluation index, which will be 0 when we reach the last evaluation
        uint64_t nexti = (i+1)%polSize;

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
                    if (i == (polSize-2))
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
            if (op[0] != 0) pols.FREE0[i] = op[0];
            if (op[1] != 0) pols.FREE1[i] = op[1];
            if (op[2] != 0) pols.FREE2[i] = op[2];
            if (op[3] != 0) pols.FREE3[i] = op[3];            

            // Mark the inFREE register as 1
            pols.inFREE[i] = 1;
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
            pols.CONST0[i] = op[0];
            pols.CONST1[i] = op[1];
            pols.CONST2[i] = op[2];
            pols.CONST3[i] = op[3];
        }

        // If inOLD_ROOT then op=OLD_ROOT
        if (rom.line[l].inOLD_ROOT)
        {
            op[0] = pols.OLD_ROOT0[i];
            op[1] = pols.OLD_ROOT1[i];
            op[2] = pols.OLD_ROOT2[i];
            op[3] = pols.OLD_ROOT3[i];
            pols.inOLD_ROOT[i] = 1;
        }

        // If inNEW_ROOT then op=NEW_ROOT
        if (rom.line[l].inNEW_ROOT)
        {
            op[0] = pols.NEW_ROOT0[i];
            op[1] = pols.NEW_ROOT1[i];
            op[2] = pols.NEW_ROOT2[i];
            op[3] = pols.NEW_ROOT3[i];
            pols.inNEW_ROOT[i] = 1;
        }

        // If inRKEY_BIT then op=RKEY_BIT
        if (rom.line[l].inRKEY_BIT)
        {
            op[0] = pols.RKEY_BIT[i];
            op[1] = 0;
            op[2] = 0;
            op[3] = 0;
            pols.inRKEY_BIT[i] = 1;
        }

        // If inVALUE_LOW then op=VALUE_LOW
        if (rom.line[l].inVALUE_LOW)
        {
            op[0] = pols.VALUE_LOW0[i];
            op[1] = pols.VALUE_LOW1[i];
            op[2] = pols.VALUE_LOW2[i];
            op[3] = pols.VALUE_LOW3[i];
            pols.inVALUE_LOW[i] = 1;
        }

        // If inVALUE_HIGH then op=VALUE_HIGH
        if (rom.line[l].inVALUE_HIGH)
        {
            op[0] = pols.VALUE_HIGH0[i];
            op[1] = pols.VALUE_HIGH1[i];
            op[2] = pols.VALUE_HIGH2[i];
            op[3] = pols.VALUE_HIGH3[i];
            pols.inVALUE_HIGH[i] = 1;
        }

        // If inRKEY then op=RKEY
        if (rom.line[l].inRKEY)
        {
            op[0] = pols.RKEY0[i];
            op[1] = pols.RKEY1[i];
            op[2] = pols.RKEY2[i];
            op[3] = pols.RKEY3[i];
            pols.inRKEY[i] = 1;
        }

        // If inSIBLING_RKEY then op=SIBLING_RKEY
        if (rom.line[l].inSIBLING_RKEY)
        {
            op[0] = pols.SIBLING_RKEY0[i];
            op[1] = pols.SIBLING_RKEY1[i];
            op[2] = pols.SIBLING_RKEY2[i];
            op[3] = pols.SIBLING_RKEY3[i];
            pols.inSIBLING_RKEY[i] = 1;
        }

        // If inSIBLING_VALUE_HASH then op=SIBLING_VALUE_HASH
        if (rom.line[l].inSIBLING_VALUE_HASH)
        {
            op[0] = pols.SIBLING_VALUE_HASH0[i];
            op[1] = pols.SIBLING_VALUE_HASH1[i];
            op[2] = pols.SIBLING_VALUE_HASH2[i];
            op[3] = pols.SIBLING_VALUE_HASH3[i];
            pols.inSIBLING_VALUE_HASH[i] = 1;
        }

        /****************/
        /* Instructions */
        /****************/

        // JMPZ: Jump if OP==0
        if (rom.line[l].iJmpz)
        {
            if (op[0]==0 && op[1]==0 && op[2]==0 && op[3]==0)
            {
                pols.PC[nexti] = rom.line[l].address;
                pols.iAddress[i] = rom.line[l].address;
                //cout << "StorageExecutor iJmpz address=" << rom.line[l].address << endl;
            }
            else
            {
                pols.PC[nexti] = pols.PC[i] + 1;
            }
            pols.iJmpz[i] = 1;
        }

        // JMP: Jump always
        else if (rom.line[l].iJmp)
        {
            pols.PC[nexti] = rom.line[l].address;
            pols.iAddress[i] = rom.line[l].address;
            //cout << "StorageExecutor iJmp address=" << rom.line[l].address << endl;
            pols.iJmp[i] = 1;
        }

        // If not any jump, then simply increment program counter
        else
        {
            pols.PC[nexti] = pols.PC[i] + 1;
        }

        // Rotate level registers values, from higher to lower
        if (rom.line[l].iRotateLevel)
        {
            uint64_t aux;
            aux = pols.LEVEL0[i];
            pols.LEVEL0[i] = pols.LEVEL1[i];
            pols.LEVEL1[i] = pols.LEVEL2[i];
            pols.LEVEL2[i] = pols.LEVEL3[i];
            pols.LEVEL3[i] = aux;
            pols.iRotateLevel[i] = 1;

#ifdef LOG_STORAGE_EXECUTOR
            cout << "StorageExecutor iRotateLevel level[3:2:1:0]=" << pols.LEVEL3[i] << ":" << pols.LEVEL2[i] << ":" << pols.LEVEL1[i] << ":" << pols.LEVEL0[i] << endl;
#endif
        }

        // Hash: op = poseidon.hash(HASH_LEFT + HASH_RIGHT + (0 or 1, depending on iHashType))
        if (rom.line[l].iHash)
        {
            // Prepare the data to hash: HASH_LEFT + HASH_RIGHT + 0 or 1, depending on iHashType
            FieldElement fea[12];
            fea[0] = pols.HASH_LEFT0[i];
            fea[1] = pols.HASH_LEFT1[i];
            fea[2] = pols.HASH_LEFT2[i];
            fea[3] = pols.HASH_LEFT3[i];
            fea[4] = pols.HASH_RIGHT0[i];
            fea[5] = pols.HASH_RIGHT1[i];
            fea[6] = pols.HASH_RIGHT2[i];
            fea[7] = pols.HASH_RIGHT3[i];
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
            op[0] = fea[0];
            op[1] = fea[1];
            op[2] = fea[2];
            op[3] = fea[3];

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
            uint64_t bit = pols.RKEY_BIT[i];
            if (pols.LEVEL0[i] == 1)
            {
                pols.RKEY0[i] = (pols.RKEY0[i]<<1) + bit;
            }
            if (pols.LEVEL1[i] == 1)
            {
                pols.RKEY1[i] = (pols.RKEY1[i]<<1) + bit;
            }
            if (pols.LEVEL2[i] == 1)
            {
                pols.RKEY2[i] = (pols.RKEY2[i]<<1) + bit;
            }
            if (pols.LEVEL3[i] == 1)
            {
                pols.RKEY3[i] = (pols.RKEY3[i]<<1) + bit;
            }
            pols.iClimbRkey[i] = 1;

#ifdef LOG_STORAGE_EXECUTOR
            FieldElement fea[4] = {pols.RKEY0[i], pols.RKEY1[i], pols.RKEY2[i], pols.RKEY3[i]};
            cout << "StorageExecutor iClimbRkey sibling bit=" << bit << " rkey=" << fea2string(fr,fea) << endl;
#endif
        }

        // Climb the sibling remaining key, by injecting the sibling bit in the register specified by LEVEL
        if (rom.line[l].iClimbSiblingRkey)
        {
#ifdef LOG_STORAGE_EXECUTOR
            FieldElement fea1[4] = {pols.SIBLING_RKEY0[i], pols.SIBLING_RKEY1[i], pols.SIBLING_RKEY2[i], pols.SIBLING_RKEY3[i]};
            cout << "StorageExecutor iClimbSiblingRkey before rkey=" << fea2string(fr,fea1) << endl;
#endif
            uint64_t bit = ctx.siblingBits[ctx.currentLevel];
            if (pols.LEVEL0[i] == 1)
            {
                pols.SIBLING_RKEY0[i] = (pols.SIBLING_RKEY0[i]<<1) + bit;
            }
            if (pols.LEVEL1[i] == 1)
            {
                pols.SIBLING_RKEY1[i] = (pols.SIBLING_RKEY1[i]<<1) + bit;
            }
            if (pols.LEVEL2[i] == 1)
            {
                pols.SIBLING_RKEY2[i] = (pols.SIBLING_RKEY2[i]<<1) + bit;
            }
            if (pols.LEVEL3[i] == 1)
            {
                pols.SIBLING_RKEY3[i] = (pols.SIBLING_RKEY3[i]<<1) + bit;
            }
            pols.iClimbSiblingRkey[i] = 1;

#ifdef LOG_STORAGE_EXECUTOR
            FieldElement fea[4] = {pols.SIBLING_RKEY0[i], pols.SIBLING_RKEY1[i], pols.SIBLING_RKEY2[i], pols.SIBLING_RKEY3[i]};
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
                FieldElement oldRoot[4] = {pols.OLD_ROOT0[i], pols.OLD_ROOT1[i], pols.OLD_ROOT2[i], pols.OLD_ROOT3[i]};
                if ( !fr.eq(oldRoot, action[a].getResult.root) )
                {
                    cerr << "Error: StorageExecutor() LATCH GET found action " << a << " pols.OLD_ROOT=" << fea2string(fr,oldRoot) << " different from action.getResult.root=" << fea2string(fr,action[a].getResult.root) << endl;
                    exit(-1);
                }

                // Check that the calculated complete key is the same as the provided action key
                if ( pols.RKEY0[i] != action[a].getResult.key[0] ||
                     pols.RKEY1[i] != action[a].getResult.key[1] ||
                     pols.RKEY2[i] != action[a].getResult.key[2] ||
                     pols.RKEY3[i] != action[a].getResult.key[3] )
                {
                    cerr << "Error: StorageExecutor() LATCH GET found action " << a << " pols.RKEY!=action.getResult.key" << endl;
                    exit(-1);                
                }

                // Check that final level state is consistent
                if ( pols.LEVEL0[i] != 1 ||
                     pols.LEVEL1[i] != 0 ||
                     pols.LEVEL2[i] != 0 ||
                     pols.LEVEL3[i] != 0 )
                {
                    cerr << "Error: StorageExecutor() LATCH GET found action " << a << " wrong level=" << pols.LEVEL3[i] << ":" << pols.LEVEL2[i] << ":" << pols.LEVEL1[i] << ":" << pols.LEVEL0[i] << endl;
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
            FieldElement oldRoot[4] = {pols.OLD_ROOT0[i], pols.OLD_ROOT1[i], pols.OLD_ROOT2[i], pols.OLD_ROOT3[i]};
            if ( !fr.eq(oldRoot, action[a].setResult.oldRoot) )
            {
                cerr << "Error: StorageExecutor() LATCH SET found action " << a << " pols.OLD_ROOT=" << fea2string(fr,oldRoot) << " different from action.setResult.oldRoot=" << fea2string(fr,action[a].setResult.oldRoot) << endl;
                exit(-1);
            }

            // Check that the calculated old root is the same as the provided action root
            FieldElement newRoot[4] = {pols.NEW_ROOT0[i], pols.NEW_ROOT1[i], pols.NEW_ROOT2[i], pols.NEW_ROOT3[i]};
            if ( !fr.eq(newRoot, action[a].setResult.newRoot) )
            {
                cerr << "Error: StorageExecutor() LATCH SET found action " << a << " pols.NEW_ROOT=" << fea2string(fr,newRoot) << " different from action.setResult.newRoot=" << fea2string(fr,action[a].setResult.newRoot) << endl;
                exit(-1);
            }

            // Check that the calculated complete key is the same as the provided action key
            if ( pols.RKEY0[i] != action[a].setResult.key[0] ||
                 pols.RKEY1[i] != action[a].setResult.key[1] ||
                 pols.RKEY2[i] != action[a].setResult.key[2] ||
                 pols.RKEY3[i] != action[a].setResult.key[3] )
            {
                cerr << "Error: StorageExecutor() LATCH SET found action " << a << " pols.RKEY!=action.setResult.key" << endl;
                exit(-1);                
            }

            // Check that final level state is consistent
            if ( pols.LEVEL0[i] != 1 ||
                 pols.LEVEL1[i] != 0 ||
                 pols.LEVEL2[i] != 0 ||
                 pols.LEVEL3[i] != 0 )
            {
                cerr << "Error: StorageExecutor() LATCH SET found action " << a << " wrong level=" << pols.LEVEL3[i] << ":" << pols.LEVEL2[i] << ":" << pols.LEVEL1[i] << ":" << pols.LEVEL0[i] << endl;
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
            pols.RKEY0[nexti] = op[0];
            pols.RKEY1[nexti] = op[1];
            pols.RKEY2[nexti] = op[2];
            pols.RKEY3[nexti] = op[3];
            pols.setRKEY[i] = 1;
        }
        else
        {
            pols.RKEY0[nexti] = pols.RKEY0[i];
            pols.RKEY1[nexti] = pols.RKEY1[i];
            pols.RKEY2[nexti] = pols.RKEY2[i];
            pols.RKEY3[nexti] = pols.RKEY3[i];
        }

        // If setRKEY_BIT then RKEY_BIT=op
        if (rom.line[l].setRKEY_BIT)
        {
            pols.RKEY_BIT[nexti] = op[0];
            pols.setRKEY_BIT[i] = 1;
        }
        else
        {
            pols.RKEY_BIT[nexti] = pols.RKEY_BIT[i];
        }
        
        // If setVALUE_LOW then VALUE_LOW=op
        if (rom.line[l].setVALUE_LOW)
        {
            pols.VALUE_LOW0[nexti] = op[0];
            pols.VALUE_LOW1[nexti] = op[1];
            pols.VALUE_LOW2[nexti] = op[2];
            pols.VALUE_LOW3[nexti] = op[3];
            pols.setVALUE_LOW[i] = 1;
        }
        else
        {
            pols.VALUE_LOW0[nexti] = pols.VALUE_LOW0[i];
            pols.VALUE_LOW1[nexti] = pols.VALUE_LOW1[i];
            pols.VALUE_LOW2[nexti] = pols.VALUE_LOW2[i];
            pols.VALUE_LOW3[nexti] = pols.VALUE_LOW3[i];
        }
        
        // If setVALUE_HIGH then VALUE_HIGH=op
        if (rom.line[l].setVALUE_HIGH)
        {
            pols.VALUE_HIGH0[nexti] = op[0];
            pols.VALUE_HIGH1[nexti] = op[1];
            pols.VALUE_HIGH2[nexti] = op[2];
            pols.VALUE_HIGH3[nexti] = op[3];
            pols.setVALUE_HIGH[i] = 1;
        }
        else
        {
            pols.VALUE_HIGH0[nexti] = pols.VALUE_HIGH0[i];
            pols.VALUE_HIGH1[nexti] = pols.VALUE_HIGH1[i];
            pols.VALUE_HIGH2[nexti] = pols.VALUE_HIGH2[i];
            pols.VALUE_HIGH3[nexti] = pols.VALUE_HIGH3[i];
        }
        
        // If setLEVEL then LEVEL=op
        if (rom.line[l].setLEVEL)
        {
            pols.LEVEL0[nexti] = op[0];
            pols.LEVEL1[nexti] = op[1];
            pols.LEVEL2[nexti] = op[2];
            pols.LEVEL3[nexti] = op[3];
            pols.setLEVEL[i] = 1;
        }
        else
        {
            pols.LEVEL0[nexti] = pols.LEVEL0[i];
            pols.LEVEL1[nexti] = pols.LEVEL1[i];
            pols.LEVEL2[nexti] = pols.LEVEL2[i];
            pols.LEVEL3[nexti] = pols.LEVEL3[i];
        }
        
        // If setOLD_ROOT then OLD_ROOT=op
        if (rom.line[l].setOLD_ROOT)
        {
            pols.OLD_ROOT0[nexti] = op[0];
            pols.OLD_ROOT1[nexti] = op[1];
            pols.OLD_ROOT2[nexti] = op[2];
            pols.OLD_ROOT3[nexti] = op[3];
            pols.setOLD_ROOT[i] = 1;
        }
        else
        {
            pols.OLD_ROOT0[nexti] = pols.OLD_ROOT0[i];
            pols.OLD_ROOT1[nexti] = pols.OLD_ROOT1[i];
            pols.OLD_ROOT2[nexti] = pols.OLD_ROOT2[i];
            pols.OLD_ROOT3[nexti] = pols.OLD_ROOT3[i];
        }
        
        // If setNEW_ROOT then NEW_ROOT=op
        if (rom.line[l].setNEW_ROOT)
        {
            pols.NEW_ROOT0[nexti] = op[0];
            pols.NEW_ROOT1[nexti] = op[1];
            pols.NEW_ROOT2[nexti] = op[2];
            pols.NEW_ROOT3[nexti] = op[3];
            pols.setNEW_ROOT[i] = 1;
        }
        else
        {
            pols.NEW_ROOT0[nexti] = pols.NEW_ROOT0[i];
            pols.NEW_ROOT1[nexti] = pols.NEW_ROOT1[i];
            pols.NEW_ROOT2[nexti] = pols.NEW_ROOT2[i];
            pols.NEW_ROOT3[nexti] = pols.NEW_ROOT3[i];
        }
        
        // If setHASH_LEFT then HASH_LEFT=op
        if (rom.line[l].setHASH_LEFT)
        {
            pols.HASH_LEFT0[nexti] = op[0];
            pols.HASH_LEFT1[nexti] = op[1];
            pols.HASH_LEFT2[nexti] = op[2];
            pols.HASH_LEFT3[nexti] = op[3];
            pols.setHASH_LEFT[i] = 1;
        }
        else
        {
            pols.HASH_LEFT0[nexti] = pols.HASH_LEFT0[i];
            pols.HASH_LEFT1[nexti] = pols.HASH_LEFT1[i];
            pols.HASH_LEFT2[nexti] = pols.HASH_LEFT2[i];
            pols.HASH_LEFT3[nexti] = pols.HASH_LEFT3[i];
        }
        
        // If setHASH_RIGHT then HASH_RIGHT=op
        if (rom.line[l].setHASH_RIGHT)
        {
            pols.HASH_RIGHT0[nexti] = op[0];
            pols.HASH_RIGHT1[nexti] = op[1];
            pols.HASH_RIGHT2[nexti] = op[2];
            pols.HASH_RIGHT3[nexti] = op[3];
            pols.setHASH_RIGHT[i] = 1;
        }
        else
        {
            pols.HASH_RIGHT0[nexti] = pols.HASH_RIGHT0[i];
            pols.HASH_RIGHT1[nexti] = pols.HASH_RIGHT1[i];
            pols.HASH_RIGHT2[nexti] = pols.HASH_RIGHT2[i];
            pols.HASH_RIGHT3[nexti] = pols.HASH_RIGHT3[i];
        }
        
        // If setSIBLING_RKEY then SIBLING_RKEY=op
        if (rom.line[l].setSIBLING_RKEY)
        {
            pols.SIBLING_RKEY0[nexti] = op[0];
            pols.SIBLING_RKEY1[nexti] = op[1];
            pols.SIBLING_RKEY2[nexti] = op[2];
            pols.SIBLING_RKEY3[nexti] = op[3];
            pols.setSIBLING_RKEY[i] = 1;
        }
        else
        {
            pols.SIBLING_RKEY0[nexti] = pols.SIBLING_RKEY0[i];
            pols.SIBLING_RKEY1[nexti] = pols.SIBLING_RKEY1[i];
            pols.SIBLING_RKEY2[nexti] = pols.SIBLING_RKEY2[i];
            pols.SIBLING_RKEY3[nexti] = pols.SIBLING_RKEY3[i];
        }
        
        // If setSIBLING_VALUE_HASH then SIBLING_VALUE_HASH=op
        if (rom.line[l].setSIBLING_VALUE_HASH)
        {
            pols.SIBLING_VALUE_HASH0[nexti] = op[0];
            pols.SIBLING_VALUE_HASH1[nexti] = op[1];
            pols.SIBLING_VALUE_HASH2[nexti] = op[2];
            pols.SIBLING_VALUE_HASH3[nexti] = op[3];
            pols.setSIBLING_VALUE_HASH[i] = 1;
        }
        else
        {
            pols.SIBLING_VALUE_HASH0[nexti] = pols.SIBLING_VALUE_HASH0[i];
            pols.SIBLING_VALUE_HASH1[nexti] = pols.SIBLING_VALUE_HASH1[i];
            pols.SIBLING_VALUE_HASH2[nexti] = pols.SIBLING_VALUE_HASH2[i];
            pols.SIBLING_VALUE_HASH3[nexti] = pols.SIBLING_VALUE_HASH3[i];
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

    // Deallocate polynomials
    pols.dealloc();

    cout << "StorageExecutor successfully processed " << action.size() << " SMT actions" << endl;

    // TODO: Pending to integrate zkevm.pil.json when storage committed polynomials are listed there
}

