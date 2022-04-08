#include "storage.hpp"
#include "utils.hpp"
#include "storage_rom.hpp"
#include "storage_pols.hpp"
#include "scalar.hpp"

void StorageExecutor::execute (vector<SmtAction> &action)
{
    json j;
    file2json("storage_sm_rom.json", j);
    StorageRom rom;
    rom.load(j);

    StoragePols pols;
    uint64_t polSize = 1<<16;
    pols.alloc(polSize);

    uint64_t l=0; // rom line
    uint64_t a=0; // action
    // TODO: Check with Jordi: when GET of an empty tree, what should we do, and what value should isOld0 have?
    //action.clear();
    bool actionListEmpty = (action.size()==0); // true if we run out of actions
    SmtActionContext ctx;

    if (!actionListEmpty)
    {
        ctx.init(action[a]);
    }

    for (uint64_t i=0; i<polSize; i++)
    {
        uint64_t op[4] = {0, 0, 0, 0};

        l = pols.PC[i];

        uint64_t nexti = (i+1)%polSize;

#ifdef LOG_STORAGE_EXECUTOR
        //rom.line[l].print(l); // Print the rom line content 
#endif

        /*************/
        /* Selectors */
        /*************/

        if (rom.line[l].inFREE)
        {
            if (rom.line[l].op == "functionCall")
            {
                /* Possible values of mode when SMT Set:
                    - update -> update existing value
                    - insertFound -> insert with found key; found a leaf node with a common set of key bits
                    - insertNotFound -> insert with no found key
                    - deleteFound -> delete with found key
                    - deleteNotFound -> delete with no found key
                    - deleteLast -> delete the last node, so root becomes 0
                    - zeroToZero -> value was zero and remains zero
                */
                if (rom.line[l].funcName=="isUpdate")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "update")
                    {
                        op[0] = 1;

#ifdef LOG_STORAGE_EXECUTOR
                        cout << "StorageExecutor isUpdate returns " << fea2string(fr, op) << endl;
#endif
                    }
                }
                else if (rom.line[l].funcName=="isInsertFound")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "insertFound")
                    {
                        op[0] = 1;

#ifdef LOG_STORAGE_EXECUTOR
                        cout << "StorageExecutor isInsertFound returns " << fea2string(fr, op) << endl;
#endif
                    }
                }
                else if (rom.line[l].funcName=="isInsertNotFound")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "insertNotFound")
                    {
                        op[0] = 1;

#ifdef LOG_STORAGE_EXECUTOR
                        cout << "StorageExecutor isInsertNotFound returns " << fea2string(fr, op) << endl;
#endif
                    }
                }
                else if (rom.line[l].funcName=="isSetReplacingZero")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "insertNotFound")
                    {
                        op[0] = 1;

#ifdef LOG_STORAGE_EXECUTOR
                        cout << "StorageExecutor isSetReplacingZero returns " << fea2string(fr, op) << endl;
#endif
                    }
                }
                else if (rom.line[l].funcName=="isSetWithSibling")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "insertFound")
                    {
                        op[0] = 1;

#ifdef LOG_STORAGE_EXECUTOR
                        cout << "StorageExecutor isSetWithSibling returns " << fea2string(fr, op) << endl;
#endif
                    }
                }
                else if (rom.line[l].funcName=="isDeleteLast")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "deleteLast")
                    {
                        op[0] = 1;

#ifdef LOG_STORAGE_EXECUTOR
                        cout << "StorageExecutor isDeleteLast returns " << fea2string(fr, op) << endl;
#endif
                    }
                }
                else if (rom.line[l].funcName=="isDeleteNotFound")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "deleteNotFound")
                    {
                        op[0] = 1;

#ifdef LOG_STORAGE_EXECUTOR
                        cout << "StorageExecutor isDeleteNotFound returns " << fea2string(fr, op) << endl;
#endif
                    }
                }
                else if (rom.line[l].funcName=="isZeroToZero")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "zeroToZero")
                    {
                        op[0] = 1;

#ifdef LOG_STORAGE_EXECUTOR
                        cout << "StorageExecutor isZeroToZero returns " << fea2string(fr, op) << endl;
#endif
                    }
                }
                else if (rom.line[l].funcName=="isGet")
                {
                    if (!actionListEmpty &&
                        !action[a].bIsSet &&
                        action[a].getResult.isOld0)
                    {
                        op[0] = 1;

#ifdef LOG_STORAGE_EXECUTOR
                        cout << "StorageExecutor isGet returns " << fea2string(fr, op) << endl;
#endif
                    }
                }
                else if (rom.line[l].funcName=="isGetZero")
                {
                    if (!actionListEmpty &&
                        !action[a].bIsSet &&
                        !action[a].getResult.isOld0)
                    {
                        op[0] = 1;

#ifdef LOG_STORAGE_EXECUTOR
                        cout << "StorageExecutor isGetZero returns " << fea2string(fr, op) << endl;
#endif
                    }
                }
                else if (rom.line[l].funcName=="isOld0")
                {
                    if (action[a].bIsSet)
                    {
                        if (action[a].setResult.isOld0) op[0]=1;
                    }
                    else
                    {
                        if (action[a].getResult.isOld0) op[0]=1;
                    }

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor isOld0 returns " << fea2string(fr, op) << endl;
#endif
                }
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
                else if (rom.line[l].funcName=="GetSiblingHash")
                {
                    op[0] = ctx.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4];
                    op[1] = ctx.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4+1];
                    op[2] = ctx.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4+2];
                    op[3] = ctx.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4+3];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetSiblingHash returns " << fea2string(fr, op) << endl;
#endif
                }
                else if (rom.line[l].funcName=="GetValueLow")
                {
                    FieldElement fea[8];
                    if (action[a].bIsSet)
                    {
                        scalar2fea(fr, action[a].setResult.newValue, fea);
                    }
                    else
                    {
                        scalar2fea(fr, action[a].getResult.value, fea);                        
                    }
                    op[0] = fea[0];
                    op[1] = fea[1];
                    op[2] = fea[2];
                    op[3] = fea[3];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetValueLow returns " << fea2string(fr, op) << endl;
#endif
                }
                else if (rom.line[l].funcName=="GetValueHigh")
                {
                    FieldElement fea[8];
                    if (action[a].bIsSet)
                    {
                        scalar2fea(fr, action[a].setResult.newValue, fea);
                    }
                    else
                    {
                        scalar2fea(fr, action[a].getResult.value, fea);                        
                    }
                    op[0] = fea[4];
                    op[1] = fea[5];
                    op[2] = fea[6];
                    op[3] = fea[7];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetValueHigh returns " << fea2string(fr, op) << endl;
#endif
                }
                #if 0
                else if (rom.line[l].funcName=="GetSiblingValueLow")
                {
                    FieldElement fea[8];
                    if (action[a].bIsSet)
                    {
                        scalar2fea(fr, action[a].setResult.oldValue, fea);
                    }
                    else
                    {
                        // Error scalar2fea(fr, action[a].getResult.insValue, fea);                        
                    }
                    op[0] = fea[0];
                    op[1] = fea[1];
                    op[2] = fea[2];
                    op[3] = fea[3];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetSiblingValueLow returns " << fea2string(fr, op) << endl;
#endif
                }
                else if (rom.line[l].funcName=="GetSiblingValueHigh")
                {
                    FieldElement fea[8];
                    if (action[a].bIsSet)
                    {
                        scalar2fea(fr, action[a].setResult.oldValue, fea);
                    }
                    else
                    {
                        // Error scalar2fea(fr, action[a].getResult.value, fea);                        
                    }
                    op[0] = fea[4];
                    op[1] = fea[5];
                    op[2] = fea[6];
                    op[3] = fea[7];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetSiblingValueHigh returns " << fea2string(fr, op) << endl;
#endif
                }
                #endif
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
                    if ( ( ctx.level & (1<<bit) ) == 0)
                    {
                        op[0] = 0;
                    }
                    else
                    {
                        op[0] = 1;
                    }

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetLevelBit(" << bit << ") returns " << fea2string(fr, op) << endl;
#endif
                }
                else if (rom.line[l].funcName=="GetTopTree")
                {
                    op[0] = (ctx.currentLevel<=0) ? 0 : 1;

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetTopTree returns " << fea2string(fr, op) << endl;
#endif
                }
                else if (rom.line[l].funcName=="GetNextKeyBit")
                {
                    ctx.currentLevel--;
                    if (ctx.currentLevel<0)
                    {
                        cerr << "Error: StorageExecutor.execute() GetNextKeyBit() found ctx.currentLevel<0" << endl;
                        exit(-1);
                    }
                    op[0] = ctx.bits[ctx.currentLevel];

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetNextKeyBit returns " << fea2string(fr, op) << endl;
#endif
                }
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
                else if (rom.line[l].funcName=="GetTopOfBranch")
                {
                    if (ctx.currentLevel > (int64_t)ctx.siblings.size() )
                    {
                        op[0] = 1;
                    }

#ifdef LOG_STORAGE_EXECUTOR
                    cout << "StorageExecutor GetTopOfBranch returns " << fea2string(fr, op) << endl;
#endif
                }
                else if (rom.line[l].funcName=="isEndPolynomial")
                {
                    if (i==polSize-1)
                    {
                        op[0] = fr.one();
#ifdef LOG_STORAGE_EXECUTOR
                        cout << "StorageExecutor isEndPolynomial returns " << fea2string(fr,op) << endl;
#endif
                    }
                    else
                    {
                        op[0] = fr.zero();
                    }
                }
                else
                {
                    cerr << "Error: StorageExecutor() unknown funcName:" << rom.line[l].funcName << endl;
                    exit(-1);
                }                
            }
            else if (rom.line[l].op=="")
            {
                // Ignore; this is just to report a list of setters
            }
            else
            {
                cerr << "Error: StorageExecutor() unknown op:" << rom.line[l].op << endl;
                exit(-1);
            }
            pols.inFREE[i] = 1;
        }

        if (rom.line[l].CONST!="")
        {
            op[0] = stoi(rom.line[l].CONST);
            op[1] = 0;
            op[2] = 0;
            op[3] = 0;
            pols.CONST[i] = op[0];
        }

        if (rom.line[l].inOLD_ROOT)
        {
            op[0] = pols.OLD_ROOT0[i];
            op[1] = pols.OLD_ROOT1[i];
            op[2] = pols.OLD_ROOT2[i];
            op[3] = pols.OLD_ROOT3[i];
            pols.inOLD_ROOT[i] = 1;
        }

        if (rom.line[l].inNEW_ROOT)
        {
            op[0] = pols.NEW_ROOT0[i];
            op[1] = pols.NEW_ROOT1[i];
            op[2] = pols.NEW_ROOT2[i];
            op[3] = pols.NEW_ROOT3[i];
            pols.inNEW_ROOT[i] = 1;
        }

        if (rom.line[l].inRKEY_BIT)
        {
            op[0] = pols.RKEY_BIT[i];
            op[1] = 0;
            op[2] = 0;
            op[3] = 0;
            pols.inRKEY_BIT[i] = 1;
        }

        if (rom.line[l].inVALUE_LOW)
        {
            op[0] = pols.VALUE_LOW0[i];
            op[1] = pols.VALUE_LOW1[i];
            op[2] = pols.VALUE_LOW2[i];
            op[3] = pols.VALUE_LOW3[i];
            pols.inVALUE_LOW[i] = 1;
        }

        if (rom.line[l].inVALUE_HIGH)
        {
            op[0] = pols.VALUE_HIGH0[i];
            op[1] = pols.VALUE_HIGH1[i];
            op[2] = pols.VALUE_HIGH2[i];
            op[3] = pols.VALUE_HIGH3[i];
            pols.inVALUE_HIGH[i] = 1;
        }

        if (rom.line[l].inRKEY)
        {
            op[0] = pols.RKEY0[i];
            op[1] = pols.RKEY1[i];
            op[2] = pols.RKEY2[i];
            op[3] = pols.RKEY3[i];
            pols.inRKEY[i] = 1;
        }

        if (rom.line[l].inSIBLING_RKEY)
        {
            op[0] = pols.SIBLING_RKEY0[i];
            op[1] = pols.SIBLING_RKEY1[i];
            op[2] = pols.SIBLING_RKEY2[i];
            op[3] = pols.SIBLING_RKEY3[i];
            pols.inSIBLING_RKEY[i] = 1;
        }

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
            //cout << "StorageExecutor iJmp address=" << rom.line[l].address << endl;
            pols.iJmp[i] = 1;
        }
        // Increment program counter
        else
        {
            pols.PC[nexti] = pols.PC[i] + 1;
        }

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

        if (rom.line[l].iHash)
        {
            // Prepare the data to hash
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

        if (rom.line[l].iClimbSiblingRkey)
        {
#ifdef LOG_STORAGE_EXECUTOR
            FieldElement fea1[4] = {pols.SIBLING_RKEY0[i], pols.SIBLING_RKEY1[i], pols.SIBLING_RKEY2[i], pols.SIBLING_RKEY3[i]};
            cout << "StorageExecutor iClimbSiblingRkey before rkey=" << fea2string(fr,fea1) << endl;
#endif
            // TODO: Check with Jordi if it is ok to use ctx.siblingBits[] internally or not
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

        if (rom.line[l].iLatchGet)
        {
            // TODO: check with Jordi
            // At this point consistency is granted: OLD_ROOT, RKEY (complete key), VALUE_LOW, VALUE_HIGH, LEVEL
            
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
            if (a>=action.size())
            {
#ifdef LOG_STORAGE_EXECUTOR
                cout << "StorageExecutor LATCH GET detected the end of the action list a=" << a << " i=" << i << endl;
#endif
                actionListEmpty = true;
            }
            else
            {
                ctx.init(action[a]);
            }

            pols.iLatchGet[i] = 1;
        }

        if (rom.line[l].iLatchSet)
        {
            // TODO: check with Jordi
            // At this point consistency is granted: OLD_ROOT, NEW_ROOT, RKEY (complete key), VALUE_LOW, VALUE_HIGH, LEVEL
            
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
            if (a>=action.size())
            {
#ifdef LOG_STORAGE_EXECUTOR
                cout << "StorageExecutor() LATCH SET detected the end of the action list a=" << a << " i=" << i << endl;
#endif
                actionListEmpty = true;
            }
            else
            {
                ctx.init(action[a]);
            }

            pols.iLatchSet[i] = 1;
        }

        /***********/
        /* Setters */
        /***********/

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

        if (rom.line[l].setRKEY_BIT)
        {
            pols.RKEY_BIT[nexti] = op[0];
            pols.setRKEY_BIT[i] = 1;
        }
        else
        {
            pols.RKEY_BIT[nexti] = pols.RKEY_BIT[i];
        }
        
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

#ifdef LOG_STORAGE_EXECUTOR
        if ((i%1000) == 0) cout << "StorageExecutor step " << i << " done" << endl;
#endif
    }
}

