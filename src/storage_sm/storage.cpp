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
    uint64_t a=2; // action
    //action.clear();
    bool actionListEmpty = (action.size()==0); // true if we run out of actions
    SmtActionContext ctx;

    if (!actionListEmpty)
    {
        ctx.init(action[a]);
    }

    for (uint64_t i=0; i<polSize; i++)
    {
        uint64_t op0, op1, op2, op3;
        op0=0;
        op1=0;
        op2=0;
        op3=0;

        l = pols.PC[i];

        uint64_t nexti = (i+1)%polSize;

        // Print the rom line content
        rom.line[l].print(l);

        /*************/
        /* Selectors */
        /*************/

        if (rom.line[l].inFREE)
        {
            if (rom.line[l].op == "functionCall")
            {
                /* Possible values of mode:
                    - update -> update existing value
                    - insertFound -> insert with found key
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
                        op0 = 1;
                    }
                }
                else if (rom.line[l].funcName=="isSetReplacingZero")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "insertNotFound")
                    {
                        op0 = 1;
                    }
                }
                else if (rom.line[l].funcName=="isSetWithSibling")
                {
                    if (!actionListEmpty &&
                        action[a].bIsSet &&
                        action[a].setResult.mode == "insertFound")
                    {
                        op0 = 1;
                    }
                }
                else if (rom.line[l].funcName=="isGet")
                {
                    if (!actionListEmpty &&
                        !action[a].bIsSet)
                    {
                        op0 = 1;
                    }
                }
                else if (rom.line[l].funcName=="isOld0")
                {
                    if (action[a].bIsSet)
                    {
                        if (action[a].setResult.isOld0) op0=1;
                    }
                    else
                    {
                        if (action[a].getResult.isOld0) op0=1;
                    }
                }
                else if (rom.line[l].funcName=="GetRKey")
                {
                    op0 = ctx.rKey[0];
                    op1 = ctx.rKey[1];
                    op2 = ctx.rKey[2];
                    op3 = ctx.rKey[3];
                }
                else if (rom.line[l].funcName=="GetSiblingRKey")
                {
                    op0 = ctx.siblings[ctx.currentLevel][ctx.bits[ctx.currentLevel]*4];
                    op1 = ctx.siblings[ctx.currentLevel][ctx.bits[ctx.currentLevel]*4+1];
                    op2 = ctx.siblings[ctx.currentLevel][ctx.bits[ctx.currentLevel]*4+2];
                    op3 = ctx.siblings[ctx.currentLevel][ctx.bits[ctx.currentLevel]*4+3];
                }
                else if (rom.line[l].funcName=="GetSiblingHash")
                {
                    op0 = ctx.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4];
                    op1 = ctx.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4+1];
                    op2 = ctx.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4+2];
                    op3 = ctx.siblings[ctx.currentLevel][(1-ctx.bits[ctx.currentLevel])*4+3];
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
                    op0 = fea[0];
                    op1 = fea[1];
                    op2 = fea[2];
                    op3 = fea[3];
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
                    op0 = fea[4];
                    op1 = fea[5];
                    op2 = fea[6];
                    op3 = fea[7];
                }
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

                    // Set the bit in op0
                    if ( ( ctx.level & (1<<bit) ) == 0)
                    {
                        op0 = 0;
                    }
                    else
                    {
                        op0 = 1;
                    }
                }
                else if (rom.line[l].funcName=="GetTopTree")
                {
                    op0 = (ctx.currentLevel<=0) ? 0 : 1;
                }
                else if (rom.line[l].funcName=="GetNextKeyBit")
                {
                    ctx.currentLevel--;
                    if (ctx.currentLevel<0)
                    {
                        cerr << "Error: StorageExecutor.execute() GetNextKeyBit() found ctx.currentLevel<0" << endl;
                        exit(-1);
                    }
                    op0 = ctx.bits[ctx.currentLevel];
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
                    op0 = fea[0];
                    op1 = fea[1];
                    op2 = fea[2];
                    op3 = fea[3];
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

                    // Take the lower 4 field elements
                    op0 = fea[4];
                    op1 = fea[5];
                    op2 = fea[6];
                    op3 = fea[7];
                }
                else if (rom.line[l].funcName=="GetTopOfBranch")
                {
                    sleep(1);
                }
                else if (rom.line[l].funcName=="isEndPolinomial")
                {
                    if (i==polSize-1)
                    {
                        op0 = fr.one();
                    }
                    else
                    {
                        op0 = fr.zero();
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
            op0 = stoi(rom.line[l].CONST);
            op1 = 0;
            op2 = 0;
            op3 = 0;
            pols.CONST[i] = op0;
        }

        if (rom.line[l].inOLD_ROOT)
        {
            op0 = pols.OLD_ROOT0[i];
            op1 = pols.OLD_ROOT1[i];
            op2 = pols.OLD_ROOT2[i];
            op3 = pols.OLD_ROOT3[i];
            pols.inOLD_ROOT[i] = 1;
        }

        if (rom.line[l].inNEW_ROOT)
        {
            op0 = pols.NEW_ROOT0[i];
            op1 = pols.NEW_ROOT1[i];
            op2 = pols.NEW_ROOT2[i];
            op3 = pols.NEW_ROOT3[i];
            pols.inNEW_ROOT[i] = 1;
        }

        if (rom.line[l].inRKEY_BIT)
        {
            op0 = pols.RKEY_BIT[i];
            op1 = 0;
            op2 = 0;
            op3 = 0;
            pols.inRKEY_BIT[i] = 1;
        }

        if (rom.line[l].inVALUE_LOW)
        {
            op0 = pols.VALUE_LOW0[i];
            op1 = pols.VALUE_LOW1[i];
            op2 = pols.VALUE_LOW2[i];
            op3 = pols.VALUE_LOW3[i];
            pols.inVALUE_LOW[i] = 1;
        }

        if (rom.line[l].inVALUE_HIGH)
        {
            op0 = pols.VALUE_HIGH0[i];
            op1 = pols.VALUE_HIGH1[i];
            op2 = pols.VALUE_HIGH2[i];
            op3 = pols.VALUE_HIGH3[i];
            pols.inVALUE_HIGH[i] = 1;
        }

        if (rom.line[l].inRKEY)
        {
            op0 = pols.RKEY0[i];
            op1 = pols.RKEY1[i];
            op2 = pols.RKEY2[i];
            op3 = pols.RKEY3[i];
            pols.inRKEY[i] = 1;
        }

        if (rom.line[l].inSIBLING_RKEY)
        {
            op0 = pols.SIBLING_RKEY0[i];
            op1 = pols.SIBLING_RKEY1[i];
            op2 = pols.SIBLING_RKEY2[i];
            op3 = pols.SIBLING_RKEY3[i];
            pols.inSIBLING_RKEY[i] = 1;
        }

        if (rom.line[l].inSIBLING_VALUE_HASH)
        {
            op0 = pols.SIBLING_VALUE_HASH0[i];
            op1 = pols.SIBLING_VALUE_HASH1[i];
            op2 = pols.SIBLING_VALUE_HASH2[i];
            op3 = pols.SIBLING_VALUE_HASH3[i];
            pols.inSIBLING_VALUE_HASH[i] = 1;
        }

        /****************/
        /* Instructions */
        /****************/

        // JMPZ: Jump if OP==0
        if (rom.line[l].iJmpz)
        {
            if (op0==0 && op1==0 && op2==0 && op3==0)
            {
                pols.PC[nexti] = rom.line[l].address;
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

            FieldElement auxFea1[4] = {fea[0], fea[1], fea[2], fea[3]};
            FieldElement auxFea2[4] = {fea[4], fea[5], fea[6], fea[7]};

            // Call poseidon
            poseidon.hash(fea);

            // Get the calculated hash from the first 4 elements
            op0 = fea[0];
            op1 = fea[1];
            op2 = fea[2];
            op3 = fea[3];

            pols.iHash[i] = 1;

            FieldElement auxFea3[4] = {op0, op1, op2, op3};
            cout << "iHash computed left=" << fea2string(fr, auxFea1) << " right=" << fea2string(fr, auxFea2) << " hash=" << fea2string(fr, auxFea3) << endl;
            cout << endl;
        }

        if (rom.line[l].iClimbRkey)
        {
            if (pols.LEVEL0[i] == 1)
            {
                pols.RKEY0[i] = (pols.RKEY0[i]<<1) + pols.RKEY_BIT[i];
            }
            if (pols.LEVEL1[i] == 1)
            {
                pols.RKEY1[i] = (pols.RKEY1[i]<<1) + pols.RKEY_BIT[i];
            }
            if (pols.LEVEL2[i] == 1)
            {
                pols.RKEY2[i] = (pols.RKEY2[i]<<1) + pols.RKEY_BIT[i];
            }
            if (pols.LEVEL3[i] == 1)
            {
                pols.RKEY3[i] = (pols.RKEY3[i]<<1) + pols.RKEY_BIT[i];
            }
            pols.iClimbRkey[i] = 1;
        }

        if (rom.line[l].iClimbSiblingRkey)
        {
            if (pols.LEVEL0[i] == 1)
            {
                pols.SIBLING_RKEY0[i] = (pols.SIBLING_RKEY0[i]<<1) + pols.RKEY_BIT[i];
            }
            if (pols.LEVEL1[i] == 1)
            {
                pols.SIBLING_RKEY1[i] = (pols.SIBLING_RKEY1[i]<<1) + pols.RKEY_BIT[i];
            }
            if (pols.LEVEL2[i] == 1)
            {
                pols.SIBLING_RKEY2[i] = (pols.SIBLING_RKEY2[i]<<1) + pols.RKEY_BIT[i];
            }
            if (pols.LEVEL3[i] == 1)
            {
                pols.SIBLING_RKEY3[i] = (pols.SIBLING_RKEY3[i]<<1) + pols.RKEY_BIT[i];
            }
            pols.iClimbSiblingRkey[i] = 1;
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

            // Check only if key was founr
            if (action[a].getResult.isOld0)
            {
                // Check that the calculated old root is the same as the provided action root
                FieldElement oldRoot[4] = {pols.OLD_ROOT0[i], pols.OLD_ROOT1[i], pols.OLD_ROOT2[i], pols.OLD_ROOT3[i]};
                if ( !fr.eq(oldRoot, action[a].getResult.root) )
                {
                    cerr << "Error: StorageExecutor() LATCH GET found action " << a << " pols.OLD_ROOT=" << fea2string(fr,oldRoot) << " different from action.getResult.root=" << fea2string(fr,action[1].getResult.root) << endl;
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

            // Increase action
            a++;
            if (a>=action.size())
            {
                cout << "StorageExecutor() LATCH GET detected the end of the action list a=" << a << " i=" << i << endl;
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
                cerr << "Error: StorageExecutor() LATCH SET found action " << a << " pols.OLD_ROOT=" << fea2string(fr,oldRoot) << " different from action.setResult.oldRoot=" << fea2string(fr,action[1].setResult.oldRoot) << endl;
                exit(-1);
            }

            // Check that the calculated old root is the same as the provided action root
            FieldElement newRoot[4] = {pols.NEW_ROOT0[i], pols.NEW_ROOT1[i], pols.NEW_ROOT2[i], pols.NEW_ROOT3[i]};
            if ( !fr.eq(newRoot, action[a].setResult.newRoot) )
            {
                cerr << "Error: StorageExecutor() LATCH SET found action " << a << " pols.NEW_ROOT=" << fea2string(fr,newRoot) << " different from action.setResult.newRoot=" << fea2string(fr,action[1].setResult.newRoot) << endl;
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

            // Increase action
            a++;
            if (a>=action.size())
            {
                cout << "StorageExecutor() LATCH SET detected the end of the action list a=" << a << " i=" << i << endl;
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
            pols.RKEY0[nexti] = op0;
            pols.RKEY1[nexti] = op1;
            pols.RKEY2[nexti] = op2;
            pols.RKEY3[nexti] = op3;
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
            pols.RKEY_BIT[nexti] = op0;
            pols.setRKEY_BIT[i] = 1;
        }
        else
        {
            pols.RKEY_BIT[nexti] = pols.RKEY_BIT[i];
        }
        
        if (rom.line[l].setVALUE_LOW)
        {
            pols.VALUE_LOW0[nexti] = op0;
            pols.VALUE_LOW1[nexti] = op1;
            pols.VALUE_LOW2[nexti] = op2;
            pols.VALUE_LOW3[nexti] = op3;
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
            pols.VALUE_HIGH0[nexti] = op0;
            pols.VALUE_HIGH1[nexti] = op1;
            pols.VALUE_HIGH2[nexti] = op2;
            pols.VALUE_HIGH3[nexti] = op3;
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
            pols.LEVEL0[nexti] = op0;
            pols.LEVEL1[nexti] = op1;
            pols.LEVEL2[nexti] = op2;
            pols.LEVEL3[nexti] = op3;
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
            pols.OLD_ROOT0[nexti] = op0;
            pols.OLD_ROOT1[nexti] = op1;
            pols.OLD_ROOT2[nexti] = op2;
            pols.OLD_ROOT3[nexti] = op3;
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
            pols.NEW_ROOT0[nexti] = op0;
            pols.NEW_ROOT1[nexti] = op1;
            pols.NEW_ROOT2[nexti] = op2;
            pols.NEW_ROOT3[nexti] = op3;
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
            pols.HASH_LEFT0[nexti] = op0;
            pols.HASH_LEFT1[nexti] = op1;
            pols.HASH_LEFT2[nexti] = op2;
            pols.HASH_LEFT3[nexti] = op3;
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
            pols.HASH_RIGHT0[nexti] = op0;
            pols.HASH_RIGHT1[nexti] = op1;
            pols.HASH_RIGHT2[nexti] = op2;
            pols.HASH_RIGHT3[nexti] = op3;
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
            pols.SIBLING_RKEY0[nexti] = op0;
            pols.SIBLING_RKEY1[nexti] = op1;
            pols.SIBLING_RKEY2[nexti] = op2;
            pols.SIBLING_RKEY3[nexti] = op3;
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
            pols.SIBLING_VALUE_HASH0[nexti] = op0;
            pols.SIBLING_VALUE_HASH1[nexti] = op1;
            pols.SIBLING_VALUE_HASH2[nexti] = op2;
            pols.SIBLING_VALUE_HASH3[nexti] = op3;
            pols.setSIBLING_VALUE_HASH[i] = 1;
        }
        else
        {
            pols.SIBLING_VALUE_HASH0[nexti] = pols.SIBLING_VALUE_HASH0[i];
            pols.SIBLING_VALUE_HASH1[nexti] = pols.SIBLING_VALUE_HASH1[i];
            pols.SIBLING_VALUE_HASH2[nexti] = pols.SIBLING_VALUE_HASH2[i];
            pols.SIBLING_VALUE_HASH3[nexti] = pols.SIBLING_VALUE_HASH3[i];
        }
        if ((i%1000) == 0) cout << "Step " << i << " done" << endl;
    }
}

