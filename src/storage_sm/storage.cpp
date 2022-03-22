#include "storage.hpp"
#include "utils.hpp"
#include "storage_rom.hpp"
#include "storage_pols.hpp"

void StorageExecutor (RawFr &fr, const Config &config, SmtActionList &smtActionList)
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
    for (uint64_t i=0; i<polSize; i++)
    {
        uint64_t op0, op1, op2, op3;
        op0=0;
        op1=0;
        op2=0;
        op3=0;

        l = pols.PC[i];

        uint64_t nexti = (i+1)%polSize;
 
        // Selectors

        if (rom.line[l].inFREE)
        {
            if (rom.line[l].op == "functionCall")
            {
                /* Possible values of mode:
                    - update
                    - insertFound
                    - insertNotFound
                    - deleteFound
                    - deleteNotFound
                    - deleteLast
                    - zeroToZero
                */
                if (rom.line[l].funcName=="GetIsUpdate")
                {
                    if (smtActionList.action[a].bIsSet &&
                        smtActionList.action[a].setResult.mode == "update")
                    {
                        op0 = 0;
                    }
                    else
                    {
                        op0 = 1;
                    }
                }
                if (rom.line[l].funcName=="GetIsSetReplacingZero")
                {
                    if (smtActionList.action[a].bIsSet &&
                        smtActionList.action[a].setResult.mode == "insertNotFound")
                    {
                        op0 = 0;
                    }
                    else
                    {
                        op0 = 1;
                    }
                }
                if (rom.line[l].funcName=="GetIsSetWithSibling")
                {
                    if (smtActionList.action[a].bIsSet &&
                        smtActionList.action[a].setResult.mode == "insertFound")
                    {
                        op0 = 0;
                    }
                    else
                    {
                        op0 = 1;
                    }
                }
                if (rom.line[l].funcName=="GetIsGet")
                {
                    if (smtActionList.action[a].bIsSet)
                    {
                        op0 = 1;
                    }
                }
                if (rom.line[l].funcName=="GetRKey")
                {
                    if (!smtActionList.action[a].bIsSet)
                    {
                        /*RawFr::Element root;
                        RawFr::Element key;
                        map< uint64_t, vector<RawFr::Element> > siblings;
                        RawFr::Element insKey;
                        mpz_class insValue;
                        bool isOld0;
                        mpz_class value;*/
                        /*cout << "GetRKey()" << endl;
                        cout << "value=" << smtActionList.action[a].getResult.value.get_str(16) << endl;
                        cout << "insValue=" << smtActionList.action[a].getResult.insValue.get_str(16) << endl;
                        cout << "isOld0=" << smtActionList.action[a].getResult.isOld0 << endl;
                        cout << "root=" << fr.toString(smtActionList.action[a].getResult.root, 16) << endl;
                        cout << "key=" << fr.toString(smtActionList.action[a].getResult.key, 16) << endl;
                        cout << "insKey=" << fr.toString(smtActionList.action[a].getResult.insKey, 16) << endl;
                        for ( map< uint64_t, vector<RawFr::Element> >::iterator it=smtActionList.action[a].getResult.siblings.begin(); it != smtActionList.action[a].getResult.siblings.end(); it++)
                        {
                            cout << "sibling first=" << it->first << " second=" << endl;
                            for (uint64_t aux=0; aux<it->second.size(); aux++)
                            {
                                cout << "  " << fr.toString(it->second[aux], 16) << endl;
                            }
                        }*/

                    }
                }
                if (rom.line[l].funcName=="GetValueLow")
                {

                }
                if (rom.line[l].funcName=="GetValueHigh")
                {

                }
                if (rom.line[l].funcName=="GetLevelBit")
                {

                }
                if (rom.line[l].funcName=="GetTopTree");
                if (rom.line[l].funcName=="GetNextKeyBit");
                if (rom.line[l].funcName=="GetSiblingHash");
                if (rom.line[l].funcName=="GetOldValueLow");
                if (rom.line[l].funcName=="GetOldValueHigh");
                if (rom.line[l].funcName=="GetTopOfBranch");
                if (rom.line[l].funcName=="isEndPolinomial");
                
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

        // Instructions

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
        else if (rom.line[l].iJmp)
        {
            pols.PC[nexti] = rom.line[l].address;
            pols.iJmp[i] = 1;
        }
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
            // TODO: hash
            pols.iHash[i] = 1;
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
            // TODO: latch get
            pols.iLatchGet[i] = 1;
        }

        if (rom.line[l].iLatchSet)
        {
            // TODO: latch set
            pols.iLatchSet[i] = 1;
        }

        // Setters

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
    }
}

