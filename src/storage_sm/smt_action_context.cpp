#include "smt_action_context.hpp"
#include "scalar.hpp"

void SmtActionContext::init (const SmtAction &action)
{

    if (action.bIsSet)
    {
        level = action.setResult.siblings.size();
        key[0] = action.setResult.key[0];
        key[1] = action.setResult.key[1];
        key[2] = action.setResult.key[2];
        key[3] = action.setResult.key[3];
        insKey[0] = action.setResult.insKey[0];
        insKey[1] = action.setResult.insKey[1];
        insKey[2] = action.setResult.insKey[2];
        insKey[3] = action.setResult.insKey[3];
        insValue = action.setResult.insValue;
        siblings = action.setResult.siblings;

        // Initial value of siblingRKey is key
        siblingRKey[0] = action.setResult.insKey[0];
        siblingRKey[1] = action.setResult.insKey[1];
        siblingRKey[2] = action.setResult.insKey[2];
        siblingRKey[3] = action.setResult.insKey[3];
    }
    else
    {
        level = action.getResult.siblings.size();
        key[0] = action.getResult.key[0];
        key[1] = action.getResult.key[1];
        key[2] = action.getResult.key[2];
        key[3] = action.getResult.key[3];
        insKey[0] = action.getResult.insKey[0];
        insKey[1] = action.getResult.insKey[1];
        insKey[2] = action.getResult.insKey[2];
        insKey[3] = action.getResult.insKey[3];
        insValue = action.getResult.insValue;
        siblings = action.getResult.siblings;
    }

    // Initial value of rKey is key
    rKey[0] = key[0];
    rKey[1] = key[1];
    rKey[2] = key[2];
    rKey[3] = key[3];

    FiniteField fr;

#ifdef LOG_STORAGE_EXECUTOR
    cout << "SmtActionContext::init()    key=" << fea2string(fr, key) << endl;
    cout << "SmtActionContext::init() insKey=" << fea2string(fr, insKey) << endl;
    cout << "SmtActionContext::init() insValue=" << insValue.get_str(16) << endl;
    cout << "SmtActionContext::init() level=" << level << endl;
    map< uint64_t, vector<FieldElement> >::iterator it;
    for (it=siblings.begin(); it!=siblings.end(); it++)
    {
        cout << "siblings[" << it->first << "]= ";
        for (uint64_t i=0; i<it->second.size(); i++)
        {
            mpz_class auxScalar;
            auxScalar = it->second[i];
            cout << auxScalar.get_str(16) << ":";
        }
        cout << endl;
    }
#endif

    // Generate bits vectors
    bits.clear();
    siblingBits.clear();

    if (!action.bIsSet ||
        (action.bIsSet && action.setResult.mode=="update") )
    {
        for (uint64_t i=0; i<level; i++)
        {
            uint64_t keyNumber = i%4; // 0, 1, 2, 3, 0, 1, 2, 3...
            uint64_t bit = rKey[keyNumber]&1;
            bits.push_back(bit);
            rKey[keyNumber] /= 2;
            //cout << "SmtActionContext::init() rKey=" << fea2string(fr, rKey) << endl;
        }

        cout << "SmtActionContext::init()   rKey=" << fea2string(fr, rKey) << endl;
    }
    if (action.bIsSet && action.setResult.mode=="insertFound")
    {
        cout << "SmtActionContext::init() before siblingRKey=" << fea2string(fr, siblingRKey) << endl;
        for (uint64_t i=0; i<256; i++)
        {
            uint64_t keyNumber = i%4; // 0, 1, 2, 3, 0, 1, 2, 3...
            uint64_t bit = rKey[keyNumber]&1;
            uint64_t siblingBit = insKey[keyNumber]&1;
            rKey[keyNumber] /= 2;
            siblingRKey[keyNumber] /= 2;
            bits.push_back(bit);
            siblingBits.push_back(siblingBit);
            cout << "SmtActionContext::init() bit=" << siblingBit << " siblingRKey=" << fea2string(fr, siblingRKey) << endl;
            if (bit!=siblingBit) break;
        }
        cout << "SmtActionContext::init()        rKey=" << fea2string(fr, rKey) << endl;
        cout << "SmtActionContext::init() siblingRKey=" << fea2string(fr, siblingRKey) << endl;
    }

    // Set current level
    currentLevel = bits.size();
    level = currentLevel;

    // Print bits vector content
#ifdef LOG_STORAGE_EXECUTOR
    cout << "SmtActionContext::init() ";
    for (uint64_t i=0; i<bits.size(); i++)
    {
        cout << "bits[" << i << "]=" << bits[i] << " ";
    }
    cout << endl;
    cout << "SmtActionContext::init() currentLevel=" << currentLevel << endl;
#endif

}