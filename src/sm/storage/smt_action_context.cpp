#include "smt_action_context.hpp"
#include "scalar.hpp"

void SmtActionContext::init (Goldilocks &fr, const SmtAction &action)
{

    if (action.bIsSet)
    {
        // Deepest, initial level
        level = action.setResult.siblings.size();

        // Initial value of rKey is key
        rKey[0] = action.setResult.key[0];
        rKey[1] = action.setResult.key[1];
        rKey[2] = action.setResult.key[2];
        rKey[3] = action.setResult.key[3];

        // Initial value of siblingRKey is insKey
        siblingRKey[0] = action.setResult.insKey[0];
        siblingRKey[1] = action.setResult.insKey[1];
        siblingRKey[2] = action.setResult.insKey[2];
        siblingRKey[3] = action.setResult.insKey[3];
        
        // Initial value of insRKey is zero
        insRKey[0] = fr.zero();
        insRKey[1] = fr.zero();
        insRKey[2] = fr.zero();
        insRKey[3] = fr.zero();

#ifdef LOG_STORAGE_EXECUTOR
        cout << "SmtActionContext::init() mode=" << action.setResult.mode << endl;
#endif

    }
    else
    {
        // Deepest, initial level
        level = action.getResult.siblings.size();

        // Initial value of rKey is key
        rKey[0] = action.getResult.key[0];
        rKey[1] = action.getResult.key[1];
        rKey[2] = action.getResult.key[2];
        rKey[3] = action.getResult.key[3];

        // Reset siblingRKey from previous actions
        siblingRKey[0] = fr.zero();
        siblingRKey[1] = fr.zero();
        siblingRKey[2] = fr.zero();
        siblingRKey[3] = fr.zero();

        // Initial value of insRKey is insKey
        insRKey[0] = action.getResult.insKey[0];
        insRKey[1] = action.getResult.insKey[1];
        insRKey[2] = action.getResult.insKey[2];
        insRKey[3] = action.getResult.insKey[3];
    }

#ifdef LOG_STORAGE_EXECUTOR
    cout << "SmtActionContext::init()    key=" << fea2string(fr, action.bIsSet ? action.setResult.key : action.getResult.key) << endl;
    cout << "SmtActionContext::init() insKey=" << fea2string(fr, action.bIsSet ? action.setResult.insKey : action.getResult.insKey) << endl;
    cout << "SmtActionContext::init() insValue=" << ( action.bIsSet ? action.setResult.insValue.get_str(16) : action.getResult.insValue.get_str(16) ) << endl;
    cout << "SmtActionContext::init() level=" << level << endl;
    map< uint64_t, vector<FieldElement> >::const_iterator it;
    for ( it = ( action.bIsSet ? action.setResult.siblings.begin() : action.getResult.siblings.begin() );
          it != ( action.bIsSet ? action.setResult.siblings.end() : action.getResult.siblings.end() );
          it++ )
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

    // Reset bits vectors
    bits.clear();
    siblingBits.clear();

    // Generate bits vectors when there is no found sibling
    if (!action.bIsSet ||
        ( action.bIsSet && (action.setResult.mode=="update") ) ||
        ( action.bIsSet && (action.setResult.mode=="deleteNotFound") ) ||
        ( action.bIsSet && (action.setResult.mode=="zeroToZero") ) ||
        ( action.bIsSet && (action.setResult.mode=="insertNotFound") ) )
    {
        for (uint64_t i=0; i<level; i++)
        {
            uint64_t keyNumber = i%4; // 0, 1, 2, 3, 0, 1, 2, 3...
            uint64_t bit = fr.toU64(rKey[keyNumber]) & 1;
            uint64_t siblingBit = fr.toU64(siblingRKey[keyNumber]) & 1;
            bits.push_back(bit);
            siblingBits.push_back(siblingBit);
            rKey[keyNumber] = fr.fromU64(fr.toU64(rKey[keyNumber]) / 2);
            insRKey[keyNumber] = fr.fromU64(fr.toU64(insRKey[keyNumber]) / 2);
            siblingRKey[keyNumber] = fr.fromU64(fr.toU64(siblingRKey[keyNumber]) / 2);
        }

#ifdef LOG_STORAGE_EXECUTOR
        cout << "SmtActionContext::init()   rKey=" << fea2string(fr, rKey) << endl;
#endif
    }

    // Generate bits vectors when there is a found sibling
    if ( ( action.bIsSet && (action.setResult.mode=="insertFound") ) ||
         ( action.bIsSet && (action.setResult.mode=="deleteFound") ) )
    {
        //cout << "SmtActionContext::init() before siblingRKey=" << fea2string(fr, siblingRKey) << endl;
        for (uint64_t i=0; i<256; i++)
        {
            uint64_t keyNumber = i%4; // 0, 1, 2, 3, 0, 1, 2, 3...
            uint64_t bit = fr.toU64(rKey[keyNumber]) & 1;
            uint64_t siblingBit = fr.toU64(siblingRKey[keyNumber]) & 1;
            rKey[keyNumber] = fr.fromU64(fr.toU64(rKey[keyNumber]) / 2);
            siblingRKey[keyNumber] = fr.fromU64(fr.toU64(siblingRKey[keyNumber]) / 2);
            bits.push_back(bit);
            siblingBits.push_back(siblingBit);
            //cout << "SmtActionContext::init() bit=" << siblingBit << " siblingRKey=" << fea2string(fr, siblingRKey) << endl;
            if (bit!=siblingBit) break;
        }
#ifdef LOG_STORAGE_EXECUTOR
        cout << "SmtActionContext::init()        rKey=" << fea2string(fr, rKey) << endl;
        cout << "SmtActionContext::init() siblingRKey=" << fea2string(fr, siblingRKey) << endl;
#endif

        // Update level
        level = bits.size();
    }

    // Init current level countdown counter
    currentLevel = level;

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