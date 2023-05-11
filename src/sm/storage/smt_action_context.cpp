#include "smt_action_context.hpp"
#include "scalar.hpp"
#include "zklog.hpp"

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

#ifdef LOG_STORAGE_EXECUTOR
        zklog.info("SmtActionContext::init() mode=" + action.setResult.mode);
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

        // Initial value of siblingRKey is insKey
        siblingRKey[0] = action.getResult.insKey[0];
        siblingRKey[1] = action.getResult.insKey[1];
        siblingRKey[2] = action.getResult.insKey[2];
        siblingRKey[3] = action.getResult.insKey[3];
    }

#ifdef LOG_STORAGE_EXECUTOR
    zklog.info("SmtActionContext::init()    key=" + fea2string(fr, action.bIsSet ? action.setResult.key : action.getResult.key));
    zklog.info("SmtActionContext::init() insKey=" + fea2string(fr, action.bIsSet ? action.setResult.insKey : action.getResult.insKey));
    zklog.info("SmtActionContext::init() insValue=" + ( action.bIsSet ? action.setResult.insValue.get_str(16) : action.getResult.insValue.get_str(16) ));
    zklog.info("SmtActionContext::init() level=" + to_string(level));
    map< uint64_t, vector<Goldilocks::Element> >::const_iterator it;
    for ( it = ( action.bIsSet ? action.setResult.siblings.begin() : action.getResult.siblings.begin() );
          it != ( action.bIsSet ? action.setResult.siblings.end() : action.getResult.siblings.end() );
          it++ )
    {
        string s = "siblings[" + to_string(it->first) + "]= ";
        for (uint64_t i=0; i<it->second.size(); i++)
        {
            s += fr.toString(it->second[i], 16) + ":";
        }
        zklog.info(s);
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
            siblingRKey[keyNumber] = fr.fromU64(fr.toU64(siblingRKey[keyNumber]) / 2);
        }

#ifdef LOG_STORAGE_EXECUTOR
        zklog.info("SmtActionContext::init()   rKey=" + fea2string(fr, rKey));
#endif
    }

    // Generate bits vectors when there is a found sibling
    if ( ( action.bIsSet && (action.setResult.mode=="insertFound") ) ||
         ( action.bIsSet && (action.setResult.mode=="deleteFound") ) )
    {
        //zklog.info("SmtActionContext::init() before siblingRKey=" + fea2string(fr, siblingRKey));
        for (uint64_t i=0; i<256; i++)
        {
            uint64_t keyNumber = i%4; // 0, 1, 2, 3, 0, 1, 2, 3...
            uint64_t bit = fr.toU64(rKey[keyNumber]) & 1;
            uint64_t siblingBit = fr.toU64(siblingRKey[keyNumber]) & 1;
            rKey[keyNumber] = fr.fromU64(fr.toU64(rKey[keyNumber]) / 2);
            siblingRKey[keyNumber] = fr.fromU64(fr.toU64(siblingRKey[keyNumber]) / 2);
            bits.push_back(bit);
            siblingBits.push_back(siblingBit);
            //zklog.info("SmtActionContext::init() bit=" + to_string(siblingBit) + " siblingRKey=" + fea2string(fr, siblingRKey));
            if (bit!=siblingBit) break;
        }
#ifdef LOG_STORAGE_EXECUTOR
        zklog.info("SmtActionContext::init()        rKey=" + fea2string(fr, rKey));
        zklog.info("SmtActionContext::init() siblingRKey=" + fea2string(fr, siblingRKey));
#endif

        // Update level
        level = bits.size();
    }

    // Init current level countdown counter
    currentLevel = level;

    // Print bits vector content
#ifdef LOG_STORAGE_EXECUTOR
    {
        string s = "SmtActionContext::init() ";
        for (uint64_t i=0; i<bits.size(); i++)
        {
            s += "bits[" + to_string(i) + "]=" + to_string(bits[i]) + " ";
        }
        zklog.info(s);
    }
    zklog.info("SmtActionContext::init() currentLevel=" + to_string(currentLevel));
#endif

}