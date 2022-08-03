#include "prover_utils.hpp"
#include "utils.hpp"
#include "scalar.hpp"


void inputProver2Input (Goldilocks &fr, const zkprover::v1::InputProver &inputProver, Input &input)
{
    // Parse public inputs
    zkprover::v1::PublicInputs publicInputs = inputProver.public_inputs();
    input.publicInputs.oldStateRoot = publicInputs.old_state_root();
    input.publicInputs.oldLocalExitRoot = publicInputs.old_local_exit_root();
    input.publicInputs.newStateRoot = publicInputs.new_state_root();
    input.publicInputs.newLocalExitRoot = publicInputs.new_local_exit_root();
    input.publicInputs.sequencerAddr = publicInputs.sequencer_addr();
    input.publicInputs.batchHashData = publicInputs.batch_hash_data();
    input.publicInputs.batchNum = publicInputs.batch_num();
    input.publicInputs.timestamp = publicInputs.eth_timestamp();

#ifdef LOG_RPC_INPUT
    cout << "input.publicInputs.oldStateRoot: " << input.publicInputs.oldStateRoot << endl;
    cout << "input.publicInputs.oldLocalExitRoot: " << input.publicInputs.oldLocalExitRoot << endl;
    cout << "input.publicInputs.newStateRoot: " << input.publicInputs.newStateRoot << endl;
    cout << "input.publicInputs.newLocalExitRoot: " << input.publicInputs.newLocalExitRoot << endl;
    cout << "input.publicInputs.sequencerAddr: " << input.publicInputs.sequencerAddr << endl;
    cout << "input.publicInputs.batchHashData: " << input.publicInputs.batchHashData << endl;
    cout << "input.publicInputs.batchNum: " << to_string(input.publicInputs.batchNum) << endl;
    cout << "input.publicInputs.blockNum: " << to_string(input.publicInputs.blockNum) << endl;
    cout << "input.publicInputs.timestamp: " << to_string(input.publicInputs.timestamp) << endl;
#endif

    // Parse global exit root
    input.globalExitRoot = inputProver.global_exit_root();
#ifdef LOG_RPC_INPUT
    cout << "input.globalExitRoot: " << input.globalExitRoot << endl;
#endif

    // Parse batch L2 data
    input.batchL2Data = Add0xIfMissing(inputProver.batch_l2_data());

#ifdef LOG_RPC_INPUT
    cout << "input.batchL2Data: " << input.batchL2Data << endl;
#endif

    // Parse aggregator address
    input.aggregatorAddress = Add0xIfMissing(inputProver.address_aggregator());

#ifdef LOG_RPC_INPUT
    cout << "input.aggregatorAddress: " << input.aggregatorAddress << endl;
#endif

    // Preprocess the transactions
    input.preprocessTxs();

    // Parse keys map
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> > db;
    db = inputProver.db();
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> >::iterator it;
    for (it=db.begin(); it!=db.end(); it++)
    {
        vector<Goldilocks::Element> dbValue;
        string concatenatedValues = it->second;
        if (concatenatedValues.size()%64!=0)
        {
            cerr << "Error: inputProver2Input() found invalid db value size: " << concatenatedValues.size() << endl;
            exit(-1); // TODO: return an error
        }
        for (uint64_t i=0; i<concatenatedValues.size(); i+=64)
        {
            Goldilocks::Element fe;
            string2fe(fr, concatenatedValues.substr(i, 64), fe);
            dbValue.push_back(fe);
        }
        input.db[it->first] = dbValue;
#ifdef LOG_RPC_INPUT
        cout << "input.db[" << it->first << "][0]: " << fr.toString(input.db[it->first][0], 16) << endl;
#endif
    }

    // Parse contracts data
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> > contractsBytecode;
    contractsBytecode = inputProver.contracts_bytecode();
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> >::iterator itp;
    for (itp=contractsBytecode.begin(); itp!=contractsBytecode.end(); itp++)
    {
        vector<uint8_t> dbValue;
        string contractValue = string2ba(itp->second);
        for (uint64_t i=0; i<contractValue.size(); i++)
        {
            dbValue.push_back(contractValue.at(i));
        }
        input.contractsBytecode[itp->first] = dbValue;
#ifdef LOG_RPC_INPUT
        cout << "input.contractsBytecode[" << itp->first << "]: " << itp->second << endl;
#endif
    }    
}

void input2InputProver (Goldilocks &fr, const Input &input, zkprover::v1::InputProver &inputProver)
{
    // Parse public inputs
    zkprover::v1::PublicInputs * pPublicInputs = new zkprover::v1::PublicInputs();
    pPublicInputs->set_old_state_root(input.publicInputs.oldStateRoot);
    pPublicInputs->set_old_local_exit_root(input.publicInputs.oldStateRoot);
    pPublicInputs->set_new_state_root(input.publicInputs.newStateRoot);
    pPublicInputs->set_new_local_exit_root(input.publicInputs.newLocalExitRoot);
    pPublicInputs->set_sequencer_addr(input.publicInputs.sequencerAddr);
    pPublicInputs->set_batch_hash_data(input.publicInputs.batchHashData);
    pPublicInputs->set_batch_num(input.publicInputs.batchNum);
    pPublicInputs->set_eth_timestamp(input.publicInputs.timestamp);
    inputProver.set_allocated_public_inputs(pPublicInputs);

    // Parse global exit root
    inputProver.set_global_exit_root(input.globalExitRoot);

    // Parse batch L2 data
    inputProver.set_batch_l2_data(input.batchL2Data);

    // Parse keys map
    map< string, vector<Goldilocks::Element>>::const_iterator it;
    for (it=input.db.begin(); it!=input.db.end(); it++)
    {
        string key = NormalizeToNFormat(it->first, 64);
        string value;
        vector<Goldilocks::Element> dbValue = it->second;
        for (uint64_t i=0; i<dbValue.size(); i++)
        {
            value += NormalizeToNFormat(fr.toString(dbValue[i], 16), 64);
        }
        (*inputProver.mutable_db())[key] = value;
    }

    // Parse contracts data
    map< string, vector<uint8_t>>::const_iterator itc;
    for (itc=input.contractsBytecode.begin(); itc!=input.contractsBytecode.end(); itc++)
    {
        string key = NormalizeToNFormat(itc->first, 64);
        string value;
        vector<uint8_t> contractValue = itc->second;
        for (uint64_t i=0; i<contractValue.size(); i++)
        {
            value += byte2string(contractValue[i]);
        }
        (*inputProver.mutable_contracts_bytecode())[key] = value;
    }    
}

void proof2ProofProver (Goldilocks &fr, const Proof &proof, zkprover::v1::Proof &proofProver)
{
    // Set proofA
    for (uint64_t i=0; i<proof.proofA.size(); i++)
    {
        proofProver.add_proof_a(proof.proofA[i]);
#ifdef LOG_RPC_OUTPUT
        cout << "RCP output proofA[" << i << "] = " << proof.proofA[i] << endl;
#endif
    }

    // Set proofB
    for (uint64_t i=0; i<proof.proofB.size(); i++)
    {
        zkprover::v1::ProofB *pProofB = proofProver.add_proof_b();
        for (uint64_t j=0; j<proof.proofB[i].proof.size(); j++)
        {
            pProofB->add_proofs(proof.proofB[i].proof[j]);
#ifdef LOG_RPC_OUTPUT
        cout << "RCP output proofB[" << i << "].proof[" << j << "] = " << proof.proofB[i].proof[j] << endl;
#endif            
        }
    }

    // Set proofC
    for (uint64_t i=0; i<proof.proofC.size(); i++)
    {
        proofProver.add_proof_c(proof.proofC[i]);
#ifdef LOG_RPC_OUTPUT
        cout << "RCP output proofC[" << i << "] = " << proof.proofC[i] << endl;
#endif
    }
}