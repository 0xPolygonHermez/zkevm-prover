#include "stark.hpp"
#include "timer.hpp"
#include "utils.hpp"

//#include "starkPols.hpp"
//#include "starkPols2ns.hpp"

#include "ntt_goldilocks.hpp"

Stark::Stark (const Config &config) : config(config), starkInfo(config)
{
    // Allocate an area of memory, mapped to file, to read all the constant polynomials,
    // and create them using the allocated address
    TimerStart(LOAD_CONST_POLS_TO_MEMORY);
    pConstPolsAddress = NULL;
    if (config.generateProof())
    {
        if (config.constPolsFile.size() == 0)
        {
            cerr << "Error: Stark::Stark() received an empty cofnig.constPolsFile" << endl;
            exit(-1);
        }
        pConstPolsAddress = mapFile(config.constPolsFile, ConstantPols::pilSize(), false);
        cout << "Stark::Stark() successfully mapped " << ConstantPols::pilSize() << " bytes from constant file " << config.constPolsFile << endl;
    }
    pConstPols = new ConstantPols(pConstPolsAddress, ConstantPols::pilDegree());
    TimerStopAndLog(LOAD_CONST_POLS_TO_MEMORY);

    // Map constants tree file to memory
    /*TimerStart(LOAD_CONST_TREE_TO_MEMORY);
    pConstTreeAddress = NULL;
    if (config.generateProof())
    {
        if (config.constantsTreeFile.size() == 0)
        {
            cerr << "Error: Stark::Stark() received an empty config.constantsTreeFile" << endl;
            exit(-1);
        }
        pConstTreeAddress = mapFile(config.constantsTreeFile, starkInfo.getConstTreeSizeInBytes(), false);
        cout << "Stark::Stark() successfully mapped " << starkInfo.getConstTreeSizeInBytes() << " bytes from constant tree file " << config.constantsTreeFile << endl;
    }
    TimerStopAndLog(LOAD_CONST_TREE_TO_MEMORY);*/
}

Stark::~Stark()
{
    if (config.generateProof())
    {
        delete pConstPols;
        unmapFile(pConstPolsAddress, ConstantPols::pilSize());
    }
}

void Stark::genProof(void *pAddress, CommitPols &cmPols, Proof &proof)
{
    // Temporary struct
    /*
    starkStruct structStark{10, 11, 8};
    structStark.extendBits = structStark.nBitsExt - structStark.nBits;
    structStark.N = (1 << structStark.nBits);
    structStark.N_Extended = (1 << structStark.nBitsExt);
    */

    uint64_t nBitsExt = 1;
    uint64_t N_Extended = (cmPols.degree() << nBitsExt);

    // StarkPols starkPols(cmPols);
    // StarkPols2ns starkPols2ns(N_Extended);

    NTT_Goldilocks ntt(cmPols.degree());

    // ntt.extendPol((Goldilocks::Element *)starkPols2ns.cmPols->address(), (Goldilocks::Element *)starkPols.cmPols->address(), starkPols2ns.cmPols->degree(), starkPols.cmPols->degree(), CommitPols::numPols());

    // HARDCODE PROOFs
    proof.proofA.push_back("13661670604050723159190639550237390237901487387303122609079617855313706601738");
    proof.proofA.push_back("318870292909531730706266902424471322193388970015138106363857068613648741679");
    proof.proofA.push_back("1");

    ProofX proofX;
    proofX.proof.push_back("697129936138216869261087581911668981951894602632341950972818743762373194907");
    proofX.proof.push_back("8382255061406857865565510718293473646307698289010939169090474571110768554297");
    proof.proofB.push_back(proofX);
    proofX.proof.clear();
    proofX.proof.push_back("15430920731683674465693779067364347784717314152940718599921771157730150217435");
    proofX.proof.push_back("9973632244944366583831174453935477607483467152902406810554814671794600888188");
    proof.proofB.push_back(proofX);
    proofX.proof.clear();
    proofX.proof.push_back("1");
    proofX.proof.push_back("0");
    proof.proofB.push_back(proofX);

    proof.proofC.push_back("19319469652444706345294120534164146052521965213898291140974711293816652378032");
    proof.proofC.push_back("20960565072144725955004735885836324119094967998861346319897532045008317265851");
    proof.proofC.push_back("1");

    proof.publicInputsExtended.inputHash = "0x1afd6eaf13538380d99a245c2acc4a25481b54556ae080cf07d1facc0638cd8e";
    proof.publicInputsExtended.publicInputs.oldStateRoot = "0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9";
    proof.publicInputsExtended.publicInputs.oldLocalExitRoot = "0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9";
    proof.publicInputsExtended.publicInputs.newStateRoot = "0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9";
    proof.publicInputsExtended.publicInputs.newLocalExitRoot = "0x17c04c3760510b48c6012742c540a81aba4bca2f78b9d14bfd2f123e2e53ea3e";
    proof.publicInputsExtended.publicInputs.sequencerAddr = "0x617b3a3528F9cDd6630fd3301B9c8911F7Bf063D";
    proof.publicInputsExtended.publicInputs.batchHashData = "0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9";
    proof.publicInputsExtended.publicInputs.batchNum = 1;
}