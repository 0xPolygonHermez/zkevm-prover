#include "starks.hpp"

template <typename ElementType>
void *genRecursiveProof(SetupCtx& setupCtx, Goldilocks::Element *pAddress, Goldilocks::Element *publicInputs, std::string proofFile) {
    TimerStart(STARK_PROOF);

    FRIProof<Goldilocks::Element> proof(setupCtx.starkInfo);

    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;
    
    using MerkleTreeType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, MerkleTreeGL, MerkleTreeBN128>;

    Starks<ElementType> starks(setupCtx);

    ExpressionsAvx expressionsAvx(setupCtx);

    uint64_t nFieldElements = setupCtx.starkInfo.starkStruct.verificationHashType == std::string("BN128") ? 1 : HASH_SIZE;

    TranscriptType transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);

    Goldilocks::Element* evals = new Goldilocks::Element[setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* challenges = new Goldilocks::Element[setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* subproofValues = new Goldilocks::Element[setupCtx.starkInfo.nSubProofValues * FIELD_EXTENSION];
    
    vector<bool> subProofValuesCalculated(setupCtx.starkInfo.nSubProofValues, false);
    vector<bool> commitsCalculated(setupCtx.starkInfo.cmPolsMap.size(), false);

    StepsParams params = {
        pols : pAddress,
        publicInputs : publicInputs,
        challenges : challenges,
        subproofValues : subproofValues,
        evals : evals,
        xDivXSub : nullptr,
    };

    for (uint64_t i = 0; i < setupCtx.starkInfo.mapSectionsN["cm1"]; ++i)
    {
        commitsCalculated[i] = true;
    }

    MerkleTreeType **treesGL = new MerkleTreeType*[setupCtx.starkInfo.nStages + 2];
    MerkleTreeType **treesFRI = new MerkleTreeType*[setupCtx.starkInfo.starkStruct.steps.size() - 1];

    treesGL[setupCtx.starkInfo.nStages + 1] = new MerkleTreeType(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, (Goldilocks::Element *)setupCtx.constPols.pConstTreeAddress);
    for (uint64_t i = 0; i < setupCtx.starkInfo.nStages + 1; i++)
    {
        std::string section = "cm" + to_string(i + 1);
        uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];
        treesGL[i] = new MerkleTreeType(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, 1 << setupCtx.starkInfo.starkStruct.nBitsExt, nCols, NULL, false);
    }
        
    for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) {
        uint64_t nGroups = 1 << setupCtx.starkInfo.starkStruct.steps[step + 1].nBits;
        uint64_t groupSize = (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) / nGroups;

        treesFRI[step] = new MerkleTreeType(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, nGroups, groupSize * FIELD_EXTENSION, NULL);
    }

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------

    TimerStart(STARK_STEP_0);
    ElementType verkey[nFieldElements];
    treesGL[setupCtx.starkInfo.nStages + 1]->getRoot(verkey);
    starks.addTranscript(transcript, &verkey[0], nFieldElements);

    if(!setupCtx.starkInfo.starkStruct.hashCommits) {
        starks.addTranscriptGL(transcript, &publicInputs[0], setupCtx.starkInfo.nPublics);
    } else {
        ElementType hash[nFieldElements];
        starks.calculateHash(hash, &publicInputs[0], setupCtx.starkInfo.nPublics);
        starks.addTranscript(transcript, hash, nFieldElements);
    }

    TimerStopAndLog(STARK_STEP_0);

    // STAGE 1
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 1) {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }
    starks.commitStage(1, pAddress, proof);
    starks.addTranscript(transcript, &proof.proof.roots[0][0], nFieldElements);
    
    // STAGE 2
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 2) {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    Goldilocks::Element *num = new Goldilocks::Element[N*FIELD_EXTENSION];
    Goldilocks::Element *den = new Goldilocks::Element[N*FIELD_EXTENSION];
    Goldilocks::Element *gprod = new Goldilocks::Element[N*FIELD_EXTENSION];

    Hint gprod_hint = setupCtx.expressionsBin.hints[0];
    auto denField = std::find_if(gprod_hint.fields.begin(), gprod_hint.fields.end(), [](const HintField& hintField) {
        return hintField.name == "denominator";
    });
    auto numField = std::find_if(gprod_hint.fields.begin(), gprod_hint.fields.end(), [](const HintField& hintField) {
        return hintField.name == "numerator";
    });
    auto gprodField = std::find_if(gprod_hint.fields.begin(), gprod_hint.fields.end(), [](const HintField& hintField) {
        return hintField.name == "reference";
    });
    
    expressionsAvx.calculateExpression(params, den, denField->id, true);
    expressionsAvx.calculateExpression(params, num, numField->id);


    Goldilocks3::copy((Goldilocks3::Element *)&gprod[0], &Goldilocks3::one());
    for(uint64_t i = 1; i < N; ++i) {
        Goldilocks::Element res[3];
        Goldilocks3::mul((Goldilocks3::Element *)res, (Goldilocks3::Element *)&num[(i - 1) * FIELD_EXTENSION], (Goldilocks3::Element *)&den[(i - 1) * FIELD_EXTENSION]);
        Goldilocks3::mul((Goldilocks3::Element *)&gprod[i * FIELD_EXTENSION], (Goldilocks3::Element *)&gprod[(i - 1) * FIELD_EXTENSION], (Goldilocks3::Element *)res);
    }

    Polinomial gprodTransposedPol;
    setupCtx.starkInfo.getPolynomial(gprodTransposedPol, pAddress, true, gprodField->id, false);
#pragma omp parallel for
    for(uint64_t j = 0; j < N; ++j) {
        std::memcpy(gprodTransposedPol[j], &gprod[j*FIELD_EXTENSION], FIELD_EXTENSION * sizeof(Goldilocks::Element));
    }
    
    delete num;
    delete den;
    delete gprod;

    commitsCalculated[gprodField->id] = true;

    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].stage == 2 && !setupCtx.starkInfo.cmPolsMap[i].imPol && !commitsCalculated[i]) {
            zklog.info("Witness polynomial " + setupCtx.starkInfo.cmPolsMap[i].name + " is not calculated");
            exitProcess();
            exit(-1);
        }
    }
    starks.calculateImPolsExpressions(2, pAddress, publicInputs, challenges, subproofValues, evals);
    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].imPol && setupCtx.starkInfo.cmPolsMap[i].stage == 2) {
            commitsCalculated[i] = true;
        }
    }

    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].stage == 2 && !commitsCalculated[i]) {
            zklog.info("Witness polynomial " + setupCtx.starkInfo.cmPolsMap[i].name + " is not calculated");
            exitProcess();
            exit(-1);
        }
    }

    starks.commitStage(2, pAddress, proof);
    starks.addTranscript(transcript, &proof.proof.roots[1][0], nFieldElements);

    TimerStart(STARK_STEP_Q);

    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1) {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }
    
    starks.calculateQuotientPolynomial(pAddress, publicInputs, challenges, subproofValues, evals);

    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].stage == setupCtx.starkInfo.nStages + 1) {
            commitsCalculated[i] = true;
        }
    }
    starks.commitStage(setupCtx.starkInfo.nStages + 1, pAddress, proof);
    starks.addTranscript(transcript, &proof.proof.roots[setupCtx.starkInfo.nStages][0], nFieldElements);
    TimerStopAndLog(STARK_STEP_Q);

    TimerStart(STARK_STEP_EVALS);

    uint64_t xiChallengeIndex = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2) {
            if(setupCtx.starkInfo.challengesMap[i].stageId == 0) xiChallengeIndex = i;
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    Goldilocks::Element *xiChallenge = &challenges[xiChallengeIndex * FIELD_EXTENSION];
    Goldilocks::Element* LEv = &pAddress[setupCtx.starkInfo.mapOffsets[make_pair("LEv", true)]];

    starks.computeLEv(xiChallenge, LEv);
    starks.computeEvals(pAddress,LEv, evals, proof);

    if(!setupCtx.starkInfo.starkStruct.hashCommits) {
        starks.addTranscriptGL(transcript, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
    } else {
        ElementType hash[nFieldElements];
        starks.calculateHash(hash, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
        starks.addTranscript(transcript, hash, nFieldElements);
    }

    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3) {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    TimerStopAndLog(STARK_STEP_EVALS);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);

    Goldilocks::Element *xDivXSub = &pAddress[setupCtx.starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)]];
    starks.calculateXDivXSub(xiChallenge, xDivXSub);
    starks.calculateFRIPolynomial(pAddress, publicInputs, challenges, subproofValues, evals, xDivXSub);

    Goldilocks::Element challenge[FIELD_EXTENSION];
    Goldilocks::Element *friPol = &pAddress[setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]];
    
    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size(); step++)
    {
        starks.computeFRIFolding(step, proof, pAddress, challenge);
        if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
        {
            starks.addTranscript(transcript, &proof.proof.fri.trees[step + 1].root[0], nFieldElements);
        }
        else
        {
            if(!setupCtx.starkInfo.starkStruct.hashCommits) {
                starks.addTranscriptGL(transcript, friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
            } else {
                ElementType hash[nFieldElements];
                starks.calculateHash(hash, friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
                starks.addTranscript(transcript, hash, nFieldElements);
            } 
            
        }
        starks.getChallenge(transcript, *challenge);
    }

    uint64_t friQueries[setupCtx.starkInfo.starkStruct.nQueries];

    TranscriptType transcriptPermutation(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    starks.addTranscriptGL(transcriptPermutation, challenge, FIELD_EXTENSION);
    transcriptPermutation.getPermutations(friQueries, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps[0].nBits);

    starks.computeFRIQueries(proof, friQueries);

    TimerStopAndLog(STARK_STEP_FRI);

    delete challenges;
    delete evals;
    delete subproofValues;

    for (uint i = 0; i < setupCtx.starkInfo.nStages + 2; i++)
    {
        delete treesGL[i];
    }
    delete[] treesGL;

    for (uint64_t i = 0; i < setupCtx.starkInfo.starkStruct.steps.size() - 1; i++)
    {
        delete treesFRI[i];
    }
    delete[] treesFRI;
        
    TimerStopAndLog(STARK_PROOF);

    nlohmann::ordered_json jProof = proof.proof.proof2json();
    nlohmann::ordered_json zkin = proof2zkinStark(jProof, setupCtx.starkInfo);

    if(!proofFile.empty()) {
        json2file(jProof, proofFile);
    }

    return (void *) new nlohmann::ordered_json(zkin);
}
