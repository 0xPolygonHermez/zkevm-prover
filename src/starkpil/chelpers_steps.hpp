#ifndef CHELPERS_STEPS_HPP
#define CHELPERS_STEPS_HPP
#include "chelpers.hpp"

#include "steps.hpp"

static int node = 0;
static int dcol = 0;

#define  _NPOS_ 100000

static int tmp1_node[_NPOS_]; //more than enough space
static int tmp3_node[_NPOS_];
static int dest_column[_NPOS_];

static int read_trace[2*_NPOS_];
static int read_challenges[_NPOS_];
static int read_publics[_NPOS_];
static int read_numbers[_NPOS_];
static int read_evals[_NPOS_];


enum VAR_TYPE {
    challenge,
    publicInput,
    number,
    eval
};

enum DEST_TYPE {
    tmp1,
    tmp3,
    dest1,
    dest3
};

enum TMP_TYPE {
    dim1,
    dim3
};
class CHelpersSteps {
public:
    uint64_t nrowsPack = 4;
    uint64_t nCols;
    vector<uint64_t> nColsStages;
    vector<uint64_t> nColsStagesAcc;
    vector<uint64_t> offsetsStages;

    inline int printOP(int op, int node1, int node2, string mode){

        switch (op)
        {
            case 0:
                std::cout<<"   "<<node<<": ADD"+mode+" "<<node1<<" "<<node2<<std::endl;
                break;
            case 1:
                std::cout<<"   "<<node<<": SUB"+mode+" "<<node1<<" "<<node2<<std::endl;
                break;
            case 2:
                std::cout<<"   "<<node<<": MULT"+mode+" "<<node1<<" "<<node2<<std::endl;
                break;
            case 3:
                std::cout<<"   "<<node<<": SUB"+mode+" "<<node2<<" "<<node1<<std::endl;
                break;
            default:
                assert(0);
                break;
        }
        ++node;
        return node-1;

    }

    inline int printTraceValue(int section, int col, int dim){

        assert(dim==1 || dim==3);
        int accum = nColsStagesAcc[section] + col;
        int node_out = -1;
        if(read_trace[accum]==-1){
            if(section < 5){
                std::cout<<"   "<<node<<": VALUE TRACE"<<dim<<" "<<section<<" "<<col<<" 0"<<std::endl;
            } else if(section <10){
                std::cout<<"   "<<node<<": VALUE TRACE"<<dim<<" "<<section-5<<" "<<col<<" 1"<<std::endl;
            } else {
                //TODO, FALTA LA F, LA Q, LA X i la Zi
            }
        } else {
            node_out = read_trace[accum];
        }
        if(node_out==-1){
            node_out = node;
            read_trace[accum] = node;
            ++node;
        }
        return node_out;
    }

    inline int printVariableValue(VAR_TYPE type, int row, int dim){
        
        int section;
        int node_out = -1;
        switch(type){
            case VAR_TYPE::challenge:{
                assert(dim==3);
                if(read_challenges[row] != -1){
                    node_out = read_challenges[row];
                }else{
                    read_challenges[row] = node;
                }
                section = 0;
                break;
            }
            case VAR_TYPE::publicInput:{
                assert(dim==1);
                if(read_publics[row] != -1){
                    node_out = read_publics[row];
                }else{
                    read_publics[row] = node;
                }
                section = 1;
                break;
            }
            case VAR_TYPE::number:{
                assert(dim==1);
                if(read_numbers[row] != -1){
                    node_out = read_numbers[row];
                }else{
                    read_numbers[row] = node;
                }
                section = 2;
                break;
            }
            case VAR_TYPE::eval:{
                assert(dim==3);
                if(read_evals[row] != -1){
                    node_out = read_evals[row];
                }else{
                    read_evals[row] = node;
                }
                section = 3;
                break;
            }
            default:{
                assert(0);
                break;
            }
        }

        if(node_out==-1){
            std::cout<<"   "<<node<<": VALUE VARIABLE"<<dim<<" "<<section<<" "<<row<<std::endl;;
            node_out = node;
            ++node;
        } 
        return node_out;
    }

    inline void printDst(DEST_TYPE type, int pos, int node_out){
        switch(type){
            case DEST_TYPE::tmp1:{
                tmp1_node[pos] = node_out;
                break;
            }
            case DEST_TYPE::tmp3:{
                tmp3_node[pos] = node_out;
                break;
            }
            case DEST_TYPE::dest1:{
                assert(dest_column[pos] == -1);
                dest_column[pos] = dcol;
                std::cout<<"   "<<node<<": COPY "<<node_out<<" "<<dcol<<std::endl;
                ++dcol;
                ++node;
                break;
            }
            case DEST_TYPE::dest3:{
                assert(dest_column[pos] == -1);
                dest_column[pos] = dcol;
                std::cout<<"   "<<node<<": COPY3 "<<node_out<<" "<<dcol<<std::endl;
                ++dcol;
                ++node;
                break;
            }
            default:{
                assert(0);
                break;
            }
        }
    }

    inline int getNode(TMP_TYPE type, int row){
        switch(type){
            case TMP_TYPE::dim1:{
                assert(tmp1_node[row] != -1);
                return tmp1_node[row];
            }
            case TMP_TYPE::dim3:{
                assert(tmp3_node[row] != -1);
                return tmp3_node[row];
            }
            default:{
                assert(0);
                break;
            }
        }
    }
    
    //=======================/

    inline virtual void setBufferTInfo(StarkInfo& starkInfo, uint64_t stage) {
        bool domainExtended = stage <= 3 ? false : true;
        nColsStagesAcc.resize(10 + 2);
        nColsStages.resize(10 + 2);
        offsetsStages.resize(10 + 2);

        nColsStages[0] = starkInfo.nConstants + 2;
        offsetsStages[0] = 0;

        for(uint64_t s = 1; s <= 3; ++s) {
            nColsStages[s] = starkInfo.mapSectionsN.section[string2section("cm" + to_string(s) + "_n")];
            if(domainExtended) {
                offsetsStages[s] = starkInfo.mapOffsets.section[string2section("cm" + to_string(s) + "_2ns")];
            } else {
                offsetsStages[s] = starkInfo.mapOffsets.section[string2section("cm" + to_string(s) + "_n")];
            }
        }
        if(domainExtended) {
            nColsStages[4] = starkInfo.mapSectionsN.section[eSection::cm4_2ns];
            offsetsStages[4] = starkInfo.mapOffsets.section[eSection::cm4_2ns];
        } else {
            nColsStages[4] = starkInfo.mapSectionsN.section[eSection::tmpExp_n];
            offsetsStages[4] = starkInfo.mapOffsets.section[eSection::tmpExp_n];
        }
        for(uint64_t o = 0; o < 2; ++o) {
            for(uint64_t s = 0; s < 5; ++s) {
                if(s == 0) {
                    if(o == 0) {
                        nColsStagesAcc[0] = 0;
                    } else {
                        nColsStagesAcc[5*o] = nColsStagesAcc[5*o - 1] + nColsStages[4];
                    }
                } else {
                    nColsStagesAcc[5*o + s] = nColsStagesAcc[5*o + (s - 1)] + nColsStages[(s - 1)];
                }
            }
        }
        nColsStagesAcc[10] = nColsStagesAcc[9] + nColsStages[4]; // Polinomials f & q
        if(stage == 4) {
            offsetsStages[10] = starkInfo.mapOffsets.section[eSection::q_2ns];
            nColsStages[10] = starkInfo.qDim;
        } else if(stage == 5) {
            offsetsStages[10] = starkInfo.mapOffsets.section[eSection::f_2ns];
            nColsStages[10] = 3;
        }
        nColsStagesAcc[11] = nColsStagesAcc[10] + nColsStages[10]; // xDivXSubXi
        nCols = nColsStagesAcc[11] + 6; // 3 for xDivXSubXi and 3 for xDivXSubWxi
    }

    inline virtual void storePolinomials(StarkInfo &starkInfo, StepsParams &params, __m256i *bufferT_, uint8_t* storePol, uint64_t row, uint64_t nrowsPack, uint64_t domainExtended) {
        if(domainExtended) {
            // Store either polinomial f or polinomial q
            for(uint64_t k = 0; k < nColsStages[10]; ++k) {
                __m256i *buffT = &bufferT_[(nColsStagesAcc[10] + k)];
                Goldilocks::store_avx(&params.pols[offsetsStages[10] + k + row * nColsStages[10]], nColsStages[10], buffT[0]);
            }
        } else {
            uint64_t nStages = 3;
            uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
            for(uint64_t s = 2; s <= nStages + 1; ++s) {
                bool isTmpPol = !domainExtended && s == 4;
                for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                    uint64_t dim = storePol[nColsStagesAcc[s] + k];
                    if(storePol[nColsStagesAcc[s] + k]) {
                        __m256i *buffT = &bufferT_[(nColsStagesAcc[s] + k)];
                        if(isTmpPol) {
                            for(uint64_t i = 0; i < dim; ++i) {
                                Goldilocks::store_avx(&params.pols[offsetsStages[s] + k * domainSize + row * dim + i], uint64_t(dim), buffT[i]);
                            }
                        } else {
                            Goldilocks::store_avx(&params.pols[offsetsStages[s] + k + row * nColsStages[s]], nColsStages[s], buffT[0]);
                        }
                    }
                }
            }
        }
    }

    inline virtual void loadPolinomials(StarkInfo &starkInfo, StepsParams &params, __m256i *bufferT_, uint64_t row, uint64_t stage, uint64_t nrowsPack, uint64_t domainExtended) {
        Goldilocks::Element bufferT[2*nrowsPack];
        ConstantPolsStarks *constPols = domainExtended ? params.pConstPols2ns : params.pConstPols;
        Polinomial &x = domainExtended ? params.x_2ns : params.x_n;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint64_t nStages = 3;
        uint64_t nextStride = domainExtended ?  1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;
        std::vector<uint64_t> nextStrides = {0, nextStride};
        for(uint64_t k = 0; k < starkInfo.nConstants; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT[nrowsPack*o + j] = ((Goldilocks::Element *)constPols->address())[l * starkInfo.nConstants + k];
                }
                Goldilocks::load_avx(bufferT_[nColsStagesAcc[5*o] + k], &bufferT[nrowsPack*o]);
            }
        }

        // Load x and Zi
        for(uint64_t j = 0; j < nrowsPack; ++j) {
            bufferT[j] = x[row + j][0];
        }
        Goldilocks::load_avx(bufferT_[starkInfo.nConstants], &bufferT[0]);
        for(uint64_t j = 0; j < nrowsPack; ++j) {
            bufferT[j] = params.zi[row + j][0];
        }

        Goldilocks::load_avx(bufferT_[starkInfo.nConstants + 1], &bufferT[0]);

        for(uint64_t s = 1; s <= nStages; ++s) {
            if(stage < s) break;
            for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                for(uint64_t o = 0; o < 2; ++o) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        bufferT[nrowsPack*o + j] = params.pols[offsetsStages[s] + l * nColsStages[s] + k];
                    }
                    Goldilocks::load_avx(bufferT_[nColsStagesAcc[5*o + s] + k], &bufferT[nrowsPack*o]);
                }
            }
        }

        if(stage == 5) {
           for(uint64_t k = 0; k < nColsStages[nStages + 1]; ++k) {
               for(uint64_t o = 0; o < 2; ++o) {
                   for(uint64_t j = 0; j < nrowsPack; ++j) {
                       uint64_t l = (row + j + nextStrides[o]) % domainSize;
                       bufferT[nrowsPack*o + j] = params.pols[offsetsStages[nStages + 1] + l * nColsStages[nStages + 1] + k];
                   }
                   Goldilocks::load_avx(bufferT_[nColsStagesAcc[5*o + nStages + 1] + k], &bufferT[nrowsPack*o]);
               }
           }

           // Load xDivXSubXi & xDivXSubWXi
           for(uint64_t d = 0; d < 2; ++d) {
               for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
                   for(uint64_t j = 0; j < nrowsPack; ++j) {
                       bufferT[j] = params.xDivXSubXi[d*domainSize + row + j][i];
                   }
                   Goldilocks::load_avx(bufferT_[nColsStagesAcc[11] + FIELD_EXTENSION*d + i], &bufferT[0]);
               }
           }
       }
    }

    virtual void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {

        //initialize dest_comuns to -1
        for(int i = 0; i < _NPOS_; i++){
            dest_column[i] = -1;
            tmp1_node[i]= -1; 
            tmp3_node[i]= -1;
            dest_column[i]= -1;
            read_trace[i]= -1;
            read_trace[i+_NPOS_]= -1;
            read_challenges[i]= -1;
            read_publics[i]= -1;
            read_numbers[i]= -1;
            read_evals[i]= -1;
        }
        assert(nrowsPack == 4);
        bool domainExtended = parserParams.stage > 3 ? true : false;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint8_t *ops = &parserArgs.ops[parserParams.opsOffset];
        uint16_t *args = &parserArgs.args[parserParams.argsOffset];
        uint64_t *numbers = &parserArgs.numbers[parserParams.numbersOffset];
        uint8_t *storePol = &parserArgs.storePols[parserParams.storePolsOffset];

        setBufferTInfo(starkInfo, parserParams.stage);
        Goldilocks3::Element_avx challenges[params.challenges.degree()];
        Goldilocks3::Element_avx challenges_ops[params.challenges.degree()];
        for(uint64_t i = 0; i < params.challenges.degree(); ++i) {
            challenges[i][0] = _mm256_set1_epi64x(params.challenges[i][0].fe);
            challenges[i][1] = _mm256_set1_epi64x(params.challenges[i][1].fe);
            challenges[i][2] = _mm256_set1_epi64x(params.challenges[i][2].fe);

            Goldilocks::Element challenges_aux[3];
            challenges_aux[0] = params.challenges[i][0] + params.challenges[i][1];
            challenges_aux[1] = params.challenges[i][0] + params.challenges[i][2];
            challenges_aux[2] = params.challenges[i][1] + params.challenges[i][2];
            challenges_ops[i][0] = _mm256_set1_epi64x(challenges_aux[0].fe);
            challenges_ops[i][1] =  _mm256_set1_epi64x(challenges_aux[1].fe);
            challenges_ops[i][2] =  _mm256_set1_epi64x(challenges_aux[2].fe);
        }

        __m256i numbers_[parserParams.nNumbers];
        for(uint64_t i = 0; i < parserParams.nNumbers; ++i) {
            numbers_[i] = _mm256_set1_epi64x(numbers[i]);
        }

        __m256i publics[starkInfo.nPublics];
        for(uint64_t i = 0; i < starkInfo.nPublics; ++i) {
            publics[i] = _mm256_set1_epi64x(params.publicInputs[i].fe);
        }

        Goldilocks3::Element_avx evals[params.evals.degree()];
        for(uint64_t i = 0; i < params.evals.degree(); ++i) {
            evals[i][0] = _mm256_set1_epi64x(params.evals[i][0].fe);
            evals[i][1] = _mm256_set1_epi64x(params.evals[i][1].fe);
            evals[i][2] = _mm256_set1_epi64x(params.evals[i][2].fe);
        }
        std::cout<<"EXPRESSIONS: "<<parserParams.nOps<<std::endl;
        int node1, node2;

    #pragma omp parallel for
        for (uint64_t i = 0; i < 1/*domainSize*/; i+= nrowsPack) {
            uint64_t i_args = 0;

            __m256i bufferT_[2*nCols];
            __m256i tmp1[parserParams.nTemp1];
            Goldilocks3::Element_avx tmp3[parserParams.nTemp3];

            loadPolinomials(starkInfo, params, bufferT_, i, parserParams.stage, nrowsPack, domainExtended);

            for (uint64_t kk = 0; kk < parserParams.nOps; ++kk) {
                //std::cout<<"=> OP: "<<kk<<" "<<uint64_t(ops[kk])<<std::endl;
                switch (ops[kk]) {
                case 0: {
                    
                    // AST format
                    assert(0);
                    
                    // COPY commit1 to commit1
                    Goldilocks::copy_avx(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 1: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 3], args[i_args + 4], 1);
                    int node2 = printTraceValue(args[i_args + 5], args[i_args + 6], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::dest1, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], bufferT_[nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]]);
                    i_args += 7;
                    break;
                }
                case 2: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 3], args[i_args + 4], 1);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 5]);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::dest1, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], tmp1[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 3: {
                    
                    // AST format
                    int node1 = printTraceValue(args[i_args + 3], args[i_args + 4], 1);
                    int node2 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 5], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::dest1, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: public
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], publics[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 4: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 3], args[i_args + 4], 1);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 5], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::dest1, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: number
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], numbers_[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 5: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 2]);
                    printDst(DEST_TYPE::dest1, nColsStagesAcc[args[i_args]] + args[i_args + 1], node1);

                    // COPY tmp1 to commit1
                    Goldilocks::copy_avx(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], tmp1[args[i_args + 2]]);

                    i_args += 3;
                    break;
                }
                case 6: {
                    
                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 3]);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 4]);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::dest1, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp1[args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 7: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 3]);
                    int node2 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::dest1, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: public
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp1[args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 8: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 3]);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::dest1, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp1[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 9: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 2], 1);
                    printDst(DEST_TYPE::dest1, nColsStagesAcc[args[i_args]] + args[i_args + 1], node1);

                    // COPY public to commit1
                    Goldilocks::copy_avx(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], publics[args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 10: {
                    
                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 3], 1);
                    int node2 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::dest1, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: public
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], publics[args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 11: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 3], 1);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::dest1, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: number
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], publics[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 12: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::number, args[i_args + 2], 1);
                    printDst(DEST_TYPE::dest1, nColsStagesAcc[args[i_args]] + args[i_args + 1], node1);

                    // COPY number to commit1
                    Goldilocks::copy_avx(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], numbers_[args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 13: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::number, args[i_args + 3], 1);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::dest1, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit1 - SRC0: number - SRC1: number
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], numbers_[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 14: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 1], args[i_args + 2], 1);
                    printDst(DEST_TYPE::tmp1, args[i_args], node1);

                    // COPY commit1 to tmp1
                    Goldilocks::copy_avx(tmp1[args[i_args]], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 15: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 2], args[i_args + 3], 1);
                    int node2 = printTraceValue(args[i_args + 4], args[i_args + 5], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::tmp1, args[i_args+1], node3);
                    
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 16: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 2], args[i_args + 3], 1);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 4]);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::tmp1, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 17: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 2], args[i_args + 3], 1);
                    int node2 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::tmp1, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 18: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 2], args[i_args + 3], 1);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::tmp1, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 19: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 1]);
                    printDst(DEST_TYPE::tmp1, args[i_args], node1);

                    // COPY tmp1 to tmp1
                    Goldilocks::copy_avx(tmp1[args[i_args]], tmp1[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 20: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 2]);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 3]);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::tmp1, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 21: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 2]);
                    int node2 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 3], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::tmp1, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 22: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 2]);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 3], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::tmp1, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 23: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 1],1);
                    printDst(DEST_TYPE::tmp1, args[i_args], node1);

                    // COPY public to tmp1
                    Goldilocks::copy_avx(tmp1[args[i_args]], publics[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 24: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 2], 1);
                    int node2 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 3], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::tmp1, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 25: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 2], 1);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 3], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::tmp1, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 26: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::number, args[i_args + 1],1);
                    printDst(DEST_TYPE::tmp1, args[i_args], node1);

                    // COPY number to tmp1
                    Goldilocks::copy_avx(tmp1[args[i_args]], numbers_[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 27: {
                    
                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::number, args[i_args + 2], 1);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 3], 1);
                    int node3 = printOP(args[i_args], node1, node2, "");
                    printDst(DEST_TYPE::tmp1, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], numbers_[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 28: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 3], args[i_args + 4], 3);
                    int node2 = printTraceValue(args[i_args + 5], args[i_args + 6], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], bufferT_[nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]]);
                    i_args += 7;
                    break;
                }
                case 29: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 3], args[i_args + 4], 3);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 5]);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp1
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], tmp1[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 30: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 3], args[i_args + 4], 3);
                    int node2 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 5], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: public
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], publics[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 31: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 3], args[i_args + 4], 3);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 5], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: number
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], numbers_[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 32: {
                    
                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 3]);
                    int node2 = printTraceValue(args[i_args + 4], args[i_args + 5], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 33: {
                    
                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 3]);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 4]);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 34: {
                    
                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 3]);
                    int node2 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: public
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 35: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 3]);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: number
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 36: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::challenge, args[i_args + 3], 3);
                    int node2 = printTraceValue(args[i_args + 4], args[i_args + 5], 1);                    
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 37: {
                    
                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::challenge, args[i_args + 3], 3);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 4]);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: tmp1
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 38: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::challenge, args[i_args + 3], 3);
                    int node2 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: public
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 39: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::challenge, args[i_args + 3], 3);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: number
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 40: {

                    // AST format
                    assert(0);

                    // COPY commit3 to commit3
                    Goldilocks3::copy_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 41: {
                    
                    // AST format
                    int node1 = printTraceValue(args[i_args + 3], args[i_args + 4], 3);
                    int node2 = printTraceValue(args[i_args + 5], args[i_args + 6], 3);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit3
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]]);
                    i_args += 7;
                    break;
                }
                case 42: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 3], args[i_args + 4], 3);
                    int node2 = getNode(TMP_TYPE::dim3, args[i_args + 5]);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);
                
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], tmp3[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 43: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 3], args[i_args + 4], 3);
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 5], 3);
                    int node3 = printOP(2, node1, node2, "33");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // MULTIPLICATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::mul_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], challenges[args[i_args + 5]], challenges_ops[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 44: {
                    
                    // AST format
                    int node1 = printTraceValue(args[i_args + 3], args[i_args + 4], 3);
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 5], 3);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], challenges[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 45: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2]);
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args]] + args[i_args + 1], node1);

                    // COPY tmp3 to commit3
                    Goldilocks3::copy_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], tmp3[args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 46: {
                    
                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 3]);
                    int node2 = getNode(TMP_TYPE::dim3, args[i_args + 4]);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], tmp3[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 47: {
                    
                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 3]);
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 4], 3);
                    int node3 = printOP(2, node1, node2, "33");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // MULTIPLICATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::mul_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 48: {

                    // AST format
                    int node1 = tmp3_node[args[i_args + 3]];
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 4], 3);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], challenges[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 49: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::challenge, args[i_args + 3], 3);
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 4], 3);
                    int node3 = printOP(2, node1, node2, "33");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // MULTIPLICATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::mul_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 50: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::challenge, args[i_args + 3], 3);
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 4], 3);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::dest3, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3);

                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], challenges[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 51: {
                    
                    // AST format
                    int node1 = printTraceValue(args[i_args + 2], args[i_args + 3], 3);
                    int node2 = printTraceValue(args[i_args + 4], args[i_args + 5], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 52: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 2], args[i_args + 3], 3);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 4]);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 53: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 2], args[i_args + 3], 3);
                    int node2 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 54: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 2], args[i_args + 3], 3);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 55: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2]);
                    int node2 = printTraceValue(args[i_args + 3], args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 56: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2]);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 3]);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::tmp3, args[i_args + 1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 57: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2]);
                    int node2 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 3],1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 58: {
                    
                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2]);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 3],1);
                    int node3 = printOP(args[i_args], node1, node2, "31");  
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 59: {
                    
                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::challenge, args[i_args + 2],3);
                    int node2 = printTraceValue(args[i_args + 3], args[i_args + 4],1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 60: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::challenge, args[i_args + 2],3);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 3]);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 61: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::challenge, args[i_args + 2], 3);
                    int node2 = printVariableValue(VAR_TYPE::publicInput, args[i_args + 3], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 62: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::challenge, args[i_args + 2], 3);
                    int node2 = printVariableValue(VAR_TYPE::number, args[i_args + 3], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 63: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 1], args[i_args + 2], 3);
                    printDst(DEST_TYPE::tmp3, args[i_args], node1);

                    // COPY commit3 to tmp3
                    Goldilocks3::copy_avx(tmp3[args[i_args]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 64: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 2], args[i_args + 3], 3);
                    int node2 = printTraceValue(args[i_args + 4], args[i_args + 5], 3);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 65: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 2], args[i_args + 3], 3);
                    int node2 = getNode(TMP_TYPE::dim3, args[i_args + 4]);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::tmp3, args[i_args + 1], node3);
                    
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp3[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 66: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 2], args[i_args + 3], 3);
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 4], 3);
                    int node3 = printOP(2, node1, node2, "33");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::mul_avx(tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 67: {

                    // AST format
                    int node1 = printTraceValue(args[i_args + 2], args[i_args + 3], 3);
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 4], 3);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], challenges[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 68: {
                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 1]);
                    printDst(DEST_TYPE::tmp3, args[i_args], node1);

                    // COPY tmp3 to tmp3
                    Goldilocks3::copy_avx(tmp3[args[i_args]], tmp3[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 69: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2]);
                    int node2 = getNode(TMP_TYPE::dim3, args[i_args + 3]);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 70: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2]);
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 3], 3);
                    int node3 = printOP(2, node1, node2, "33");
                    printDst(DEST_TYPE::tmp3, args[i_args + 1 ], node3);

                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 71: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2]);
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 3], 3);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::tmp3, args[i_args + 1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 72: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::challenge, args[i_args + 2], 3);
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 3], 3);
                    int node3 = printOP(2, node1, node2, "33");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::mul_avx(tmp3[args[i_args + 1]], challenges[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 73: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::challenge, args[i_args + 2], 3);
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 3], 3);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], challenges[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 74: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::eval, args[i_args + 1], 3);
                    printDst(DEST_TYPE::tmp3, args[i_args], node1);

                    // COPY eval to tmp3
                    Goldilocks3::copy_avx(tmp3[args[i_args]], evals[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 75: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::eval, args[i_args + 2], 3);
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 3], 3);
                    int node3 = printOP(2, node1, node2, "33");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: eval - SRC1: challenge
                    Goldilocks3::mul_avx(tmp3[args[i_args + 1]], evals[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 76: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::challenge, args[i_args + 2], 3);
                    int node2 = printVariableValue(VAR_TYPE::eval, args[i_args + 3], 3);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], evals[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 77: {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2]);
                    int node2 = printVariableValue(VAR_TYPE::eval, args[i_args + 3], 3);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);
                    
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], evals[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 78: {

                    // AST format
                    int node1 = printVariableValue(VAR_TYPE::eval, args[i_args + 2], 3);
                    int node2 = printTraceValue(args[i_args + 3], args[i_args + 4], 1);
                    int node3 = printOP(args[i_args], node1, node2, "31");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 79: {
                    
                    // AST format
                    int node1 = printTraceValue(args[i_args + 2], args[i_args + 3], 3);
                    int node2 = printVariableValue(VAR_TYPE::challenge, args[i_args + 4], 3);
                    int node3 = printOP(args[i_args], node1, node2, "33");
                    printDst(DEST_TYPE::tmp3, args[i_args+1], node3);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], evals[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                    default: {
                        std::cout << " Wrong operation!" << std::endl;
                        exit(1);
                    }
                }
            }
            storePolinomials(starkInfo, params, bufferT_, storePol, i, nrowsPack, domainExtended);
            if (i_args != parserParams.nArgs) std::cout << " " << i_args << " - " << parserParams.nArgs << std::endl;
            assert(i_args == parserParams.nArgs);
        }
    }
};

#endif