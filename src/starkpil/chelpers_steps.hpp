#ifndef CHELPERS_STEPS_HPP
#define CHELPERS_STEPS_HPP
#include "chelpers.hpp"
#include "steps.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
using json = nlohmann::json;

static int node = 0;
static int dcol = 0;

#define _NPOS_ 100000

static int tmp1_node[_NPOS_]; // more than enough space
static int tmp3_node[_NPOS_];
static int dest_column[_NPOS_];

static int read_trace[2 * _NPOS_];
static int read_challenges[_NPOS_];
static int read_publics[_NPOS_];
static int read_numbers[_NPOS_];
static int read_evals[_NPOS_];

static int used_sections[100];
static int used_variables[100];

static json outputJSON;

enum VAR_TYPE
{
    challenge,
    publicInput,
    number,
    eval
};

enum DEST_TYPE
{
    tmp1,
    tmp3,
        expression
};

enum TMP_TYPE
{
    dim1,
    dim3
};
class CHelpersSteps
{
public:
    uint64_t nrowsPack = 4;
    uint64_t nCols;
    vector<uint64_t> nColsStages;
    vector<uint64_t> nColsStagesAcc;
    vector<uint64_t> offsetsStages;
    bool genAST = false;
    string fileAST = "ast.json";

    inline void addGlobalVars()
    {
        node = 0;
        dcol = 0;

        // initialize dest_comuns to -1
        for (int i = 0; i < _NPOS_; i++)
        {
            tmp1_node[i] = -1;
            tmp3_node[i] = -1;
            dest_column[i] = -1;
            read_trace[i] = -1;
            read_trace[i + _NPOS_] = -1;
            read_challenges[i] = -1;
            read_publics[i] = -1;
            read_numbers[i] = -1;
            read_evals[i] = -1;
        }
        for (int i = 0; i < 100; i++)
        {
            used_sections[i] = 0;
            used_variables[i] = 0;
        }
        // clean outputJSON
        outputJSON.clear();
        outputJSON["nodes"] = json::array();
        outputJSON["expressions"] = json::array();
    }

    inline void addMetadata(StarkInfo &starkInfo, StepsParams &params, ParserParams &parserParams)
    {

        std::cout << "   " << std::endl;
        std::cout << "   nNodes: " << node << std::endl;
        std::cout << "   nStages: " << 7 << std::endl;
        std::cout << "   nVariables: " << 4 << std::endl;
        std::cout << "   nEvals: " << dcol << std::endl;
        std::cout << std::endl;

        // Create the trace_widths array
        json traceWidths = json::array();
        for (int i = 0; i < 5; i++)
        {
            traceWidths.push_back(nColsStages[i] * used_sections[i]);
        }
        traceWidths.push_back(nColsStages[10] * used_sections[5]);
        traceWidths.push_back(6 * used_sections[6]);

        // Num variables
        json numVariables = json::array();
        numVariables.push_back(params.challenges.degree() * used_variables[0] * 3);
        numVariables.push_back(starkInfo.nPublics * used_variables[1]);
        numVariables.push_back(params.evals.degree() * used_variables[2] * 3);

        outputJSON["metadata"] = {
            {"field",
             {{"name", "Goldilocks"},
              {"modulus", "18446744069414584321"},
              {"root_of_unity", "7277203076849721926"},
              {"coset_offset", "7"},
              {"extension",
               {{"degree", 3},
                {"polynom", "x^3 - x + 2"}}}}},
            {"trace_widths", traceWidths},
            {"num_variables", numVariables},
        };

        json zerofiers = json::array();
        outputJSON["zerofiers"] = zerofiers;

        json periodic = json::array();
        outputJSON["periodic"] = periodic;

        ///// old stuff
        // outputJSON["nNodes"] = node;
        // outputJSON["nStages"] = 7;
        // outputJSON["nVariables"] = 4;
        // outputJSON["nEvals"] = dcol;
        // outputJSON["max_nodes_base"] = parserParams.nTemp1;
        // outputJSON["max_nodes_extension"] = parserParams.nTemp3;
    }

    inline int addOP(int op, int node1, int node2, string mode, int irow)
    {
        if (irow == 0)
        {
            json valueNode;
            assert(mode == "" || mode == "33" || mode == "31");
            if (mode == "33" || mode == "31")
                valueNode["value"] = "ext";
            else
                valueNode["value"] = "base";

            switch (op)
            {
            case 0:
                std::cout << "   " << node << ": add " << node1 << " " << node2 << std::endl;
                valueNode["type"] = "add";
                valueNode["args"] = {{"lhs", node1}, {"rhs", node2}};
                break;
            case 1:
                std::cout << "   " << node << ": sub " << node1 << " " << node2 << std::endl;
                valueNode["type"] = "sub";
                valueNode["args"] = {{"lhs", node1}, {"rhs", node2}};
                break;
            case 2:
                std::cout << "   " << node << ": mul " << node1 << " " << node2 << std::endl;
                valueNode["type"] = "mul";
                valueNode["args"] = {{"lhs", node1}, {"rhs", node2}};
                break;
            case 3:
                std::cout << "   " << node << ": sub " << node2 << " " << node1 << std::endl;
                valueNode["type"] = "sub";
                valueNode["args"] = {{"lhs", node2}, {"rhs", node1}};
                break;
            default:
                assert(0);
                break;
            }
            outputJSON["nodes"].push_back(valueNode);
            ++node;
            return node - 1;
        }
        else
        {
            return 0;
        }
    }

    inline int addTrace(int section, int col, string dim, int irow)
    {
        if (irow == 0)
        {
            json valueNode;
            valueNode["type"] = "trace";
            if (dim == "3")
            {
                valueNode["value"] = "ext";
            }
            else
            {
                valueNode["value"] = "base";
            }
            assert(dim == "1" || dim == "3");
            int accum = nColsStagesAcc[section] + col;
            int node_out = -1;
            if (read_trace[accum] == -1)
            {
                if (section < 5)
                {
                    used_sections[section] = 1;
                    std::cout << "   " << node << ": trace " << section << " " << col << " 0" << std::endl;
                    valueNode["args"] = {{"segment", section}, {"col_offset", col}, {"row_offset", 0}};
                }
                else if (section < 10)
                {
                    used_sections[section - 5] = 1;
                    std::cout << "   " << node << ": trace " << section - 5 << " " << col << " 1" << std::endl;
                    valueNode["args"] = {{"segment", section - 5}, {"col_offset", col}, {"row_offset", 1}};
                }
                else
                {
                    assert(section < 12);
                    used_sections[section - 5] = 1;
                    std::cout << "   " << node << ": trace " << section - 5 << " " << col << " 0" << std::endl;
                    valueNode["args"] = {{"segment", section - 5}, {"col_offset", col}, {"row_offset", 0}};
                }
            }
            else
            {
                node_out = read_trace[accum];
            }
            if (node_out == -1)
            {
                node_out = node;
                read_trace[accum] = node;
                outputJSON["nodes"].push_back(valueNode);
                ++node;
            }
            return node_out;
        }
        else
        {
            return 0;
        }
    }

    inline int addVariable(VAR_TYPE type, int row, string dim, int irow)
    {
        int offset;
        if (irow == 0)
        {
            int section;
            int node_out = -1;

            json valueNode;
            valueNode["type"] = "var";
            assert(dim == "1" || dim == "3");
            if (dim == "3")
            {
                valueNode["value"] = "ext";
            }
            else
            {
                valueNode["value"] = "base";
            }

            switch (type)
            {
            case VAR_TYPE::challenge:
            {
                assert(dim == "3");
                offset = row * 3;
                if (read_challenges[row] != -1)
                {
                    node_out = read_challenges[row];
                }
                else
                {
                    read_challenges[row] = node;
                }
                section = 0;
                break;
            }
            case VAR_TYPE::publicInput:
            {
                assert(dim == "1");
                offset = row;
                if (read_publics[row] != -1)
                {
                    node_out = read_publics[row];
                }
                else
                {
                    read_publics[row] = node;
                }
                section = 1;
                break;
            }
            case VAR_TYPE::eval:
            {
                assert(dim == "3");
                offset = row * 3;
                if (read_evals[row] != -1)
                {
                    node_out = read_evals[row];
                }
                else
                {
                    read_evals[row] = node;
                }
                section = 2;
                break;
            }
            default:
            {
                assert(0);
                break;
            }
            }
            used_variables[section] = 1;
            if (node_out == -1)
            {
                std::cout << "   " << node << ": var " << section << " " << row << std::endl;
                valueNode["args"] = {{"group", section}, {"offset", offset}};
                outputJSON["nodes"].push_back(valueNode);
                node_out = node;
                ++node;
            }
            return node_out;
        }
        else
        {
            return 0;
        }
    }

    inline int addConstant(int offset, uint64_t value, int irow)
    {
        if (irow == 0)
        {
            int node_out = -1;
            json constNode;
            if (read_numbers[offset] != -1)
            {
                node_out = read_numbers[offset];
            }
            else
            {
                // uint64_t val = 2 * 67 + offset * 11;
                read_numbers[offset] = node;
                constNode["type"] = "const";
                constNode["value"] = "base";
                constNode["args"] = {{"value", value}}; /*val*/
                outputJSON["nodes"].push_back(constNode);
                node_out = node;
                ++node;
            }
            return node_out;
        }
        else
        {
            return 0;
        }
    }

    inline void addDst(DEST_TYPE type, int pos, int node_out, int irow)
    {

        if (irow == 0)
        {

            switch (type)
            {
            case DEST_TYPE::tmp1:
            {
                tmp1_node[pos] = node_out;
                break;
            }
            case DEST_TYPE::tmp3:
            {
                tmp3_node[pos] = node_out;
                break;
            }
            case DEST_TYPE::expression:
            {
                json expression;
                assert(dest_column[pos] == -1);
                dest_column[pos] = dcol;
                expression["node_id"] = node_out;
                outputJSON["expressions"].push_back(expression);
                ++dcol;
                break;
            }
            default:
            {
                assert(0);
                break;
            }
            }
        }
    }

    inline int getNode(TMP_TYPE type, int row, int irow)
    {
        if (irow == 0)
        {
            switch (type)
            {
            case TMP_TYPE::dim1:
            {
                assert(tmp1_node[row] != -1);
                return tmp1_node[row];
            }
            case TMP_TYPE::dim3:
            {
                assert(tmp3_node[row] != -1);
                return tmp3_node[row];
            }
            default:
            {
                assert(0);
                break;
            }
            }
        }
        else
        {
            return 0;
        }
    }

    //=======================/

    inline virtual void setBufferTInfo(StarkInfo &starkInfo, uint64_t stage)
    {
        bool domainExtended = stage <= 3 ? false : true;
        nColsStagesAcc.resize(10 + 2);
        nColsStages.resize(10 + 2);
        offsetsStages.resize(10 + 2);

        nColsStages[0] = starkInfo.nConstants + 2;
        offsetsStages[0] = 0;

        for (uint64_t s = 1; s <= 3; ++s)
        {
            nColsStages[s] = starkInfo.mapSectionsN.section[string2section("cm" + to_string(s) + "_n")];
            if (domainExtended)
            {
                offsetsStages[s] = starkInfo.mapOffsets.section[string2section("cm" + to_string(s) + "_2ns")];
            }
            else
            {
                offsetsStages[s] = starkInfo.mapOffsets.section[string2section("cm" + to_string(s) + "_n")];
            }
        }
        if (domainExtended)
        {
            nColsStages[4] = starkInfo.mapSectionsN.section[eSection::cm4_2ns];
            offsetsStages[4] = starkInfo.mapOffsets.section[eSection::cm4_2ns];
        }
        else
        {
            nColsStages[4] = starkInfo.mapSectionsN.section[eSection::tmpExp_n];
            offsetsStages[4] = starkInfo.mapOffsets.section[eSection::tmpExp_n];
        }
        for (uint64_t o = 0; o < 2; ++o)
        {
            for (uint64_t s = 0; s < 5; ++s)
            {
                if (s == 0)
                {
                    if (o == 0)
                    {
                        nColsStagesAcc[0] = 0;
                    }
                    else
                    {
                        nColsStagesAcc[5 * o] = nColsStagesAcc[5 * o - 1] + nColsStages[4];
                    }
                }
                else
                {
                    nColsStagesAcc[5 * o + s] = nColsStagesAcc[5 * o + (s - 1)] + nColsStages[(s - 1)];
                }
            }
        }
        nColsStagesAcc[10] = nColsStagesAcc[9] + nColsStages[4]; // Polinomials f & q
        if (stage == 4)
        {
            offsetsStages[10] = starkInfo.mapOffsets.section[eSection::q_2ns];
            nColsStages[10] = starkInfo.qDim;
        }
        else if (stage == 5)
        {
            offsetsStages[10] = starkInfo.mapOffsets.section[eSection::f_2ns];
            nColsStages[10] = 3;
        }
        nColsStagesAcc[11] = nColsStagesAcc[10] + nColsStages[10]; // xDivXSubXi
        nCols = nColsStagesAcc[11] + 6;                            // 3 for xDivXSubXi and 3 for xDivXSubWxi
    }

    inline virtual void storePolinomials(StarkInfo &starkInfo, StepsParams &params, __m256i *bufferT_, uint8_t *storePol, uint64_t row, uint64_t nrowsPack, uint64_t domainExtended)
    {
        if (domainExtended)
        {
            // Store either polinomial f or polinomial q
            if (row < 10)
            {
                std::cout << "Q " << row << " ";
            }
            for (uint64_t k = 0; k < nColsStages[10]; ++k)
            {
                __m256i *buffT = &bufferT_[(nColsStagesAcc[10] + k)];
                Goldilocks::store_avx(&params.pols[offsetsStages[10] + k + row * nColsStages[10]], nColsStages[10], buffT[0]);
                if (row < 10)
                {
                    std::cout << " " << params.pols[offsetsStages[10] + k + row * nColsStages[10]].fe;
                }
            }
            if (row < 10)
            {
                std::cout << std::endl;
            }
        }
        else
        {
            uint64_t nStages = 3;
            uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
            for (uint64_t s = 2; s <= nStages + 1; ++s)
            {
                bool isTmpPol = !domainExtended && s == 4;
                for (uint64_t k = 0; k < nColsStages[s]; ++k)
                {
                    uint64_t dim = storePol[nColsStagesAcc[s] + k];
                    if (storePol[nColsStagesAcc[s] + k])
                    {

                        __m256i *buffT = &bufferT_[(nColsStagesAcc[s] + k)];
                        if (isTmpPol)
                        {
                            for (uint64_t i = 0; i < dim; ++i)
                            {
                                Goldilocks::store_avx(&params.pols[offsetsStages[s] + k * domainSize + row * dim + i], uint64_t(dim), buffT[i]);
                            }
                        }
                        else
                        {
                            Goldilocks::store_avx(&params.pols[offsetsStages[s] + k + row * nColsStages[s]], nColsStages[s], buffT[0]);
                        }
                    }
                }
            }
        }
    }

    inline virtual void loadPolinomials(StarkInfo &starkInfo, StepsParams &params, __m256i *bufferT_, uint64_t row, uint64_t stage, uint64_t nrowsPack, uint64_t domainExtended)
    {

        Goldilocks::Element bufferT[2 * nrowsPack];
        ConstantPolsStarks *constPols = domainExtended ? params.pConstPols2ns : params.pConstPols;
        Polinomial &x = domainExtended ? params.x_2ns : params.x_n;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint64_t nStages = 3;
        uint64_t nextStride = domainExtended ? 1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;
        std::vector<uint64_t> nextStrides = {0, nextStride};
        for (uint64_t k = 0; k < starkInfo.nConstants; ++k)
        {
            for (uint64_t o = 0; o < 2; ++o)
            {
                for (uint64_t j = 0; j < nrowsPack; ++j)
                {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    //Goldilocks::Element val = Goldilocks::fromU64(0 * uint64_t(11) + l * uint64_t(7) + k * uint64_t(21));
                    bufferT[nrowsPack * o + j] = /*val; */((Goldilocks::Element *)constPols->address())[l * starkInfo.nConstants + k];
                }
                Goldilocks::load_avx(bufferT_[nColsStagesAcc[5 * o] + k], &bufferT[nrowsPack * o]);
            }
        }

        // Load x and Zi
        for (uint64_t j = 0; j < nrowsPack; ++j)
        {
            uint64_t l = row + j;
            Goldilocks::Element val = Goldilocks::fromU64(0 * uint64_t(11) + l * uint64_t(7) + starkInfo.nConstants * uint64_t(21));
            bufferT[j] = /*val; */ x[row + j][0];
        }
        Goldilocks::load_avx(bufferT_[starkInfo.nConstants], &bufferT[0]);
        for (uint64_t j = 0; j < nrowsPack; ++j)
        {
            uint64_t l = row + j;
            Goldilocks::Element val = Goldilocks::fromU64(0 * uint64_t(11) + l * uint64_t(7) + (starkInfo.nConstants + 1) * uint64_t(21));
            bufferT[j] = /*val; */ params.zi[row + j][0];
        }

        Goldilocks::load_avx(bufferT_[starkInfo.nConstants + 1], &bufferT[0]);

        for (uint64_t s = 1; s <= nStages; ++s)
        {
            if (stage < s)
                break;
            if (row == 0)
            {
                std::cout << "nColsStage " << s << " " << nColsStages[s] << std::endl;
            }
            for (uint64_t k = 0; k < nColsStages[s]; ++k)
            {

                for (uint64_t o = 0; o < 2; ++o)
                {
                    for (uint64_t j = 0; j < nrowsPack; ++j)
                    {

                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        Goldilocks::Element val = Goldilocks::fromU64(s * uint64_t(11) + l * uint64_t(7) + k * uint64_t(21));
                        bufferT[nrowsPack * o + j] = /*val; */ params.pols[offsetsStages[s] + l * nColsStages[s] + k];
                    }
                    Goldilocks::load_avx(bufferT_[nColsStagesAcc[5 * o + s] + k], &bufferT[nrowsPack * o]);
                }
            }
        }

        if (stage == 5)
        {
            for (uint64_t k = 0; k < nColsStages[nStages + 1]; ++k)
            {
                for (uint64_t o = 0; o < 2; ++o)
                {
                    for (uint64_t j = 0; j < nrowsPack; ++j)
                    {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        bufferT[nrowsPack * o + j] = params.pols[offsetsStages[nStages + 1] + l * nColsStages[nStages + 1] + k];
                    }
                    Goldilocks::load_avx(bufferT_[nColsStagesAcc[5 * o + nStages + 1] + k], &bufferT[nrowsPack * o]);
                }
            }

            // Load xDivXSubXi & xDivXSubWXi
            for (uint64_t d = 0; d < 2; ++d)
            {
                for (uint64_t i = 0; i < FIELD_EXTENSION; ++i)
                {
                    for (uint64_t j = 0; j < nrowsPack; ++j)
                    {
                        bufferT[j] = params.xDivXSubXi[d * domainSize + row + j][i];
                    }
                    Goldilocks::load_avx(bufferT_[nColsStagesAcc[11] + FIELD_EXTENSION * d + i], &bufferT[0]);
                }
            }
        }
    }

    virtual void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams)
    {

        addGlobalVars();

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
        for (uint64_t i = 0; i < params.challenges.degree(); ++i)
        {
            /*params.challenges[i][0] = Goldilocks::fromU64(0 * uint64_t(31) + (i * 3) * uint64_t(101));
            params.challenges[i][1] = Goldilocks::fromU64(0 * uint64_t(31) + (i * 3 + 1) * uint64_t(101));
            params.challenges[i][2] = Goldilocks::fromU64(0 * uint64_t(31) + (i * 3 + 2) * uint64_t(101));*/
            challenges[i][0] = _mm256_set1_epi64x(params.challenges[i][0].fe);
            challenges[i][1] = _mm256_set1_epi64x(params.challenges[i][1].fe);
            challenges[i][2] = _mm256_set1_epi64x(params.challenges[i][2].fe);

            Goldilocks::Element challenges_aux[3];
            challenges_aux[0] = params.challenges[i][0] + params.challenges[i][1];
            challenges_aux[1] = params.challenges[i][0] + params.challenges[i][2];
            challenges_aux[2] = params.challenges[i][1] + params.challenges[i][2];
            challenges_ops[i][0] = _mm256_set1_epi64x(challenges_aux[0].fe);
            challenges_ops[i][1] = _mm256_set1_epi64x(challenges_aux[1].fe);
            challenges_ops[i][2] = _mm256_set1_epi64x(challenges_aux[2].fe);
        }

        __m256i numbers_[parserParams.nNumbers];
        for (uint64_t i = 0; i < parserParams.nNumbers; ++i)
        {
            //numbers[i] = 2 * 31 + i * 101;
            numbers_[i] = _mm256_set1_epi64x(numbers[i]);
        }

        __m256i publics[starkInfo.nPublics];
        for (uint64_t i = 0; i < starkInfo.nPublics; ++i)
        {
            /*params.publicInputs[i] = Goldilocks::fromU64(1 * uint64_t(31) + i * uint64_t(101));*/
            publics[i] = _mm256_set1_epi64x(params.publicInputs[i].fe);
        }

        Goldilocks3::Element_avx evals[params.evals.degree()];
        for (uint64_t i = 0; i < params.evals.degree(); ++i)
        {
            evals[i][0] = _mm256_set1_epi64x(params.evals[i][0].fe);
            evals[i][1] = _mm256_set1_epi64x(params.evals[i][1].fe);
            evals[i][2] = _mm256_set1_epi64x(params.evals[i][2].fe);
        }
        std::cout << std::endl
                  << "   EXPRESSIONS: " << parserParams.nOps << std::endl
                  << std::endl;

        #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i += nrowsPack)
        {
            uint64_t i_args = 0;

            __m256i bufferT_[2 * nCols];
            __m256i tmp1[parserParams.nTemp1];
            Goldilocks3::Element_avx tmp3[parserParams.nTemp3];

            loadPolinomials(starkInfo, params, bufferT_, i, parserParams.stage, nrowsPack, domainExtended);

            for (uint64_t kk = 0; kk < parserParams.nOps; ++kk)
            {
                if (i == 0)
                    std::cout << "=> OP: " << kk << " " << uint64_t(ops[kk]) << std::endl;
                switch (ops[kk])
                {
                case 0:
                {

                    // AST format
                    assert(0);

                    // COPY commit1 to commit1
                    Goldilocks::copy_avx(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 1:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 3], args[i_args + 4], "1", i);
                    int node2 = addTrace(args[i_args + 5], args[i_args + 6], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], bufferT_[nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]]);
                    i_args += 7;
                    break;
                }
                case 2:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 3], args[i_args + 4], "1", i);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 5], i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], tmp1[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 3:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 3], args[i_args + 4], "1", i);
                    int node2 = addVariable(VAR_TYPE::publicInput, args[i_args + 5], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: public
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], publics[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 4:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 3], args[i_args + 4], "1", i);
                    int node2 = addConstant(args[i_args + 5], numbers[args[i_args + 5]], i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: number
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], numbers_[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 5:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 2], i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args]] + args[i_args + 1], node1, i);

                    // COPY tmp1 to commit1
                    Goldilocks::copy_avx(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], tmp1[args[i_args + 2]]);

                    i_args += 3;
                    break;
                }
                case 6:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 3], i);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 4], i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp1[args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 7:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 3], i);
                    int node2 = addVariable(VAR_TYPE::publicInput, args[i_args + 4], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: public
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp1[args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 8:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 3], i);
                    int node2 = addConstant(args[i_args + 4], numbers[args[i_args + 4]], i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp1[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 9:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::publicInput, args[i_args + 2], "1", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args]] + args[i_args + 1], node1, i);

                    // COPY public to commit1
                    Goldilocks::copy_avx(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], publics[args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 10:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::publicInput, args[i_args + 3], "1", i);
                    int node2 = addVariable(VAR_TYPE::publicInput, args[i_args + 4], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: public
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], publics[args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 11:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::publicInput, args[i_args + 3], "1", i);
                    int node2 = addConstant(args[i_args + 4], numbers[args[i_args + 4]], i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: number
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], publics[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 12:
                {

                    // AST format
                    int node1 = addConstant(args[i_args + 2], numbers[args[i_args + 2]], i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args]] + args[i_args + 1], node1, i);

                    // COPY number to commit1
                    Goldilocks::copy_avx(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], numbers_[args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 13:
                {

                    // AST format
                    int node1 = addConstant(args[i_args + 3], numbers[args[i_args + 3]], i);
                    int node2 = addConstant(args[i_args + 4], numbers[args[i_args + 4]], i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit1 - SRC0: number - SRC1: number
                    Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], numbers_[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 14:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 1], args[i_args + 2], "1", i);
                    addDst(DEST_TYPE::tmp1, args[i_args], node1, i);

                    // COPY commit1 to tmp1
                    // print fisrt value of argument
                    // 1) store from avx
                    Goldilocks::Element arg[4];
                    Goldilocks::store_avx(arg, 1, bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]]);
                    if (i == 0)
                        std::cout << " _op14: " << arg[0].fe << std::endl;
                    Goldilocks::copy_avx(tmp1[args[i_args]], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]]);
                    //print first value of the result
                    Goldilocks::Element res[4];
                    Goldilocks::store_avx(res, 1, tmp1[args[i_args]]);
                    if (i == 0)
                        std::cout << " _op14: " << res[0].fe << std::endl; 
                    i_args += 3;
                    break;
                }
                case 15:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 2], args[i_args + 3], "1", i);
                    int node2 = addTrace(args[i_args + 4], args[i_args + 5], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::tmp1, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                    // print fisrt value of two arguments
                    // 1) store from avx
                    Goldilocks::Element arg1[4];
                    Goldilocks::Element arg2[4];
                    Goldilocks::store_avx(arg1, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    Goldilocks::store_avx(arg2, 1, bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    if (i == 0)
                        std::cout << " _op15: " << arg1[0].fe << " " << arg2[0].fe << std::endl;

                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);

                    // print first value of the result
                    Goldilocks::Element res[4];
                    Goldilocks::store_avx(res, 1, tmp1[args[i_args + 1]]);
                    if (i == 0)
                        std::cout << " _op15: " << res[0].fe << std::endl;

                    i_args += 6;
                    break;
                }
                case 16:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 2], args[i_args + 3], "1", i);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 4], i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::tmp1, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1

                    //print fisrt value of two arguments
                    // 1) store from avx
                    Goldilocks::Element arg1[4];
                    Goldilocks::Element arg2[4];
                    Goldilocks::store_avx(arg1, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    Goldilocks::store_avx(arg2, 1, tmp1[args[i_args + 4]]);
                    if (i == 0)
                        std::cout << " _op16: " << arg1[0].fe << " " << arg2[0].fe << std::endl;

                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp1[args[i_args + 4]]);

                    //print first value of the result
                    Goldilocks::Element res[4];
                    Goldilocks::store_avx(res, 1, tmp1[args[i_args + 1]]);
                    if (i == 0)
                        std::cout << " _op16: " << res[0].fe << std::endl;

                    i_args += 5;
                    break;
                }
                case 17:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 2], args[i_args + 3], "1", i);
                    int node2 = addVariable(VAR_TYPE::publicInput, args[i_args + 4], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::tmp1, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                    //print fisrt value of two arguments
                    // 1) store from avx
                    Goldilocks::Element arg1[4];
                    Goldilocks::Element arg2[4];
                    Goldilocks::store_avx(arg1, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    Goldilocks::store_avx(arg2, 1, publics[args[i_args + 4]]);
                    if (i == 0)
                        std::cout << " _op17: " << arg1[0].fe << " " << arg2[0].fe << std::endl;
                    
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], publics[args[i_args + 4]]);

                    //print first value of the result
                    Goldilocks::Element res[4];
                    Goldilocks::store_avx(res, 1, tmp1[args[i_args + 1]]);
                    if (i == 0)
                        std::cout << " _op17: " << res[0].fe << std::endl;

                    i_args += 5;
                    break;
                }
                case 18:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 2], args[i_args + 3], "1", i);
                    int node2 = addConstant(args[i_args + 4], numbers[args[i_args + 4]], i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::tmp1, args[i_args + 1], node3, i);

                    // print fisrt value of two arguments
                    //  1) store from avx
                    Goldilocks::Element arg1[4];
                    Goldilocks::Element arg2[4];
                    Goldilocks::store_avx(arg1, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    Goldilocks::store_avx(arg2, 1, numbers_[args[i_args + 4]]);
                    if (i == 0)
                        std::cout << " _op18: " << arg1[0].fe << " " << arg2[0].fe << std::endl;

                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], numbers_[args[i_args + 4]]);

                    // print first value of the result
                    Goldilocks::Element res[4];
                    Goldilocks::store_avx(res, 1, tmp1[args[i_args + 1]]);
                    if (i == 0)
                        std::cout << " _op18: " << res[0].fe << std::endl;

                    i_args += 5;
                    break;
                }
                case 19:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 1], i);
                    addDst(DEST_TYPE::tmp1, args[i_args], node1, i);

                    // COPY tmp1 to tmp1
                    Goldilocks::copy_avx(tmp1[args[i_args]], tmp1[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 20:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 2], i);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 3], i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::tmp1, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                    // print fisrt value of two arguments
                    // 1) store from avx
                    Goldilocks::Element arg1[4];
                    Goldilocks::Element arg2[4];
                    Goldilocks::store_avx(arg1, 1, tmp1[args[i_args + 2]]);
                    Goldilocks::store_avx(arg2, 1, tmp1[args[i_args + 3]]);
                    if (i == 0)
                        std::cout << " _op20: " << arg1[0].fe << " " << arg2[0].fe << std::endl;

                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);

                    // print first value of the result
                    Goldilocks::Element res[4];
                    Goldilocks::store_avx(res, 1, tmp1[args[i_args + 1]]);
                    if (i == 0)
                        std::cout << " _op20: " << res[0].fe << std::endl;
                    i_args += 4;

                    break;
                }
                case 21:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 2], i);
                    int node2 = addVariable(VAR_TYPE::publicInput, args[i_args + 3], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::tmp1, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 22:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim1, args[i_args + 2], i);
                    int node2 = addConstant(args[i_args + 3], numbers[args[i_args + 3]], i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::tmp1, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                    // print fisrt value of two arguments
                    // 1) store from avx
                    Goldilocks::Element arg1[4];
                    Goldilocks::Element arg2[4];
                    Goldilocks::store_avx(arg1, 1, tmp1[args[i_args + 2]]);
                    Goldilocks::store_avx(arg2, 1, numbers_[args[i_args + 3]]);
                    if (i == 0)
                        std::cout << " _op22: " << arg1[0].fe << " " << arg2[0].fe << std::endl;

                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], numbers_[args[i_args + 3]]);

                    // print first value of the result
                    Goldilocks::Element res[4];
                    Goldilocks::store_avx(res, 1, tmp1[args[i_args + 1]]);
                    if (i == 0)
                        std::cout << " _op22: " << res[0].fe << std::endl;
                    i_args += 4;
                    break;
                }
                case 23:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::publicInput, args[i_args + 1], "1", i);
                    addDst(DEST_TYPE::tmp1, args[i_args], node1, i);

                    // COPY public to tmp1
                    Goldilocks::copy_avx(tmp1[args[i_args]], publics[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 24:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::publicInput, args[i_args + 2], "1", i);
                    int node2 = addVariable(VAR_TYPE::publicInput, args[i_args + 3], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::tmp1, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 25:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::publicInput, args[i_args + 2], "1", i);
                    int node2 = addConstant(args[i_args + 3], numbers[args[i_args + 3]], i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::tmp1, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 26:
                {

                    // AST format
                    int node1 = addConstant(args[i_args + 1], numbers[args[i_args + 1]], i);
                    addDst(DEST_TYPE::tmp1, args[i_args], node1, i);

                    // COPY number to tmp1
                    Goldilocks::copy_avx(tmp1[args[i_args]], numbers_[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 27:
                {

                    // AST format
                    int node1 = addConstant(args[i_args + 2], numbers[args[i_args + 2]], i);
                    int node2 = addConstant(args[i_args + 3], numbers[args[i_args + 3]], i);
                    int node3 = addOP(args[i_args], node1, node2, "", i);
                    addDst(DEST_TYPE::tmp1, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], numbers_[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 28:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 3], args[i_args + 4], "3", i);
                    int node2 = addTrace(args[i_args + 5], args[i_args + 6], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], bufferT_[nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]]);
                    i_args += 7;
                    break;
                }
                case 29:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 3], args[i_args + 4], "3", i);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 5], i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp1
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], tmp1[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 30:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 3], args[i_args + 4], "3", i);
                    int node2 = addVariable(VAR_TYPE::publicInput, args[i_args + 5], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: public
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], publics[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 31:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 3], args[i_args + 4], "3", i);
                    int node2 = addConstant(args[i_args + 5], numbers[args[i_args + 5]], i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: number
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], numbers_[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 32:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 3], i);
                    int node2 = addTrace(args[i_args + 4], args[i_args + 5], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
                    // print fisrt value of two arguments
                    // 1) store from avx
                    Goldilocks::Element arg1_1[4];
                    Goldilocks::Element arg1_2[4];
                    Goldilocks::Element arg1_3[4];
                    Goldilocks::Element arg2[4];
                    Goldilocks::store_avx(arg1_1, 1, tmp3[args[i_args + 3]][0]);
                    Goldilocks::store_avx(arg1_2, 1, tmp3[args[i_args + 3]][1]);
                    Goldilocks::store_avx(arg1_3, 1, tmp3[args[i_args + 3]][2]);
                    Goldilocks::store_avx(arg2, 1, bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    if (i == 0)
                        std::cout << " _op32: " << arg1_1[0].fe << " " << arg1_2[0].fe << " " << arg1_3[0].fe << " " << arg2[0].fe << std::endl;
                                        
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    //print result
                    Goldilocks::Element res_1[4];
                    Goldilocks::Element res_2[4];
                    Goldilocks::Element res_3[4];
                    Goldilocks::store_avx(res_1, 1, bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]]);
                    Goldilocks::store_avx(res_2, 1, bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]+1]);
                    Goldilocks::store_avx(res_3, 1, bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]+2]);
                    if (i == 0)
                        std::cout << " _op32: " << res_1[0].fe << " " << res_2[0].fe << " " << res_3[0].fe << std::endl;

                    i_args += 6;
                    break;
                }
                case 33:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 3], i);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 4], i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 34:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 3], i);
                    int node2 = addVariable(VAR_TYPE::publicInput, args[i_args + 4], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: public
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 35:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 3], i);
                    int node2 = addConstant(args[i_args + 4], numbers[args[i_args + 4]], i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: number
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 36:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::challenge, args[i_args + 3], "3", i);
                    int node2 = addTrace(args[i_args + 4], args[i_args + 5], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 37:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::challenge, args[i_args + 3], "3", i);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 4], i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: tmp1
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 38:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::challenge, args[i_args + 3], "3", i);
                    int node2 = addVariable(VAR_TYPE::publicInput, args[i_args + 4], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: public
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 39:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::challenge, args[i_args + 3], "3", i);
                    int node2 = addConstant(args[i_args + 4], numbers[args[i_args + 4]], i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: number
                    Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 40:
                {

                    // AST format
                    assert(0);

                    // COPY commit3 to commit3
                    Goldilocks3::copy_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 41:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 3], args[i_args + 4], "3", i);
                    int node2 = addTrace(args[i_args + 5], args[i_args + 6], "3", i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit3
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]]);
                    i_args += 7;
                    break;
                }
                case 42:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 3], args[i_args + 4], "3", i);
                    int node2 = getNode(TMP_TYPE::dim3, args[i_args + 5], i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], tmp3[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 43:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 3], args[i_args + 4], "3", i);
                    int node2 = addVariable(VAR_TYPE::challenge, args[i_args + 5], "3", i);
                    int node3 = addOP(2, node1, node2, "33", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // MULTIPLICATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::mul_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], challenges[args[i_args + 5]], challenges_ops[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 44:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 3], args[i_args + 4], "3", i);
                    int node2 = addVariable(VAR_TYPE::challenge, args[i_args + 5], "3", i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], challenges[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 45:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2], i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args]] + args[i_args + 1], node1, i);

                    // COPY tmp3 to commit3
                    Goldilocks3::copy_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], tmp3[args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 46:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 3], i);
                    int node2 = getNode(TMP_TYPE::dim3, args[i_args + 4], i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], tmp3[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 47:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 3], i);
                    int node2 = addVariable(VAR_TYPE::challenge, args[i_args + 4], "3", i);
                    int node3 = addOP(2, node1, node2, "33", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // MULTIPLICATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::mul_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 48:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 3], i); 
                    int node2 = addVariable(VAR_TYPE::challenge, args[i_args + 4], "3", i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], challenges[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 49:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::challenge, args[i_args + 3], "3", i);
                    int node2 = addVariable(VAR_TYPE::challenge, args[i_args + 4], "3", i);
                    int node3 = addOP(2, node1, node2, "33", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // MULTIPLICATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::mul_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 50:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::challenge, args[i_args + 3], "3", i);
                    int node2 = addVariable(VAR_TYPE::challenge, args[i_args + 4], "3", i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::expression, nColsStagesAcc[args[i_args + 1]] + args[i_args + 2], node3, i);

                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], challenges[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 51:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 2], args[i_args + 3], "3", i);
                    int node2 = addTrace(args[i_args + 4], args[i_args + 5], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 52:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 2], args[i_args + 3], "3", i);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 4], i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 53:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 2], args[i_args + 3], "3", i);
                    int node2 = addVariable(VAR_TYPE::publicInput, args[i_args + 4], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 54:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 2], args[i_args + 3], "3", i);
                    int node2 = addConstant(args[i_args + 4], numbers[args[i_args + 4]], i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                    //print arguments

                    // 1) store from avx
                    Goldilocks::Element arg1_1[4];
                    Goldilocks::Element arg1_2[4];
                    Goldilocks::Element arg1_3[4];
                    Goldilocks::Element arg2[4];
                    Goldilocks::store_avx(arg1_1, 1,bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    Goldilocks::store_avx(arg1_2, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3] + 1]);
                    Goldilocks::store_avx(arg1_3, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3] + 2]);
                    Goldilocks::store_avx(arg2, 1, numbers_[args[i_args + 4]]);
                    if (i == 0)
                        std::cout << " _op54: (" << arg1_1[0].fe << ", " << arg1_2[0].fe<<", "<<arg1_3[0].fe <<") " << arg2[0].fe << std::endl;

                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], numbers_[args[i_args + 4]]);

                    //print the result
                    Goldilocks::Element res_1[4];
                    Goldilocks::Element res_2[4];
                    Goldilocks::Element res_3[4];

                    Goldilocks::store_avx(res_1, 1, tmp3[args[i_args + 1]][0]);
                    Goldilocks::store_avx(res_2, 1, tmp3[args[i_args + 1]][1]);
                    Goldilocks::store_avx(res_3, 1, tmp3[args[i_args + 1]][2]);
                    if (i == 0)
                        std::cout << " _op54: (" << res_1[0].fe << ", " << res_2[0].fe << ", " << res_3[0].fe << ") " << std::endl;
                    
                    i_args += 5;
                    break;
                }
                case 55:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2], i);
                    int node2 = addTrace(args[i_args + 3], args[i_args + 4], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1

                    //print arguments
                    // 1) store from avx
                    Goldilocks::Element arg1_1[4];
                    Goldilocks::Element arg1_2[4];
                    Goldilocks::Element arg1_3[4];
                    Goldilocks::Element arg2[4];
                    Goldilocks::store_avx(arg1_1, 1, tmp3[args[i_args + 2]][0]);
                    Goldilocks::store_avx(arg1_2, 1, tmp3[args[i_args + 2]][1]);
                    Goldilocks::store_avx(arg1_3, 1, tmp3[args[i_args + 2]][2]);
                    Goldilocks::store_avx(arg2, 1, bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                    if (i == 0)
                        std::cout << " _op55: (" << arg1_1[0].fe << ", " << arg1_2[0].fe << ", " << arg1_3[0].fe << ") " << arg2[0].fe << std::endl;
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);

                    //print the result
                    Goldilocks::Element res_1[4];
                    Goldilocks::Element res_2[4];
                    Goldilocks::Element res_3[4];
                    Goldilocks::store_avx(res_1, 1, tmp3[args[i_args + 1]][0]);
                    Goldilocks::store_avx(res_2, 1, tmp3[args[i_args + 1]][1]);
                    Goldilocks::store_avx(res_3, 1, tmp3[args[i_args + 1]][2]);
                    
                    if (i == 0)
                        std::cout << " _op54: (" << res_1[0].fe << ", " << res_2[0].fe << ", " << res_3[0].fe << ") " << std::endl;
                    

                    i_args += 5;
                    break;
                }
                case 56:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2], i);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 3], i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                    //print arguments
                    // 1) store from avx
                    Goldilocks::Element arg1_1[4];
                    Goldilocks::Element arg1_2[4];
                    Goldilocks::Element arg1_3[4];
                    Goldilocks::Element arg2[4];
                    Goldilocks::store_avx(arg1_1, 1, tmp3[args[i_args + 2]][0]);
                    Goldilocks::store_avx(arg1_2, 1, tmp3[args[i_args + 2]][1]);
                    Goldilocks::store_avx(arg1_3, 1, tmp3[args[i_args + 2]][2]);
                    Goldilocks::store_avx(arg2, 1, tmp1[args[i_args + 3]]);
                    if (i == 0)
                        std::cout << " _op56: (" << arg1_1[0].fe << ", " << arg1_2[0].fe << ", " << arg1_3[0].fe << ") " << arg2[0].fe << std::endl;
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    //print the result
                    Goldilocks::Element res_1[4];
                    Goldilocks::Element res_2[4];
                    Goldilocks::Element res_3[4];
                    Goldilocks::store_avx(res_1, 1, tmp3[args[i_args + 1]][0]);
                    Goldilocks::store_avx(res_2, 1, tmp3[args[i_args + 1]][1]);
                    Goldilocks::store_avx(res_3, 1, tmp3[args[i_args + 1]][2]);
                    if (i == 0)
                        std::cout << " _op56: (" << res_1[0].fe << ", " << res_2[0].fe << ", " << res_3[0].fe << ") " << std::endl;
                    i_args += 4;
                    break;
                }
                case 57:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2], i);
                    int node2 = addVariable(VAR_TYPE::publicInput, args[i_args + 3], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 58:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2], i);
                    int node2 = addConstant(args[i_args + 3], numbers[args[i_args + 3]], i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 59:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::challenge, args[i_args + 2], "3", i);
                    int node2 = addTrace(args[i_args + 3], args[i_args + 4], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 60:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::challenge, args[i_args + 2], "3", i);
                    int node2 = getNode(TMP_TYPE::dim1, args[i_args + 3], i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                    //print arguments
                    // 1) store from avx
                    Goldilocks::Element arg1_1[4];
                    Goldilocks::Element arg1_2[4];
                    Goldilocks::Element arg1_3[4];
                    Goldilocks::Element arg2[4];
                    Goldilocks::store_avx(arg1_1, 1, challenges[args[i_args + 2]][0]);
                    Goldilocks::store_avx(arg1_2, 1, challenges[args[i_args + 2]][1]);
                    Goldilocks::store_avx(arg1_3, 1, challenges[args[i_args + 2]][2]);
                    Goldilocks::store_avx(arg2, 1, tmp1[args[i_args + 3]]);
                    if (i == 0)
                        std::cout << " _op60: (" << arg1_1[0].fe << ", " << arg1_2[0].fe << ", " << arg1_3[0].fe << ") " << arg2[0].fe << std::endl;
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    //print the result
                    Goldilocks::Element res_1[4];
                    Goldilocks::Element res_2[4];
                    Goldilocks::Element res_3[4];
                    Goldilocks::store_avx(res_1, 1, tmp3[args[i_args + 1]][0]);
                    Goldilocks::store_avx(res_2, 1, tmp3[args[i_args + 1]][1]);
                    Goldilocks::store_avx(res_3, 1, tmp3[args[i_args + 1]][2]);
                    if (i == 0)
                        std::cout << " _op60: (" << res_1[0].fe << ", " << res_2[0].fe << ", " << res_3[0].fe << ") " << std::endl;
                    i_args += 4;
                    break;
                }
                case 61:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::challenge, args[i_args + 2], "3", i);
                    int node2 = addVariable(VAR_TYPE::publicInput, args[i_args + 3], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 62:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::challenge, args[i_args + 2], "3", i);
                    int node2 = addConstant(args[i_args + 3], numbers[args[i_args + 3]], i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                    //print arguments
                    // 1) store from avx
                    Goldilocks::Element arg1_1[4];
                    Goldilocks::Element arg1_2[4];
                    Goldilocks::Element arg1_3[4];
                    Goldilocks::Element arg2[4];
                    Goldilocks::store_avx(arg1_1, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    Goldilocks::store_avx(arg1_2, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3] + 1]);
                    Goldilocks::store_avx(arg1_3, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3] + 2]);
                    Goldilocks::store_avx(arg2, 1, numbers_[args[i_args + 3]]);
                    if (i == 0)
                        std::cout << " _op62: (" << arg1_1[0].fe << ", " << arg1_2[0].fe << ", " << arg1_3[0].fe << ") " << arg2[0].fe << std::endl;
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    //print the result
                    Goldilocks::Element res_1[4];
                    Goldilocks::Element res_2[4];
                    Goldilocks::Element res_3[4];
                    Goldilocks::store_avx(res_1, 1, tmp3[args[i_args + 1]][0]);
                    Goldilocks::store_avx(res_2, 1, tmp3[args[i_args + 1]][1]);
                    Goldilocks::store_avx(res_3, 1, tmp3[args[i_args + 1]][2]);
                    if (i == 0)
                        std::cout << " _op62: (" << res_1[0].fe << ", " << res_2[0].fe << ", " << res_3[0].fe << ") " << std::endl;
                    i_args += 4;
                    break;
                }
                case 63:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 1], args[i_args + 2], "3", i);
                    addDst(DEST_TYPE::tmp3, args[i_args], node1, i);

                    // COPY commit3 to tmp3
                    Goldilocks3::copy_avx(tmp3[args[i_args]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 64:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 2], args[i_args + 3], "3", i);
                    int node2 = addTrace(args[i_args + 4], args[i_args + 5], "3", i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                    //print arguments
                    // 1) store from avx
                    Goldilocks::Element arg1_1[4];
                    Goldilocks::Element arg1_2[4];
                    Goldilocks::Element arg1_3[4];
                    Goldilocks::Element arg2_1[4];
                    Goldilocks::Element arg2_2[4];
                    Goldilocks::Element arg2_3[4];
                    Goldilocks::store_avx(arg1_1, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    Goldilocks::store_avx(arg1_2, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3] + 1]);
                    Goldilocks::store_avx(arg1_3, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3] + 2]);
                    Goldilocks::store_avx(arg2_1, 1, bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    Goldilocks::store_avx(arg2_2, 1, bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5] + 1]);
                    Goldilocks::store_avx(arg2_3, 1, bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5] + 2]);
                    if (i == 0)
                        std::cout << " _op64: (" << arg1_1[0].fe << ", " << arg1_2[0].fe << ", " << arg1_3[0].fe << ") (" << arg2_1[0].fe << ", " << arg2_2[0].fe << ", " << arg2_3[0].fe << ") " << std::endl;
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    //print the result
                    Goldilocks::Element res_1[4];
                    Goldilocks::Element res_2[4];
                    Goldilocks::Element res_3[4];
                    Goldilocks::store_avx(res_1, 1, tmp3[args[i_args + 1]][0]);
                    Goldilocks::store_avx(res_2, 1, tmp3[args[i_args + 1]][1]);
                    Goldilocks::store_avx(res_3, 1, tmp3[args[i_args + 1]][2]);
                    if (i == 0)
                        std::cout << " _op64: (" << res_1[0].fe << ", " << res_2[0].fe << ", " << res_3[0].fe << ") " << std::endl;
                    i_args += 6;
                    break;
                }
                case 65:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 2], args[i_args + 3], "3", i);
                    int node2 = getNode(TMP_TYPE::dim3, args[i_args + 4], i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                    //print arguments
                    // 1) store from avx
                    Goldilocks::Element arg1_1[4];
                    Goldilocks::Element arg1_2[4];
                    Goldilocks::Element arg1_3[4];
                    Goldilocks::Element arg2_1[4];
                    Goldilocks::Element arg2_2[4];
                    Goldilocks::Element arg2_3[4];
                    Goldilocks::store_avx(arg1_1, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    Goldilocks::store_avx(arg1_2, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3] + 1]);
                    Goldilocks::store_avx(arg1_3, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3] + 2]);
                    Goldilocks::store_avx(arg2_1, 1, tmp3[args[i_args + 4]][0]);
                    Goldilocks::store_avx(arg2_2, 1, tmp3[args[i_args + 4]][1]);
                    Goldilocks::store_avx(arg2_3, 1, tmp3[args[i_args + 4]][2]);
                    if (i == 0)
                        std::cout << " _op65: (" << arg1_1[0].fe << ", " << arg1_2[0].fe << ", " << arg1_3[0].fe << ") (" << arg2_1[0].fe << ", " << arg2_2[0].fe << ", " << arg2_3[0].fe << ") " << std::endl;
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp3[args[i_args + 4]]);
                    //print the result
                    Goldilocks::Element res_1[4];
                    Goldilocks::Element res_2[4];
                    Goldilocks::Element res_3[4];
                    Goldilocks::store_avx(res_1, 1, tmp3[args[i_args + 1]][0]);
                    Goldilocks::store_avx(res_2, 1, tmp3[args[i_args + 1]][1]);
                    Goldilocks::store_avx(res_3, 1, tmp3[args[i_args + 1]][2]);
                    if (i == 0)
                        std::cout << " _op65: (" << res_1[0].fe << ", " << res_2[0].fe << ", " << res_3[0].fe << ") " << std::endl;
                    i_args += 5;
                    break;
                }
                case 66:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 2], args[i_args + 3], "3", i);
                    int node2 = addVariable(VAR_TYPE::challenge, args[i_args + 4], "3", i);
                    int node3 = addOP(2, node1, node2, "33", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                    //ptint arguments
                    // 1) store from avx
                    Goldilocks::Element arg1_1[4];
                    Goldilocks::Element arg1_2[4];
                    Goldilocks::Element arg1_3[4];
                    Goldilocks::Element arg2_1[4];
                    Goldilocks::Element arg2_2[4];
                    Goldilocks::Element arg2_3[4];
                    Goldilocks::store_avx(arg1_1, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    Goldilocks::store_avx(arg1_2, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3] + 1]);
                    Goldilocks::store_avx(arg1_3, 1, bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3] + 2]);
                    Goldilocks::store_avx(arg2_1, 1, challenges[args[i_args + 4]][0]);
                    Goldilocks::store_avx(arg2_2, 1, challenges[args[i_args + 4]][1]);
                    Goldilocks::store_avx(arg2_3, 1, challenges[args[i_args + 4]][2]);
                    if (i == 0)
                        std::cout << " _op66: (" << arg1_1[0].fe << ", " << arg1_2[0].fe << ", " << arg1_3[0].fe << ") (" << arg2_1[0].fe << ", " << arg2_2[0].fe << ", " << arg2_3[0].fe << ") " << std::endl;
                    Goldilocks3::mul_avx(tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                    //print the result
                    Goldilocks::Element res_1[4];
                    Goldilocks::Element res_2[4];
                    Goldilocks::Element res_3[4];
                    Goldilocks::store_avx(res_1, 1, tmp3[args[i_args + 1]][0]);
                    Goldilocks::store_avx(res_2, 1, tmp3[args[i_args + 1]][1]);
                    Goldilocks::store_avx(res_3, 1, tmp3[args[i_args + 1]][2]);
                    if (i == 0)
                        std::cout << " _op66: (" << res_1[0].fe << ", " << res_2[0].fe << ", " << res_3[0].fe << ") " << std::endl;
                    i_args += 5;
                    break;
                }
                case 67:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 2], args[i_args + 3], "3", i);
                    int node2 = addVariable(VAR_TYPE::challenge, args[i_args + 4], "3", i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], challenges[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 68:
                {
                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 1], i);
                    addDst(DEST_TYPE::tmp3, args[i_args], node1, i);

                    // COPY tmp3 to tmp3
                    Goldilocks3::copy_avx(tmp3[args[i_args]], tmp3[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 69:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2], i);
                    int node2 = getNode(TMP_TYPE::dim3, args[i_args + 3], i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                    //print arguments
                    // 1) store from avx
                    Goldilocks::Element arg1_1[4];
                    Goldilocks::Element arg1_2[4];
                    Goldilocks::Element arg1_3[4];
                    Goldilocks::Element arg2_1[4];
                    Goldilocks::Element arg2_2[4];
                    Goldilocks::Element arg2_3[4];
                    Goldilocks::store_avx(arg1_1, 1, tmp3[args[i_args + 2]][0]);
                    Goldilocks::store_avx(arg1_2, 1, tmp3[args[i_args + 2]][1]);
                    Goldilocks::store_avx(arg1_3, 1, tmp3[args[i_args + 2]][2]);
                    Goldilocks::store_avx(arg2_1, 1, tmp3[args[i_args + 3]][0]);
                    Goldilocks::store_avx(arg2_2, 1, tmp3[args[i_args + 3]][1]);
                    Goldilocks::store_avx(arg2_3, 1, tmp3[args[i_args + 3]][2]);
                    if (i == 0)
                        std::cout << " _op69: (" << arg1_1[0].fe << ", " << arg1_2[0].fe << ", " << arg1_3[0].fe << ") (" << arg2_1[0].fe << ", " << arg2_2[0].fe << ", " << arg2_3[0].fe << ") " << std::endl;
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                    //print the result
                    Goldilocks::Element res_1[4];
                    Goldilocks::Element res_2[4];
                    Goldilocks::Element res_3[4];
                    Goldilocks::store_avx(res_1, 1, tmp3[args[i_args + 1]][0]);
                    Goldilocks::store_avx(res_2, 1, tmp3[args[i_args + 1]][1]);
                    Goldilocks::store_avx(res_3, 1, tmp3[args[i_args + 1]][2]);
                    if (i == 0)
                        std::cout << " _op69: (" << res_1[0].fe << ", " << res_2[0].fe << ", " << res_3[0].fe << ") " << std::endl;
                    i_args += 4;
                    break;
                }
                case 70:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2], i);
                    int node2 = addVariable(VAR_TYPE::challenge, args[i_args + 3], "3", i);
                    int node3 = addOP(2, node1, node2, "33", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                    //print arguments
                    // 1) store from avx
                    Goldilocks::Element arg1_1[4];
                    Goldilocks::Element arg1_2[4];
                    Goldilocks::Element arg1_3[4];
                    Goldilocks::Element arg2_1[4];
                    Goldilocks::Element arg2_2[4];
                    Goldilocks::Element arg2_3[4];
                    Goldilocks::store_avx(arg1_1, 1, tmp3[args[i_args + 2]][0]);
                    Goldilocks::store_avx(arg1_2, 1, tmp3[args[i_args + 2]][1]);
                    Goldilocks::store_avx(arg1_3, 1, tmp3[args[i_args + 2]][2]);
                    Goldilocks::store_avx(arg2_1, 1, challenges[args[i_args + 3]][0]);
                    Goldilocks::store_avx(arg2_2, 1, challenges[args[i_args + 3]][1]);
                    Goldilocks::store_avx(arg2_3, 1, challenges[args[i_args + 3]][2]);
                    if (i == 0)
                        std::cout << " _op70: (" << arg1_1[0].fe << ", " << arg1_2[0].fe << ", " << arg1_3[0].fe << ") (" << arg2_1[0].fe << ", " << arg2_2[0].fe << ", " << arg2_3[0].fe << ") " << std::endl;

                    Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    //print the result
                    Goldilocks::Element res_1[4];
                    Goldilocks::Element res_2[4];
                    Goldilocks::Element res_3[4];
                    Goldilocks::store_avx(res_1, 1, tmp3[args[i_args + 1]][0]);
                    Goldilocks::store_avx(res_2, 1, tmp3[args[i_args + 1]][1]);
                    Goldilocks::store_avx(res_3, 1, tmp3[args[i_args + 1]][2]);
                    if (i == 0)
                        std::cout << " _op70: (" << res_1[0].fe << ", " << res_2[0].fe << ", " << res_3[0].fe << ") " << std::endl;
                    i_args += 4;
                    break;
                }
                case 71:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2], i);
                    int node2 = addVariable(VAR_TYPE::challenge, args[i_args + 3], "3", i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                    //print arguments
                    // 1) store from avx
                    Goldilocks::Element arg1_1[4];
                    Goldilocks::Element arg1_2[4];
                    Goldilocks::Element arg1_3[4];
                    Goldilocks::Element arg2_1[4];
                    Goldilocks::Element arg2_2[4];
                    Goldilocks::Element arg2_3[4];
                    Goldilocks::store_avx(arg1_1, 1, tmp3[args[i_args + 2]][0]);
                    Goldilocks::store_avx(arg1_2, 1, tmp3[args[i_args + 2]][1]);
                    Goldilocks::store_avx(arg1_3, 1, tmp3[args[i_args + 2]][2]);
                    Goldilocks::store_avx(arg2_1, 1, challenges[args[i_args + 3]][0]);
                    Goldilocks::store_avx(arg2_2, 1, challenges[args[i_args + 3]][1]);
                    Goldilocks::store_avx(arg2_3, 1, challenges[args[i_args + 3]][2]);
                    if (i == 0)
                        std::cout << " _op71: (" << arg1_1[0].fe << ", " << arg1_2[0].fe << ", " << arg1_3[0].fe << ") (" << arg2_1[0].fe << ", " << arg2_2[0].fe << ", " << arg2_3[0].fe << ") " << std::endl;
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]]);
                    //print the result
                    Goldilocks::Element res_1[4];
                    Goldilocks::Element res_2[4];
                    Goldilocks::Element res_3[4];
                    Goldilocks::store_avx(res_1, 1, tmp3[args[i_args + 1]][0]);
                    Goldilocks::store_avx(res_2, 1, tmp3[args[i_args + 1]][1]);
                    Goldilocks::store_avx(res_3, 1, tmp3[args[i_args + 1]][2]);
                    if (i == 0)
                        std::cout << " _op71: (" << res_1[0].fe << ", " << res_2[0].fe << ", " << res_3[0].fe << ") " << std::endl;
                    i_args += 4;
                    break;
                }
                case 72:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::challenge, args[i_args + 2], "3", i);
                    int node2 = addVariable(VAR_TYPE::challenge, args[i_args + 3], "3", i);
                    int node3 = addOP(2, node1, node2, "33", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::mul_avx(tmp3[args[i_args + 1]], challenges[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 73:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::challenge, args[i_args + 2], "3", i);
                    int node2 = addVariable(VAR_TYPE::challenge, args[i_args + 3], "3", i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], challenges[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 74:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::eval, args[i_args + 1], "3", i);
                    addDst(DEST_TYPE::tmp3, args[i_args], node1, i);

                    // COPY eval to tmp3
                    Goldilocks3::copy_avx(tmp3[args[i_args]], evals[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 75:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::eval, args[i_args + 2], "3", i);
                    int node2 = addVariable(VAR_TYPE::challenge, args[i_args + 3], "3", i);
                    int node3 = addOP(2, node1, node2, "33", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: eval - SRC1: challenge
                    Goldilocks3::mul_avx(tmp3[args[i_args + 1]], evals[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 76:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::challenge, args[i_args + 2], "3", i);
                    int node2 = addVariable(VAR_TYPE::eval, args[i_args + 3], "3", i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], evals[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 77:
                {

                    // AST format
                    int node1 = getNode(TMP_TYPE::dim3, args[i_args + 2], i);
                    int node2 = addVariable(VAR_TYPE::eval, args[i_args + 3], "3", i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], evals[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 78:
                {

                    // AST format
                    int node1 = addVariable(VAR_TYPE::eval, args[i_args + 2], "3", i);
                    int node2 = addTrace(args[i_args + 3], args[i_args + 4], "1", i);
                    int node3 = addOP(args[i_args], node1, node2, "31", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 79:
                {

                    // AST format
                    int node1 = addTrace(args[i_args + 2], args[i_args + 3], "3", i);
                    int node2 = addVariable(VAR_TYPE::eval, args[i_args + 4], "3", i);
                    int node3 = addOP(args[i_args], node1, node2, "33", i);
                    addDst(DEST_TYPE::tmp3, args[i_args + 1], node3, i);

                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], evals[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                default:
                {
                    std::cout << " Wrong operation!" << std::endl;
                    exit(1);
                }
                }
            }
            storePolinomials(starkInfo, params, bufferT_, storePol, i, nrowsPack, domainExtended);
            if (i_args != parserParams.nArgs)
                std::cout << " " << i_args << " - " << parserParams.nArgs << std::endl;
            assert(i_args == parserParams.nArgs);
        }
        addMetadata(starkInfo, params, parserParams);
        std::ofstream file(fileAST);
        file << outputJSON.dump(2);
        file.close();
    }
};

#endif