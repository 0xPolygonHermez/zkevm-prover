#include <iostream>
#include "climb_key_test.hpp"
#include "climb_key_executor.hpp"
#include "goldilocks_base_field.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"

using namespace std;
uint64_t ClimbKeySMTest (Goldilocks &fr, const Config &config)
{
    uint64_t numberOfErrors = 0;

    cout << "ClimbKeySMTest starting..." << endl;

    const Goldilocks::Element PGL = fr.fromU64(0xFFFFFFFF00000000ULL);
    const Goldilocks::Element SGL = fr.fromU64(0x7FFFFFFF80000000ULL);
    const Goldilocks::Element P2_18 = fr.fromU64(1ULL << 18);

    vector<ClimbKeyAction> input;
    vector<ClimbKeyAction> output;

    void * pAddress = malloc(CommitPols::pilSize());
    if (pAddress == NULL)
    {
        zklog.error("ClimbKeySMTest::execute() failed calling malloc() of size=" + to_string(CommitPols::pilSize()));
        exitProcess();
    }
    CommitPols cmPols(pAddress, CommitPols::pilDegree());

    for (int i = 0; i < 5; ++i) {
        ClimbKeyAction climbKeyAction;
        for (int j = 0; j < 4; ++j) {
            climbKeyAction.key[j] = fr.zero();
        }
        climbKeyAction.level = i + 3;
        climbKeyAction.bit = i == 0 ? 0 : 1;
        input.push_back(climbKeyAction);
        for (int j = 0; j < 4; ++j) {
            climbKeyAction.key[j] = ((i - 1) == j) ? fr.one() : fr.zero();
        }
        output.push_back(climbKeyAction);
    }

    for (int i = 0; i < 4; ++i) {
        ClimbKeyAction climbKeyAction;
        for (int j = 0; j < 4; ++j) {
            climbKeyAction.key[j] = (i == j) ? SGL : PGL;
        }
        climbKeyAction.level = i;
        climbKeyAction.bit = 0;
        input.push_back(climbKeyAction);
        for (int j = 0; j < 4; ++j) {
            climbKeyAction.key[j] = PGL;
        }
        output.push_back(climbKeyAction);
    }

    const uint64_t values[6] = {0x3FFFFULL, 0x1FFFFULL, 0xFFFFC0000ULL, 0x7FFFC0000ULL, 0x3FF000000000ULL, 0x1FF000000000ULL};
    for (int level = 0; level < 256; ++level) {
        int zlevel = level % 4;
        for (int bit = 0; bit < 2; ++bit) {
            for (int valueIndex = 0; valueIndex < 6; ++valueIndex) {
                Goldilocks::Element fill = fr.zero();
                for (int fillIndex = 0; fillIndex < 2; ++fillIndex) {
                    ClimbKeyAction climbKeyAction;
                    for (int j = 0; j < 4; ++j) {
                        climbKeyAction.key[j] = (zlevel == j) ? fr.fromU64(values[valueIndex]) : fill;
                    }
                    climbKeyAction.level = level;
                    climbKeyAction.bit = bit;
                    input.push_back(climbKeyAction);
                    for (int j = 0; j < 4; ++j) {
                        climbKeyAction.key[j] = (zlevel == j) ? fr.fromU64(values[valueIndex] * 2 + bit) : fill;
                    }
                    output.push_back(climbKeyAction);
                    fill = PGL;
                }
            }
        }
    }

    ClimbKeyExecutor climbKeyExecutor(fr, config);

    climbKeyExecutor.execute(input, cmPols.ClimbKey);

    for (unsigned int inputIndex = 0; inputIndex < input.size(); ++inputIndex) {
        int rclk = inputIndex * 4 + 2;
        if (!fr.equal(cmPols.ClimbKey.key0[rclk], input[inputIndex].key[0]) ||
            !fr.equal(cmPols.ClimbKey.key1[rclk], input[inputIndex].key[1]) ||
            !fr.equal(cmPols.ClimbKey.key2[rclk], input[inputIndex].key[2]) ||
            !fr.equal(cmPols.ClimbKey.key3[rclk], input[inputIndex].key[3])) {
            cout << "Fail input " << inputIndex << " on input key verification" << endl;
            cout << "> INPUT-KEY0 " << fr.toString(cmPols.ClimbKey.key0[rclk], 16) << " vs " << fr.toString(input[inputIndex].key[0], 16) << endl;
            cout << "> INPUT-KEY1 " << fr.toString(cmPols.ClimbKey.key1[rclk], 16) << " vs " << fr.toString(input[inputIndex].key[1], 16) << endl;
            cout << "> INPUT-KEY2 " << fr.toString(cmPols.ClimbKey.key2[rclk], 16) << " vs " << fr.toString(input[inputIndex].key[2], 16) << endl;
            cout << "> INPUT-KEY3 " << fr.toString(cmPols.ClimbKey.key3[rclk], 16) << " vs " << fr.toString(input[inputIndex].key[3], 16) << endl;
            ++numberOfErrors;
        }
        if (!fr.equal(cmPols.ClimbKey.key0[rclk+1], output[inputIndex].key[0]) ||
            !fr.equal(cmPols.ClimbKey.key1[rclk+1], output[inputIndex].key[1]) ||
            !fr.equal(cmPols.ClimbKey.key2[rclk+1], output[inputIndex].key[2]) ||
            !fr.equal(cmPols.ClimbKey.key3[rclk+1], output[inputIndex].key[3])) {
            cout << "Fail input " << inputIndex << " on output key verification" << endl;
            cout << "> OUTPUT-KEY0 " << fr.toString(cmPols.ClimbKey.key0[rclk+1], 16) << " vs " << fr.toString(output[inputIndex].key[0], 16) << endl;
            cout << "> OUTPUT-KEY1 " << fr.toString(cmPols.ClimbKey.key1[rclk+1], 16) << " vs " << fr.toString(output[inputIndex].key[1], 16) << endl;
            cout << "> OUTPUT-KEY2 " << fr.toString(cmPols.ClimbKey.key2[rclk+1], 16) << " vs " << fr.toString(output[inputIndex].key[2], 16) << endl;
            cout << "> OUTPUT-KEY3 " << fr.toString(cmPols.ClimbKey.key3[rclk+1], 16) << " vs " << fr.toString(output[inputIndex].key[3], 16) << endl;
            ++numberOfErrors;
        }
        uint64_t _level = fr.toU64(cmPols.ClimbKey.level[rclk+1]);
        if (_level != input[inputIndex].level) {
            cout << "Fail input " << inputIndex << " not match level (" << _level << " vs " <<  input[inputIndex].level << ") verification" << endl;
            ++numberOfErrors;
         }

        uint64_t _bit = fr.toU64(cmPols.ClimbKey.bit[rclk]);
        if (_bit != input[inputIndex].bit) {
            cout << "Fail input " << inputIndex << " not match bit (" << _bit << " vs " <<  input[inputIndex].bit << ") verification" << endl;
            ++numberOfErrors;
        }

        // constraints verification


        Goldilocks::Element _keyIn = fr.zero();
        Goldilocks::Element _factor = fr.one();
        for (int j = 0; j < 4; ++j) {
            int clk = inputIndex * 4 + j;
            int zlevel = input[inputIndex].level % 4;
            if (!fr.equal(cmPols.ClimbKey.keySel0[clk], (j == 3 && zlevel == 0) ? fr.one() : fr.zero())) {
                cout << "Fail input " << inputIndex << " keySel0 with invalid value (" << fr.toString(cmPols.ClimbKey.keySel0[clk]) << ") on clock " << j << endl;
                ++numberOfErrors;
            }
            if (!fr.equal(cmPols.ClimbKey.keySel1[clk], (j == 3 && zlevel == 1) ? fr.one() : fr.zero())) {
                cout << "Fail input " << inputIndex << " keySel1 with invalid value (" << fr.toString(cmPols.ClimbKey.keySel1[clk]) << ") on clock " << j << endl;
                ++numberOfErrors;
            }
            if (!fr.equal(cmPols.ClimbKey.keySel2[clk], (j == 3 && zlevel == 2) ? fr.one() : fr.zero())) {
                cout << "Fail input " << inputIndex << " keykeySel2Sel0 with invalid value (" << fr.toString(cmPols.ClimbKey.keySel2[clk]) << ") on clock " << j << endl;
                ++numberOfErrors;
            }
            if (!fr.equal(cmPols.ClimbKey.keySel3[clk], (j == 3 && zlevel == 3) ? fr.one() : fr.zero())) {
                cout << "Fail input " << inputIndex << " keySel3 with invalid value (" << fr.toString(cmPols.ClimbKey.keySel3[clk]) << ") on clock " << j << endl;
                ++numberOfErrors;
            }
            if (j != 2 && !fr.isZero(cmPols.ClimbKey.result[clk])) {
                cout << "Fail input " << inputIndex << " result not zero on clock " << j << endl;
                ++numberOfErrors;
            }

            uint64_t _bit = fr.toU64(cmPols.ClimbKey.bit[clk]);
            if (_bit != input[inputIndex].bit) {
                cout << "Fail input " << inputIndex << " latch not match bit (" << _bit << " vs " <<  input[inputIndex].bit << ")" << endl;
                ++numberOfErrors;
            }

            _keyIn = fr.add(_keyIn, fr.mul(cmPols.ClimbKey.keyInChunk[clk], _factor));
            if (!fr.equal(cmPols.ClimbKey.keyIn[clk], _keyIn)) {
                cout << "Fail input " << inputIndex << " keyIn with invalid value (0x" << fr.toString(cmPols.ClimbKey.keyIn[clk], 16) << " vs 0x" << fr.toString(_keyIn, 16) << ") on clock " << j << endl;
                cout << "> keyInChunk 0x" << fr.toString(cmPols.ClimbKey.keyInChunk[clk], 16) << " input key 0x" << fr.toString(input[inputIndex].key[zlevel], 16) << " _factor 0x" << fr.toString(_factor, 16) << endl;
                ++numberOfErrors;
            }
            _factor = fr.mul(_factor, P2_18);

            if (clk == 3 && !fr.equal(cmPols.ClimbKey.keyIn[clk], input[inputIndex].key[zlevel])) {
                cout << "Fail input " << inputIndex << " keyIn not match with input key (0x" << fr.toString(cmPols.ClimbKey.keyIn[clk], 16) << " vs 0x" <<  fr.toString(input[inputIndex].key[zlevel], 16) << endl;
                ++numberOfErrors;
            }

            if (j >= 2) continue;

            if (!fr.equal(cmPols.ClimbKey.key0[clk], input[inputIndex].key[0]) ||
                !fr.equal(cmPols.ClimbKey.key1[clk], input[inputIndex].key[1]) ||
                !fr.equal(cmPols.ClimbKey.key2[clk], input[inputIndex].key[2]) ||
                !fr.equal(cmPols.ClimbKey.key3[clk], input[inputIndex].key[3])) {
                cout << "Fail input " << inputIndex << " on input key latch clock " << j << endl;
                cout << "> INPUT-KEY0 " << fr.toString(cmPols.ClimbKey.key0[clk], 16) << " vs " << fr.toString(input[inputIndex].key[0], 16) << endl;
                cout << "> INPUT-KEY1 " << fr.toString(cmPols.ClimbKey.key1[clk], 16) << " vs " << fr.toString(input[inputIndex].key[1], 16) << endl;
                cout << "> INPUT-KEY2 " << fr.toString(cmPols.ClimbKey.key2[clk], 16) << " vs " << fr.toString(input[inputIndex].key[2], 16) << endl;
                cout << "> INPUT-KEY3 " << fr.toString(cmPols.ClimbKey.key3[clk], 16) << " vs " << fr.toString(input[inputIndex].key[3], 16) << endl;
                ++numberOfErrors;
            }
        }
    }
    free(pAddress);
    cout << "ClimbKeySMTest done" << endl;
    return numberOfErrors;
}
