#include "mem_align_test.hpp"
#include "mem_align_executor.hpp"
#include "goldilocks_base_field.hpp"
#include "smt.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "sm/pols_generated/commit_pols.hpp"

using namespace std;

uint64_t compareValue (Goldilocks &fr, uint64_t index, const char* label, CommitPol t[8], mpz_class r)
{
    mpz_class value = 0;
    for (uint8_t i = 0; i < 8; ++i)
    {
        value = (value << 32) + fr.toU64(t[7-i][(index+1) * 64]);
    }
    //cout << label << " on INPUT " << index << " " << r.get_str(16) << " " << value.get_str(16) << endl;
    if (value != r)
    {
        zklog.error("MemAlignSMTest compareValue() DIFF " + string(label) + " on INPUT " + to_string(index) + " " + r.get_str(16) + " " + value.get_str(16));
        return 1;
    }
    return 0;
}

uint64_t MemAlignSMTest (Goldilocks &fr, const Config &config)
{
    uint64_t numberOfErrors = 0;

    zklog.info("MemAlignSMTest starting...");

    Smt smt(fr);
    Database db(fr, config);
    db.init();

    vector<MemAlignAction> input;

    MemAlignAction action;
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.w1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.v.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.offset = 0;
    action.wr256 = 0;    
    input.push_back(action);

    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0",16);
    action.w1.set_str("0",16);
    action.v.set_str("060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021A0A1A2A3A4",16);
    action.offset = 5;
    action.wr256 = 0;    
    input.push_back(action);

    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0",16);
    action.w1.set_str("0",16);
    action.v.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.offset = 0;
    action.wr256 = 0;    
    input.push_back(action);

    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF",16);
    action.w1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.v.set_str("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF",16);
    action.offset = 0;
    action.wr256 = 1;    
    input.push_back(action);

    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.w1.set_str("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF",16);
    action.v.set_str("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF",16);
    action.offset = 32;
    action.wr256 = 1;    
    input.push_back(action);

    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("01020304C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADB",16);
    action.w1.set_str("DCDDDEDFA4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.v.set_str("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF",16);
    action.offset = 4;
    action.wr256 = 1;    
    input.push_back(action);

    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("01C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDE",16);
    action.w1.set_str("DFA1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.v.set_str("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF",16);
    action.offset = 1;
    action.wr256 = 1;    
    input.push_back(action);

    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2000",16);
    action.w1.set_str("00000000000000000000000000000000000000000000000000000000000000BF",16);
    action.v.set_str("0",16);
    action.offset = 31;
    action.wr256 = 1;    
    input.push_back(action);    

    void * pAddress = malloc(CommitPols::pilSize());
    if (pAddress == NULL)
    {
        zklog.error("MemAlignSMTest() failed calling malloc() of size=" + to_string(CommitPols::pilSize()));
        exitProcess();
    }
    CommitPols cmPols(pAddress, CommitPols::pilDegree());

    MemAlignExecutor memAlignExecutor(fr, config);
    memAlignExecutor.execute(input,cmPols.MemAlign);

    mpz_class value;
    for (uint64_t index = 0; index < input.size(); index++)
    {
        numberOfErrors += compareValue (fr, index, "m0", cmPols.MemAlign.m0, input[index].m0);
        numberOfErrors += compareValue (fr, index, "m1", cmPols.MemAlign.m1, input[index].m1);
        numberOfErrors += compareValue (fr, index, "v", cmPols.MemAlign.v, input[index].v);

        if (input[index].wr256)
        {
            numberOfErrors += compareValue (fr, index, "w0", cmPols.MemAlign.w0, input[index].w0);
            numberOfErrors += compareValue (fr, index, "w1", cmPols.MemAlign.w1, input[index].w1);
        }
    }

    free(pAddress);

    zklog.info("MemAlignSMTest done with errors=" + to_string(numberOfErrors));
    return numberOfErrors;
};

