#include "mem_align_test.hpp"
#include "mem_align_executor.hpp"
#include "goldilocks_base_field.hpp"
#include "smt.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "sm/pols_generated/commit_pols.hpp"

using namespace std;

mpz_class getValue (Goldilocks &fr, uint64_t index, CommitPol t[8])
{
    mpz_class value = 0;
    for (uint8_t i = 0; i < 8; ++i)
    {
        value = (value << 32) + fr.toU64(t[7-i][index * 32 + 31]);
    }
    return value;
}

uint64_t compareValue (Goldilocks &fr, uint64_t index, const char* label, CommitPol t[8], mpz_class r)
{
    mpz_class value = 0;
    for (uint8_t i = 0; i < 8; ++i)
    {
        value = (value << 32) + fr.toU64(t[7-i][index * 32 + 31]);
    }
    //cout << label << " on INPUT " << index << " " << r.get_str(16) << " " << value.get_str(16) << endl;
    if (value != r)
    {
        zklog.error("MemAlignSMTest compareValue() DIFF " + string(label) + " on INPUT " + to_string(index) + " " + r.get_str(16) + " " + value.get_str(16));
        return 1;
    }
    return 0;
}

// CommitPol inM[2];
// CommitPol inV[2];
// CommitPol inV_V;
// CommitPol wr;
// CommitPol mode;
// CommitPol m0[8];
// CommitPol m1[8];
// CommitPol w0[8];
// CommitPol w1[8];
// CommitPol v[8];
// CommitPol selM0;
// CommitPol selM1;
// CommitPol bytePos;
// CommitPol result;
// CommitPol selV;


uint64_t checkRange(const string &label, uint64_t value, uint64_t fromValue, uint64_t toValue, uint64_t index) 
{
    if (value < fromValue || value > toValue) {
        zklog.error("MemAlignSMTest invalid range "+label+"("+to_string(fromValue)+","+to_string(toValue)+")="+to_string(value)
                    +" (expected) on input #" + to_string(index/32) + "/" + to_string(index % 32));
        return 1;
    }
    return 0;
}

uint64_t checkEquals(const string &label, uint64_t value, uint64_t expectedValue, uint64_t index) 
{
    if (value != expectedValue) {
        zklog.error("MemAlignSMTest not equal value "+label+" "+to_string(value)+" vs "+to_string(expectedValue)
                    +" (expected) on input #" + to_string(index/32) + "/" + to_string(index % 32));
        return 1;
    }
    return 0;
}

uint64_t verifyAction (Goldilocks &fr, uint64_t baseIndex, MemAlignAction &input, MemAlignCommitPols &pols)
{
    mpz_class value = 0;
    uint64_t errorCount = 0;
    uint64_t _M0[32],_M1[32],_W0[32],_W1[32],_V[32];
    mpz_class m0 = getValue(fr, baseIndex, pols.m0);
    mpz_class m1 = getValue(fr, baseIndex, pols.m1);
    mpz_class w0 = getValue(fr, baseIndex, pols.w0);
    mpz_class w1 = getValue(fr, baseIndex, pols.w1);
    mpz_class v = getValue(fr, baseIndex, pols.v);

    for (uint8_t i = 0; i < 32; ++i) 
    {
        uint32_t bits = 8 * (31 - i);
        value = (m0 >> bits) & 0xFF;
        _M0[i] = value.get_ui();
        value = (m1 >> bits) & 0xFF;
        _M1[i] = value.get_ui();
        value = (w0 >> bits) & 0xFF;
        _W0[i] = value.get_ui();
        value = (w1 >> bits) & 0xFF;
        _W1[i] = value.get_ui();
        value = (v >> bits) & 0xFF;
        _V[i] = value.get_ui();
    }
    for (uint8_t i = 0; i < 32; ++i)
    {
        uint32_t index = baseIndex * 32 + i;
        uint64_t inV_0 = fr.toU64(pols.inV[0][index]);
        uint64_t inV_1 = fr.toU64(pols.inV[1][index]);
        uint64_t inV = inV_0 + inV_1 * 64;
        uint64_t inV_V = fr.toU64(pols.inV_V[index]);
        uint64_t selM0 = fr.toU64(pols.selM0[index]);
        uint64_t selM1 = fr.toU64(pols.selM1[index]);
        uint64_t selV = fr.toU64(pols.selV[index]);
        uint64_t bytePos = fr.toU64(pols.bytePos[index]);
        uint64_t result = fr.toU64(pols.result[index]);
        uint64_t wr = fr.toU64(pols.wr[index]);
        uint64_t mode = fr.toU64(pols.mode[index]);
        uint64_t inM0 = fr.toU64(pols.inM[0][index]);
        uint64_t inM1 = fr.toU64(pols.inM[1][index]);
        uint64_t inW0 = selM0 ? inV : inM0;
        uint64_t inW1 = selM1 ? inV : inM1;
        errorCount += checkRange("result", result, 0, 1, index);
        errorCount += checkRange("wr", wr, 0, 1, index);
        errorCount += checkRange("inV[0]", inV_0, 0, 255, index);
        errorCount += checkRange("inV[1]", inV_1, 0, 255, index);
        errorCount += checkRange("inV", inV, 0, 255, index);
        errorCount += checkRange("inM0", inM0, 0, 255, index);
        errorCount += checkRange("inM1", inM1, 0, 255, index);
        errorCount += checkRange("bytePos", bytePos, 0, 31, index);
        errorCount += checkRange("selM0", selM0, 0, 1, index);
        errorCount += checkRange("selM1", selM1, 0, 1, index);
        errorCount += checkRange("mode", mode, 0, 64 + 64 * 128 + 8192 + 16384, index);
                
        // TODO: check mode
        errorCount += checkEquals("wr", mode, input.mode, index);
        errorCount += checkEquals("mode", wr, input.wr, index);
        errorCount += checkEquals("result", result, i == 31 ? 1:0, index);
        errorCount += checkEquals("inM0", inM0, _M0[i], index);
        errorCount += checkEquals("inM1", inM1, _M1[i], index);
        errorCount += checkEquals("inV", inV_V, _V[i], index);
        errorCount += checkEquals("inW0", inW0, _W0[i], index);
        errorCount += checkEquals("inW1", inW1, _W1[i], index);
        // { ID, bytePos, selV * inV_M } is { ID, Global.STEP32, inV_V };
        errorCount += checkEquals("inV_V/inV_M", selV ? inV: 0 , fr.toU64(pols.inV_V[baseIndex * 32 + bytePos]), index);
    }

    return errorCount;
}

uint64_t mainExecutorFreeInCheck(mpz_class m0, mpz_class m1, uint64_t mode, mpz_class v, uint64_t index) 
{
    uint64_t offset = mode & 0x7F;
    uint64_t len = (mode >> 7) & 0x3F;
    bool leftAlignment = mode & 0x2000;
    bool littleEndian = mode & 0x4000;

    if (offset>64 || len > 32 || mode > 0x7FFFF)
    {
        zklog.error("MemAlign out of range mode="+to_string(mode)+" offset=" + to_string(offset)+" len="+to_string(len)+ " on input #" + to_string(index));
        return 1;
    }
    uint64_t _len = len == 0 ? 32 : len;
    if ((_len + offset) > 64) 
    {
        _len = 64 - offset;
    }
    mpz_class m = (m0 << 256) | m1;
    // mpz_class maskV = ScalarMask256 >> (32 - _len);
    // mpz_class maskV = ScalarMask256 >> (32 - _len);
    mpz_class maskV = ScalarMask256 >> (8 * (32 - _len));
    uint64_t shiftBits = (64 - offset - _len) * 8;
    if (shiftBits > 0) 
    {
        m = m >> shiftBits;
    }
    mpz_class _v = m & maskV;
    if (littleEndian) 
    {
        // reverse bytes
        mpz_class _tmpv = 0;
        for (int ilen = 0; ilen < _len; ++ilen) 
        {
            _tmpv = (_tmpv << 8) | (_v & 0xFF);
            _v = _v >> 8;
        }
        _v = _tmpv;
    }
    if (leftAlignment && _len < 32) 
    {
        _v = _v << ((32 - len) * 8);
    }
    if (_v != v) {
        zklog.error("MemAlign freeIn calculation fails v="+_v.get_str(16)+" expectedV=" + v.get_str(16)+ " on input #" + to_string(index));
        return 1;
    }
    return 0;
}


uint64_t mainExecutorCheck(mpz_class m0, mpz_class m1, uint64_t mode, mpz_class v, bool wr, mpz_class w0, mpz_class w1, uint64_t index) 
{
    uint64_t offset = mode & 0x7F;
    uint64_t len = (mode >> 7) & 0x3F;
    bool leftAlignment = mode & 0x2000;
    bool littleEndian = mode & 0x4000;

    if (offset>64 || len > 32 || mode > 0x7FFFF)
    {
        zklog.error("MemAlign out of range mode="+to_string(mode)+" offset=" + to_string(offset)+" len="+to_string(len)+ 
                    " on input #" + to_string(index));
        return 1;
    }
    uint64_t _len = len == 0 ? 32 : len;
    if ((_len + offset) > 64) 
    {
        _len = 64 - offset;
    }
    mpz_class m = (m0 << 256) | m1;
//    mpz_class maskV = ScalarMask256 >> (32 - _len);
    mpz_class maskV = ScalarMask256 >> (8 * (32 - _len));
    uint64_t shiftBits = (64 - offset - _len) * 8;

    if (wr)
    {
        mpz_class _v = v;
        if (leftAlignment && _len < 32) 
        {
            _v = _v >> (8* (32 - len));
        }
        _v = _v & maskV;
        if (littleEndian) 
        {
            // reverse bytes
            mpz_class _tmpv = 0;
            for (int ilen = 0; ilen < _len; ++ilen) 
            {
                _tmpv = (_tmpv << 8) | (_v & 0xFF);
                _v = _v >> 8;
            }
            _v = _tmpv;
        }
        mpz_class _W = (m & (ScalarMask512 ^ (maskV << shiftBits))) | (_v << shiftBits);

        mpz_class _W0 = _W >> 256;
        mpz_class _W1 = _W & ScalarMask256;
        if ( (w0 != _W0) || (w1 != _W1) )
        {
            zklog.error("MemAlign verification calculation fails w0="+_W0.get_str(16)+" expectedw0=" + w0.get_str(16)+" w1="+_W0.get_str(16)+
                        " expectedw0=" + w0.get_str(16)+ " on input #" + to_string(index));
            return 1;
        }
        return 0;
    }

    if (shiftBits > 0) 
    {
        m = m >> shiftBits;
    }
    mpz_class _v = m & maskV;
    if (littleEndian)
    {
        // reverse bytes
        mpz_class _tmpv = 0;
        for (int ilen = 0; ilen < _len; ++ilen) 
        {
        _tmpv = (_tmpv << 8) | (_v & 0xFF);
            _v = _v >> 8;
        }
        _v = _tmpv;
    }
    if (leftAlignment && _len < 32)
    {
        _v = _v << ((32 - len) * 8);
    }
    if (v != _v)
    {
        zklog.error("MemAlign verification calculation fails v="+_v.get_str(16)+" expectedV=" + _v.get_str(16)+ 
                    " on input #" + to_string(index));
        return 1;
    }
    return 0;
}

uint64_t helperTestMemAlignWR_W0 (mpz_class m0, mpz_class value, uint64_t mode, mpz_class w0, uint64_t index)
{
    uint64_t offset = mode & 0x7F;
    uint64_t len = (mode >> 7) & 0x3F;
    bool leftAlignment = mode & 0x2000;
    bool littleEndian = mode & 0x4000;

    if (offset>64 || len > 32 || mode > 0x7FFFF)
    {
        zklog.error("MemAlign helperTestMemAlignWR_W0 fails, invalid mode="+to_string(mode) + " on input #" + to_string(index));
        return 1;
    }
    uint64_t _len = len == 0 ? 32 : len;
    if ((_len + offset) > 64) 
    {
        _len = 64 - offset;
    }
    // mpz_class maskV = ScalarMask256 >> (32 - _len);
    mpz_class maskV = ScalarMask256 >> (8*(32 - _len));
    uint64_t shiftBits = (64 - offset - _len) * 8;

    if (leftAlignment && _len < 32) 
    {
        value = value >> (8* (32 - len));
    }
    value = value & maskV;
    if (littleEndian) 
    {
        // reverse bytes
        mpz_class _tmpv = 0;
        for (uint64_t ilen = 0; ilen < _len; ++ilen) 
        {
            _tmpv = (_tmpv << 8) | (value & 0xFF);
            value = value >> 8;
        }
        value = _tmpv;
    }
    mpz_class _w0 = ((m0 << 256 & (ScalarMask512 ^ (maskV << shiftBits))) | (value << shiftBits)) >> 256;
    if (w0 != _w0)
    {
        zklog.error("MemAlign helperTestMemAlignWR_W0 fails w0="+_w0.get_str(16)+" expectedW0=" + w0.get_str(16)+ 
                    " on input #" + to_string(index));
        return 1;
    }
    return 0;
}

uint64_t helperTestMemAlignWR_W1 (mpz_class m1, mpz_class value, uint64_t mode, mpz_class w1, uint64_t index)
{
    uint64_t offset = mode & 0x7F;
    uint64_t len = (mode >> 7) & 0x3F;
    bool leftAlignment = mode & 0x2000;
    bool littleEndian = mode & 0x4000;

    if (offset>64 || len > 32 || mode > 0x7FFFF)
    {
        zklog.error("MemAlign helperTestMemAlignWR_W1 fails, invalid mode="+to_string(mode) + " on input #" + to_string(index));
        return 1;
    }
    uint64_t _len = len == 0 ? 32 : len;
    if ((_len + offset) > 64) 
    {
        _len = 64 - offset;
    }
    // mpz_class maskV = ScalarMask256 >> (32 - _len);
    mpz_class maskV = ScalarMask256 >> (8*(32 - _len));
    uint64_t shiftBits = (64 - offset - _len) * 8;

    if (leftAlignment && _len < 32) 
    {
        value = value >> (8* (32 - len));
    }
    value = value & maskV;
    if (littleEndian) 
    {
        // reverse bytes
        mpz_class _tmpv = 0;
        for (uint64_t ilen = 0; ilen < _len; ++ilen) 
        {
            _tmpv = (_tmpv << 8) | (value & 0xFF);
            value = value >> 8;
        }
        value = _tmpv;
    }
    mpz_class _w1 = ((m1 & (ScalarMask512 ^ (maskV << shiftBits))) | (value << shiftBits)) &  ScalarMask256;
    if (w1 != _w1)
    {
        zklog.error("MemAlign helperTestMemAlignWR_W1 fails w1="+_w1.get_str(16)+" expectedW1=" + w1.get_str(16)+ 
                    " on input #" + to_string(index));
        return 1;
    }
    return 0;
}

uint64_t helperTestMemAlign_RD (mpz_class m0, mpz_class m1, uint64_t mode, mpz_class value, uint64_t index)
{
    uint64_t offset = mode & 0x7F;
    uint64_t len = (mode >> 7) & 0x3F;
    bool leftAlignment = mode & 0x2000;
    bool littleEndian = mode & 0x4000;

    if (offset>64 || len > 32 || mode > 0x7FFFF)
    {
        zklog.error("MemAlign helperTestMemAlignRD fails, invalid mode="+to_string(mode) + " on input #" + to_string(index));
        return 1;
    }
    uint64_t _len = len == 0 ? 32 : len;
    if ((_len + offset) > 64) 
    {
        _len = 64 - offset;
    }
    mpz_class m = (m0 << 256) | m1;
    // mpz_class maskV = ScalarMask256 >> (32 - _len);
    mpz_class maskV = ScalarMask256 >> (8 * (32 - _len));
    uint64_t shiftBits = (64 - offset - _len) * 8;
    if (shiftBits > 0) 
    {
        m = m >> shiftBits;
    }
    mpz_class _v = m & maskV;
    if (littleEndian) 
    {
        // reverse bytes
        mpz_class _tmpv = 0;
        for (uint64_t ilen = 0; ilen < _len; ++ilen) 
        {
            _tmpv = (_tmpv << 8) | (_v & 0xFF);
            _v = _v >> 8;
        }
        _v = _tmpv;
    }
    if (leftAlignment && _len < 32) 
    {
        _v = _v << ((32 - len) * 8);
    }
    if (value != _v)
    {
        zklog.error("MemAlign helperTestMemAlignRD fails value="+_v.get_str(16)+" expectedValue=" + value.get_str(16)+ 
                    " on input #" + to_string(index));
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

    // #0] 32 bytes, offset = 0
    MemAlignAction action;
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.w1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.v.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.mode = 0;
    action.wr = 0;    
    input.push_back(action);

    // #1] 32 bytes, offset = 0, little endian
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.w1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.v.set_str("21201E1D1C1B1A191817161514131211100F0E0D0C0B0A090807060504030201",16);
    action.mode = 0 + 16384;
    action.wr = 0;    
    input.push_back(action);

    // #2] 32 bytes, offset = 0, left alignment
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.w1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.v.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.mode = 0 + 8192;
    action.wr = 0;    
    input.push_back(action);

    // #3] 32 bytes, offset = 0, left alignment + little endian
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.w1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.v.set_str("21201E1D1C1B1A191817161514131211100F0E0D0C0B0A090807060504030201",16);
    action.mode = 0 + 8192 + 16384;
    action.wr = 0;    
    input.push_back(action);

    // #4] 32 bytes, offset = 5
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0",16);
    action.w1.set_str("0",16);
    action.v.set_str("060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021A0A1A2A3A4",16);
    action.mode = 5;
    action.wr = 0;    
    input.push_back(action);

    // #5] 32 bytes, offset = 5, left alignment
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0",16);
    action.w1.set_str("0",16);
    action.v.set_str("060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021A0A1A2A3A4",16);
    action.mode = 5 + 8192;
    action.wr = 0;    
    input.push_back(action);

    // #6] 32 bytes, offset = 5, litle endian
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0",16);
    action.w1.set_str("0",16);
    action.v.set_str("A4A3A2A1A021201E1D1C1B1A191817161514131211100F0E0D0C0B0A09080706",16);
    action.mode = 5 + 16384;
    action.wr = 0;    
    input.push_back(action);

    // #7] 32 bytes, offset = 5, left alignment + litle endian
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0",16);
    action.w1.set_str("0",16);
    action.v.set_str("A4A3A2A1A021201E1D1C1B1A191817161514131211100F0E0D0C0B0A09080706",16);
    action.mode = 5 + 8192 + 16384;
    action.wr = 0;    
    input.push_back(action);

    // #8] 30 bytes, offset = 5
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0",16);
    action.w1.set_str("0",16);
    action.v.set_str("0000060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021A0A1A2",16);
    action.mode = 5 + 30 * 128;
    action.wr = 0;    
    input.push_back(action);

    // #9] 30 bytes, offset = 5, left alignment
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0",16);
    action.w1.set_str("0",16);
    action.v.set_str("060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021A0A1A20000",16);
    action.mode = 5 + 30 * 128 + 8192;
    action.wr = 0;    
    input.push_back(action);

    // #10] 30 bytes, offset = 5, litle endian
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0",16);
    action.w1.set_str("0",16);
    action.v.set_str("0000A2A1A021201E1D1C1B1A191817161514131211100F0E0D0C0B0A09080706",16);
    action.mode = 5 + 30 * 128 + 16384;
    action.wr = 0;    
    input.push_back(action);

    // #11] 30 bytes, offset = 5, left alignment + litle endian
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0",16);
    action.w1.set_str("0",16);
    action.v.set_str("A2A1A021201E1D1C1B1A191817161514131211100F0E0D0C0B0A090807060000",16);
    action.mode = 5 +  30 * 128 + 8192 + 16384;
    action.wr = 0;    
    input.push_back(action);

    // #12] write 30 bytes, offset = 5
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("010203040533445566778899AABBCCDDEEFFE000E1E2E3E4E5E6E7E8E9EAEBEC",16);
    action.w1.set_str("EDEEEFA3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.v.set_str("112233445566778899AABBCCDDEEFFE000E1E2E3E4E5E6E7E8E9EAEBECEDEEEF",16);
    action.mode = 5 + 30 * 128;
    action.wr = 1;    
    input.push_back(action);

    // #13] write 30 bytes, offset = 5, left alignment
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0102030405112233445566778899AABBCCDDEEFFE000E1E2E3E4E5E6E7E8E9EA",16);
    action.w1.set_str("EBECEDA3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.v.set_str("112233445566778899AABBCCDDEEFFE000E1E2E3E4E5E6E7E8E9EAEBECEDEEEF",16);
    action.mode = 5 + 30 * 128 + 8192;
    action.wr = 1;    
    input.push_back(action);

    // #14] write 30 bytes, offset = 5, litle endian
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0102030405EFEEEDECEBEAE9E8E7E6E5E4E3E2E100E0FFEEDDCCBBAA99887766",16);
    action.w1.set_str("554433A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    // EFEEEDECEBEAE9E8E7E6E5E4E3E2E100E0FFEEDDCCBBAA998877665544332211
    action.v.set_str("112233445566778899AABBCCDDEEFFE000E1E2E3E4E5E6E7E8E9EAEBECEDEEEF",16);
    action.mode = 5 + 30 * 128 + 16384;
    action.wr = 1;    
    input.push_back(action);

    // #15] write 30 bytes, offset = 5, left alignment + litle endian
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0102030405EDECEBEAE9E8E7E6E5E4E3E2E100E0FFEEDDCCBBAA998877665544",16);
    action.w1.set_str("332211A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    // EFEEEDECEBEAE9E8E7E6E5E4E3E2E100E0FFEEDDCCBBAA998877665544332211
    action.v.set_str("112233445566778899AABBCCDDEEFFE000E1E2E3E4E5E6E7E8E9EAEBECEDEEEF",16);
    action.mode = 5 +  30 * 128 + 8192 + 16384;
    action.wr = 1;    
    input.push_back(action);

    // #16] 
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0",16);
    action.w1.set_str("0",16);
    action.v.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.mode = 0;
    action.wr = 0;    
    input.push_back(action);

    // #9]
    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF",16);
    action.w1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.v.set_str("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF",16);
    action.mode = 0;
    action.wr = 1;    
    input.push_back(action);

    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.w1.set_str("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF",16);
    action.v.set_str("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF",16);
    action.mode = 32;
    action.wr = 1;    
    input.push_back(action);

    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("01020304C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADB",16);
    action.w1.set_str("DCDDDEDFA4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.v.set_str("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF",16);
    action.mode = 4;
    action.wr = 1;    
    input.push_back(action);

    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("01C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDE",16);
    action.w1.set_str("DFA1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.v.set_str("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF",16);
    action.mode = 1;
    action.wr = 1;    
    input.push_back(action);

    action.m0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2021",16);
    action.m1.set_str("A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF",16);
    action.w0.set_str("0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E2000",16);
    action.w1.set_str("00000000000000000000000000000000000000000000000000000000000000BF",16);
    action.v.set_str("0",16);
    action.mode = 31;
    action.wr = 1;    
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
// uint64_t mainExecutorFreeInCheck(mpz_class m0, mpz_class m1, uint64_t mode, mpz_class v, uint64_t index);
// uint64_t mainExecutorCheck(mpz_class m0, mpz_class m1, uint64_t mode, mpz_class v, bool wr, mpz_class w0, mpz_class w1, uint64_t index);

    for (uint64_t index = 0; index < input.size(); index++)
    {
        numberOfErrors += compareValue (fr, index, "m0", cmPols.MemAlign.m0, input[index].m0);
        numberOfErrors += compareValue (fr, index, "m1", cmPols.MemAlign.m1, input[index].m1);
        numberOfErrors += compareValue (fr, index, "v", cmPols.MemAlign.v, input[index].v);

        if (input[index].wr)
        {
            numberOfErrors += compareValue (fr, index, "w0", cmPols.MemAlign.w0, input[index].w0);
            numberOfErrors += compareValue (fr, index, "w1", cmPols.MemAlign.w1, input[index].w1);
            numberOfErrors += helperTestMemAlignWR_W0 (input[index].m0, input[index].v, input[index].mode, input[index].w0, index);
            numberOfErrors += helperTestMemAlignWR_W1 (input[index].m1, input[index].v, input[index].mode, input[index].w1, index);
        } 
        else 
        {
            numberOfErrors += helperTestMemAlign_RD (input[index].m0, input[index].m1, input[index].mode, input[index].v, index);
            numberOfErrors += mainExecutorFreeInCheck(input[index].m0, input[index].m1, input[index].mode, input[index].v, index);                   
        }
        numberOfErrors += verifyAction (fr, index, input[index], cmPols.MemAlign);        
        numberOfErrors += mainExecutorCheck(input[index].m0, input[index].m1, input[index].mode, input[index].v, input[index].wr, input[index].w0, input[index].w1, index);
    }

    free(pAddress);

    zklog.info("MemAlignSMTest done with errors=" + to_string(numberOfErrors));
    return numberOfErrors;
};

