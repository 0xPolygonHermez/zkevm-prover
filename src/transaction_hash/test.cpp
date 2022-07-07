#include <stdint.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "getTransactionHash.hpp"

using namespace std;
/*
    {
      batchL2Data: "0xea80843b9aca00830186a0941275fbb540c8efc58b812ba83b0d0b8b9917ae988084159278198203e98080b22577e12dd4e9c8c7c9f68a63d6ab3ec6fb612ee6b206dbcd7817257d816e752e69f816289c68eab050ddc279714bd05dc567ec3fc883148a0338334f3d5e771b",
      signedTx: "0xf86a80843b9aca00830186a0941275fbb540c8efc58b812ba83b0d0b8b9917ae988084159278198207f5a0b22577e12dd4e9c8c7c9f68a63d6ab3ec6fb612ee6b206dbcd7817257d816e75a02e69f816289c68eab050ddc279714bd05dc567ec3fc883148a0338334f3d5e77",
      txHash: "0xe73241ad9d642066483b1c85439d1f3fe77f854b2c61292e31c185bdefe21308",
      nonce: 0,
      gasPrice: "0x3b9aca00",
      gasLimit: "0x0186a0",
      to: "0x1275fbb540c8efC58b812ba83B0D0B8b9917AE98",
      value: "0x00",
      data: "0x15927819",
      chainId: 1001,
      // "v": 2037,
      v: 27,
      r: "0xb22577e12dd4e9c8c7c9f68a63d6ab3ec6fb612ee6b206dbcd7817257d816e75",
      s: "0x2e69f816289c68eab050ddc279714bd05dc567ec3fc883148a0338334f3d5e77"
    }
*/




void testGetTransactionHash(void)
{
    Context_ ctx;
    string from, to, data, res, expected;
    stringstream hex;

    ctx.v = 27;
    ctx.r.set_str("1186622d03b6b8da7cf111d1ccba5bb185c56deae6a322cebc6dda0556f3cb97", 16);
    ctx.s.set_str("00910c26408b64b51c5da36ba2f38ef55ba1cee719d5a6c01225968799907432", 16);

    from = "";
    to = "0x4d5Cf5032B2a844602278b01199ED191A86c93ff";
    data = "0x";
    res = getTransactionHash(ctx, from, to, 0x016345785d8a0000, 0, 0x0186a0, 0x3b9aca00, data, 1001);

    hex.str("");
    hex << "0x";
    for (string::size_type i = 0; i < res.length(); ++i)
        hex << std::hex << std::setfill('0') << std::setw(2) << (int)((unsigned char *)res.c_str())[i];

    cout << hex.str() << endl;
    expected = "0xee80843b9aca00830186a0944d5cf5032b2a844602278b01199ed191a86c93ff88016345785d8a0000808203e98080";
    expected = "0xf86d80843b9aca00830186a0944d5cf5032b2a844602278b01199ed191a86c93ff88016345785d8a0000808207f5a01186622d03b6b8da7cf111d1ccba5bb185c56deae6a322cebc6dda0556f3cb979f910c26408b64b51c5da36ba2f38ef55ba1cee719d5a6c01225968799907432";
    cout << expected << endl;
    cout << "TEST1 " << (hex.str() == expected ? "OK" : "FAIL") << endl << endl;

    ctx.v = 27;
    ctx.r.set_str("1186622d03b6b8da7cf111d1ccba5bb185c56deae6a322cebc6dda0556f3cb97", 16);
    ctx.s.set_str("00910c26408b64b51c5da36ba2f38ef55ba1cee719d5a6c01225968799907432", 16);

    from = "";
    to = "0x4d5Cf5032B2a844602278b01199ED191A86c93ff";
    data = "0x";
    res = getTransactionHash(ctx, from, to, 0x016345785d8a0000, 0, 0x0186a0, 0x3b9aca00, data, 1001);

    hex.str("");
    hex << "0x";
    for (string::size_type i = 0; i < res.length(); ++i)
        hex << std::hex << std::setfill('0') << std::setw(2) << (int)((unsigned char *)res.c_str())[i];

    cout << hex.str() << endl;
    expected = "0xee80843b9aca00830186a0944d5cf5032b2a844602278b01199ed191a86c93ff88016345785d8a0000808203e98080";
    expected = "0xf86d80843b9aca00830186a0944d5cf5032b2a844602278b01199ed191a86c93ff88016345785d8a0000808207f5a01186622d03b6b8da7cf111d1ccba5bb185c56deae6a322cebc6dda0556f3cb979f910c26408b64b51c5da36ba2f38ef55ba1cee719d5a6c01225968799907432";
    cout << expected << endl;
    cout << "TEST2 " << (hex.str() == expected ? "OK" : "FAIL") << endl << endl;


    ctx.v = 27;
    ctx.r.set_str("b22577e12dd4e9c8c7c9f68a63d6ab3ec6fb612ee6b206dbcd7817257d816e75", 16);
    ctx.s.set_str("2e69f816289c68eab050ddc279714bd05dc567ec3fc883148a0338334f3d5e77", 16);

    from = "";
    to = "0x1275fbb540c8efC58b812ba83B0D0B8b9917AE98";
    data = "0x15927819";
    res = getTransactionHash(ctx, from, to, 0, 0, 0x0186a0, 0x3b9aca00, data, 1001);

    hex.str("");
    hex << "0x";
    for (string::size_type i = 0; i < res.length(); ++i)
        hex << std::hex << std::setfill('0') << std::setw(2) << (int)((unsigned char *)res.c_str())[i];

    cout << hex.str() << endl;
    expected = "0xea80843b9aca00830186a0941275fbb540c8efc58b812ba83b0d0b8b9917ae988084159278198203e98080";
    expected = "0xf86a80843b9aca00830186a0941275fbb540c8efc58b812ba83b0d0b8b9917ae988084159278198207f5a0b22577e12dd4e9c8c7c9f68a63d6ab3ec6fb612ee6b206dbcd7817257d816e75a02e69f816289c68eab050ddc279714bd05dc567ec3fc883148a0338334f3d5e77";
    cout << expected << endl;
    cout << "TEST3 " << (hex.str() == expected ? "OK" : "FAIL") << endl;


    /*
        "blocknumber": "3500000",
        "hash": "e0be81f8d506dbe3a5549e720b51eb79492378d6638087740824f168667e5239",
        "rlp": "0xf864808504a817c800825208943535353535353535353535353535353535353535808025a0044852b2a670ade5407e78fb2863c51de9fcb96542a07186fe3aeda6bb8a116da0044852b2a670ade5407e78fb2863c51de9fcb96542a07186fe3aeda6bb8a116d",
        "sender": "f0f6f18bca1b28cd68e4357452947e021241e9ce",
        "transaction": {
        "data": "",
        "gasLimit": "0x5208",
        "gasPrice": "0x04a817c800",
        "nonce": "0x",
        "r": "0x044852b2a670ade5407e78fb2863c51de9fcb96542a07186fe3aeda6bb8a116d",
        "s": "0x044852b2a670ade5407e78fb2863c51de9fcb96542a07186fe3aeda6bb8a116d",
        "to": "0x3535353535353535353535353535353535353535",
        "v": "0x25",
        "value": "0x00"
    */
    ctx.v = 0x25;
    ctx.r.set_str("044852b2a670ade5407e78fb2863c51de9fcb96542a07186fe3aeda6bb8a116d", 16);
    ctx.s.set_str("044852b2a670ade5407e78fb2863c51de9fcb96542a07186fe3aeda6bb8a116d", 16);

    from = "";
    to = "0x3535353535353535353535353535353535353535";
    data = "";
    res = getTransactionHash(ctx, from, to, 0, 0, 0x5208, 0x04a817c800, data, 1001);

    hex.str("");
    hex << "0x";
    for (string::size_type i = 0; i < res.length(); ++i)
        hex << std::hex << std::setfill('0') << std::setw(2) << (int)((unsigned char *)res.c_str())[i];

    cout << hex.str() << endl;
    expected = "0xe0be81f8d506dbe3a5549e720b51eb79492378d6638087740824f168667e5239";
    cout << expected << endl;
    cout << "TEST4 " << (hex.str() == expected ? "OK" : "FAIL") << endl;

    cout << endl;
}
