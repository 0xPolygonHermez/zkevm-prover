#include <stdint.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "rlp.hpp"
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
    bool isOk;

    ctx.v = 27;
    ctx.r.set_str("1186622d03b6b8da7cf111d1ccba5bb185c56deae6a322cebc6dda0556f3cb97", 16);
    ctx.s.set_str("00910c26408b64b51c5da36ba2f38ef55ba1cee719d5a6c01225968799907432", 16);

    from = "";
    to = "0x4d5Cf5032B2a844602278b01199ED191A86c93ff";
    data = "0x";
    res = getTransactionHash(from, to, 0x016345785d8a0000, 0, 0x0186a0, 0x3b9aca00, data, 1001);
    cout << "res: " << res << endl;

    expected = "0x0cc3dd49b941271b19df83ba6733bed4023fb82c28d40e6ef863ca589d4a933a";
    isOk = (strcasecmp(res.c_str(), expected.c_str()) == 0);
    cout << "TEST1 " << ( isOk ? "OK" : "FAIL") << endl;
    cout << "res: " << res << endl;
    if (!isOk) {
        cout << "exp: " << expected << endl;
    }
    cout << endl;

    ctx.v = 27;
    ctx.r.set_str("1186622d03b6b8da7cf111d1ccba5bb185c56deae6a322cebc6dda0556f3cb97", 16);
    ctx.s.set_str("00910c26408b64b51c5da36ba2f38ef55ba1cee719d5a6c01225968799907432", 16);

    from = "";
    to = "0x4d5Cf5032B2a844602278b01199ED191A86c93ff";
    data = "0x";
    res = getTransactionHash(from, to, 0x016345785d8a0000, 0, 0x0186a0, 0x3b9aca00, data, 1001);
    cout << "res: " << res << endl;

    expected = "0x0cc3dd49b941271b19df83ba6733bed4023fb82c28d40e6ef863ca589d4a933a";
    isOk = (strcasecmp(res.c_str(), expected.c_str()) == 0);
    cout << "TEST2 " << ( isOk ? "OK" : "FAIL") << endl;
    cout << "res: " << res << endl;
    if (!isOk) {
        cout << "exp: " << expected << endl;
    }
    cout << endl;

    ctx.v = 27;
    ctx.r.set_str("b22577e12dd4e9c8c7c9f68a63d6ab3ec6fb612ee6b206dbcd7817257d816e75", 16);
    ctx.s.set_str("2e69f816289c68eab050ddc279714bd05dc567ec3fc883148a0338334f3d5e77", 16);

    from = "";
    to = "0x1275fbb540c8efC58b812ba83B0D0B8b9917AE98";
    data = "0x15927819";
    res = getTransactionHash(from, to, 0, 0, 0x0186a0, 0x3b9aca00, data, 1001);
    cout << "res: " << res << endl;

    expected = "0xe73241ad9d642066483b1c85439d1f3fe77f854b2c61292e31c185bdefe21308";
    isOk = (strcasecmp(res.c_str(), expected.c_str()) == 0);
    cout << "TEST3 " << ( isOk ? "OK" : "FAIL") << endl;
    cout << "res: " << res << endl;
    if (!isOk) {
        cout << "exp: " << expected << endl;
    }
    cout << endl;

    ctx.v = 0x25;
    ctx.r.set_str("044852b2a670ade5407e78fb2863c51de9fcb96542a07186fe3aeda6bb8a116d", 16);
    ctx.s.set_str("044852b2a670ade5407e78fb2863c51de9fcb96542a07186fe3aeda6bb8a116d", 16);

    from = "";
    to = "0x3535353535353535353535353535353535353535";
    data = "";
    res = getTransactionHash(from, to, 0, 0, 0x5208, 0x04a817c800, data, 1);

    expected = "0xe0be81f8d506dbe3a5549e720b51eb79492378d6638087740824f168667e5239";
    isOk = (strcasecmp(res.c_str(), expected.c_str()) == 0);
    cout << "TEST4 " << ( isOk ? "OK" : "FAIL") << endl;
    cout << "res: " << res << endl;
    if (!isOk) {
        cout << "exp: " << expected << endl;
    }
    cout << endl;

    ctx.v = 0x25;
    ctx.r.set_str("64b1702d9298fee62dfeccc57d322a463ad55ca201256d01f62b45b2e1c21c12", 16);
    ctx.s.set_str("64b1702d9298fee62dfeccc57d322a463ad55ca201256d01f62b45b2e1c21c12", 16);

    from = "";
    to = "0x3535353535353535353535353535353535353535";
    data = "";
    res = getTransactionHash(from, to, 0x0200, 0x08, 0x02e248, 0x04a817c808, data, 1);

    expected = "0x50b6e7b58320c885ab7b2ee0d0b5813a697268bd2494a06de792790b13668c08";
    isOk = (strcasecmp(res.c_str(), expected.c_str()) == 0);
    cout << "TEST5 " << ( isOk ? "OK" : "FAIL") << endl;
    cout << "res: " << res << endl;
    if (!isOk) {
        cout << "exp: " << expected << endl;
    }
    cout << endl;
}
