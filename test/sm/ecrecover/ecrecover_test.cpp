#include <string>
#include <gmpxx.h>
#include "ecrecover.hpp"
#include "ecrecover_test.hpp"
#include "zklog.hpp"
#include "timer.hpp"
#include <iostream>
#include <cassert>

using namespace std;

struct ECRecoverTestVector
{
    string signature;
    string r;
    string s;
    string v;
    bool precompiled;
    ECRecoverResult result;
    string address;
};
#define NTESTS 47
#define REPETITIONS 1
#define BENCHMARK_MODE 0 // 0: test mode, 1: benchmark mode

#if BENCHMARK_MODE == 1
#undef NTESTS
#define NTESTS 30
#undef REPETITIONS
#define REPETITIONS 1000
#endif

ECRecoverTestVector ecrecoverTestVectors[] = {
    // 0
    {"d9eba16ed0ecae432b71fe008c98cc872bb4cc214d3220a36f365326cf807d68",
     "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
     "265e99e47ad31bb2cab9646c504576b3abc6939a1710afc08cbf3034d73214b8",
     "1c", false, ECR_NO_ERROR,
     "BC44674AD5868F642EAD3FDF94E2D9C9185EAFB7"},
    // 1
    {"d9eba16ed0ecae432b71fe008c98cc872bb4cc214d3220a36f365326cf807d68",
     "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
     "265e99e47ad31bb2cab9646c504576b3abc6939a1710afc08cbf3034d73214b8",
     "1b", false, ECR_NO_ERROR,
     "EE3FEFB38D4E5C7337818F635DEE7609F67CFDB8"},
    // 2
    {"d9eba16ed0ecae432b71fe008c98cc872bb4cc214d3220a36f365326cf807d68",
     "ddd0a7290af9526056b4e35a077b9a11b513aa0028ec6c9880948544508f3c63",
     "265e99e47ad31bb2cab9646c504576b3abc6939a1710afc08cbf3034d73214b8",
     "1c", false, ECR_NO_ERROR,
     "14791697260e4c9a71f18484c9f997b308e59325"},
    // 3
    {"3cc4cb050478c49877188e4fbd022f35ccb41cee02d9d4417194cbf7ebc1eebe",
     "7dff8b06f4914ff0be0e02edf967ce8f13224cb4819e3833b777867db61f8a62",
     "2bcf13b5e4f34a04a77344d5943e0228ba2e787583036325450c889d597a16c4",
     "1b", false, ECR_NO_ERROR,
     "bec80D04A24CD4D811876fF40F31260C339d63C2"},
    // 4
    {"ee43d51baa54831bfd9b03c4b17b59a594378d777e418020c3358f1822cec07d",
     "ea678b4b3ebde1e877401b1bf67fd59e17c0eb75e6da8ecd9cc38620099e0c65",
     "2b954333cfe2b4bf97c4abbcede04a5edda0688b9e965a4bff3d6a45bed2a6e8",
     "1b", false, ECR_NO_ERROR,
     "d571a2180a8647e77adfc109C49bd2137c7a71b5"},
    // 5
    {"fd98b0bfc9eecc813b958263c08db17c932bfe1c2ae7c21a12bb42860d9d9c1f",
     "86b2f6d4bae0e1e1139bc29378fd243d33de35822652144b0b04af346cbf3ad2",
     "5ec4a8672d44b21bdc2203ab8adab6ae53d4c57a2d832090cab710f26600e2fc",
     "1b", false, ECR_NO_ERROR,
     "52855436E41c2671759d54103e71A3d5Fe27439C"},
    // 6
    {"c401c7d7baa568f5928028def2818ec62b70db9b738ab3f4d36326dac17839e5",
     "4a95890f1102c6a6684815f4274d68d64d6deb59e5d524d1a6394256dd188b33",
     "3e53619fe2f4c67ac8ff4103da0740fd7c912be5b176329a74764bd790e42f51",
     "1b", false, ECR_NO_ERROR,
     "A4E24c3ea459D50409dd31Cf1C35A222ADA889fa"},
    // 7
    {"60049b234e7fd86ba5354613066d0d67ccdd993d5322656c56562d48c208002a",
     "f06fbfb021b185a599ec35b0c4a85f5c329abee4f91c4027cb90210c66868e99",
     "1e0e24cab3e119fea57b012b25bfd65f710fe567a53d1a414208f0385f37be1c",
     "1b", false, ECR_NO_ERROR,
     "2302Fdd6dB52D32b84cc7648Fb9d8978a005E382"},
    // 8
    {"fda892e54ef8d0fc5d34f3c78fd9262ebc97c448eec72348340c902ff8f82d86",
     "e9713e52d8ca16c7723ea73ff89244d1c5bfe6611f56d8a2cebcf25f546c0bb3",
     "627566979c5b6e3a5b8e06ba56f22eed367a9b54ed9c3c54d01a29012435fe02",
     "1c", false, ECR_NO_ERROR,
     "Cc665b2CaA43684c99f0EB59751DC1a8C4dBd9Ee"},
    // 9
    {"1df5d6cd09999848b734ef91a35a31840b6f4f7bc42fa4005ef44a3a0da75fd7",
     "8c661a2e0ae1ffe716d34e6d1ceacfd75326da91bf3af225ed675602a00ebf21",
     "3de21452fd054750ae68326d7781ed903b278bb1ac1f52fb8f55e8a4c9954000",
     "1b", false, ECR_NO_ERROR,
     "1F34358a423C2FED839090Dd943A728Dbd711e62"},
    // 10
    {"84254d72d3a17a61a1ff7dbef2cbb063f2913a407a15ee6012168ce0ce5bc500",
     "4937d15fdde73a18520a70b8e8f6fb05e9c172d8b04b02c0246cd5c9fa76ff97",
     "70ac180d31a5336bc902e317c6b5850b151ea7a2cff0595c61d67bf49d1fc210",
     "1b", false, ECR_NO_ERROR,
     "0C470e0895Bbe22A5cFeD06002e29e8ADCEB59E1"},
    // 11
    {"cbfb5477b23d3f14e821799805060a4be3dc7ec51f007ac78b810941a5693723",
     "465e8a566e29f97b29980b458bff07cdeff5e020f38d541c3a980eb4642b4606",
     "531eaeeeec12998af811c435a79a737d3dbb7411ed4d937080b6201a46703bf0",
     "1b", false, ECR_NO_ERROR,
     "6777D222b69b69F41ee3DBFCf6baee332999f347"},
    // 12
    {"f27fb8414a5aaed947f91ff458ea71f6eed0b8949d3da51ad7325047aba2e081",
     "493b8ca6e09c06450508b8eae88bf8103548b71135e1d8721b935a8822f28890",
     "4dea2362974bb205c48e92bd88761b454f2f4e085dfb9beb0706588b6eb852a8",
     "1c", false, ECR_NO_ERROR,
     "1f56A9F3Ca1F284978478D9363D2b345B23B198B"},
    // 13
    {"7d7e073c17eb159ad8372f82f9fa700a77a8a09dbfa7ccc2561504993be9ccca",
     "8c23081e8211029ca641fe88235387a54d875b2fd64df1dbe87fb438316db708",
     "41c8e2f71966a6553c03d9477145366764399a6ad23c2efdabbdf7004d21a7b7",
     "1b", false, ECR_NO_ERROR,
     "6a7C3804E52D9B089767eA5D26232F9096F7B50e"},
    // 14
    {"8aec5831f7b02d4f714878d3729975afd5435e61e738a119ddca2e9777e04183",
     "49b0018d05b6f36d461339633082c8a032543c4f4c5ae6af56ff5e48fd943372",
     "34f096b3ee06d00f330697bddbe5e25d9207e5ccac214eb94ce96ed4507393c1",
     "1b", false, ECR_NO_ERROR,
     "dD6F178Fa6ed5859154301C686C3cd12192CcC0A"},
    // 15
    {"c2c0f5c01b3e244bd9f40da627cfefbc45c53e76f26798cc195de61e3438585f",
     "4a06e1f0012017c68294911be669e953e01c41f97b31e4f6830af4200d4bf253",
     "2eafb33fbb92b512175812972ce89e54e50f28ecf27a051b4f2291da2cf0e1f3",
     "1c", false, ECR_NO_ERROR,
     "5fFd05d8565FA51BB1aE1E4c7b980e1C91d2c939"},
    // 16
    {"1430fc999d7bfa8954983e43f80a27a07eb4845037bc344fc128b453f76877a3",
     "88843aea48fff6790cb6f9657e1650e1386aa73a6bed50e8db2467d06445bed8",
     "779ce085139a5fc49f33431c9d4794a16732d497158570690662ed779fb22ead",
     "1c", false, ECR_NO_ERROR,
     "F329e07AFade0Fb45a599993dF4313DFaB5f80A6"},
    // 17
    {"8bc6619ef176973b22e4a0f0f81a6cd5badf091e452f910de38162b18c2f2ff6",
     "347faebc1fc4c014692ffc7284fa28596d1494c29f29cdc52fa1c8d57ded6a5d",
     "597a18da2e6c3c1eaa56d9611f50bf3d161a05e542afd29cf416b80f82ee5ef9",
     "1c", false, ECR_NO_ERROR,
     "411559B990Fd50F1DF20B4D3780C4185d113A9Ee"},
    // 18
    {"b3dc90532569a73ab8fc3d41d6df176833133be5a697f076392f51001c715006",
     "af4592a0ec76c84bb587b1bdd3b77b1207f7fd572e118f4278a80ec7c89666b9",
     "0cb4bd62edd2d286b883d32006b799532737b5b90ed4a03ce0a6868a2f10edb6",
     "1b", false, ECR_NO_ERROR,
     "B2E1a1FbfdAE0743f4539166309B72BB737716b1"},
    // 19
    {"c7367ffedb65fe022fa06a8c2855c69547cca19ee828d6615e5f7decafccbdc8",
     "b7fd2fa91dd0ba816b37f4ad9b507ad3b32ec2a445544fabb0e231c5aeeefb18",
     "244c004f8b78dc3a3c3db9bda9b39d6e6b11b9397baaf29d0529b7bac9490742",
     "1c", false, ECR_NO_ERROR,
     "95DdaFE17e5550FF8ac7b2EF9D365d067aa7Ae91"},
    // 20
    {"898a015948f4e12fe71bb3f6015410f33d0b67842b551599762b51fecad5577c",
     "a0d427856db72164abc59666e34cd4accb689f1800f71384d797a729c003598b",
     "514d41eca750ebe03d75ed7182d81168af198760e3ac5077c5d88d1f85656039",
     "1c", false, ECR_NO_ERROR,
     "9D39Cfa3F2e98CA109c45F28735436798CBBDEF8"},
    // 21
    {"add47d80decfc50037d37430ccdda08b3d39831e5355e99bfe5e08e00531182e",
     "f74a32e691a8e0ee1b55fa4aeeebfb3eed091108ce03f7ef449846044f228d4a",
     "308af50fef22673ae4a83c5e80c4f316829490b57a561c9f45a1570286d94e41",
     "1b", false, ECR_NO_ERROR,
     "4AB0c3d24dfF7E98e16Edd61269997D089E97c9f"},
    // 22
    {"8fc27577b5290a3ab0d3407969123c719c475a28e687d943f88ac5abadf83420",
     "8b9d9e9c4201b9b2d351b5dde35c0ebefa5446c2dbdd5e30dca02af69d8c29bd",
     "047fb2ba1e4cc0a059ee64aae40d4ba48e6d3265426f602be275d286e45917f6",
     "1b", false, ECR_NO_ERROR,
     "d02c6aAB18d3f40AA994A5B7F3c2be14B34EAB6e"},
    // 23
    {"256e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
     "1b", false, ECR_NO_ERROR,
     "34E325D8023eb901c39747338C587b098fB75dF4"},
    // 24
    {"0000000000000000000000000000000000000000000000000000000000000000",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
     "1b", false, ECR_NO_ERROR,
     "2a558C4cD662E0b74E289d746AEA2f8cf8e54f7c"},
    // 25
    {"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
     "1b", false, ECR_NO_ERROR,
     "2a558C4cD662E0b74E289d746AEA2f8cf8e54f7c"},
    // 26
    {"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
     "1b", false, ECR_NO_ERROR,
     "c41ABa9e06fac6976618820d04D247FfD38f62FF"},
    // 27
    {"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364142",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
     "1b", false, ECR_NO_ERROR,
     "7a343F50dd8fAFa76406F6Ee4dA1796FF9A06109"},
    // 28
    {"0000000000000000000000000000000000000000000000000000000000000001",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
     "1b", false, ECR_NO_ERROR,
     "7a343F50dd8fAFa76406F6Ee4dA1796FF9A06109"},
    // 29
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
     "1b", false, ECR_NO_ERROR,
     "E077fd3C958303e36309B9EE20AE9D3D59817232"},
    // 30 v < 27
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
     "1a", false, ECR_V_INVALID,
     "0000000000000000000000000000000000000000"},
    // 31 v > 28
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
     "1d", false, ECR_V_INVALID,
     "0000000000000000000000000000000000000000"},
    // 32 r == 0
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "0000000000000000000000000000000000000000000000000000000000000000",
     "4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
     "1c", false, ECR_R_IS_ZERO,
     "0000000000000000000000000000000000000000"},
    // 33 r == field
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
     "4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
     "1c", false, ECR_R_IS_TOO_BIG,
     "0000000000000000000000000000000000000000"},
    // 34 r > field
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364142",
     "4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
     "1c", false, ECR_R_IS_TOO_BIG,
     "0000000000000000000000000000000000000000"},
    // 35 r = field - 1, y^2 has no root (invalid)
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140",
     "4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
     "1c", false, ECR_NO_SQRT_Y,
     "0000000000000000000000000000000000000000"},
    // 36 s == 0
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "0000000000000000000000000000000000000000000000000000000000000000",
     "1c", false, ECR_S_IS_ZERO,
     "0000000000000000000000000000000000000000"},
    // 37 s == field/2 + 1. Valid (precompiled)
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a1",
     "1c", true, ECR_NO_ERROR,
     "4ef445CADd8bEe8A02bc79b30A97e6Fe3AE3B7a3"},
    // 38 s == field/2 + 1. Invalid (tx)
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a1",
     "1c", false, ECR_S_IS_TOO_BIG,
     "000000000000000000000000000000000000000"},
    // 39  s == field/2. Valid
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0",
     "1c", false, ECR_NO_ERROR,
     "B29F65aA401660dfa96ecD7eB28134d87E9a618D"},
    // 40 field/2 + 2. Valid (precompiled)
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a2",
     "1c", true, ECR_NO_ERROR,
     "fE706AA7fe3455F29e0F5553D9C780Be3Bd54564"},
    // 41 s == field/2 + 2. Invalid (tx)
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a2",
     "1c", false, ECR_S_IS_TOO_BIG,
     "0000000000000000000000000000000000000000"},
    // 42 s == field
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
     "1c", false, ECR_S_IS_TOO_BIG,
     "0000000000000000000000000000000000000000"},
    // 43 s == field - 1
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140",
     "1c", true, ECR_NO_ERROR,
     "c846e2E4Ab85A761042265B9A8d995345432A60e"},
    // 44 s == field - 1
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140",
     "1c", false, ECR_S_IS_TOO_BIG,
     "0000000000000000000000000000000000000000"},
    // 45 s == field + 1
    {"456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3",
     "9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608",
     "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364142",
     "1c", false, ECR_S_IS_TOO_BIG,
     "0000000000000000000000000000000000000000"},
     // 46 tests jacobian comparison
     {"362C64CAC78B18F0528AA9D9270116CB12F14B1C3920465855FA6EE0FC35BEB8",
     "C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5",
     "4296A072F8ADB4A62ABAE833A1B15350196EB00AE7C5260098628C6C68D95392",
     "1b", false, ECR_NO_ERROR,
     "f39Fd6e51aad88F6F4ce6aB8827279cffFb92266"}};

void ECRecoverTest(void)
{
    TimerStart(ECRECOVER_TEST);

    mpz_class signature, r, s, v, address;

    ECRecoverResult result;
    int failedTests = 0;
    for (int k = 0; k < REPETITIONS; k++)
        for (uint64_t i = 0; i < NTESTS; i++)
        {
            signature.set_str(ecrecoverTestVectors[i].signature, 16);
            r.set_str(ecrecoverTestVectors[i].r, 16);
            s.set_str(ecrecoverTestVectors[i].s, 16);
            v.set_str(ecrecoverTestVectors[i].v, 16);
            address.set_str(ecrecoverTestVectors[i].address, 16);

            result = ECRecover(signature, r, s, v, ecrecoverTestVectors[i].precompiled, address);
#if BENCHMARK_MODE != 1
            bool failed = false;
            if (result != ecrecoverTestVectors[i].result)
            {
                zklog.error("ECRecoverTest() failed i=" + to_string(i) + " signature=" + ecrecoverTestVectors[i].signature + " result=" + to_string(result) + " expectedResult=" + to_string(ecrecoverTestVectors[i].result));
                failed = true;
            }

            if (address != mpz_class(ecrecoverTestVectors[i].address, 16))
            {
                zklog.error("ECRecoverTest() failed i=" + to_string(i) + " signature=" + ecrecoverTestVectors[i].signature + " address=" + address.get_str(16) + " expectedAddress=" + ecrecoverTestVectors[i].address);
                failed = true;
            }
            RawFec::Element buffer[1026];
            int precres = ECRecoverPrecalc(signature, r, s, v, ecrecoverTestVectors[i].precompiled, buffer, 2);
            if(result != ECR_NO_ERROR )
            {   
                if(precres != -1)
                {
                    zklog.error("ECRecoverPrecalc() failed i=" + to_string(i) + " signature=" + ecrecoverTestVectors[i].signature + " result=" + to_string(precres) + " expectedResult= -1" );
                    failed = true;
                }
            }else{
                if(precres < 1)
                {
                    zklog.error("ECRecoverPrecalc() failed i=" + to_string(i) + " signature=" + ecrecoverTestVectors[i].signature + " result=" + to_string(precres) + " expectedResult > 0" );
                    failed = true;
                }
            }
            if (failed)
                failedTests++;
#endif
        }
    TimerStopAndLog(ECRECOVER_TEST);
#if BENCHMARK_MODE == 0
    zklog.info("    Failed ECRECOVER_TEST " + to_string(failedTests));
#else
    assert(result == 0 && failedTests == 0); // to avoid warnings
    uint64_t time_bench = TimeDiff(ECRECOVER_TEST_start, ECRECOVER_TEST_stop) / 1000;
    zklog.info("    Benchmark: " + to_string((double)time_bench / (NTESTS * REPETITIONS)) + " ms per ECRecover()");
#endif
}
