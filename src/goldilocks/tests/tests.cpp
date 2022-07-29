#include <gtest/gtest.h>
#include <iostream>

#include "../src/goldilocks_base_field.hpp"
#include "../src/goldilocks_cubic_extension.hpp"
#include "../src/poseidon_goldilocks.hpp"
#include "../src/ntt_goldilocks.hpp"

#define GOLDILOCKS_PRIME 0xFFFFFFFF00000001ULL

#define FFT_SIZE (1 << 4)
#define NUM_REPS 5
#define BLOWUP_FACTOR 1
#define NUM_COLUMNS 8
#define NPHASES 4

TEST(GOLDILOCKS_TEST, one)
{
    uint64_t a = 1;
    uint64_t b = 1 + GOLDILOCKS_PRIME;
    std::string c = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1

    Goldilocks::Element ina1 = Goldilocks::fromU64(a);
    Goldilocks::Element ina2 = Goldilocks::fromS32(a);
    Goldilocks::Element ina3 = Goldilocks::fromString(std::to_string(a));
    Goldilocks::Element inb1 = Goldilocks::fromU64(b);
    Goldilocks::Element inc1 = Goldilocks::fromString(c);

    ASSERT_EQ(Goldilocks::toU64(ina1), a);
    ASSERT_EQ(Goldilocks::toU64(ina2), a);
    ASSERT_EQ(Goldilocks::toU64(ina3), a);
    ASSERT_EQ(Goldilocks::toU64(inb1), a);
    ASSERT_EQ(Goldilocks::toU64(inc1), a);
}

TEST(GOLDILOCKS_TEST, add)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element p_1 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    ASSERT_EQ(Goldilocks::toU64(p_1 + p_1), 2);

    Goldilocks::Element max = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFF);
    ASSERT_EQ(Goldilocks::toU64(max + max), 0X1FFFFFFFC);

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);

    ASSERT_EQ(Goldilocks::toU64(inE1 + inE2), in1 + in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 + inE2 + inE3), in1 + in2 + 1);
    ASSERT_EQ(Goldilocks::toU64(inE1 + inE2 + inE3 + inE4), 1);

    // Edge case (double carry)
    Goldilocks::Element a1 = Goldilocks::fromU64(Goldilocks::from_montgomery(0xFFFFFFFF00000000));
    Goldilocks::Element a2 = Goldilocks::fromU64(Goldilocks::from_montgomery(0xFFFFFFFF));
    Goldilocks::Element b1 = (a1 + a2);
    Goldilocks::Element b2 = (b1 + b1);
    ASSERT_EQ(Goldilocks::toU64(b2), 0x200000002);
}

TEST(GOLDILOCKS_TEST, sub)
{

    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);

    ASSERT_EQ(Goldilocks::toU64(inE1 - inE2), GOLDILOCKS_PRIME + in1 - in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 - inE2 - inE3), GOLDILOCKS_PRIME + in1 - in2 - 1);
    ASSERT_EQ(Goldilocks::toU64(inE1 - inE2 - inE3 - inE4), 5);

    Goldilocks::Element a1 = Goldilocks::fromU64(Goldilocks::from_montgomery(0xFFFFFFFF00000000LL));
    Goldilocks::Element a2 = Goldilocks::fromU64(Goldilocks::from_montgomery(0xFFFFFFFFLL));

    Goldilocks::Element a3 = (a1 + a2);
    Goldilocks::Element b2 = Goldilocks::zero() - a3;
    ASSERT_EQ(Goldilocks::toU64(b2), Goldilocks::from_montgomery(0XFFFFFFFE00000003LL));
}

TEST(GOLDILOCKS_TEST, mul)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);

    ASSERT_EQ(Goldilocks::toU64(inE1 * inE2), in1 * in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 * inE2 * inE3), in1 * in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 * inE2 * inE3 * inE4), 0XFFFFFFFEFFFFFEBDLL);
}

TEST(GOLDILOCKS_TEST, div)
{
    uint64_t in1 = 10;
    int32_t in2 = 5;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element inE5 = Goldilocks::fromS32(in4);
    Goldilocks::Element inE6 = Goldilocks::fromS32(in4);

    ASSERT_EQ(Goldilocks::toU64(inE1 / inE2), in1 / in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 / inE2 / inE3), in1 / in2); // 10 / 2 / ( 0 + 1 ) = 10 / 2
    ASSERT_EQ(Goldilocks::toU64(inE1 / inE2 / inE3 / inE4), 0X2AAAAAAA80000000);
    ASSERT_EQ(Goldilocks::toU64(inE5 / inE6), 1);
    ASSERT_EQ(Goldilocks::toU64(Goldilocks::one() / inE6), 0X1555555540000000);
}

TEST(GOLDILOCKS_TEST, inv)
{
    uint64_t in1 = 5;
    std::string in2 = "18446744069414584326"; // 0xFFFFFFFF00000001n + 5n

    Goldilocks::Element input1 = Goldilocks::one();
    Goldilocks::Element inv1 = Goldilocks::inv(input1);
    Goldilocks::Element res1 = input1 * inv1;

    Goldilocks::Element input5 = Goldilocks::fromU64(in1);
    Goldilocks::Element inv5 = Goldilocks::inv(input5);
    Goldilocks::Element res5 = input5 * inv5;

    ASSERT_EQ(res1, Goldilocks::one());
    ASSERT_EQ(res5, Goldilocks::one());

    Goldilocks::Element inE1 = Goldilocks::fromString(std::to_string(in1));
    Goldilocks::Element inE1_plus_p = Goldilocks::fromString(in2);

    ASSERT_EQ(Goldilocks::inv(inE1_plus_p) * inE1, Goldilocks::one());
    ASSERT_EQ(Goldilocks::inv(inE1), Goldilocks::inv(inE1_plus_p));
}

TEST(GOLDILOCKS_TEST, poseidon_full)
{

    Goldilocks::Element fibonacci[SPONGE_WIDTH];
    Goldilocks::Element result[SPONGE_WIDTH];

    fibonacci[0] = Goldilocks::zero();
    fibonacci[1] = Goldilocks::one();

    for (uint64_t i = 2; i < SPONGE_WIDTH; i++)
    {
        fibonacci[i] = fibonacci[i - 1] + fibonacci[i - 2];
    }

    PoseidonGoldilocks::hash_full_result(result, fibonacci);

    ASSERT_EQ(Goldilocks::toU64(result[0]), 0X3095570037F4605D);
    ASSERT_EQ(Goldilocks::toU64(result[1]), 0X3D561B5EF1BC8B58);
    ASSERT_EQ(Goldilocks::toU64(result[2]), 0X8129DB5EC75C3226);
    ASSERT_EQ(Goldilocks::toU64(result[3]), 0X8EC2B67AFB6B87ED);
    ASSERT_EQ(Goldilocks::toU64(result[4]), 0XFC591F17D0FAB161);
    ASSERT_EQ(Goldilocks::toU64(result[5]), 0X1D2B045CC2FEA1AD);
    ASSERT_EQ(Goldilocks::toU64(result[6]), 0X8A4E3B0CB12D4527);
    ASSERT_EQ(Goldilocks::toU64(result[7]), 0XFF217A756AE2211);
    ASSERT_EQ(Goldilocks::toU64(result[8]), 0X78F6E79CFC407293);
    ASSERT_EQ(Goldilocks::toU64(result[9]), 0X3DE827E086AE61C9);
    ASSERT_EQ(Goldilocks::toU64(result[10]), 0X921456F6D2D11E27);
    ASSERT_EQ(Goldilocks::toU64(result[11]), 0XF58A41D4028C66A5);

    Goldilocks::Element zero[SPONGE_WIDTH] = {Goldilocks::zero()};
    Goldilocks::Element result0[SPONGE_WIDTH];

    PoseidonGoldilocks::hash_full_result(result0, zero);

    ASSERT_EQ(Goldilocks::toU64(result0[0]), 0X3C18A9786CB0B359);
    ASSERT_EQ(Goldilocks::toU64(result0[1]), 0XC4055E3364A246C3);
    ASSERT_EQ(Goldilocks::toU64(result0[2]), 0X7953DB0AB48808F4);
    ASSERT_EQ(Goldilocks::toU64(result0[3]), 0XC71603F33A1144CA);
    ASSERT_EQ(Goldilocks::toU64(result0[4]), 0XD7709673896996DC);
    ASSERT_EQ(Goldilocks::toU64(result0[5]), 0X46A84E87642F44ED);
    ASSERT_EQ(Goldilocks::toU64(result0[6]), 0XD032648251EE0B3C);
    ASSERT_EQ(Goldilocks::toU64(result0[7]), 0X1C687363B207DF62);
    ASSERT_EQ(Goldilocks::toU64(result0[8]), 0XDF8565563E8045FE);
    ASSERT_EQ(Goldilocks::toU64(result0[9]), 0X40F5B37FF4254DAE);
    ASSERT_EQ(Goldilocks::toU64(result0[10]), 0XD070F637B431067C);
    ASSERT_EQ(Goldilocks::toU64(result0[11]), 0X1792B1C4342109D7);
}

TEST(GOLDILOCKS_TEST, poseidon)
{

    Goldilocks::Element fibonacci[SPONGE_WIDTH];
    Goldilocks::Element result[CAPACITY];

    fibonacci[0] = Goldilocks::zero();
    fibonacci[1] = Goldilocks::one();

    for (uint64_t i = 2; i < SPONGE_WIDTH; i++)
    {
        fibonacci[i] = fibonacci[i - 1] + fibonacci[i - 2];
    }

    PoseidonGoldilocks::hash(result, fibonacci);

    ASSERT_EQ(Goldilocks::toU64(result[0]), 0X3095570037F4605D);
    ASSERT_EQ(Goldilocks::toU64(result[1]), 0X3D561B5EF1BC8B58);
    ASSERT_EQ(Goldilocks::toU64(result[2]), 0X8129DB5EC75C3226);
    ASSERT_EQ(Goldilocks::toU64(result[3]), 0X8EC2B67AFB6B87ED);

    Goldilocks::Element zero[SPONGE_WIDTH] = {Goldilocks::zero()};
    Goldilocks::Element result0[CAPACITY];

    PoseidonGoldilocks::hash(result0, zero);

    ASSERT_EQ(Goldilocks::toU64(result0[0]), 0X3C18A9786CB0B359);
    ASSERT_EQ(Goldilocks::toU64(result0[1]), 0XC4055E3364A246C3);
    ASSERT_EQ(Goldilocks::toU64(result0[2]), 0X7953DB0AB48808F4);
    ASSERT_EQ(Goldilocks::toU64(result0[3]), 0XC71603F33A1144CA);
}

TEST(GOLDILOCKS_TEST, ntt)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc(FFT_SIZE * sizeof(Goldilocks::Element));
    Goldilocks::Element *initial = (Goldilocks::Element *)malloc(FFT_SIZE * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE);

    a[0] = Goldilocks::one();
    a[1] = Goldilocks::one();
    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        a[i] = a[i - 1] + a[i - 2];
    }

    std::memcpy(initial, a, FFT_SIZE * sizeof(Goldilocks::Element));

    for (int i = 0; i < NUM_REPS; i++)
    {
        gntt.NTT(a, a, FFT_SIZE);
        gntt.INTT(a, a, FFT_SIZE);
    }

    for (int i = 0; i < FFT_SIZE; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(initial[i]));
    }
    free(a);
    free(initial);
}

TEST(GOLDILOCKS_TEST, ntt_block)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc(FFT_SIZE * NUM_COLUMNS * sizeof(Goldilocks::Element));
    Goldilocks::Element *initial = (Goldilocks::Element *)malloc(FFT_SIZE * NUM_COLUMNS * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE);

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            Goldilocks::add(a[i * NUM_COLUMNS + j], Goldilocks::one(), Goldilocks::fromU64(j));
        }
    }

    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * (i - 1) + j] + a[NUM_COLUMNS * (i - 2) + j];
        }
    }

    std::memcpy(initial, a, FFT_SIZE * NUM_COLUMNS * sizeof(Goldilocks::Element));

    // Option 1: dst is a NULL pointer
    for (int i = 0; i < NUM_REPS; i++)
    {
        gntt.NTT(NULL, a, FFT_SIZE, NUM_COLUMNS);
        gntt.INTT(NULL, a, FFT_SIZE, NUM_COLUMNS);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(initial[i]));
    }

    // Option 2: dst = src
    for (int i = 0; i < NUM_REPS; i++)
    {
        gntt.NTT(a, a, FFT_SIZE, NUM_COLUMNS);
        gntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(initial[i]));
    }

    // Option 3: dst != src
    Goldilocks::Element *dst = (Goldilocks::Element *)malloc(FFT_SIZE * NUM_COLUMNS * sizeof(Goldilocks::Element));
    for (int i = 0; i < NUM_REPS; i++)
    {
        gntt.NTT(dst, a, FFT_SIZE, NUM_COLUMNS);
        for (uint64_t k = 0; k < FFT_SIZE * NUM_COLUMNS; ++k)
            a[k] = Goldilocks::zero();
        gntt.INTT(a, dst, FFT_SIZE, NUM_COLUMNS);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(initial[i]));
    }

    // Option 4: different configurations of phases and blocks
    for (int i = 0; i < NUM_REPS; i++)
    {
        gntt.NTT(a, a, FFT_SIZE, NUM_COLUMNS, 3, 5);
        gntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS, 4, 3);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(initial[i]));
    }

    // Option 5: out of range parameters
    for (int i = 0; i < NUM_REPS; i++)
    {
        gntt.NTT(a, a, FFT_SIZE, NUM_COLUMNS, 3, 3000);
        gntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS, 4, -1);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(initial[i]));
    }
    free(a);
    free(initial);
}

TEST(GOLDILOCKS_TEST, LDE)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc((FFT_SIZE << BLOWUP_FACTOR) * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE);
    NTT_Goldilocks gntt_extension((FFT_SIZE << BLOWUP_FACTOR));

    Goldilocks::Element *zeros_array = (Goldilocks::Element *)malloc(((FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE) * sizeof(Goldilocks::Element));
#pragma omp parallel for
    for (uint i = 0; i < ((FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE); i++)
    {
        zeros_array[i] = Goldilocks::zero();
    }

    a[0] = Goldilocks::one();
    a[1] = Goldilocks::one();
    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        a[i] = a[i - 1] + a[i - 2];
    }

    Goldilocks::Element shift = Goldilocks::fromU64(49); // TODO: ask for this number, where to put it how to calculate it
    gntt.INTT(a, a, FFT_SIZE);

    // TODO: This can be pre-generated
    Goldilocks::Element *r = (Goldilocks::Element *)malloc(FFT_SIZE * sizeof(Goldilocks::Element));
    r[0] = Goldilocks::one();
    for (int i = 1; i < FFT_SIZE; i++)
    {
        r[i] = r[i - 1] * shift;
    }

#pragma omp parallel for
    for (int i = 0; i < FFT_SIZE; i++)
    {
        a[i] = a[i] * r[i];
    }

    std::memcpy(&a[FFT_SIZE], zeros_array, ((FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE) * sizeof(Goldilocks::Element));

    gntt_extension.NTT(a, a, (FFT_SIZE << BLOWUP_FACTOR));

    ASSERT_EQ(Goldilocks::toU64(a[0]), 0X5C7F9E08245DBA11);
    ASSERT_EQ(Goldilocks::toU64(a[1]), 0X90D1DFB0589ABF6);
    ASSERT_EQ(Goldilocks::toU64(a[2]), 0XF8B3928DED48A98F);
    ASSERT_EQ(Goldilocks::toU64(a[3]), 0XC1918A78E4345E88);
    ASSERT_EQ(Goldilocks::toU64(a[4]), 0XF6E69C9842AA2E22);
    ASSERT_EQ(Goldilocks::toU64(a[5]), 0X5ADBBE450C79CDAD);
    ASSERT_EQ(Goldilocks::toU64(a[6]), 0X60A2A349428A0DA);
    ASSERT_EQ(Goldilocks::toU64(a[7]), 0X4A218E1A5E4B64C4);
    ASSERT_EQ(Goldilocks::toU64(a[8]), 0XB8AA93BF9B77357D);
    ASSERT_EQ(Goldilocks::toU64(a[9]), 0XC5E4FD1C23783A86);
    ASSERT_EQ(Goldilocks::toU64(a[10]), 0X5059D5ACFEFD1C4E);
    ASSERT_EQ(Goldilocks::toU64(a[11]), 0X84BFB1AF052262DC);
    ASSERT_EQ(Goldilocks::toU64(a[12]), 0X267CA8D006A0D83B);
    ASSERT_EQ(Goldilocks::toU64(a[13]), 0X85FFE94AD79AB9D8);
    ASSERT_EQ(Goldilocks::toU64(a[14]), 0XC929E62672F3B564);
    ASSERT_EQ(Goldilocks::toU64(a[15]), 0XF1F6FB9811E8B6D9);
    ASSERT_EQ(Goldilocks::toU64(a[16]), 0X303E9B9EE7F5018C);
    ASSERT_EQ(Goldilocks::toU64(a[17]), 0X85656D5B36F8B64A);
    ASSERT_EQ(Goldilocks::toU64(a[18]), 0X4EED2DDC4ABB9788);
    ASSERT_EQ(Goldilocks::toU64(a[19]), 0X9B19FA8666AFA997);
    ASSERT_EQ(Goldilocks::toU64(a[20]), 0XA02461E0BCDDB962);
    ASSERT_EQ(Goldilocks::toU64(a[21]), 0XEB1585E707FD372A);
    ASSERT_EQ(Goldilocks::toU64(a[22]), 0XD1B3B074B5FDC807);
    ASSERT_EQ(Goldilocks::toU64(a[23]), 0X8AEE07B925BE1179);
    ASSERT_EQ(Goldilocks::toU64(a[24]), 0XBEE1035F312C4BC3);
    ASSERT_EQ(Goldilocks::toU64(a[25]), 0X5C1FE1437308D938);
    ASSERT_EQ(Goldilocks::toU64(a[26]), 0X75FCDA707D67FB90);
    ASSERT_EQ(Goldilocks::toU64(a[27]), 0XE3BD3C32E5635D9F);
    ASSERT_EQ(Goldilocks::toU64(a[28]), 0X3E0A945C57D94083);
    ASSERT_EQ(Goldilocks::toU64(a[29]), 0X48161FC7B47B998E);
    ASSERT_EQ(Goldilocks::toU64(a[30]), 0X5144C235578455C6);
    ASSERT_EQ(Goldilocks::toU64(a[31]), 0XAF5244B5C1134635);

    free(a);
    free(zeros_array);
    free(r);
}

TEST(GOLDILOCKS_TEST, LDE_block)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc((FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE);
    NTT_Goldilocks gntt_extension((FFT_SIZE << BLOWUP_FACTOR));

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            Goldilocks::add(a[i * NUM_COLUMNS + j], Goldilocks::one(), Goldilocks::fromU64(j));
        }
    }

    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * (i - 1) + j] + a[NUM_COLUMNS * (i - 2) + j];
        }
    }

    Goldilocks::Element shift = Goldilocks::fromU64(49); // TODO: ask for this number, where to put it how to calculate it

    gntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS, NPHASES);

    // TODO: This can be pre-generated
    Goldilocks::Element *r = (Goldilocks::Element *)malloc(FFT_SIZE * sizeof(Goldilocks::Element));
    r[0] = Goldilocks::one();
    for (int i = 1; i < FFT_SIZE; i++)
    {
        r[i] = r[i - 1] * shift;
    }

#pragma omp parallel for
    for (uint64_t i = 0; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * i + j] * r[i];
        }
    }
#pragma omp parallel for schedule(static)
    for (uint i = FFT_SIZE * NUM_COLUMNS; i < (FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS; i++)
    {
        a[i] = Goldilocks::zero();
    }

    gntt_extension.NTT(a, a, (FFT_SIZE << BLOWUP_FACTOR), NUM_COLUMNS, NUM_PHASES);

    ASSERT_EQ(Goldilocks::toU64(a[0 * NUM_COLUMNS]), 0X5C7F9E08245DBA11);
    ASSERT_EQ(Goldilocks::toU64(a[1 * NUM_COLUMNS]), 0X90D1DFB0589ABF6);
    ASSERT_EQ(Goldilocks::toU64(a[2 * NUM_COLUMNS]), 0XF8B3928DED48A98F);
    ASSERT_EQ(Goldilocks::toU64(a[3 * NUM_COLUMNS]), 0XC1918A78E4345E88);
    ASSERT_EQ(Goldilocks::toU64(a[4 * NUM_COLUMNS]), 0XF6E69C9842AA2E22);
    ASSERT_EQ(Goldilocks::toU64(a[5 * NUM_COLUMNS]), 0X5ADBBE450C79CDAD);
    ASSERT_EQ(Goldilocks::toU64(a[6 * NUM_COLUMNS]), 0X60A2A349428A0DA);
    ASSERT_EQ(Goldilocks::toU64(a[7 * NUM_COLUMNS]), 0X4A218E1A5E4B64C4);
    ASSERT_EQ(Goldilocks::toU64(a[8 * NUM_COLUMNS]), 0XB8AA93BF9B77357D);
    ASSERT_EQ(Goldilocks::toU64(a[9 * NUM_COLUMNS]), 0XC5E4FD1C23783A86);
    ASSERT_EQ(Goldilocks::toU64(a[10 * NUM_COLUMNS]), 0X5059D5ACFEFD1C4E);
    ASSERT_EQ(Goldilocks::toU64(a[11 * NUM_COLUMNS]), 0X84BFB1AF052262DC);
    ASSERT_EQ(Goldilocks::toU64(a[12 * NUM_COLUMNS]), 0X267CA8D006A0D83B);
    ASSERT_EQ(Goldilocks::toU64(a[13 * NUM_COLUMNS]), 0X85FFE94AD79AB9D8);
    ASSERT_EQ(Goldilocks::toU64(a[14 * NUM_COLUMNS]), 0XC929E62672F3B564);
    ASSERT_EQ(Goldilocks::toU64(a[15 * NUM_COLUMNS]), 0XF1F6FB9811E8B6D9);
    ASSERT_EQ(Goldilocks::toU64(a[16 * NUM_COLUMNS]), 0X303E9B9EE7F5018C);
    ASSERT_EQ(Goldilocks::toU64(a[17 * NUM_COLUMNS]), 0X85656D5B36F8B64A);
    ASSERT_EQ(Goldilocks::toU64(a[18 * NUM_COLUMNS]), 0X4EED2DDC4ABB9788);
    ASSERT_EQ(Goldilocks::toU64(a[19 * NUM_COLUMNS]), 0X9B19FA8666AFA997);
    ASSERT_EQ(Goldilocks::toU64(a[20 * NUM_COLUMNS]), 0XA02461E0BCDDB962);
    ASSERT_EQ(Goldilocks::toU64(a[21 * NUM_COLUMNS]), 0XEB1585E707FD372A);
    ASSERT_EQ(Goldilocks::toU64(a[22 * NUM_COLUMNS]), 0XD1B3B074B5FDC807);
    ASSERT_EQ(Goldilocks::toU64(a[23 * NUM_COLUMNS]), 0X8AEE07B925BE1179);
    ASSERT_EQ(Goldilocks::toU64(a[24 * NUM_COLUMNS]), 0XBEE1035F312C4BC3);
    ASSERT_EQ(Goldilocks::toU64(a[25 * NUM_COLUMNS]), 0X5C1FE1437308D938);
    ASSERT_EQ(Goldilocks::toU64(a[26 * NUM_COLUMNS]), 0X75FCDA707D67FB90);
    ASSERT_EQ(Goldilocks::toU64(a[27 * NUM_COLUMNS]), 0XE3BD3C32E5635D9F);
    ASSERT_EQ(Goldilocks::toU64(a[28 * NUM_COLUMNS]), 0X3E0A945C57D94083);
    ASSERT_EQ(Goldilocks::toU64(a[29 * NUM_COLUMNS]), 0X48161FC7B47B998E);
    ASSERT_EQ(Goldilocks::toU64(a[30 * NUM_COLUMNS]), 0X5144C235578455C6);
    ASSERT_EQ(Goldilocks::toU64(a[31 * NUM_COLUMNS]), 0XAF5244B5C1134635);

    free(a);
    free(r);
}

TEST(GOLDILOCKS_CUBIC_TEST, one)
{
    uint64_t a[3] = {1, 1, 1};
    int32_t b[3] = {1, 1, 1};
    std::string c[3] = {"92233720347072921606", "92233720347072921606", "92233720347072921606"}; // GOLDILOCKS_PRIME * 5 + 1
    uint64_t d[3] = {1 + GOLDILOCKS_PRIME, 1 + GOLDILOCKS_PRIME, 1 + GOLDILOCKS_PRIME};

    Goldilocks3::Element ina1;
    Goldilocks3::Element ina2;
    Goldilocks3::Element ina3;
    Goldilocks3::Element inb1;
    Goldilocks3::Element inc1;

    Goldilocks3::fromU64(ina1, a);
    Goldilocks3::fromS32(ina2, b);
    Goldilocks3::fromString(ina3, c);
    Goldilocks3::fromU64(inb1, d);
    Goldilocks3::fromString(inc1, c);

    uint64_t ina1_res[3];
    uint64_t ina2_res[3];
    uint64_t ina3_res[3];
    uint64_t inb1_res[3];
    uint64_t inc1_res[3];

    Goldilocks3::toU64(ina1_res, ina1);
    Goldilocks3::toU64(ina2_res, ina2);
    Goldilocks3::toU64(ina3_res, ina3);
    Goldilocks3::toU64(inb1_res, inb1);
    Goldilocks3::toU64(inc1_res, inc1);

    ASSERT_EQ(ina1_res[0], a[0]);
    ASSERT_EQ(ina1_res[1], a[1]);
    ASSERT_EQ(ina1_res[2], a[2]);

    ASSERT_EQ(ina2_res[0], a[0]);
    ASSERT_EQ(ina2_res[1], a[1]);
    ASSERT_EQ(ina2_res[2], a[2]);

    ASSERT_EQ(ina3_res[0], a[0]);
    ASSERT_EQ(ina3_res[1], a[1]);
    ASSERT_EQ(ina3_res[2], a[2]);

    ASSERT_EQ(inb1_res[0], a[0]);
    ASSERT_EQ(inb1_res[1], a[1]);
    ASSERT_EQ(inb1_res[2], a[2]);

    ASSERT_EQ(inc1_res[0], a[0]);
    ASSERT_EQ(inc1_res[1], a[1]);
    ASSERT_EQ(inc1_res[2], a[2]);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

// Build command: g++ tests/tests.cpp src/* -lgtest -lgmp -lomp -o test -g  -Wall -pthread -fopenmp -L/usr/lib/llvm-13/lib/