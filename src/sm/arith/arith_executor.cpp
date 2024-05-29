#include <nlohmann/json.hpp>
#include "arith_executor.hpp"
#include "arith_action_bytes.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "zklog.hpp"
#include "goldilocks_precomputed.hpp"
#include "zkglobals.hpp"

using json = nlohmann::json;

int64_t eq0  (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq1  (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq2  (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq3  (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq4  (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq5  (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq6  (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq7  (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq8  (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq9  (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq10 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq11 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq12 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq13 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq14 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq15 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq16 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq17 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq18 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq19 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);

const uint64_t ARITH_CYCLE = 32;
const uint64_t INPUT_CHUNKS = 16;

const uint16_t PRIME_SECP256K1_CHUNKS[16] = { 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                                              0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFE, 0xFFFF, 0xFC2F };

const uint16_t PRIME_BN254_CHUNKS[16] = { 0x3064, 0x4E72, 0xE131, 0xA029, 0xB850, 0x45B6, 0x8181, 0x585D, 
                                          0x9781, 0x6A91, 0x6871, 0xCA8D, 0x3C20, 0x8C16, 0xD87C, 0xFD47 };

const uint32_t PRIME_BLS12381_CHUNKS[16] = { 0x1A0111, 0xEA397F, 0xE69A4B, 0x1BA7B6, 0x434BAC, 0xD76477, 0x4B84F3, 0x8512BF, 
                                             0x6730D2, 0xA0F6B0, 0xF6241E, 0xABFFFE, 0xB153FF, 0xFFB9FE, 0xFFFFFF, 0xFFAAAB };

const uint64_t CARRY_OFFSET = 0x10000000; // 2^28

class ArithExecutorContext
{
public:
    uint64_t arithEquation;
    uint64_t offset;
    ArithCommitPols &pols;
    bool moduleCheck;
    mpz_class forcedS;
    RawFec::Element threeFec;
    RawFec::Element twoFec;
    ArithExecutorContext(uint64_t arithEquation, uint64_t offset, ArithCommitPols &pols) :
        arithEquation(arithEquation),
        offset(offset),
        pols(pols)
    {
        moduleCheck = (arithEquation != ARITH_BASE) && (arithEquation != ARITH_256TO384);
        fec.fromUI(threeFec, 3);
        fec.fromUI(twoFec, 2);
    };
    void setSelectorPols (void);
    void calculateEquationStep (uint64_t step);
    void calculateCarryAndEquationStep (uint64_t eqIndex, uint64_t carryIndex, uint64_t step, uint64_t bits = 16);
    int64_t getCarry (uint64_t carryIndex, uint64_t step);
    void setCarry (uint64_t carryIndex, uint64_t step, int64_t value);
    Goldilocks::Element getPrimeChunk(uint64_t step, Goldilocks::Element &moduleChunk);
    void inputToPols (CommitPol (&pol)[16], Goldilocks::Element (&input)[8], uint64_t bits);
    void prepareInputPols (ArithAction &input, mpz_class &x1, mpz_class &y1, mpz_class &x2, mpz_class &y2, mpz_class &x3, mpz_class &y3);
    void calculateSQPols (const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, const mpz_class &x3, const mpz_class &y3, mpz_class &s, mpz_class &q0, mpz_class &q1, mpz_class &q2);
    void splitChunkRangeCheck(CommitPol (&pol)[16], uint64_t from, uint64_t count, uint64_t relative_offset, CommitPol &hpols, CommitPol &lpols);

    void getChunkBits (const uint64_t arithEquation, uint64_t &bits1, uint64_t &bits2);



    mpz_class calculateQ (const mpz_class &module, const mpz_class &eqvalue, const mpz_class &offset, const char * title, const mpz_class &sign = 1);
    void calculateAddPointSQ (const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, mpz_class &s, mpz_class &q0);
    void calculateDblPointSQ (const mpz_class &x1, const mpz_class &y1, mpz_class &s, mpz_class &q0);
    void calculateAddPointQs (const mpz_class &s, const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, const mpz_class &x3, const mpz_class &y3, mpz_class &q1, mpz_class &q2);
    void calculateMulFp2Qs (const mpz_class &module, const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, const mpz_class &x3, const mpz_class &y3, const mpz_class &offset1, const mpz_class &offset2, mpz_class &q1, mpz_class &q2);
    void calculateAddFp2Qs (const mpz_class &module, const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, const mpz_class &x3, const mpz_class &y3, const mpz_class &offset1, const mpz_class &offset2, mpz_class &q1, mpz_class &q2);
    void calculateSubFp2Qs (const mpz_class &module, const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, const mpz_class &x3, const mpz_class &y3, const mpz_class &offset1, const mpz_class &offset2, mpz_class &q1, mpz_class &q2);
    void calculateModularQs (const uint64_t bits, const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, const mpz_class &y3, mpz_class &q1, mpz_class &q2);
    void feaToScalar(Goldilocks::Element (&fea)[8], uint64_t bits, mpz_class &result);

    void valueToPols(mpz_class value, uint64_t chunkBits, CommitPol (&pol)[16], uint64_t chunks = 16);
};


void ArithExecutor::execute (vector<ArithAction> &inputs, ArithCommitPols &pols)
{
    // Check that we have enough room in polynomials  TODO: Do this check in JS
    if (inputs.size()*32 > N)
    {
        zklog.error("ArithExecutor::execute() Too many Arith inputs=" + to_string(inputs.size()) + " > N/32=" + to_string(N/32));
        exitProcess();
    }

    for (uint64_t i = 0; i < inputs.size(); i++)
    {
        //zklog.info("ArithExecutor::execute() i=" + to_string(i) + " " + inputs[i].toString());

        /* setupEquation */
        ArithExecutorContext ctx(inputs[i].equation, i * ARITH_CYCLE, pols);
        ctx.setSelectorPols();
        mpz_class x1, y1, x2, y2, x3, y3;
        ctx.prepareInputPols(inputs[i], x1, y1, x2, y2, x3, y3);
        //zklog.info("ArithExecutor::execute() x1=" + x1.get_str(16) + " y1=" + y1.get_str(16) + " x2=" + x2.get_str(16) + " y2=" + y2.get_str(16) + " x3=" + x3.get_str(16) + " y3=" + y3.get_str(16));
        
        // Therefore, as the quotient needs to be represented in our VM, we need to know
        // the worst negative case and add an offset so that the resulting name is never negative.
        // Then, this offset is also added in the PIL constraint to ensure the equality.
        //
        // Note1: Since we can choose whether the quotient is positive or negative, we choose it so
        //        that the added offset is the lowest.
        //
        // Note2: x1,x2,y1,y2,x3,y3,s are well-composed with chunks of 16 or 24 bits verified with
        //        range check. For this reason we could assume that x1,x2,y1,y2,s ∈ [0, 2^256-1] in
        //        case of 256 bits equation and x1,x2,y1,y2,s ∈ [0, 2^384-1] in case of 384 bit equations.

        mpz_class s, q0, q1, q2;
        ctx.calculateSQPols(x1, y1, x2, y2, x3, y3, s, q0, q1, q2);
        //zklog.info("s=" + s.get_str(16) + " q0=" + q0.get_str(16) + " q1=" + q1.get_str(16) + " q2=" + q2.get_str(16));

        bool xAreDifferent = false;
        bool valueLtPrime = false;
        
        for (uint64_t step = 0; step < ARITH_CYCLE; step++)
        {
            uint64_t index = ctx.offset + step;
            uint64_t nextIndex = (index + 1) % N;
            uint64_t chunkStep = step % INPUT_CHUNKS;
            if (chunkStep == 0)
            {
                valueLtPrime = false;
            }
            pols.y2_clock[index] = pols.y2[15 - chunkStep][ctx.offset];

            // ARITH_ECADD_DIFFERENT is select need to check that points are diferent
            if ((ctx.arithEquation == ARITH_ECADD_DIFFERENT) && (step < INPUT_CHUNKS))
            {
                if (xAreDifferent == false)
                {
                    Goldilocks::Element delta = fr.sub(pols.x2[step][index], pols.x1[step][index]);
                    if (!fr.isZero(delta))
                    {
                        pols.xDeltaChunkInverse[index] = fr.inv(delta);
                        xAreDifferent = true;
                        pols.xAreDifferent[nextIndex] = fr.one();
                    }
                }
            }

            // If either selEq1,selEq2,selEq3,selEq4,selEq5,selEq6 is selected, we need to ensure that x3, y3 is alias free.
            // Recall that selEq1,selEq2 work over the base field of the Secp256k1 curve, selEq3,selEq4,selEq5 works over the
            // base field of the BN254 curve and selEq6 works modulo y2.

            Goldilocks::Element chunkValue = (step < INPUT_CHUNKS) ? pols.x3[15 - chunkStep][ctx.offset] : pols.y3[15 - chunkStep][ctx.offset];
            pols.x3y3_clock[index] = chunkValue;

            if (ctx.moduleCheck)
            {
                Goldilocks::Element primeChunk = ctx.getPrimeChunk(chunkStep, pols.y2[15 - chunkStep][ctx.offset]);

                uint64_t chunkLtPrime = valueLtPrime ? 0 : (fr.toU64(chunkValue) < fr.toU64(primeChunk));
                valueLtPrime = valueLtPrime || chunkLtPrime;
                pols.primeChunk[index] = primeChunk;
                pols.chunkLtPrime[index] = fr.fromU64(chunkLtPrime);
                if (valueLtPrime) pols.valueLtPrime[nextIndex] = fr.one();
                uint64_t delta = fr.toU64(fr.fromU64(fr.toU64(primeChunk) - fr.toU64(chunkValue) - chunkLtPrime));
                pols.hs_bit_delta[index] = fr.fromU64((delta >> 23) & 0x01);
                pols.ls_bits_delta[index] = fr.fromU64(delta & 0x7FFFF);
            }
            else
            {
                pols.primeChunk[index] = chunkValue;
            }
            if (step == (ARITH_CYCLE - 1))
            {
                pols.resultEq[index] = fr.one();
            }
        }

        // calculateEquation need all pols calculated
        for (uint64_t step = 0; step < ARITH_CYCLE; step++)
        {
            ctx.calculateEquationStep(step);
        }
    }
}

void ArithExecutorContext::setSelectorPols (void)
{
    for (uint64_t isel = 0; isel < 11; isel++)
    {
        uint64_t value = ((arithEquation - 1) == isel) ? 1 : 0;
        if (value != 0)
        {
            for (uint64_t step = 0; step < ARITH_CYCLE; ++step)
            {
                pols.selEq[isel][offset + step] = fr.fromU64(value);
            }
        }
    }
}

void ArithExecutorContext::calculateEquationStep (uint64_t step)
{
    switch (arithEquation)
    {
        case ARITH_BASE:
            calculateCarryAndEquationStep(0, 0, step);
            break;

        case ARITH_ECADD_DIFFERENT:
            calculateCarryAndEquationStep(1, 0, step);
            calculateCarryAndEquationStep(3, 1, step);
            calculateCarryAndEquationStep(4, 2, step);
            break;

        case ARITH_ECADD_SAME:
            calculateCarryAndEquationStep(2, 0, step);
            calculateCarryAndEquationStep(3, 1, step);
            calculateCarryAndEquationStep(4, 2, step);
            break;

        case ARITH_BN254_MULFP2:
            calculateCarryAndEquationStep(5, 0, step);
            calculateCarryAndEquationStep(6, 1, step);
            break;

        case ARITH_BN254_ADDFP2:
            calculateCarryAndEquationStep(7, 0, step);
            calculateCarryAndEquationStep(8, 1, step);
            break;

        case ARITH_BN254_SUBFP2:
            calculateCarryAndEquationStep(9, 0, step);
            calculateCarryAndEquationStep(10, 1, step);
            break;

        case ARITH_MOD:
            calculateCarryAndEquationStep(11, 0, step);
            break;

        case ARITH_384_MOD:
            calculateCarryAndEquationStep(12, 0, step, 24);
            break;

        case ARITH_BLS12381_MULFP2:
            calculateCarryAndEquationStep(13, 0, step, 24);
            calculateCarryAndEquationStep(14, 1, step, 24);
            break;

        case ARITH_BLS12381_ADDFP2:
            calculateCarryAndEquationStep(15, 0, step, 24);
            calculateCarryAndEquationStep(16, 1, step, 24);
            break;

        case ARITH_BLS12381_SUBFP2:
            calculateCarryAndEquationStep(17, 0, step, 24);
            calculateCarryAndEquationStep(18, 1, step, 24);
            break;

        case ARITH_256TO384:
            calculateCarryAndEquationStep(19, 0, step, 24);
            break;
        default:
            zklog.error("ArithExecutorContext::calculateEquationStep() found invalid arithEquation=" + to_string(arithEquation));
            exitProcess();
    }
}

void ArithExecutorContext::calculateCarryAndEquationStep (uint64_t eqIndex, uint64_t carryIndex, uint64_t step, uint64_t bits)
{
    int64_t eqValue;
    switch(eqIndex)
    {
        case 0:  eqValue = eq0(fr, pols, step, offset); break;
        case 1:  eqValue = eq1(fr, pols, step, offset); break;
        case 2:  eqValue = eq2(fr, pols, step, offset); break;
        case 3:  eqValue = eq3(fr, pols, step, offset); break;
        case 4:  eqValue = eq4(fr, pols, step, offset); break;
        case 5:  eqValue = eq5(fr, pols, step, offset); break;
        case 6:  eqValue = eq6(fr, pols, step, offset); break;
        case 7:  eqValue = eq7(fr, pols, step, offset); break;
        case 8:  eqValue = eq8(fr, pols, step, offset); break;
        case 9:  eqValue = eq9(fr, pols, step, offset); break;
        case 10: eqValue = eq10(fr, pols, step, offset); break;
        case 11: eqValue = eq11(fr, pols, step, offset); break;
        case 12: eqValue = eq12(fr, pols, step, offset); break;
        case 13: eqValue = eq13(fr, pols, step, offset); break;
        case 14: eqValue = eq14(fr, pols, step, offset); break;
        case 15: eqValue = eq15(fr, pols, step, offset); break;
        case 16: eqValue = eq16(fr, pols, step, offset); break;
        case 17: eqValue = eq17(fr, pols, step, offset); break;
        case 18: eqValue = eq18(fr, pols, step, offset); break;
        case 19: eqValue = eq19(fr, pols, step, offset); break;
        default:
            zklog.error("ArithExecutorContext::calculateCarryAndEquationStep() invalid eqIndex=" + to_string(eqIndex));
            exitProcess();
    }
    int64_t carry = (step > 0) ? getCarry(carryIndex, step) : 0;
    uint64_t mask = (1 << bits) - 1;
    if (((eqValue + carry) & mask) != 0)
    {
        zklog.error("ArithExecutorContext::calculateCarryAndEquationStep() mismatch eqValue=" + to_string(eqValue) + " carry=" + to_string(carry) + " mask=" + to_string(mask));
        exitProcess();
    }
    setCarry(carryIndex, step + 1, (eqValue + carry) >> bits);
}

int64_t ArithExecutorContext::getCarry (uint64_t carryIndex, uint64_t step)
{
    int64_t result = 0;
    switch (carryIndex)
    {
        case 0: result = fr.toS64(pols.ls_carry0[offset + step]) + (fr.toS64(pols.hs_carry0[offset + step]) << 23) - CARRY_OFFSET;
        break;
        case 1: result = fr.toS64(pols.ls_carry1[offset + step]) + (fr.toS64(pols.hs_carry1[offset + step]) << 23) - CARRY_OFFSET;
        break;
        case 2: result = fr.toS64(pols.carry2[offset + step]);
        break;
        default:
            zklog.error("ArithExecutorContext::getCarry() invalid carryIndex=" + to_string(carryIndex));
            exitProcess();
    }
    /*zklog.info("getCarry() carryIndex=" + to_string(carryIndex) +
        " offset=" + to_string(offset) + " step=" + to_string(step) +
        " result=" + to_string(result) +
        " pols.ls_carry0=" + fr.toString(pols.ls_carry0[offset + step], 16) +
        " pols.hs_carry0=" + fr.toString(pols.hs_carry0[offset + step], 16) +
        " pols.ls_carry1=" + fr.toString(pols.ls_carry1[offset + step], 16) +
        " pols.hs_carry1=" + fr.toString(pols.hs_carry1[offset + step], 16) +
        " pols.carry2=" + fr.toString(pols.carry2[offset + step], 16));*/
    return result;
}

void ArithExecutorContext::setCarry (uint64_t carryIndex, uint64_t step, int64_t value)
{
    switch (carryIndex)
    {
        case 0: 
            pols.ls_carry0[offset + step] = fr.fromS64((value + CARRY_OFFSET) & 0x7FFFFF);
            pols.hs_carry0[offset + step] = fr.fromS64((value + CARRY_OFFSET) >> 23);
            break;

        case 1: 
            pols.ls_carry1[offset + step] = fr.fromS64((value + CARRY_OFFSET) & 0x7FFFFF);
            pols.hs_carry1[offset + step] = fr.fromS64((value + CARRY_OFFSET) >> 23);
            break;

        case 2:
            pols.carry2[offset + step] = fr.fromS64(value);
            break;
    
        default:
            zklog.error("ArithExecutorContext::setCarry() invalid carryIndex=" + to_string(carryIndex));
            exitProcess();
    }
    /*zklog.info("setCarry() carryIndex=" + to_string(carryIndex) +
        " offset=" + to_string(offset) + " step=" + to_string(step) +
        " value=" + to_string(value) +
        " pols.ls_carry0=" + fr.toString(pols.ls_carry0[offset + step], 16) +
        " pols.hs_carry0=" + fr.toString(pols.hs_carry0[offset + step], 16) +
        " pols.ls_carry1=" + fr.toString(pols.ls_carry1[offset + step], 16) +
        " pols.hs_carry1=" + fr.toString(pols.hs_carry1[offset + step], 16) +
        " pols.carry2=" + fr.toString(pols.carry2[offset + step], 16));*/
}

Goldilocks::Element ArithExecutorContext::getPrimeChunk (uint64_t step, Goldilocks::Element &moduleChunk)
{
    switch (arithEquation)
    {
        case ARITH_ECADD_DIFFERENT:
        case ARITH_ECADD_SAME:
            return fr.fromU64(PRIME_SECP256K1_CHUNKS[step]);

        case ARITH_BN254_MULFP2:
        case ARITH_BN254_ADDFP2:
        case ARITH_BN254_SUBFP2:
            return fr.fromU64(PRIME_BN254_CHUNKS[step]);

        case ARITH_MOD:
        case ARITH_384_MOD:
            return moduleChunk;

        case ARITH_BLS12381_MULFP2:
        case ARITH_BLS12381_ADDFP2:
        case ARITH_BLS12381_SUBFP2:
            return fr.fromU64(PRIME_BLS12381_CHUNKS[step]);
        default:
            zklog.error("ArithExecutorContext::getPrimeChunk() invalid arithEquation=" + to_string(arithEquation));
            exitProcess();
    }
    return fr.zero();
}

void ArithExecutorContext::calculateSQPols (const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, const mpz_class &x3, const mpz_class &y3, mpz_class &s, mpz_class &q0, mpz_class &q1, mpz_class &q2)
{
    uint64_t chunkBits = 16;
    switch (arithEquation)
    {
        case ARITH_ECADD_DIFFERENT:
            calculateAddPointSQ(x1, y1, x2, y2, s, q0);
            calculateAddPointQs(s, x1, y1, x2, y2, x3, y3, q1, q2);
            break;
        case ARITH_ECADD_SAME:
            calculateDblPointSQ(x1, y1, s, q0);
            calculateAddPointQs(s, x1, y1, x2, y2, x3, y3, q1, q2);
            break;
        case ARITH_BN254_MULFP2:
            calculateMulFp2Qs(BN254p_prime, x1, y1, x2, y2, x3, y3, ScalarTwoTo259, 0, q1, q2);
            break;
        case ARITH_BN254_ADDFP2:
            calculateAddFp2Qs(BN254p_prime, x1, y1, x2, y2, x3, y3, 0, 0, q1, q2);
            break;
        case ARITH_BN254_SUBFP2:
            calculateSubFp2Qs(BN254p_prime, x1, y1, x2, y2, x3, y3, 8, 8, q1, q2);
            break;
        case ARITH_MOD:
            calculateModularQs(256, x1, y1, x2, y2, y3, q0, q1);
            break;
        case ARITH_384_MOD:
            calculateModularQs(384, x1, y1, x2, y2, y3, q0, q1);
            chunkBits = 24;
            break;
        case ARITH_BLS12381_MULFP2:
            calculateMulFp2Qs(BLS12_381p_prime, x1, y1, x2, y2, x3, y3, ScalarTwoTo388, 0, q1, q2);
            chunkBits = 24;
            break;
        case ARITH_BLS12381_ADDFP2:
            calculateAddFp2Qs(BLS12_381p_prime, x1, y1, x2, y2, x3, y3, 0, 0, q1, q2);
            chunkBits = 24;
            break;
        case ARITH_BLS12381_SUBFP2:
            calculateSubFp2Qs(BLS12_381p_prime, x1, y1, x2, y2, x3, y3, 16, 16, q1, q2);
            chunkBits = 24;
            break;
    }
    
    valueToPols(s, chunkBits, pols.s);
    valueToPols(q0, chunkBits, pols.q0);
    valueToPols(q1, chunkBits, pols.q1);
    valueToPols(q2, chunkBits, pols.q2);
    
    // Since chunk values could have 24 bits (or more in the case of qs high significant chunk) it is
    // necessary to divide them in two chunk to do a range check.

    splitChunkRangeCheck(pols.s,  0, INPUT_CHUNKS, 0, pols.hsc_sq0q1, pols.lsc_sq0q1);
    splitChunkRangeCheck(pols.q0, 0, INPUT_CHUNKS - 1, INPUT_CHUNKS, pols.hsc_sq0q1, pols.lsc_sq0q1);
    splitChunkRangeCheck(pols.q1, 0, 1, INPUT_CHUNKS * 2 - 1, pols.hsc_sq0q1, pols.lsc_sq0q1);

    splitChunkRangeCheck(pols.q1, 1, INPUT_CHUNKS - 2, 0, pols.hsc_q1q2qh, pols.lsc_q1q2qh);
    splitChunkRangeCheck(pols.q2, 0, INPUT_CHUNKS, INPUT_CHUNKS - 2, pols.hsc_q1q2qh, pols.lsc_q1q2qh);
    splitChunkRangeCheck(pols.q0, INPUT_CHUNKS - 1, 1, INPUT_CHUNKS * 2 - 2, pols.hsc_q1q2qh, pols.lsc_q1q2qh);
    splitChunkRangeCheck(pols.q1, INPUT_CHUNKS - 1, 1, INPUT_CHUNKS * 2 - 1, pols.hsc_q1q2qh, pols.lsc_q1q2qh);
}

// method to split chunk of 24 bits o more:
// pols = hpols * 2**16 + lpols 
// less 16 significant bits and other with the rest
void ArithExecutorContext::splitChunkRangeCheck(CommitPol (&pol)[16], uint64_t from, uint64_t count, uint64_t relative_offset, CommitPol &hpols, CommitPol &lpols)
{
    for (uint64_t index = 0; index < count; index++)
    {
        uint64_t value = fr.toU64(pol[from + index][offset]);
        hpols[offset + relative_offset + index] = fr.fromU64(value >> 16);
        lpols[offset + relative_offset + index] = fr.fromU64(value & 0xFFFF);
    }
}

void ArithExecutorContext::getChunkBits (const uint64_t arithEquation, uint64_t &bits1, uint64_t &bits2)
{
    switch (arithEquation)
    {
        case ARITH_BASE:
        case ARITH_ECADD_DIFFERENT:
        case ARITH_ECADD_SAME:
        case ARITH_BN254_MULFP2:
        case ARITH_BN254_ADDFP2:
        case ARITH_BN254_SUBFP2:
        case ARITH_MOD:
            bits1 = 16;
            bits2 = 16;
            return;

        case ARITH_384_MOD:
        case ARITH_BLS12381_MULFP2:
        case ARITH_BLS12381_ADDFP2:
        case ARITH_BLS12381_SUBFP2:
            bits1 = 24;
            bits2 = 24;
            return;

        case ARITH_256TO384:
            bits1 = 16;
            bits2 = 24;
            return;

        default:
            zklog.error("ArithExecutorContext::getChunkBits() Invalid arithmetic operation=" + to_string(arithEquation));
            exitProcess();
    }
}

void ArithExecutorContext::prepareInputPols (ArithAction &input, mpz_class &x1, mpz_class &y1, mpz_class &x2, mpz_class &y2, mpz_class &x3, mpz_class &y3)
{
    uint64_t bits1;
    uint64_t bits2;
    getChunkBits(arithEquation, bits1, bits2);

    //this.forcedS = input.s ?? false;

    inputToPols(pols.x1, input.x1, bits1);
    inputToPols(pols.y1, input.y1, bits1);

    inputToPols(pols.x2, input.x2, bits2);
    inputToPols(pols.y2, input.y2, bits2);

    inputToPols(pols.x3, input.x3, bits2);
    inputToPols(pols.y3, input.y3, bits2);

    // how chunk values could has 24 bits is necessary divide in two chunk to do range check.

    splitChunkRangeCheck(pols.x1, 0, INPUT_CHUNKS, 0, pols.hsc_x1y1, pols.lsc_x1y1);
    splitChunkRangeCheck(pols.y1, 0, INPUT_CHUNKS, INPUT_CHUNKS, pols.hsc_x1y1, pols.lsc_x1y1);

    splitChunkRangeCheck(pols.x2, 0, INPUT_CHUNKS, 0, pols.hsc_x2y2, pols.lsc_x2y2);
    splitChunkRangeCheck(pols.y2, 0, INPUT_CHUNKS, INPUT_CHUNKS, pols.hsc_x2y2, pols.lsc_x2y2);      

    splitChunkRangeCheck(pols.x3, 0, INPUT_CHUNKS, 0, pols.hsc_x3y3, pols.lsc_x3y3);
    splitChunkRangeCheck(pols.y3, 0, INPUT_CHUNKS, INPUT_CHUNKS, pols.hsc_x3y3, pols.lsc_x3y3);

    uint64_t _bits1 = 2*bits1;
    uint64_t _bits2 = 2*bits2;

    feaToScalar(input.x1, _bits1, x1);
    feaToScalar(input.y1, _bits1, y1);
    feaToScalar(input.x2, _bits2, x2);
    feaToScalar(input.y2, _bits2, y2);
    feaToScalar(input.x3, _bits2, x3);
    feaToScalar(input.y3, _bits2, y3);
}

mpz_class ArithExecutorContext::calculateQ (const mpz_class &module, const mpz_class &eqvalue, const mpz_class &offset, const char * title, const mpz_class &sign)
{
    if (module == 0)
    {
        zklog.error("ArithExecutorContext::calculateQ() module=0 " + string(title));
        exitProcess();
    }
    const mpz_class _q = eqvalue/module;

    if ((eqvalue - module*_q) != 0)
    {
        zklog.error("ArithExecutorContext::calculateQ() with the calculated q the residual is not zero " + string(title));
        exitProcess();
    }
    const mpz_class q = offset + sign * _q;
    if (q < 0)
    {
        zklog.error("ArithExecutorContext::calculateQ() the q with offset is negative q=" + q.get_str(16) + " _q=" + _q.get_str(16) + " " + string(title));
        exitProcess();
    }
    return q;
}

void ArithExecutorContext::calculateAddPointSQ (const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, mpz_class &s, mpz_class &q0)
{
    // Check x1 and x2 are not equal
    if (x1 == x2)
    {
        zklog.error("ArithExecutorContext::calculateAddPointSQ() x1 and x2 are equals, but ADD_EC_DIFFERENT is called");
        exitProcess();
    }

    // Convert to finite field elements
    RawFec::Element x1fec;
    RawFec::Element y1fec;
    RawFec::Element x2fec;
    RawFec::Element y2fec;
    scalar2fec(fec, x1fec, x1);
    scalar2fec(fec, y1fec, y1);
    scalar2fec(fec, x2fec, x2);
    scalar2fec(fec, y2fec, y2);

    // Calculate s
    RawFec::Element sfec;
    scalar2fec(fec, sfec, forcedS);
    if (forcedS == 0)
    {
        fec.div(sfec, fec.sub(y2fec, y1fec), fec.sub(x2fec, x1fec));
    }
    else
    {
        scalar2fec(fec, sfec, forcedS);
    }
    fec2scalar(fec, sfec, s);

    // Calculate equation value
    mpz_class eqvalue = s * x2 - s * x1 - y2 + y1;

    // Calculate Q
    q0 = calculateQ(Secp256k1p_prime, eqvalue, ScalarTwoTo257, "q0 diff point");
}

void ArithExecutorContext::calculateDblPointSQ (const mpz_class &x1, const mpz_class &y1, mpz_class &s, mpz_class &q0)
{
    // Convert to finite field elements
    RawFec::Element x1fec;
    RawFec::Element y1fec;
    scalar2fec(fec, x1fec, x1);
    scalar2fec(fec, y1fec, y1);

    // Calculate s
    RawFec::Element sfec;
    scalar2fec(fec, sfec, forcedS);
    if (forcedS == 0)
    {
        fec.div(sfec, fec.mul( threeFec, fec.mul(x1fec, x1fec) ), fec.add(y1fec, y1fec));
    }
    else
    {
        scalar2fec(fec, sfec, forcedS);
    }
    fec2scalar(fec, sfec, s);

    // Calculate equation value
    mpz_class eqvalue = s * 2 * y1 - 3 * x1 * x1;

    // Calculate Q
    q0 = calculateQ(Secp256k1p_prime, eqvalue, ScalarTwoTo258, "q0 same point", -1);
}

void ArithExecutorContext::calculateAddPointQs (const mpz_class &s, const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, const mpz_class &x3, const mpz_class &y3, mpz_class &q1, mpz_class &q2)
{
    // Calculate Qs
    q1 = calculateQ(Secp256k1p_prime, s*s - x1 - x2 - x3, 4, "q1");
    q2 = calculateQ(Secp256k1p_prime, s*x1 - s*x3 - y1 - y3, ScalarTwoTo257, "q2", -1);
}

void ArithExecutorContext::calculateMulFp2Qs (const mpz_class &module, const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, const mpz_class &x3, const mpz_class &y3, const mpz_class &offset1, const mpz_class &offset2, mpz_class &q1, mpz_class &q2)
{
    // Calculate Qs
    q1 = calculateQ(module, x1*x2 - y1*y2 - x3, offset1, "q1", -1);
    q2 = calculateQ(module, y1*x2 + x1*y2 - y3, offset2, "q2");
}

void ArithExecutorContext::calculateAddFp2Qs (const mpz_class &module, const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, const mpz_class &x3, const mpz_class &y3, const mpz_class &offset1, const mpz_class &offset2, mpz_class &q1, mpz_class &q2)
{
    // Calculate Qs
    q1 = calculateQ(module, x1 + x2 - x3, offset1, "q1");
    q2 = calculateQ(module, y1 + y2 - y3, offset2, "q2");
}

void ArithExecutorContext::calculateSubFp2Qs (const mpz_class &module, const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, const mpz_class &x3, const mpz_class &y3, const mpz_class &offset1, const mpz_class &offset2, mpz_class &q1, mpz_class &q2)
{
    // Calculate Qs
    q1 = calculateQ(module, x1 - x2 - x3, offset1, "q1", -1);
    q2 = calculateQ(module, y1 - y2 - y3, offset2, "q2", -1);
}

void ArithExecutorContext::calculateModularQs (const uint64_t bits, const mpz_class &x1, const mpz_class &y1, const mpz_class &x2, const mpz_class &y2, const mpz_class &y3, mpz_class &q1, mpz_class &q2)
{
    // Calculate Qs
    mpz_class _q = calculateQ(y2, x1 * y1 + x2 - y3, 0, "q");
    q1 = _q & ((ScalarOne << bits) - 1);
    q2 = _q >> bits;
}

void ArithExecutorContext::inputToPols (CommitPol (&pol)[16], Goldilocks::Element (&input)[8], uint64_t bits)
{
    uint64_t mask = (1 << bits) - 1;
    for (uint64_t index = 0; index < 8; index++)
    {
        uint64_t lchunk = fr.toU64(input[index]) & mask;
        uint64_t hchunk = fr.toU64(input[index]) >> bits;
        for (uint64_t step = 0; step < ARITH_CYCLE; step++)
        {
            pol[index*2][offset + step] = fr.fromU64(lchunk);
            pol[index*2 + 1][offset + step] = fr.fromU64(hchunk);
        }
    }
}

void ArithExecutorContext::feaToScalar(Goldilocks::Element (&fea)[8], uint64_t bits, mpz_class &result)
{
    result = 0;
    for (int64_t index = 8 - 1; index >= 0; index--)
    {
        result = (result << bits) + fr.toU64(fea[index]);
    }
}

void ArithExecutorContext::valueToPols (mpz_class value, uint64_t chunkBits, CommitPol (&pol)[16], uint64_t chunks)
{
    mpz_class mask = (1 << chunkBits) - 1;
    for (uint64_t index = 0; index < chunks; index++)
    {
        mpz_class pvalue = (index < (chunks - 1) ? (value & mask) : value);
        for (uint64_t step = 0; step < ARITH_CYCLE; step++)
        {
            pol[index][offset + step] = fr.fromU64(pvalue.get_ui());
        }
        value = value >> chunkBits;
    }
}