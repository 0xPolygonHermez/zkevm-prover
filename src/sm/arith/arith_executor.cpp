#include <nlohmann/json.hpp>
#include "arith_executor.hpp"
#include "arith_action_bytes.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "zklog.hpp"
#include "goldilocks_precomputed.hpp"
#include "zkglobals.hpp"

using json = nlohmann::json;

int64_t eq0 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq1 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq2 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq3 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq4 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq5 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq6 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq7 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq8 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq9 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq10 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq11 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq12 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq13 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq14 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);

const uint64_t ARITH = 1;
const uint64_t ARITH_ECADD_DIFFERENT = 2;
const uint64_t ARITH_ECADD_SAME = 3;
const uint64_t ARITH_BN254_MULFP2 = 4;
const uint64_t ARITH_BN254_ADDFP2 = 5;
const uint64_t ARITH_BN254_SUBFP2 = 6;
const uint64_t ARITH_SECP256R1_ECADD_DIFFERENT = 7;
const uint64_t ARITH_SECP256R1_ECADD_SAME = 8;

const uint16_t PRIME_SECP256K1_CHUNKS[16] = { 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                                              0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                                              0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                                              0xFFFF, 0xFFFE, 0xFFFF, 0xFC2F };

const uint16_t PRIME_BN254_CHUNKS[16] = { 0x3064, 0x4E72, 0xE131, 0xA029,
                                          0xB850, 0x45B6, 0x8181, 0x585D,
                                          0x9781, 0x6A91, 0x6871, 0xCA8D,
                                          0x3C20, 0x8C16, 0xD87C, 0xFD47 };

const uint16_t PRIME_SECP256R1_CHUNKS[16] = { 0xFFFF, 0xFFFF, 0x0000, 0x0001,
                                              0x0000, 0x0000, 0x0000, 0x0000,
                                              0x0000, 0x0000, 0xFFFF, 0xFFFF,
                                              0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF };

const uint64_t PRIME_CHUNKS[3][16] = {
    {
        PRIME_SECP256K1_CHUNKS[0],
        PRIME_SECP256K1_CHUNKS[1],
        PRIME_SECP256K1_CHUNKS[2],
        PRIME_SECP256K1_CHUNKS[3],
        PRIME_SECP256K1_CHUNKS[4],
        PRIME_SECP256K1_CHUNKS[5],
        PRIME_SECP256K1_CHUNKS[6],
        PRIME_SECP256K1_CHUNKS[7],
        PRIME_SECP256K1_CHUNKS[8],
        PRIME_SECP256K1_CHUNKS[9],
        PRIME_SECP256K1_CHUNKS[10],
        PRIME_SECP256K1_CHUNKS[11],
        PRIME_SECP256K1_CHUNKS[12],
        PRIME_SECP256K1_CHUNKS[13],
        PRIME_SECP256K1_CHUNKS[14],
        PRIME_SECP256K1_CHUNKS[15],
    },
    {
        PRIME_BN254_CHUNKS[0],
        PRIME_BN254_CHUNKS[1],
        PRIME_BN254_CHUNKS[2],
        PRIME_BN254_CHUNKS[3],
        PRIME_BN254_CHUNKS[4],
        PRIME_BN254_CHUNKS[5],
        PRIME_BN254_CHUNKS[6],
        PRIME_BN254_CHUNKS[7],
        PRIME_BN254_CHUNKS[8],
        PRIME_BN254_CHUNKS[9],
        PRIME_BN254_CHUNKS[10],
        PRIME_BN254_CHUNKS[11],
        PRIME_BN254_CHUNKS[12],
        PRIME_BN254_CHUNKS[13],
        PRIME_BN254_CHUNKS[14],
        PRIME_BN254_CHUNKS[15],
    },
    {
        PRIME_SECP256R1_CHUNKS[0],
        PRIME_SECP256R1_CHUNKS[1],
        PRIME_SECP256R1_CHUNKS[2],
        PRIME_SECP256R1_CHUNKS[3],
        PRIME_SECP256R1_CHUNKS[4],
        PRIME_SECP256R1_CHUNKS[5],
        PRIME_SECP256R1_CHUNKS[6],
        PRIME_SECP256R1_CHUNKS[7],
        PRIME_SECP256R1_CHUNKS[8],
        PRIME_SECP256R1_CHUNKS[9],
        PRIME_SECP256R1_CHUNKS[10],
        PRIME_SECP256R1_CHUNKS[11],
        PRIME_SECP256R1_CHUNKS[12],
        PRIME_SECP256R1_CHUNKS[13],
        PRIME_SECP256R1_CHUNKS[14],
        PRIME_SECP256R1_CHUNKS[15],
    }
};

const uint64_t PRIME_SECP256K1_INDEX = 0;
const uint64_t PRIME_BN254_INDEX = 1;
const uint64_t PRIME_SECP256R1_INDEX = 2;

const uint64_t EQ_INDEX_TO_CARRY_INDEX[15] = { 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 0, 0, 1, 2 };

class ArithInfo
{
public:
    uint64_t selEq[8];
    vector<uint64_t> eqIndexes;
    uint64_t primeIndex;
    bool checkAliasFree;
    bool checkDifferent;
    mpz_class prime;
    mpz_class a;
    string name;
    string curve;
    ArithInfo() :
        primeIndex(false),
        checkAliasFree(false),
        checkDifferent(false) {
            for (uint64_t i=0; i<8; i++)
            {
                selEq[i] = 0;
            }
        };
    string toString(void)
    {
        string s;

        s += "name=" + name;

        s += " curve=" + curve;

        s += " prime=" + prime.get_str(16);

        if (a != 0)
            s += " a=" + a.get_str(16);

        s += " selEq=[";
        for (uint64_t i=0; i<8; i++)
        {
            s += to_string(selEq[i]);
            if (i != 7) s += ",";
            else s += "]";
        }
        
        s += " eqIndexes=[";
        for (uint64_t i=0; i<eqIndexes.size(); i++)
        {
            s += to_string(eqIndexes[i]);
            if (i != (eqIndexes.size() - 1)) s += ",";
            else s += "]";
        }

        s += " primeIndex=" + to_string(primeIndex);

        s += " checkAliasFree=" + to_string(checkAliasFree);

        s += " checkDifferent=" + to_string(checkDifferent);

        return s;
    }
};

void getArithInfo(uint64_t arithEq, ArithInfo &arithInfo)
{
    switch (arithEq)
    {
        case ARITH:
            arithInfo.selEq[0] = 1;
            arithInfo.eqIndexes.push_back(0);
            arithInfo.name = "ARITH";
            break;

        case ARITH_ECADD_DIFFERENT:
            arithInfo.selEq[1] = 1;
            arithInfo.eqIndexes.push_back(1); // s.diff, x3, y3
            arithInfo.eqIndexes.push_back(3);
            arithInfo.eqIndexes.push_back(4);
            arithInfo.primeIndex = PRIME_SECP256K1_INDEX;
            arithInfo.checkAliasFree = true;
            arithInfo.checkDifferent = true;
            arithInfo.prime = Secp256k1p_prime;
            arithInfo.name = "ARITH_ECADD_DIFFERENT";
            arithInfo.curve = "SECP256K1";
            break;

        case ARITH_ECADD_SAME:
            arithInfo.selEq[2] = 1;
            arithInfo.eqIndexes.push_back(2); // s.diff, x3, y3
            arithInfo.eqIndexes.push_back(3);
            arithInfo.eqIndexes.push_back(4);
            arithInfo.primeIndex = PRIME_SECP256K1_INDEX;
            arithInfo.checkAliasFree = true;
            arithInfo.checkDifferent = false;
            arithInfo.prime = Secp256k1p_prime;
            arithInfo.name = "ARITH_ECADD_SAME";
            arithInfo.curve = "SECP256K1";
            break;

        case ARITH_BN254_MULFP2:
            arithInfo.selEq[3] = 1;
            arithInfo.eqIndexes.push_back(5); // x3, y3
            arithInfo.eqIndexes.push_back(6);
            arithInfo.primeIndex = PRIME_BN254_INDEX;
            arithInfo.checkAliasFree = true;
            arithInfo.checkDifferent = false;
            arithInfo.prime = BN254p_prime;
            arithInfo.name = "ARITH_BN254_MULFP2";
            arithInfo.curve = "BN254";
            break;

        case ARITH_BN254_ADDFP2:
            arithInfo.selEq[4] = 1;
            arithInfo.eqIndexes.push_back(7); // x3, y3
            arithInfo.eqIndexes.push_back(8);
            arithInfo.primeIndex = PRIME_BN254_INDEX;
            arithInfo.checkAliasFree = true;
            arithInfo.checkDifferent = false;
            arithInfo.prime = BN254p_prime;
            arithInfo.name = "ARITH_BN254_ADDFP2";
            arithInfo.curve = "BN254";
            break;

        case ARITH_BN254_SUBFP2:
            arithInfo.selEq[5] = 1;
            arithInfo.eqIndexes.push_back(9); // x3, y3
            arithInfo.eqIndexes.push_back(10);
            arithInfo.primeIndex = PRIME_BN254_INDEX;
            arithInfo.checkAliasFree = true;
            arithInfo.checkDifferent = false;
            arithInfo.prime = BN254p_prime;
            arithInfo.name = "ARITH_BN254_SUBFP2";
            arithInfo.curve = "BN254";
            break;

        case ARITH_SECP256R1_ECADD_DIFFERENT:
            arithInfo.selEq[6] = 1;
            arithInfo.eqIndexes.push_back(11); // s.diff, x3, y3
            arithInfo.eqIndexes.push_back(13);
            arithInfo.eqIndexes.push_back(14);
            arithInfo.primeIndex = PRIME_SECP256R1_INDEX;
            arithInfo.checkAliasFree = true;
            arithInfo.checkDifferent = true;
            arithInfo.prime = pSecp256r1_prime;
            arithInfo.name = "ARITH_SECP256R1_ECADD_DIFFERENT";
            arithInfo.curve = "SECP256R1";
            break;

        case ARITH_SECP256R1_ECADD_SAME:
            arithInfo.selEq[7] = 1;
            arithInfo.eqIndexes.push_back(12); // s.diff, x3, y3
            arithInfo.eqIndexes.push_back(13);
            arithInfo.eqIndexes.push_back(14);
            arithInfo.primeIndex = PRIME_SECP256R1_INDEX;
            arithInfo.checkAliasFree = true;
            arithInfo.checkDifferent = false;
            arithInfo.a = aSecp256r1;
            arithInfo.prime = pSecp256r1_prime;
            arithInfo.name = "ARITH_SECP256R1_ECADD_SAME";
            arithInfo.curve = "SECP256R1";
            break;

        default:
            zklog.error("ArithExecutor::getArithInfo() invalid arithEq=" + to_string(arithEq));
            exitProcess();
    }
}

void ArithExecutor::execute (vector<ArithAction> &action, ArithCommitPols &pols)
{
    // Get a scalar with the bn254 prime
    mpz_class pBN254;
    pBN254.set_str(fq.toString(fq.negOne(), 16), 16);
    pBN254++;

    // Check that we have enough room in polynomials  TODO: Do this check in JS
    if (action.size()*32 > N)
    {
        zklog.error("ArithExecutor::execute() Too many Arith entries=" + to_string(action.size()) + " > N/32=" + to_string(N/32));
        exitProcess();
    }

    // Split actions into bytes
    vector<ArithActionBytes> input;
    for (uint64_t i=0; i<action.size(); i++)
    {
        uint64_t dataSize;
        ArithActionBytes actionBytes;

        actionBytes.x1 = action[i].x1;
        actionBytes.y1 = action[i].y1;
        actionBytes.x2 = action[i].x2;
        actionBytes.y2 = action[i].y2;
        actionBytes.x3 = action[i].x3;
        actionBytes.y3 = action[i].y3;
        actionBytes.arithEq = action[i].arithEq;

        dataSize = 16;
        scalar2ba16(actionBytes._x1, dataSize, action[i].x1);
        dataSize = 16;
        scalar2ba16(actionBytes._y1, dataSize, action[i].y1);
        dataSize = 16;
        scalar2ba16(actionBytes._x2, dataSize, action[i].x2);
        dataSize = 16;
        scalar2ba16(actionBytes._y2, dataSize, action[i].y2);
        dataSize = 16;
        scalar2ba16(actionBytes._x3, dataSize, action[i].x3);
        dataSize = 16;
        scalar2ba16(actionBytes._y3, dataSize, action[i].y3);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq0, dataSize, action[i].selEq0);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq1, dataSize, action[i].selEq1);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq2, dataSize, action[i].selEq2);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq3, dataSize, action[i].selEq3);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq4, dataSize, action[i].selEq4);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq5, dataSize, action[i].selEq5);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq6, dataSize, action[i].selEq6);

        memset(actionBytes._s, 0, sizeof(actionBytes._s));
        memset(actionBytes._q0, 0, sizeof(actionBytes._q0));
        memset(actionBytes._q1, 0, sizeof(actionBytes._q1));
        memset(actionBytes._q2, 0, sizeof(actionBytes._q2));

        input.push_back(actionBytes);
    }

    mpz_class sScalar, q0, q1, q2;

    // Process all the inputs
//#pragma omp parallel for // TODO: Disabled since OMP decreases performance, probably due to cache invalidations
    for (uint64_t i = 0; i < input.size(); i++)
    {
#ifdef LOG_BINARY_EXECUTOR
        if (i%10000 == 0)
        {
            zklog.info("Computing binary pols " + to_string(i) + "/" + to_string(input.size()));
        }
#endif
        // TODO: if not have x1, need to componse it


        ArithInfo arithInfo;
        getArithInfo(input[i].arithEq, arithInfo);

        // In the following, recall that we can only work with unsiged integers of 256 bits.
        // Therefore, as the quotient needs to be represented in our VM, we need to know
        // the worst negative case and add an offset so that the resulting name is never negative.
        // Then, this offset is also added in the PIL constraint to ensure the equality.
        // Note1: Since we can choose whether the quotient is positive or negative, we choose it so
        //        that the added offset is the lowest.
        // Note2: x1,x2,y1,y2 can be assumed to be alias free, as this is the pre condition in the Arith SM.
        //        I.e, x1,x2,y1,y2 ∈ [0, 2^256-1].

        bool calculateS = false;
        
        if (input[i].arithEq == ARITH_ECADD_DIFFERENT || input[i].arithEq == ARITH_SECP256R1_ECADD_DIFFERENT)
        {
            calculateS = true;

            switch (arithInfo.primeIndex)
            {
                case PRIME_SECP256K1_INDEX:
                {
                    // Convert scalars to field elements
                    RawFec::Element x1;
                    RawFec::Element y1;
                    RawFec::Element x2;
                    RawFec::Element y2;
                    Secp256k1p.fromMpz(x1, input[i].x1.get_mpz_t());
                    Secp256k1p.fromMpz(y1, input[i].y1.get_mpz_t());
                    Secp256k1p.fromMpz(x2, input[i].x2.get_mpz_t());
                    Secp256k1p.fromMpz(y2, input[i].y2.get_mpz_t());

                    // Calculate s=(y2-y1)/(x2-x1)
                    RawFec::Element s;
                    RawFec::Element aux1, aux2;
                    Secp256k1p.sub(aux1, y2, y1);
                    Secp256k1p.sub(aux2, x2, x1);
                    if (fec.isZero(aux2))
                    {
                        zklog.error("ArithExecutor::execute() PRIME_SECP256K1_INDEX divide by zero calculating S for input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                        exitProcess();
                    }
                    Secp256k1p.div(s, aux1, aux2);

                    // Get s as a scalar
                    Secp256k1p.toMpz(sScalar.get_mpz_t(), s);

                    // Check result
                    mpz_class pq0;
                    pq0 = sScalar*input[i].x2 - sScalar*input[i].x1 - input[i].y2 + input[i].y1; // Worst values are {-2^256*(2^256-1),2^256*(2^256-1)}
                    q0 = pq0/arithInfo.prime;
                    if ((pq0 - arithInfo.prime*q0) != 0)
                    {
                        zklog.error("ArithExecutor::execute() PRIME_SECP256K1_INDEX with the calculated q0 the residual is not zero (diff point) input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                        exitProcess();
                    } 
                    q0 += ScalarTwoTo257;
                    if(q0 < 0)
                    {
                        zklog.error("ArithExecutor::execute() PRIME_SECP256K1_INDEX the q0 with offset is negative (diff point). Actual value: " + q0.get_str(16) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                        exitProcess();
                    }
                    break;
                }
                case PRIME_SECP256R1_INDEX:
                {
                    // Convert scalars to field elements
                    RawpSecp256r1::Element x1;
                    RawpSecp256r1::Element y1;
                    RawpSecp256r1::Element x2;
                    RawpSecp256r1::Element y2;
                    pSecp256r1.fromMpz(x1, input[i].x1.get_mpz_t());
                    pSecp256r1.fromMpz(y1, input[i].y1.get_mpz_t());
                    pSecp256r1.fromMpz(x2, input[i].x2.get_mpz_t());
                    pSecp256r1.fromMpz(y2, input[i].y2.get_mpz_t());

                    // Calculate s=(y2-y1)/(x2-x1)
                    RawpSecp256r1::Element s;
                    RawpSecp256r1::Element aux1, aux2;
                    pSecp256r1.sub(aux1, y2, y1);
                    pSecp256r1.sub(aux2, x2, x1);
                    if (pSecp256r1.isZero(aux2))
                    {
                        zklog.error("ArithExecutor::execute() PRIME_SECP256R1_INDEX divide by zero calculating S for input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                        exitProcess();
                    }
                    pSecp256r1.div(s, aux1, aux2);

                    // Get s as a scalar
                    pSecp256r1.toMpz(sScalar.get_mpz_t(), s);

                    // Check result
                    mpz_class pq0;
                    pq0 = sScalar*input[i].x2 - sScalar*input[i].x1 - input[i].y2 + input[i].y1; // Worst values are {-2^256*(2^256-1),2^256*(2^256-1)}
                    q0 = pq0/arithInfo.prime;
                    if ((pq0 - arithInfo.prime*q0) != 0)
                    {
                        zklog.error("ArithExecutor::execute() PRIME_SECP256R1_INDEX with the calculated q0 the residual is not zero (diff point) input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                        exitProcess();
                    } 
                    q0 += ScalarTwoTo257;
                    if(q0 < 0)
                    {
                        zklog.error("ArithExecutor::execute() PRIME_SECP256R1_INDEX the q0 with offset is negative (diff point). Actual value: " + q0.get_str(16) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                        exitProcess();
                    }
                    break;
                }
                default:
                {
                    zklog.error("ArithExecutor::execute() Invalid arithInfo.primeIndex=" + to_string(arithInfo.primeIndex) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                    exitProcess();
                }
            }
        }
        else if (input[i].arithEq == ARITH_ECADD_SAME)
        {
            // Check prime index
            if (arithInfo.primeIndex != PRIME_SECP256K1_INDEX)
            {
                zklog.error("ArithExecutor::execute() ARITH_ECADD_SAME invalid arithInfo.primeIndex=" + to_string(arithInfo.primeIndex) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }

            calculateS = true;

            // Convert scalars to field elements
            RawFec::Element x1;
            RawFec::Element y1;
            RawFec::Element x2;
            RawFec::Element y2;
            Secp256k1p.fromMpz(x1, input[i].x1.get_mpz_t());
            Secp256k1p.fromMpz(y1, input[i].y1.get_mpz_t());
            Secp256k1p.fromMpz(x2, input[i].x2.get_mpz_t());
            Secp256k1p.fromMpz(y2, input[i].y2.get_mpz_t());

            // Calculate s = 3*x1*x1/(y1+y1)
            RawFec::Element s;
            RawFec::Element aux1, aux2;
            Secp256k1p.mul(aux1, x1, x1);
            Secp256k1p.fromUI(aux2, 3);
            Secp256k1p.mul(aux1, aux1, aux2);
            Secp256k1p.add(aux2, y1, y1);
            Secp256k1p.div(s, aux1, aux2);

            // Get s as a scalar
            Secp256k1p.toMpz(sScalar.get_mpz_t(), s);

            // Check result
            mpz_class pq0;
            pq0 = sScalar*2*input[i].y1 - 3*input[i].x1*input[i].x1; // Worst values are {-3*(2^256-1)**2,2*(2^256-1)**2}
                                                                     // with |-3*(2^256-1)**2| > 2*(2^256-1)**2
            q0 = -(pq0/arithInfo.prime);
            if ((pq0 + arithInfo.prime*q0) != 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_ECADD_SAME with the calculated q0 the residual is not zero (same point) input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            } 
            q0 += ScalarTwoTo258;
            if (q0 < 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_ECADD_SAME the q0 with offset is negative (same point). Actual value: " + q0.get_str(16) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
        }
        else if (input[i].arithEq == ARITH_SECP256R1_ECADD_SAME)
        {
            // Check prime index
            if (arithInfo.primeIndex != PRIME_SECP256R1_INDEX)
            {
                zklog.error("ArithExecutor::execute() ARITH_SECP256R1_ECADD_SAME invalid arithInfo.primeIndex=" + to_string(arithInfo.primeIndex) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }

            calculateS = true;

            // Get a from arithInfo
            if (arithInfo.a == 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_SECP256R1_ECADD_SAME got arithInfo.a=0 input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
            mpz_class a = arithInfo.a;

            // Convert scalars to field elements
            RawpSecp256r1::Element x1;
            RawpSecp256r1::Element y1;
            RawpSecp256r1::Element x2;
            RawpSecp256r1::Element y2;
            pSecp256r1.fromMpz(x1, input[i].x1.get_mpz_t());
            pSecp256r1.fromMpz(y1, input[i].y1.get_mpz_t());
            pSecp256r1.fromMpz(x2, input[i].x2.get_mpz_t());
            pSecp256r1.fromMpz(y2, input[i].y2.get_mpz_t());

            // Calculate s = (3*x1*x1 + a)/(y1+y1)
            RawpSecp256r1::Element s;
            RawpSecp256r1::Element aux1, aux2;
            pSecp256r1.mul(aux1, x1, x1);
            pSecp256r1.fromUI(aux2, 3);
            pSecp256r1.mul(aux1, aux1, aux2);
            RawpSecp256r1::Element afe;
            pSecp256r1.fromMpz(afe, a.get_mpz_t());
            pSecp256r1.add(aux1, aux1, afe);
            pSecp256r1.add(aux2, y1, y1);
            pSecp256r1.div(s, aux1, aux2);

            // Get s as a scalar
            pSecp256r1.toMpz(sScalar.get_mpz_t(), s);

            // Check result
            mpz_class pq0;
            pq0 = sScalar*2*input[i].y1 - 3*input[i].x1*input[i].x1 - a; // Worst values are {-3*(2^256-1)**2 - a,2*(2^256-1)**2 - a}
                                                                         // with |-3*(2^256-1)**2| > 2*(2^256-1)**2
            q0 = -(pq0/arithInfo.prime);
            if ((pq0 + arithInfo.prime*q0) != 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_SECP256R1_ECADD_SAME with the calculated q0 the residual is not zero (same point) input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            } 
            q0 += ScalarTwoTo258;
            if(q0 < 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_SECP256R1_ECADD_SAME the q0 with offset is negative (same point). Actual value: " + q0.get_str(16) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
        }
        else
        {
            //fec.fromUI(s, 0);
            q0 = 0;
        }

        if (calculateS)
        {
            // Check q1
            mpz_class pq1;
            pq1 = sScalar*sScalar - input[i].x1 - input[i].x2 - input[i].x3; /// Worst values are {-3*(2^256-1),(2^256-1)**2}
                                                                             // with (2^256-1)**2 > |-3*(2^256-1)|
            q1 = pq1/arithInfo.prime;
            if ((pq1 - arithInfo.prime*q1) != 0)
            {
                zklog.error("ArithExecutor::execute() calculateS with the calculated q1 the residual is not zero input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
            // offset 
            q1 += 4; //2**2
            if(q1 < 0)
            {
                zklog.error("ArithExecutor::execute() calculateS the q1 with offset is negative (point addition). Actual value: " + q1.get_str(16) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }

            // Check q2
            mpz_class pq2;
            pq2 = sScalar*input[i].x1 - sScalar*input[i].x3 - input[i].y1 - input[i].y3; // Worst values are {-(2^256+1)*(2^256-1),(2^256-1)**2}
                                                                                         // with |-(2^256+1)*(2^256-1)| > (2^256-1)**2
            q2 = -(pq2/arithInfo.prime);
            if ((pq2 + arithInfo.prime*q2) != 0)
            {
                zklog.error("ArithExecutor::execute() calculateS with the calculated q2 the residual is not zero input=" + to_string(i) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
            //offset 
            q2 += ScalarTwoTo257;
            if(q2 < 0)
            {
                zklog.error("ArithExecutor::execute() calculateS the q2 with offset is negative (point addition). Actual value: " + q2.get_str(16) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }

        }        
        else if (input[i].arithEq == ARITH_BN254_MULFP2)
        {
            // Check q1
            mpz_class pq1;
            pq1 = input[i].x1*input[i].x2 - input[i].y1*input[i].y2 - input[i].x3; /// Worst values are {-2^256*(2^256-1),(2^256-1)**2}
                                                                                   // with |-2^256*(2^256-1)| > (2^256-1)**2
            q1 = -(pq1/pBN254);
            if ((pq1 + pBN254*q1) != 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_BN254_MULFP2 with the calculated q1 the residual is not zero input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
            // offset
            q1 += ScalarTwoTo259;
            if(q1 < 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_BN254_MULFP2 for the q1 with offset is negative (complex mul). Actual value: " + q1.get_str(16) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }

            // Check q2
            mpz_class pq2;
            pq2 = input[i].y1*input[i].x2 + input[i].x1*input[i].y2 - input[i].y3; // Worst values are {-(2^256-1),2*(2^256-1)}
                                                                                   // with 2*(2^256-1) > |-(2^256-1)|
            q2 = pq2/pBN254;
            if ((pq2 - pBN254*q2) != 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_BN254_MULFP2 with the calculated q2 the residual is not zero input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
            // offset
            q2 += 8; //2**3
            if(q2 < 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_BN254_MULFP2 the q2 with offset is negative (complex mul). Actual value: " + q2.get_str(16) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
        }
        else if (input[i].arithEq == ARITH_BN254_ADDFP2)
        {
            // Check q1
            mpz_class pq1;
            pq1 = input[i].x1 + input[i].x2 - input[i].x3; // Worst values are {-(2^256-1),2*(2^256-1)}
                                                           // with 2*(2^256-1) > |-(2^256-1)|
            q1 = pq1/pBN254;
            if ((pq1 - pBN254*q1) != 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_BN254_ADDFP2 with the calculated q1 the residual is not zero input=" + to_string(i) + " input=" + input[i].toString());
                exitProcess();
            }
            //offset
            q1 += 8; //2**3
            if(q1 < 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_BN254_ADDFP2 the q1 with offset is negative (complex add). Actual value: " + q1.get_str(16) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }

            // Check q2
            mpz_class pq2;
            pq2 = input[i].y1 + input[i].y2 - input[i].y3; // Worst values are {-(2^256-1),2*(2^256-1)}
                                                           // with 2*(2^256-1) > |-(2^256-1)|
            q2 = pq2/pBN254;
            if ((pq2 - pBN254*q2) != 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_BN254_ADDFP2 with the calculated q2 the residual is not zero input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
            //offset
            q2 += 8; //2**3
            if(q2 < 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_BN254_ADDFP2 the q2 with offset is negative (complex add). Actual value: " + q2.get_str(16) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
        }
        else if (input[i].arithEq == ARITH_BN254_SUBFP2)
        {
            // Check q1
            mpz_class pq1;
            pq1 = input[i].x1 - input[i].x2 - input[i].x3; // Worst values are {-2*(2^256-1),(2^256-1)}
                                                           // with |-2*(2^256-1)| > (2^256-1)
            q1 = -(pq1/pBN254);
            if ((pq1 + pBN254*q1) != 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_BN254_SUBFP2 with the calculated q1 the residual is not zero input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
            //offset
            q1 += 8; //2**3
            if(q1 < 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_BN254_SUBFP2 the q1 with offset is negative (complex sub). Actual value: " + q1.get_str(16) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
            // Check q2
            mpz_class pq2;
            pq2 = input[i].y1 - input[i].y2 - input[i].y3; // Worst values are {-2*(2^256-1),(2^256-1)}
                                                           // with |-2*(2^256-1)| > (2^256-1)
                                                           
            q2 = -(pq2/pBN254);
            if ((pq2 + pBN254*q2) != 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_BN254_SUBFP2 with the calculated q2 the residual is not zero input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
            //offset
            q2 += 8; //2**3
            if(q2 < 0)
            {
                zklog.error("ArithExecutor::execute() ARITH_BN254_SUBFP2 the q2 with offset is negative (complex sub). Actual value: " + q2.get_str(16) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                exitProcess();
            }
        }
        else
        {
            q1 = 0;
            q2 = 0;
        }

        uint64_t dataSize;
        dataSize = 16;
        scalar2ba16(input[i]._s, dataSize, sScalar);
        dataSize = 16;
        scalar2ba16(input[i]._q0, dataSize, q0);
        dataSize = 16;
        scalar2ba16(input[i]._q1, dataSize, q1);
        dataSize = 16;
        scalar2ba16(input[i]._q2, dataSize, q2);
    }
    
    // Process all the inputs
//#pragma omp parallel for // TODO: Disabled since OMP decreases performance, probably due to cache invalidations
    for (uint64_t i = 0; i < input.size(); i++)
    {
        uint64_t offset = i*32;
        bool xAreDifferent = false;
        bool valueLtPrime = false;
        
        // Get arith info
        ArithInfo arithInfo;
        getArithInfo(input[i].arithEq, arithInfo);

        for (uint64_t step=0; step<32; step++)
        {
            uint64_t index = offset + step;
            uint64_t nextIndex = (index + 1) % N;
            uint64_t step16 = step % 16;
            if (step16 == 0)
            {
                valueLtPrime = false;
            }
            for (uint64_t j=0; j<16; j++)
            {
                pols.x1[j][index] = fr.fromU64(input[i]._x1[j]);
                pols.y1[j][index] = fr.fromU64(input[i]._y1[j]);
                pols.x2[j][index] = fr.fromU64(input[i]._x2[j]);
                pols.y2[j][index] = fr.fromU64(input[i]._y2[j]);
                pols.x3[j][index] = fr.fromU64(input[i]._x3[j]);
                pols.y3[j][index] = fr.fromU64(input[i]._y3[j]);
                pols.s[j][index]  = fr.fromU64(input[i]._s[j]);
                pols.q0[j][index] = fr.fromU64(input[i]._q0[j]);
                pols.q1[j][index] = fr.fromU64(input[i]._q1[j]);
                pols.q2[j][index] = fr.fromU64(input[i]._q2[j]);
                if (j < (sizeof(arithInfo.selEq)/sizeof(uint64_t)))
                {
                    pols.selEq[j][index] = fr.fromU64(arithInfo.selEq[j]);
                }
            }

            // selEq1 (addition different points) is select need to check that points are diferent
            if (arithInfo.checkDifferent && (step < 16))
            {
                if (xAreDifferent == false)
                {
                    Goldilocks::Element delta = fr.sub(pols.x2[step][index], pols.x1[step][index]);
                    pols.xDeltaChunkInverse[index] = fr.isZero(delta) ? fr.zero() : glp.inv(delta);
                    xAreDifferent = fr.isZero(delta) ? false : true;
                }
                pols.xAreDifferent[nextIndex] = xAreDifferent ? fr.one() : fr.zero();
            }

            // If either selEq3,selEq4,selEq5,selEq6 is selected, we need to ensure that x3, y3 is alias free.
            // Recall that selEq3 work over the Secp256k1 curve, and selEq4,selEq5,selEq6 work over the BN254 curve.
            if (arithInfo.checkAliasFree)
            {
                Goldilocks::Element chunkValue = step < 16 ? pols.x3[15 - step16][offset] : pols.y3[15 - step16][offset];
                uint64_t chunkPrime = PRIME_CHUNKS[arithInfo.primeIndex][step16];

                bool chunkLtPrime = valueLtPrime ? false : (fr.toU64(chunkValue) < chunkPrime);
                valueLtPrime = valueLtPrime || chunkLtPrime;
                pols.chunkLtPrime[index] = chunkLtPrime ? fr.one() : fr.zero();
                pols.valueLtPrime[nextIndex] = valueLtPrime ? fr.one() : fr.zero();
            }
        }

        mpz_class carry[3] = {0, 0, 0};
        mpz_class eq[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        mpz_class auxScalar;
        for (uint64_t step=0; step<32; step++)
        {
            for (uint64_t k=0; k<arithInfo.eqIndexes.size(); k++)
            {
                uint64_t eqIndex = arithInfo.eqIndexes[k];
                uint64_t carryIndex = EQ_INDEX_TO_CARRY_INDEX[eqIndex];
                switch(eqIndex)
                {
                    case 0:  eq[eqIndex] = eq0(fr, pols, step, offset); break;
                    case 1:  eq[eqIndex] = eq1(fr, pols, step, offset); break;
                    case 2:  eq[eqIndex] = eq2(fr, pols, step, offset); break;
                    case 3:  eq[eqIndex] = eq3(fr, pols, step, offset); break;
                    case 4:  eq[eqIndex] = eq4(fr, pols, step, offset); break;
                    case 5:  eq[eqIndex] = eq5(fr, pols, step, offset); break;
                    case 6:  eq[eqIndex] = eq6(fr, pols, step, offset); break;
                    case 7:  eq[eqIndex] = eq7(fr, pols, step, offset); break;
                    case 8:  eq[eqIndex] = eq8(fr, pols, step, offset); break;
                    case 9:  eq[eqIndex] = eq9(fr, pols, step, offset); break;
                    case 10: eq[eqIndex] = eq10(fr, pols, step, offset); break;
                    case 11: eq[eqIndex] = eq11(fr, pols, step, offset); break;
                    case 12: eq[eqIndex] = eq12(fr, pols, step, offset); break;
                    case 13: eq[eqIndex] = eq13(fr, pols, step, offset); break;
                    case 14: eq[eqIndex] = eq14(fr, pols, step, offset); break;
                    default:
                        zklog.error("ArithExecutor::execute() invalid eqIndex=" + to_string(eqIndex) + " input[" + to_string(i) + "]=[" + input[i].toString() + "]");
                        exitProcess();
                }
                pols.carry[carryIndex][offset + step] = fr.fromScalar(carry[carryIndex]);
                if (((eq[eqIndex] + carry[carryIndex]) % ScalarTwoTo16) != 0)
                {
                    zklog.error("ArithExecutor::execute() For input " + to_string(i) +
                        " step=" + to_string(step) + " k=" + to_string(k) + 
                        " eq[" + to_string(eqIndex) + "]=" + eq[eqIndex].get_str(16) +
                        " and carry[" + to_string(carryIndex) + "]=" + carry[carryIndex].get_str(16) +
                        " do not sum 0 mod 2 to 16; input[" + to_string(i) + "]=[" + input[i].toString() + "] arithInfo=[" + arithInfo.toString() + "]");
                    exitProcess();
                }
                carry[carryIndex] = (eq[eqIndex] + carry[carryIndex]) / ScalarTwoTo16;
            }
        }

        pols.resultEq[offset + 31] = fr.one();
    }
    
    zklog.info("ArithExecutor successfully processed " + to_string(action.size()) + " arith actions (" + to_string((double(action.size())*32*100)/N) + "%)");
}