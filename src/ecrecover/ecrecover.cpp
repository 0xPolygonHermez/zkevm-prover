#include "ecrecover.hpp"
#include "zklog.hpp"
#include "ffiasm/fnec.hpp"
#include "ffiasm/fec.hpp"
#include "exit_process.hpp"
#include "definitions.hpp"
#include "keccak_wrapper.hpp"
#include "zkglobals.hpp"

mpz_class FNEC("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
mpz_class FNEC_MINUS_ONE = FNEC - 1;
mpz_class FNEC_DIV_TWO("0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0");
mpz_class FPEC("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
mpz_class FPEC_NON_SQRT("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");

mpz_class ECGX("0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
mpz_class ECGY("0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
mpz_class ADDRESS_MASK("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");

//
// Field operations:
//
mpz_class invFpEc(const mpz_class &a);
mpz_class invFnEc(const mpz_class &a);
mpz_class mulFpEc(const mpz_class &a, const mpz_class &b);
mpz_class mulFnEc(const mpz_class &a, const mpz_class &b);
mpz_class addFpEc(const mpz_class &a, const mpz_class &b);
mpz_class sqFpEc(const mpz_class &a);

//
// EC operations with affine format:
//
void dblPointEc(const mpz_class &p1_x, const mpz_class &p1_y,
                mpz_class &p3_x, mpz_class &p3_y);
void addPointEc(const mpz_class &p1_x, const mpz_class &p1_y,
                const mpz_class &p2_x, const mpz_class &p2_y,
                mpz_class &p3_x, mpz_class &p3_y);
// double scalar multiplication
void mulPointEc(const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &k1,
                const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &k2,
                mpz_class &p3_x, mpz_class &p3_y);
//
// EC operations with Jabobian format:
//
void dblPointEcJacobian(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                        RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3);
void addPointEcJacobian(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                        const RawFec::Element &x2, const RawFec::Element &y2, const RawFec::Element &z2,
                        RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3);
// pass some precomputed values to speed up the addition
void addPointEcJacobian(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                        const RawFec::Element &zz1, const RawFec::Element &zzz1,
                        const RawFec::Element &x2, const RawFec::Element &y2, const RawFec::Element &z2,
                        RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3);
// doubles or adds depending on arguments
void generalAddPointEcJacobian(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                               const bool p1_empty,
                               const RawFec::Element &x2, const RawFec::Element &y2, const RawFec::Element &z2,
                               const bool p2_empty,
                               RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3,
                               bool &p3_empty);
// doubles or adds depending on arguments, passing precomputed values to speed up the addition
void generalAddPointEcJacobian(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                               const RawFec::Element &zz1, const RawFec::Element &zzz1,
                               const bool p1_empty,
                               const RawFec::Element &x2, const RawFec::Element &y2, const RawFec::Element &z2,
                               const bool p2_empty,
                               RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3,
                               bool &p3_empty);
// double scalar multiplication, operations in chunks of 2 bits
void mulPointEcJacobian(const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &k1,
                        const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &k2,
                        mpz_class &p3_x, mpz_class &p3_y); // Jacobian conversion inside
// double scalar multiplication, operations in chunks of 2 bits
void mulPointEcJacobian(const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &p1_z, const mpz_class &k1,
                        const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &p2_z, const mpz_class &k2,
                        mpz_class &p3_x, mpz_class &p3_y, mpz_class &p3_z);
// double scalar multiplication, bit by bit
void mulPointEcJacobian1bit(const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &p1_z, const mpz_class &k1,
                            const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &p2_z, const mpz_class &k2,
                            mpz_class &p3_x, mpz_class &p3_y, mpz_class &p3_z);
// double scalar multiplication, bit by bit
void mulPointEcJacobian1bit(const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &k1,
                            const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &k2,
                            mpz_class &p3_x, mpz_class &p3_y); // Jacobian conversion inside
// double scalar multiplication, bit by bit SAVE 
int mulPointEcJacobian1bitSave(const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &k1,
                            const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &k2,
                            mpz_class &p3_x, mpz_class &p3_y, RawFec::Element* buffer, int nthreads = 16); // Jacobian conversion inside

inline void Jacobian2Affine(const mpz_class &x, const mpz_class &y, const mpz_class &z, mpz_class &x_out, mpz_class &y_out);

inline void Jacobian2Affine(const RawFec::Element &x, const RawFec::Element &y, const RawFec::Element &z, RawFec::Element &x_out,
                     RawFec::Element &y_out);
//
// ECrecover:
//
ECRecoverResult ECRecover(mpz_class &signature, mpz_class &r, mpz_class &s, mpz_class &v, bool bPrecompiled, mpz_class &address)
{
    // Set the ECRecover s upper limit
    mpz_class ecrecover_s_upperlimit;
    if (bPrecompiled)
    {
        ecrecover_s_upperlimit = FNEC_MINUS_ONE;
    }
    else
    {
        ecrecover_s_upperlimit = FNEC_DIV_TWO;
    }
    // Check that r is in the range [1, FNEC-1]
    if (r == 0)
    {
        zklog.error("ECRecover() found r=0");
        return ECR_R_IS_ZERO;
    }
    if (r > FNEC_MINUS_ONE)
    {
        zklog.error("ECRecover() found r>FNEC_MINUS_ONE r=" + r.get_str(16));
        return ECR_R_IS_TOO_BIG;
    }

    // Check that s is in the range [1, ecrecover_s_upperlimit]
    if (s == 0)
    {
        zklog.error("ECRecover() found s=0");
        return ECR_S_IS_ZERO;
    }
    if (s > ecrecover_s_upperlimit)
    {
        zklog.error("ECRecover() found s>ecrecover_s_upperlimit s=" + s.get_str(16) + " ecrecover_s_upperlimit=" + ecrecover_s_upperlimit.get_str(16));
        return ECR_S_IS_TOO_BIG;
    }

    // Calculate the inverse of r
    mpz_class ecrecover_r_inv;
    ecrecover_r_inv = invFnEc(r);

    // Calculate the parity of v
    int ecrecover_v_parity;
    if (v == 0x1b)
    {
        ecrecover_v_parity = 0;
    }
    else if (v == 0x1c)
    {
        ecrecover_v_parity = 1;
    }
    else
    {
        zklog.error("ECRecover() found invalid v=" + v.get_str(16));
        return ECR_V_INVALID;
    }
    // Curve is y^2 = x^3 + 7  -->  Calculate y from x=r
    mpz_class ecrecover_y;
    mpz_class aux1, aux2, aux3, aux4;
    aux1 = mulFpEc(r, r);
    aux2 = mulFpEc(aux1, r);
    aux1 = 7;
    aux3 = addFpEc(aux2, aux1);

    if (aux3 == 0)
    {
        ecrecover_y = 0;
    }
    else
    {
        sqrtF3mod4(ecrecover_y, aux3);
        if (ecrecover_y == 0)
        {
            zklog.error("ECRecover() found y^2 without root=" + aux3.get_str(16));
            return ECR_NO_SQRT_Y;
        }
    }
    assert(ecrecover_y < FPEC);

    // parity:
    int bit0 = mpz_tstbit(ecrecover_y.get_mpz_t(), 0);
    if (bit0 + ecrecover_v_parity - 1 == 0)
    {
        ecrecover_y = FPEC - ecrecover_y;
    }

    // Calculate the point of the curve
    mpz_class k1 = FNEC - mulFnEc(signature, ecrecover_r_inv);
    mpz_class k2 = mulFnEc(s, ecrecover_r_inv);

    mpz_class p1_x = ECGX;
    mpz_class p1_y = ECGY;

    mpz_class p2_x = r;
    mpz_class p2_y = ecrecover_y;

    mpz_class p3_x;
    mpz_class p3_y;
#if 0
    mulPointEc(p1_x, p1_y, k1, p2_x, p2_y, k2, p3_x, p3_y);
#else
    mpz_class p1_z = 1;
    mpz_class p2_z = 1;
    mpz_class p3_z;
    mulPointEcJacobian(p1_x, p1_y, k1, p2_x, p2_y, k2, p3_x, p3_y);
    /*RawFec::Element buffer[6+4+8*(512)]; // 6: p1_x, p1_y, p1_z, p2_x, p2_y, p2_z
                                         // 4: 0, p1, p2, p1+p2
                                         // 8*(512): 512 points of the curve
    mulPointEcJacobian1bit_save(p1_x, p1_y, k1, p2_x, p2_y, k2, p3_x, p3_y, buffer);
    mulPointEcJacobian1bit_assert(p1_x, p1_y, k1, p2_x, p2_y, k2, p3_x, p3_y, buffer);*/

#endif

    assert(p3_x < FPEC);
    assert(p3_y < FPEC);

    // generate keccak of public key to obtain ethereum address
    unsigned char outputHash[32];
    unsigned char inputHash[64];
    mpz_export(inputHash, nullptr, 0, 1, 0, 0, p3_x.get_mpz_t());
    mpz_export(inputHash + 32, nullptr, 0, 1, 0, 0, p3_y.get_mpz_t());
    keccak(inputHash, 64, outputHash, 32);
    mpz_class keccakHash;
    mpz_import(keccakHash.get_mpz_t(), 32, 0, 1, 0, 0, outputHash);

    // for address take only last 20 bytes
    address = keccakHash & ADDRESS_MASK;

    return ECR_NO_ERROR;
}


int ECRecoverPrecalc(mpz_class &signature, mpz_class &r, mpz_class &s, mpz_class &v, bool bPrecompiled, RawFec::Element* buffer, int nthreads){

    // Set the ECRecoverPrecalc s upper limit
    mpz_class ecrecover_s_upperlimit;
    if (bPrecompiled)
    {
        ecrecover_s_upperlimit = FNEC_MINUS_ONE;
    }
    else
    {
        ecrecover_s_upperlimit = FNEC_DIV_TWO;
    }
    // Check that r is in the range [1, FNEC-1]
    if (r == 0)
    {
        zklog.error("ECRecoverPrecalc() found r=0");
        return -1; //ECR_R_IS_ZERO
    }
    if (r > FNEC_MINUS_ONE)
    {
        zklog.error("ECRecoverPrecalc() found r>FNEC_MINUS_ONE r=" + r.get_str(16));
        return -1; //ECR_R_IS_TOO_BIG;
    }

    // Check that s is in the range [1, ecrecover_s_upperlimit]
    if (s == 0)
    {
        zklog.error("ECRecoverPrecalc() found s=0");
        return -1;//ECR_S_IS_ZERO;
    }
    if (s > ecrecover_s_upperlimit)
    {
        zklog.error("ECRecoverPrecalc() found s>ecrecover_s_upperlimit s=" + s.get_str(16) + " ecrecover_s_upperlimit=" + ecrecover_s_upperlimit.get_str(16));
        return -1;//ECR_S_IS_TOO_BIG;
    }

    // Calculate the inverse of r
    mpz_class ecrecover_r_inv;
    ecrecover_r_inv = invFnEc(r);

    // Calculate the parity of v
    int ecrecover_v_parity;
    if (v == 0x1b)
    {
        ecrecover_v_parity = 0;
    }
    else if (v == 0x1c)
    {
        ecrecover_v_parity = 1;
    }
    else
    {
        zklog.error("ECRecoverPrecalc() found invalid v=" + v.get_str(16));
        return -1;//ECR_V_INVALID;
    }
    // Curve is y^2 = x^3 + 7  -->  Calculate y from x=r
    mpz_class ecrecover_y;
    mpz_class aux1, aux2, aux3, aux4;
    aux1 = mulFpEc(r, r);
    aux2 = mulFpEc(aux1, r);
    aux1 = 7;
    aux3 = addFpEc(aux2, aux1);

    if (aux3 == 0)
    {
        ecrecover_y = 0;
    }
    else
    {
        sqrtF3mod4(ecrecover_y, aux3);
        if (ecrecover_y == 0)
        {
            zklog.error("ECRecoverPrecalc() found y^2 without root=" + aux3.get_str(16));
            return -1;//ECR_NO_SQRT_Y;
        }
    }
    assert(ecrecover_y < FPEC);

    // parity:
    int bit0 = mpz_tstbit(ecrecover_y.get_mpz_t(), 0);
    if (bit0 + ecrecover_v_parity - 1 == 0)
    {
        ecrecover_y = FPEC - ecrecover_y;
    }

    // Calculate the point of the curve
    mpz_class k1 = FNEC - mulFnEc(signature, ecrecover_r_inv);
    mpz_class k2 = mulFnEc(s, ecrecover_r_inv);

    mpz_class p1_x = ECGX;
    mpz_class p1_y = ECGY;

    mpz_class p2_x = r;
    mpz_class p2_y = ecrecover_y;

    mpz_class p3_x;
    mpz_class p3_y;

    mpz_class p1_z = 1;
    mpz_class p2_z = 1;
    mpz_class p3_z;
    return mulPointEcJacobian1bitSave(p1_x, p1_y, k1, p2_x, p2_y, k2, p3_x, p3_y, buffer, nthreads);
   
}

//
// Implementation fiel operations:
//
mpz_class invFpEc(const mpz_class &value)
{
    RawFec::Element a;
    fec.fromMpz(a, value.get_mpz_t());
    if (fec.isZero(a))
    {
        zklog.error("invFpEc() Division by zero");
        exitProcess();
    }

    RawFec::Element r;
    fec.inv(r, a);

    mpz_class result;
    fec.toMpz(result.get_mpz_t(), r);

    return result;
}

mpz_class invFnEc(const mpz_class &value)
{
    RawFnec::Element a;
    fnec.fromMpz(a, value.get_mpz_t());
    if (fnec.isZero(a))
    {
        zklog.error("invFnEc() Division by zero");
        exitProcess();
    }

    RawFnec::Element r;
    fnec.inv(r, a);

    mpz_class result;
    fnec.toMpz(result.get_mpz_t(), r);

    return result;
}

mpz_class mulFpEc(const mpz_class &a, const mpz_class &b)
{
    return (a * b) % FPEC;
}

mpz_class mulFnEc(const mpz_class &a, const mpz_class &b)
{
    return (a * b) % FNEC;
}

mpz_class addFpEc(const mpz_class &a, const mpz_class &b)
{
    return (a + b) % FPEC;
}

mpz_class sqFpEc(const mpz_class &a)
{
    return (a * a) % FPEC;
}

//
// EC operations with affine format:
//
void dblPointEc(const mpz_class &p1_x, const mpz_class &p1_y,
                mpz_class &p3_x, mpz_class &p3_y)
{

    RawFec::Element x1, y1, x2, y2, x3, y3;
    fec.fromMpz(x1, p1_x.get_mpz_t());
    fec.fromMpz(y1, p1_y.get_mpz_t());
    x2 = x1;
    y2 = y1;

    RawFec::Element aux1, aux2, s;

    // s = 3*x1*x1/2*y1
    fec.mul(aux1, x1, x1);
    fec.fromUI(aux2, 3);
    fec.mul(aux1, aux1, aux2);
    fec.add(aux2, y1, y1);
    fec.div(s, aux1, aux2);
    // TODO: y1 == 0 => division by zero ==> how manage?

    // x3 = s*s - (x1+x2)
    fec.mul(aux1, s, s);
    fec.add(aux2, x1, x2);
    fec.sub(x3, aux1, aux2);

    // y3 = s*(x1-x3) - y1
    fec.sub(aux1, x1, x3);
    ;
    fec.mul(aux1, aux1, s);
    fec.sub(y3, aux1, y1);

    fec.toMpz(p3_x.get_mpz_t(), x3);
    fec.toMpz(p3_y.get_mpz_t(), y3);
}

void addPointEc(const mpz_class &p1_x, const mpz_class &p1_y,
                const mpz_class &p2_x, const mpz_class &p2_y,
                mpz_class &p3_x, mpz_class &p3_y)
{
    RawFec::Element x1, y1, x2, y2, x3, y3;
    fec.fromMpz(x1, p1_x.get_mpz_t());
    fec.fromMpz(y1, p1_y.get_mpz_t());
    fec.fromMpz(x2, p2_x.get_mpz_t());
    fec.fromMpz(y2, p2_y.get_mpz_t());

    RawFec::Element aux1, aux2, s;

    // s = (y2-y1)/(x2-x1)
    fec.sub(aux1, y2, y1);
    fec.sub(aux2, x2, x1);
    fec.div(s, aux1, aux2);
    // TODO: deltaX == 0 => division by zero ==> how manage?

    // x3 = s*s - (x1+x2)
    fec.mul(aux1, s, s);
    fec.add(aux2, x1, x2);
    fec.sub(x3, aux1, aux2);

    // y3 = s*(x1-x3) - y1
    fec.sub(aux1, x1, x3);
    ;
    fec.mul(aux1, aux1, s);
    fec.sub(y3, aux1, y1);

    fec.toMpz(p3_x.get_mpz_t(), x3);
    fec.toMpz(p3_y.get_mpz_t(), y3);
}

void mulPointEc(const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &k1,
                const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &k2,
                mpz_class &p3_x, mpz_class &p3_y)
{

    p3_x = 0;
    p3_y = 0;

    mpz_class p12_x;
    mpz_class p12_y;

    bool p12_empty = 0;
    if (p1_x != p2_x)
    {
        // p2.x != p1.x ==> p2 != p1
        p12_empty = false;
        addPointEc(p1_x, p1_y, p2_x, p2_y, p12_x, p12_y);
    }
    else if (p1_y == p2_y)
    {
        // p2 == p1
        p12_empty = false;
        dblPointEc(p1_x, p1_y, p12_x, p12_y);
    }
    else
    {
        // p2 == -p1
        p12_empty = true;
    }

    // start the loop
    mpz_t rawK1;
    mpz_t rawK2;

    mpz_init(rawK1);
    mpz_init(rawK2);

    mpz_set(rawK1, k1.get_mpz_t());
    mpz_set(rawK2, k2.get_mpz_t());

    bool p3_empty = true;

    for (int i = 255; i >= 0; --i)
    {

        // take next bits
        int bitk1 = mpz_tstbit(rawK1, i);
        int bitk2 = mpz_tstbit(rawK2, i);

        // add contribution depending on bits
        if (bitk1 == 1 && bitk2 == 0)
        {
            if (!p3_empty)
            {
                if (p3_x != p1_x)
                {
                    addPointEc(p3_x, p3_y, p1_x, p1_y, p3_x, p3_y);
                }
                else
                {
                    if (p3_y != p1_y)
                    {
                        p3_empty = true;
                    }
                    else
                    {
                        dblPointEc(p3_x, p3_y, p3_x, p3_y);
                    }
                }
            }
            else
            {
                p3_empty = false;
                mpz_set(p3_x.get_mpz_t(), p1_x.get_mpz_t());
                mpz_set(p3_y.get_mpz_t(), p1_y.get_mpz_t());
            }
        }
        else if (bitk1 == 0 && bitk2 == 1)
        {
            if (!p3_empty)
            {
                if (p3_x != p2_x)
                {
                    addPointEc(p3_x, p3_y, p2_x, p2_y, p3_x, p3_y);
                }
                else
                {
                    if (p3_y != p2_y)
                    {
                        p3_empty = true;
                    }
                    else
                    {
                        dblPointEc(p3_x, p3_y, p3_x, p3_y);
                    }
                }
            }
            else
            {
                p3_empty = false;
                mpz_set(p3_x.get_mpz_t(), p2_x.get_mpz_t());
                mpz_set(p3_y.get_mpz_t(), p2_y.get_mpz_t());
            }
        }
        else if (bitk1 == 1 && bitk2 == 1)
        {
            if (!p12_empty)
            {
                if (!p3_empty)
                {
                    if (p3_x != p12_x)
                    {
                        addPointEc(p3_x, p3_y, p12_x, p12_y, p3_x, p3_y);
                    }
                    else
                    {
                        if (p3_y != p12_y)
                        {
                            p3_empty = true;
                        }
                        else
                        {
                            dblPointEc(p3_x, p3_y, p3_x, p3_y);
                        }
                    }
                }
                else
                {
                    p3_empty = false;
                    mpz_set(p3_x.get_mpz_t(), p12_x.get_mpz_t());
                    mpz_set(p3_y.get_mpz_t(), p12_y.get_mpz_t());
                }
            }
        }
        // double p3
        if (!p3_empty and i != 0)
        {
            dblPointEc(p3_x, p3_y, p3_x, p3_y);
        }
    }
    mpz_clear(rawK1);
    mpz_clear(rawK2);
}

//
// EC operations with Jabobian format:
//
// p3=2*p1:  https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
void dblPointEcJacobian(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                        RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3)
{

    RawFec::Element a, b, c, d, e, f, aux1, aux2;
    RawFec::Element y1_; // avoid errors when y3 is the same as y1
    fec.copy(y1_, y1);

    fec.square(a, x1);  // a
    fec.square(b, y1_); // b
    fec.square(c, b);   // c
    // d
    fec.add(d, x1, b);
    fec.square(d, d);
    fec.sub(d, d, a);
    fec.sub(d, d, c);
    fec.add(d, d, d);
    // e
    fec.add(e, a, a);
    fec.add(e, e, a);
    // f
    fec.square(f, e);
    // x3
    fec.add(aux1, d, d);
    fec.sub(x3, f, aux1);
    // y3
    fec.sub(aux1, d, x3);
    fec.mul(aux1, aux1, e);
    fec.add(aux2, c, c);
    fec.add(aux2, aux2, aux2);
    fec.add(aux2, aux2, aux2);
    fec.sub(y3, aux1, aux2);
    // z3
    fec.mul(aux1, y1_, z1);
    fec.add(z3, aux1, aux1);
}
// p3=2*p1:  https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
void dblPointEcJacobianZ2Is1(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                        RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3)
{

    RawFec::Element xx, yy, yyyy, s, m, t, aux1, aux2;
    RawFec::Element y1_; // avoid errors when y3 is the same as y1
    fec.copy(y1_, y1);

    fec.square(xx, x1);     // xx
    fec.square(yy, y1_);    // yy
    fec.square(yyyy, yy);   // yyyy
    // s
    fec.add(s, x1, yy);
    fec.square(s, s);
    fec.sub(s, s, xx);
    fec.sub(s, s, yyyy);
    fec.add(s, s, s);
    // m
    fec.add(m, xx, xx);
    fec.add(m, m, xx);
    // t
    fec.square(t, m);
    fec.sub(t, t, s);
    fec.sub(t, t, s);
    // x3
    fec.copy(x3, t);
    // y3
    fec.add(aux1, yyyy, yyyy);
    fec.add(aux1, aux1, aux1);
    fec.add(aux1, aux1, aux1);
    fec.sub(aux2, s,t);
    fec.mul(aux2, aux2, m);
    fec.sub(y3, aux2, aux1);
    // z3
    fec.add(z3, y1_, y1_);
}

// p3=p2+p1:  https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
void addPointEcJacobian(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                        const RawFec::Element &x2, const RawFec::Element &y2, const RawFec::Element &z2,
                        RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3)
{

    RawFec::Element z1z1, z2z2, u1, u2, s1, s2, h, i, j, r, v, rr, aux1, aux2;

    fec.square(z1z1, z1);  // z1z1
    fec.square(z2z2, z2);  // z2z2
    fec.mul(u1, x1, z2z2); // u1
    fec.mul(u2, x2, z1z1); // u2
    // s1
    fec.mul(s1, y1, z2);
    fec.mul(s1, s1, z2z2);
    // s2
    fec.mul(s2, y2, z1);
    fec.mul(s2, s2, z1z1);
    // h
    fec.sub(h, u2, u1);
    // i
    fec.add(i, h, h);
    fec.square(i, i);
    // j
    fec.mul(j, h, i);
    // r
    fec.sub(r, s2, s1);
    fec.add(r, r, r);
    // v
    fec.mul(v, u1, i);
    // x3
    fec.square(rr, r);
    fec.add(aux1, v, v);
    fec.sub(x3, rr, j);
    fec.sub(x3, x3, aux1);
    // y3
    fec.sub(aux1, v, x3);
    fec.mul(aux1, aux1, r);
    fec.mul(aux2, s1, j);
    fec.add(aux2, aux2, aux2);
    fec.sub(y3, aux1, aux2);
    // z3
    fec.add(aux1, z1, z2);
    fec.square(aux1, aux1);
    fec.sub(aux1, aux1, z1z1);
    fec.sub(aux1, aux1, z2z2);
    fec.mul(z3, aux1, h);
}

// p3=p2+p1:  https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
void addPointEcJacobianZ2Is1(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                        const RawFec::Element &x2, const RawFec::Element &y2,   const RawFec::Element &z2,
                        RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3)
{

    RawFec::Element z1z1, u2, s2, hh, h, i, j, r, v, rr, aux1, aux2;

    fec.square(z1z1, z1);  // z1z1
    fec.mul(u2, x2, z1z1); // u2
    // s2
    fec.mul(s2, y2, z1);
    fec.mul(s2, s2, z1z1);
    // h
    fec.sub(h, u2, x1);
    // hh
    fec.square(hh, h);
    // i
    fec.add(i, hh, hh);
    fec.add(i, i, i);
    // j
    fec.mul(j, h, i);
    // r
    fec.sub(r, s2, y1);
    fec.add(r, r, r);
    // v
    fec.mul(v, x1, i);
    // x3
    fec.square(rr, r);
    fec.add(aux1, v, v);
    fec.sub(x3, rr, j);
    fec.sub(x3, x3, aux1);
    // y3
    fec.sub(aux1, v, x3);
    fec.mul(aux1, aux1, r);
    fec.mul(aux2, y1, j);
    fec.add(aux2, aux2, aux2);
    fec.sub(y3, aux1, aux2);
    // z3
    fec.add(aux1, z1, h);
    fec.square(aux1, aux1);
    fec.sub(aux1, aux1, z1z1);
    fec.sub(z3, aux1, hh);
}

// p3=p2+p1:  https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
void addPointEcJacobian(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                        const RawFec::Element &zz1, const RawFec::Element &zzz1,
                        const RawFec::Element &x2, const RawFec::Element &y2, const RawFec::Element &z2,
                        RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3)
{

    RawFec::Element z2z2, u1, u2, s1, s2, h, i, j, r, v, rr, aux1, aux2;

    fec.square(z2z2, z2);  // z2z2
    fec.mul(u1, x1, z2z2); // u1
    fec.mul(u2, x2, zz1);  // u2
    // s1
    fec.mul(s1, y1, z2);
    fec.mul(s1, s1, z2z2);
    // s2
    fec.mul(s2, y2, zzz1);
    // h
    fec.sub(h, u2, u1);
    // i
    fec.add(i, h, h);
    fec.square(i, i);
    // j
    fec.mul(j, h, i);
    // r
    fec.sub(r, s2, s1);
    fec.add(r, r, r);
    // v
    fec.mul(v, u1, i);
    // x3
    fec.square(rr, r);
    fec.add(aux1, v, v);
    fec.sub(x3, rr, j);
    fec.sub(x3, x3, aux1);
    // y3
    fec.sub(aux1, v, x3);
    fec.mul(aux1, aux1, r);
    fec.mul(aux2, s1, j);
    fec.add(aux2, aux2, aux2);
    fec.sub(y3, aux1, aux2);
    // z3
    fec.add(aux1, z1, z2);
    fec.square(aux1, aux1);
    fec.sub(aux1, aux1, zz1);
    fec.sub(aux1, aux1, z2z2);
    fec.mul(z3, aux1, h);
}

void generalAddPointEcJacobian(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                               const bool p1_empty,
                               const RawFec::Element &x2, const RawFec::Element &y2, const RawFec::Element &z2,
                               const bool p2_empty,
                               RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3,
                               bool &p3_empty)
{
    RawFec::Element z1_2 = fec.mul(z1, z1);
    RawFec::Element z2_2 = fec.mul(z2, z2);
    RawFec::Element z1_3 = fec.mul(z1_2, z1);
    RawFec::Element z2_3 = fec.mul(z2_2, z2);

    if (p1_empty && p2_empty)
    {
        p3_empty = true;
        return;
    }
    else
    {
        if (p1_empty)
        {
            fec.copy(x3, x2);
            fec.copy(y3, y2);
            fec.copy(z3, z2);
            p3_empty = p2_empty;
            return;
        }
        else
        {
            if (p2_empty)
            {
                fec.copy(x3, x1);
                fec.copy(y3, y1);
                fec.copy(z3, z1);
                p3_empty = p1_empty;
                return;
            }
            else
            {
                if (fec.eq(fec.mul(x1, z2_2), fec.mul(x2, z1_2)) == 0)
                {
                    addPointEcJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3);
                    if (fec.isZero(z3) == 1)
                    {
                        p3_empty = true;
                    }
                    else
                    {
                        p3_empty = false;
                    }
                }
                else
                {
                    if (fec.eq(fec.mul(y1, z2_3), fec.mul(y2, z1_3)) == 0)
                    {
                        p3_empty = true;
                    }
                    else
                    {
                        dblPointEcJacobian(x1, y1, z1, x3, y3, z3);
                        if (fec.isZero(z3) == 1)
                        {
                            p3_empty = true;
                        }
                        else
                        {
                            p3_empty = false;
                        }
                    }
                }
            }
        }
    }
}

void generalAddPointEcJacobianZ2Is1(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                               const bool p1_empty,
                               const RawFec::Element &x2, const RawFec::Element &y2, const RawFec::Element &z2,
                               const bool p2_empty,
                               RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3,
                               bool &p3_empty)
{

    RawFec::Element z1_2 = fec.mul(z1, z1);
    RawFec::Element z2_2 = fec.mul(z2, z2);
    RawFec::Element z1_3 = fec.mul(z1_2, z1);
    RawFec::Element z2_3 = fec.mul(z2_2, z2);

    if (p1_empty && p2_empty)
    {
        p3_empty = true;
        return;
    }
    else
    {
        if (p1_empty)
        {
            fec.copy(x3, x2);
            fec.copy(y3, y2);
            fec.copy(z3, z2);
            p3_empty = p2_empty;
            return;
        }
        else
        {
            if (p2_empty)
            {
                fec.copy(x3, x1);
                fec.copy(y3, y1);
                fec.copy(z3, z1);
                p3_empty = p1_empty;
                return;
            }
            else
            {
                if (fec.eq(fec.mul(x1, z2_2), fec.mul(x2, z1_2)) == 0)
                {
                    addPointEcJacobianZ2Is1(x1, y1, z1, x2, y2, z2, x3, y3, z3);
                    if (fec.isZero(z3) == 1)
                    {
                        p3_empty = true;
                    }
                    else
                    {
                        p3_empty = false;
                    }
                }
                else
                {
                    if (fec.eq(fec.mul(y1, z2_3), fec.mul(y2, z1_3)) == 0)
                    {
                        p3_empty = true;
                    }
                    else
                    {
                        dblPointEcJacobianZ2Is1(x1, y1, z1, x3, y3, z3);
                        if (fec.isZero(z3) == 1)
                        {
                            p3_empty = true;
                        }
                        else
                        {
                            p3_empty = false;
                        }
                    }
                }
            }
        }
    }
}

void generalAddPointEcJacobian(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                               const RawFec::Element &zz1, const RawFec::Element &zzz1,
                               const bool p1_empty,
                               const RawFec::Element &x2, const RawFec::Element &y2, const RawFec::Element &z2,
                               const bool p2_empty,
                               RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3,
                               bool &p3_empty)
{

    RawFec::Element z1_2 = fec.mul(z1, z1);
    RawFec::Element z2_2 = fec.mul(z2, z2);
    RawFec::Element z1_3 = fec.mul(z1_2, z1);
    RawFec::Element z2_3 = fec.mul(z2_2, z2);

    if (p1_empty && p2_empty)
    {
        p3_empty = true;
        return;
    }
    else
    {
        if (p1_empty)
        {
            fec.copy(x3, x2);
            fec.copy(y3, y2);
            fec.copy(z3, z2);
            p3_empty = p2_empty;
            return;
        }
        else
        {
            if (p2_empty)
            {
                fec.copy(x3, x1);
                fec.copy(y3, y1);
                fec.copy(z3, z1);
                p3_empty = p1_empty;
                return;
            }
            else
            {
                if (fec.eq(fec.mul(x1, z2_2), fec.mul(x2, z1_2)) == 0)
                {
                    addPointEcJacobian(x1, y1, z1, zz1, zzz1, x2, y2, z2, x3, y3, z3);
                    if (fec.isZero(z3) == 1)
                    {
                        p3_empty = true;
                    }
                    else
                    {
                        p3_empty = false;
                    }
                }
                else
                {
                    if (fec.eq(fec.mul(y1, z2_3), fec.mul(y2, z1_3)) == 0)
                    {
                        p3_empty = true;
                    }
                    else
                    {
                        dblPointEcJacobian(x1, y1, z1, x3, y3, z3);
                        if (fec.isZero(z3) == 1)
                        {
                            p3_empty = true;
                        }
                        else
                        {
                            p3_empty = false;
                        }
                    }
                }
            }
        }
    }
}

void mulPointEcJacobian(const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &k1,
                        const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &k2,
                        mpz_class &p3_x, mpz_class &p3_y)
{

    RawFec::Element x3, y3, z3;
    RawFec::Element p[80]; // 16*ndata
    int ndata = 5;
    bool isz[16];
    int out0, out1;
    int ina0, ina1;
    int inb0, inb1;
    int vina[] = {0, 0, 1, 1, 0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3};
    int vinb[] = {0, 0, 1, 2, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12};

    // 0000
    isz[0] = true;

    // 0001
    out0 = 1;
    out1 = out0 * ndata;
    isz[out0] = false;
    fec.fromMpz(p[out1], p1_x.get_mpz_t());
    fec.fromMpz(p[out1 + 1], p1_y.get_mpz_t());
    p[out1 + 2] = fec.one();
    p[out1 + 3] = fec.one();
    p[out1 + 4] = fec.one();

    // 0100
    out0 = 4;
    out1 = out0 * ndata;
    isz[out0] = false;
    fec.fromMpz(p[out1], p2_x.get_mpz_t());
    fec.fromMpz(p[out1 + 1], p2_y.get_mpz_t());
    p[out1 + 2] = fec.one();
    p[out1 + 3] = fec.one();
    p[out1 + 4] = fec.one();

    for (int k = 2; k < 16; ++k)
    {
        if (k == 4)
        {
            continue;
        }
        out0 = k;
        ina0 = vina[k];
        inb0 = vinb[k];
        out1 = out0 * ndata;
        ina1 = ina0 * ndata;
        inb1 = inb0 * ndata;
        generalAddPointEcJacobian(p[ina1], p[ina1 + 1], p[ina1 + 2], isz[ina0], p[inb1], p[inb1 + 1], p[inb1 + 2], isz[inb0], p[out1], p[out1 + 1], p[out1 + 2], isz[out0]);
        p[out1 + 3] = fec.mul(p[out1 + 2], p[out1 + 2]);
        p[out1 + 4] = fec.mul(p[out1 + 3], p[out1 + 2]);
    }

    // start the loop
    mpz_t rawK1;
    mpz_t rawK2;

    mpz_init(rawK1);
    mpz_init(rawK2);
    mpz_set(rawK1, k1.get_mpz_t());
    mpz_set(rawK2, k2.get_mpz_t());

    bool p3_empty = true;

    for (int i = 255; i >= 0; i -= 2)
    {

        // double p3
        if (!p3_empty)
        {
            dblPointEcJacobian(x3, y3, z3, x3, y3, z3);
            dblPointEcJacobian(x3, y3, z3, x3, y3, z3);
        }
        // take next bits
        int bitk10 = mpz_tstbit(rawK1, i - 1);
        int bitk11 = mpz_tstbit(rawK1, i);
        int bitk20 = mpz_tstbit(rawK2, i - 1);
        int bitk21 = mpz_tstbit(rawK2, i);

        int out0 = 8 * bitk21 + 4 * bitk20 + 2 * bitk11 + bitk10;
        int out1 = ndata * out0;
        generalAddPointEcJacobian(p[out1], p[out1 + 1], p[out1 + 2], p[out1 + 3], p[out1 + 4], isz[out0], x3, y3, z3, p3_empty, x3, y3, z3, p3_empty);
    }

    mpz_clear(rawK1);
    mpz_clear(rawK2);

    // save results
    Jacobian2Affine(x3, y3, z3, x3, y3);
    fec.toMpz(p3_x.get_mpz_t(), x3);
    fec.toMpz(p3_y.get_mpz_t(), y3);
}

void mulPointEcJacobian(const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &p1_z, const mpz_class &k1,
                        const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &p2_z, const mpz_class &k2,
                        mpz_class &p3_x, mpz_class &p3_y, mpz_class &p3_z)
{

    RawFec::Element x3, y3, z3;
    RawFec::Element p[48];
    bool isz[16];

    // 0000
    isz[0] = true;

    // 0001
    int out0 = 1;
    int out1 = 3;
    isz[out0] = false;
    fec.fromMpz(p[out1], p1_x.get_mpz_t());
    fec.fromMpz(p[out1 + 1], p1_y.get_mpz_t());
    fec.fromMpz(p[out1 + 2], p1_z.get_mpz_t());

    // 0010
    out0 = 2;
    out1 = 6;
    isz[out0] = false;
    dblPointEcJacobian(p[3], p[4], p[5], p[out1], p[out1 + 1], p[out1 + 2]);

    // 0011
    out0 = 3;
    out1 = 9;
    generalAddPointEcJacobian(p[3], p[4], p[5], isz[1], p[6], p[7], p[8], isz[2], p[out1], p[out1 + 1], p[out1 + 2], isz[out0]);

    // 0100
    out0 = 4;
    out1 = 12;
    isz[out0] = false;
    fec.fromMpz(p[out1], p2_x.get_mpz_t());
    fec.fromMpz(p[out1 + 1], p2_y.get_mpz_t());
    fec.fromMpz(p[out1 + 2], p2_z.get_mpz_t());

    // 0101
    out0 = 5;
    out1 = 15;
    generalAddPointEcJacobian(p[3], p[4], p[5], isz[1], p[12], p[13], p[14], isz[4], p[out1], p[out1 + 1], p[out1 + 2], isz[out0]);

    // 0110
    out0 = 6;
    out1 = 18;
    generalAddPointEcJacobian(p[6], p[7], p[8], isz[2], p[12], p[13], p[14], isz[4], p[out1], p[out1 + 1], p[out1 + 2], isz[out0]);

    // 0111
    out0 = 7;
    out1 = 21;
    generalAddPointEcJacobian(p[9], p[10], p[11], isz[3], p[12], p[13], p[14], isz[4], p[out1], p[out1 + 1], p[out1 + 2], isz[out0]);

    // 1000
    out0 = 8;
    out1 = 24;
    isz[out0] = false;
    dblPointEcJacobian(p[12], p[13], p[14], p[out1], p[out1 + 1], p[out1 + 2]);

    // 1001
    out0 = 9;
    out1 = 27;
    generalAddPointEcJacobian(p[3], p[4], p[5], isz[1], p[24], p[25], p[26], isz[8], p[out1], p[out1 + 1], p[out1 + 2], isz[out0]);

    // 1010
    out0 = 10;
    out1 = 30;
    generalAddPointEcJacobian(p[6], p[7], p[8], isz[2], p[24], p[25], p[26], isz[8], p[out1], p[out1 + 1], p[out1 + 2], isz[out0]);

    // 1011
    out0 = 11;
    out1 = 33;
    generalAddPointEcJacobian(p[9], p[10], p[11], isz[3], p[24], p[25], p[26], isz[8], p[out1], p[out1 + 1], p[out1 + 2], isz[out0]);

    // 1100
    out0 = 12;
    out1 = 36;
    generalAddPointEcJacobian(p[12], p[13], p[14], isz[4], p[24], p[25], p[26], isz[8], p[out1], p[out1 + 1], p[out1 + 2], isz[out0]);

    // 1101
    out0 = 13;
    out1 = 39;
    generalAddPointEcJacobian(p[3], p[4], p[5], isz[1], p[36], p[37], p[38], isz[12], p[out1], p[out1 + 1], p[out1 + 2], isz[out0]);

    // 1110
    out0 = 14;
    out1 = 42;
    generalAddPointEcJacobian(p[6], p[7], p[8], isz[2], p[36], p[37], p[38], isz[12], p[out1], p[out1 + 1], p[out1 + 2], isz[out0]);

    // 1111
    out0 = 15;
    out1 = 45;
    generalAddPointEcJacobian(p[9], p[10], p[11], isz[3], p[36], p[37], p[38], isz[12], p[out1], p[out1 + 1], p[out1 + 2], isz[out0]);

    // start the loop
    mpz_t rawK1;
    mpz_t rawK2;

    mpz_init(rawK1);
    mpz_init(rawK2);
    mpz_set(rawK1, k1.get_mpz_t());
    mpz_set(rawK2, k2.get_mpz_t());

    bool p3_empty = true;

    for (int i = 255; i >= 0; i -= 2)
    {

        // double p3
        if (!p3_empty)
        {
            dblPointEcJacobian(x3, y3, z3, x3, y3, z3);
            dblPointEcJacobian(x3, y3, z3, x3, y3, z3);
        }
        // take next bits
        int bitk10 = mpz_tstbit(rawK1, i - 1);
        int bitk11 = mpz_tstbit(rawK1, i);
        int bitk20 = mpz_tstbit(rawK2, i - 1);
        int bitk21 = mpz_tstbit(rawK2, i);

        int out0 = 8 * bitk21 + 4 * bitk20 + 2 * bitk11 + bitk10;
        int out1 = 3 * out0;
        generalAddPointEcJacobian(p[out1], p[out1 + 1], p[out1 + 2], isz[out0], x3, y3, z3, p3_empty, x3, y3, z3, p3_empty);
    }

    mpz_clear(rawK1);
    mpz_clear(rawK2);

    // save results
    fec.toMpz(p3_x.get_mpz_t(), x3);
    fec.toMpz(p3_y.get_mpz_t(), y3);
    fec.toMpz(p3_z.get_mpz_t(), z3);
}

void mulPointEcJacobian1bit(const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &p1_z, const mpz_class &k1,
                            const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &p2_z, const mpz_class &k2,
                            mpz_class &p3_x, mpz_class &p3_y, mpz_class &p3_z)
{

    RawFec::Element x3, y3, z3;
    RawFec::Element p[12];
    bool isz[4];

    // 00
    isz[0] = true;

    // 01
    isz[1] = false;
    int id = 3;
    fec.fromMpz(p[id], p1_x.get_mpz_t());
    fec.fromMpz(p[id + 1], p1_y.get_mpz_t());
    fec.fromMpz(p[id + 2], p1_z.get_mpz_t());

    // 10
    isz[2] = false;
    id = 6;
    fec.fromMpz(p[id], p2_x.get_mpz_t());
    fec.fromMpz(p[id + 1], p2_y.get_mpz_t());
    fec.fromMpz(p[id + 2], p2_z.get_mpz_t());

    // 11
    generalAddPointEcJacobian(p[3], p[4], p[5], isz[1], p[6], p[7], p[8], isz[2], p[9], p[10], p[11], isz[3]);

    // start the loop
    mpz_t rawK1;
    mpz_t rawK2;

    mpz_init(rawK1);
    mpz_init(rawK2);
    mpz_set(rawK1, k1.get_mpz_t());
    mpz_set(rawK2, k2.get_mpz_t());

    bool p3_empty = true;

    for (int i = 255; i >= 0; --i)
    {
        // double p3
        if (!p3_empty)
        {
            dblPointEcJacobian(x3, y3, z3, x3, y3, z3);
        }
        // take next bits
        int bitk1 = mpz_tstbit(rawK1, i);
        int bitk2 = mpz_tstbit(rawK2, i);
        int out0 = 2 * bitk2 + bitk1;
        int out1 = 3 * out0;
        generalAddPointEcJacobian(p[out1], p[out1 + 1], p[out1 + 2], isz[out0], x3, y3, z3, p3_empty, x3, y3, z3, p3_empty);
    }
    mpz_clear(rawK1);
    mpz_clear(rawK2);

    // save results
    fec.toMpz(p3_x.get_mpz_t(), x3);
    fec.toMpz(p3_y.get_mpz_t(), y3);
    fec.toMpz(p3_z.get_mpz_t(), z3);
}

void mulPointEcJacobian1bit(const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &k1,
                            const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &k2,
                            mpz_class &p3_x, mpz_class &p3_y)
{

    RawFec::Element x3, y3, z3;
    RawFec::Element p[12];
    bool isz[4];

    // 00
    isz[0] = true;

    // 01
    isz[1] = false;
    int id = 3;
    fec.fromMpz(p[id], p1_x.get_mpz_t());
    fec.fromMpz(p[id + 1], p1_y.get_mpz_t());
    p[id + 2]=fec.one();

    // 10
    isz[2] = false;
    id = 6;
    fec.fromMpz(p[id], p2_x.get_mpz_t());
    fec.fromMpz(p[id + 1], p2_y.get_mpz_t());
    p[id + 2] =fec.one();

    // 11
    generalAddPointEcJacobianZ2Is1(p[3], p[4], p[5], isz[1], p[6], p[7], p[8], isz[2], p[9], p[10], p[11], isz[3]);
    if(!isz[3]){
        Jacobian2Affine(p[9], p[10], p[11], p[9], p[10]);
        p[11]=fec.one();
    }

    // start the loop
    mpz_t rawK1;
    mpz_t rawK2;

    mpz_init(rawK1);
    mpz_init(rawK2);
    mpz_set(rawK1, k1.get_mpz_t());
    mpz_set(rawK2, k2.get_mpz_t());

    bool p3_empty = true;

    for (int i = 255; i >= 0; --i)
    {
        // double p3
        if (!p3_empty)
        {
            dblPointEcJacobian(x3, y3, z3, x3, y3, z3);
        }
        // take next bits
        int bitk1 = mpz_tstbit(rawK1, i);
        int bitk2 = mpz_tstbit(rawK2, i);
        int out0 = 2 * bitk2 + bitk1;
        int out1 = 3 * out0;
        generalAddPointEcJacobianZ2Is1( x3, y3, z3, p3_empty,p[out1], p[out1 + 1], p[out1 + 2], isz[out0], x3, y3, z3, p3_empty);
    }
    mpz_clear(rawK1);
    mpz_clear(rawK2);

    // save results
    Jacobian2Affine(x3, y3, z3, x3, y3);
    fec.toMpz(p3_x.get_mpz_t(), x3);
    fec.toMpz(p3_y.get_mpz_t(), y3);
}

int mulPointEcJacobian1bitSave(const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &k1,
                            const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &k2,
                            mpz_class &p3_x, mpz_class &p3_y, RawFec::Element* buffer, int nthreads)
{

    RawFec::Element x3, y3, z3;
    RawFec::Element p[12];
    bool isz[4];
    RawFec::Element buffer_[1536];
    int pcont_in = 0;
    int pcont_out = 0;
    int npoint_p11 = 0;
    int npoint = 0;


    // 00
    isz[0] = true;

    // 01
    isz[1] = false;
    int id = 3;
    fec.fromMpz(p[id], p1_x.get_mpz_t());
    fec.fromMpz(p[id + 1], p1_y.get_mpz_t());
    p[id + 2]=fec.one();

    // 10
    isz[2] = false;
    id = 6;
    fec.fromMpz(p[id], p2_x.get_mpz_t());
    fec.fromMpz(p[id + 1], p2_y.get_mpz_t());
    p[id + 2] =fec.one();

    // 11
    generalAddPointEcJacobianZ2Is1(p[3], p[4], p[5], isz[1], p[6], p[7], p[8], isz[2], p[9], p[10], p[11], isz[3]);
    if(!isz[3]){
        Jacobian2Affine(p[9], p[10], p[11], p[9], p[10]);
        p[11]=fec.one();
        buffer[pcont_out++] = p[9];
        buffer[pcont_out++] = p[10];
        npoint_p11=1;
    }

    // start the loop
    mpz_t rawK1;
    mpz_t rawK2;

    mpz_init(rawK1);
    mpz_init(rawK2);
    mpz_set(rawK1, k1.get_mpz_t());
    mpz_set(rawK2, k2.get_mpz_t());

    bool p3_empty = true;

    for (int i = 255; i >= 0; --i)
    {
        // double p3
        if (!p3_empty)
        {
            dblPointEcJacobian(x3, y3, z3, x3, y3, z3);
            buffer_[pcont_in++] = x3;
            buffer_[pcont_in++] = y3;
            buffer_[pcont_in++] = z3;
            ++npoint;
        }
        
        // take next bits
        int bitk1 = mpz_tstbit(rawK1, i);
        int bitk2 = mpz_tstbit(rawK2, i);
        int out0 = 2 * bitk2 + bitk1;
        int out1 = 3 * out0;
        bool aux = p3_empty;
        generalAddPointEcJacobianZ2Is1( x3, y3, z3, p3_empty,p[out1], p[out1 + 1], p[out1 + 2], isz[out0], x3, y3, z3, p3_empty);
        if( !aux && !p3_empty && !isz[out0]){
            buffer_[pcont_in++] = x3;
            buffer_[pcont_in++] = y3;
            buffer_[pcont_in++] = z3;
            ++npoint;
        }

    }
    mpz_clear(rawK1);
    mpz_clear(rawK2);

     // save results
     if(nthreads < 1) nthreads = 1;
#pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < npoint; i++)
    {
        int id1 = 3*i;
        int id2 = 2*(i+npoint_p11);
        assert(fec.eq(buffer_[id1 + 2], fec.zero()) == 0);
        Jacobian2Affine(buffer_[id1 ], buffer_[id1 + 1], buffer_[id1 + 2], buffer[id2 ], buffer[id2 + 1]);
    }
    return 2*(npoint+npoint_p11);
}

void Jacobian2Affine(const RawFec::Element &x, const RawFec::Element &y, const RawFec::Element &z, RawFec::Element &x_out, RawFec::Element &y_out)
{
    RawFec::Element z_inv;
    fec.inv(z_inv, z);
    RawFec::Element z_inv_sq, z_inv_cube;
    fec.square(z_inv_sq, z_inv);
    fec.mul(z_inv_cube, z_inv_sq, z_inv);
    x_out = fec.mul(x, z_inv_sq);
    y_out = fec.mul(y, z_inv_cube);
}

void Jacobian2Affine(const mpz_class &x, const mpz_class &y, const mpz_class &z, mpz_class &x_out, mpz_class &y_out)
{
    mpz_class z_inv = invFpEc(z);
    mpz_class z_inv_sq = sqFpEc(z_inv);
    x_out = mulFpEc(x, z_inv_sq);
    y_out = mulFpEc(y, mulFpEc(z_inv, z_inv_sq));
}
