#include "ecrecover.hpp"
#include "zklog.hpp"
#include "ffiasm/fnec.hpp"
#include "ffiasm/fec.hpp"
#include "exit_process.hpp"
#include "definitions.hpp"
#include "main_sm/fork_5/main/eval_command.hpp"
#include "keccak_wrapper.hpp"


RawFnec fnec;
RawFec fec;

mpz_class FNEC("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
mpz_class FNEC_MINUS_ONE = FNEC - 1;
mpz_class FNEC_DIV_TWO("0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0");
mpz_class FPEC("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
mpz_class FPEC_NON_SQRT("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");

mpz_class ECGX("0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
mpz_class ECGY("0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
mpz_class ADDRESS_MASK("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");

mpz_class invFpEc  (const mpz_class &a);
mpz_class invFnEc  (const mpz_class &a);
mpz_class mulFpEc  (const mpz_class &a, const mpz_class &b);
mpz_class mulFnEc  (const mpz_class &a, const mpz_class &b);
mpz_class addFpEc  (const mpz_class &a, const mpz_class &b);
mpz_class sqrtFpEc (const mpz_class &a);
mpz_class sqFpEc   (const mpz_class &a);

void Jacobian2Affine (const mpz_class &x, const mpz_class &y, const mpz_class &z, mpz_class &x_out, mpz_class &y_out);

void mulPointEc ( const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &k1, 
                  const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &k2, 
                  mpz_class &p3_x, mpz_class &p3_y); 
void addPointEc ( const mpz_class &p1_x, const mpz_class &p1_y, 
                  const mpz_class &p2_x, const mpz_class &p2_y, 
                  mpz_class &p3_x, mpz_class &p3_y);
void dblPointEc ( const mpz_class &p1_x, const mpz_class &p1_y, 
                  mpz_class &p3_x, mpz_class &p3_y);

void mulPointEcJacobian ( const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &p1_z, const mpz_class &k1, 
                          const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &p2_z, const mpz_class &k2, 
                          mpz_class &p3_x, mpz_class &p3_y, mpz_class &p3_z); 

void addPointEcJacobian ( const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &p1_z,
                          const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &p2_z,
                          mpz_class &p3_x, mpz_class &p3_y, mpz_class &p3_z);
void dblPointEcJacobian ( const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &p1_z,
                          mpz_class &p3_x, mpz_class &p3_y, mpz_class &p3_z);

void addPointEcJacobian(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                        const RawFec::Element &x2, const RawFec::Element &y2, const RawFec::Element &z2,
                        RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3);
void dblPointEcJacobian(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                        RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3);

inline void generalAddPointEcJacobian(  const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                                        const bool p1_empty,
                                        const RawFec::Element &x2, const RawFec::Element &y2, const RawFec::Element &z2,
                                        const bool p2_empty,
                                        RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3,
                                        bool &p3_empty);


ECRecoverResult ECRecover (mpz_class &signature, mpz_class &r, mpz_class &s, mpz_class &v, bool bPrecompiled, mpz_class &address)
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
    
    if(aux3==0)
    {
        ecrecover_y = 0; 
    }
    else
    {
        ecrecover_y = sqrtFpEc(aux3);
        if(ecrecover_y==0)
        {
            zklog.error("ECRecover() found y^2 without root=" + aux3.get_str(16));
            return ECR_NO_SQRT_Y;
        }
    }
    assert(ecrecover_y < FPEC);

    // pending: check indeed y^2 has an square root

    // parity:
    int bit0 = mpz_tstbit(ecrecover_y.get_mpz_t(), 0); 
    if (bit0 + ecrecover_v_parity -1 == 0)
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
    mulPointEcJacobian(p1_x, p1_y, p1_z, k1, p2_x, p2_y, p2_z, k2, p3_x, p3_y, p3_z);
    Jacobian2Affine(p3_x, p3_y, p3_z, p3_x, p3_y);
#endif

    assert(p3_x < FPEC);
    assert(p3_y < FPEC);

    //generate keccak of public key to obtain ethereum address
    unsigned char outputHash[32];
    unsigned char inputHash[64];
    mpz_export(inputHash, nullptr, 0, 1, 0, 0, p3_x.get_mpz_t());
    mpz_export(inputHash + 32, nullptr, 0, 1, 0, 0, p3_y.get_mpz_t());
    keccak(inputHash, 64, outputHash, 32);
    mpz_class keccakHash;
    mpz_import(keccakHash.get_mpz_t(), 32, 0, 1, 0, 0, outputHash);
        
    //for address take only last 20 bytes
    address = keccakHash & ADDRESS_MASK; 

    return ECR_NO_ERROR;
}

mpz_class invFpEc (const mpz_class &value)
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

mpz_class invFnEc (const mpz_class &value)
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

mpz_class mulFpEc  (const mpz_class &a, const mpz_class &b)
{
    return (a*b)%FPEC;
}

mpz_class mulFnEc  (const mpz_class &a, const mpz_class &b)
{
    return (a*b)%FNEC;
}

mpz_class addFpEc  (const mpz_class &a, const mpz_class &b)
{
    return (a+b)%FPEC;
}

mpz_class sqrtFpEc (const mpz_class &a)
{
    //We use that p = 3 mod 4, so r = a^((p+1)/4) is a square root of a: https://www.rieselprime.de/ziki/Modular_square_root
    mpz_class result;
    mpz_class n = (FPEC + 1) / 4;
    mpz_powm(result.get_mpz_t(), a.get_mpz_t(), n.get_mpz_t(), FPEC.get_mpz_t());
    if((result*result) % FPEC != a)
    {
        return 0;
    }
    return result;
}

mpz_class sqFpEc (const mpz_class &a)
{
    return (a*a)%FPEC;
}

void Jacobian2Affine(const mpz_class &x, const mpz_class &y, const mpz_class &z, mpz_class &x_out, mpz_class &y_out)
{
    mpz_class z_inv = invFpEc(z);
    mpz_class z_inv_sq = sqFpEc(z_inv);
    x_out = mulFpEc(x, z_inv_sq);
    y_out = mulFpEc(y, mulFpEc(z_inv, z_inv_sq));
}

void mulPointEc ( const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &k1, 
                  const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &k2, 
                  mpz_class &p3_x, mpz_class &p3_y){

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

    for(int i=255; i>=0; --i){

        // take next bits
        int bitk1 = mpz_tstbit(rawK1, i);
        int bitk2 = mpz_tstbit(rawK2, i);
        
        // add contribution depending on bits
        if( bitk1==1 && bitk2==0){
            if(!p3_empty){
                if(p3_x != p1_x){
                    addPointEc(p3_x, p3_y, p1_x, p1_y, p3_x, p3_y);
                }else{
                    if(p3_y != p1_y){
                        p3_empty = true;
                    }else{
                        dblPointEc(p3_x, p3_y, p3_x, p3_y);
                    }
                }
            }else{
                p3_empty = false;
                mpz_set(p3_x.get_mpz_t(), p1_x.get_mpz_t());
                mpz_set(p3_y.get_mpz_t(), p1_y.get_mpz_t());
            }
        }else if( bitk1==0 && bitk2==1){
            if(!p3_empty){
                if(p3_x != p2_x){
                    addPointEc(p3_x, p3_y, p2_x, p2_y, p3_x, p3_y);
                }else{
                    if(p3_y != p2_y){
                        p3_empty = true;
                    }else{
                        dblPointEc(p3_x, p3_y, p3_x, p3_y);
                    }
                }
            }else{
                p3_empty = false;
                mpz_set(p3_x.get_mpz_t(), p2_x.get_mpz_t());
                mpz_set(p3_y.get_mpz_t(), p2_y.get_mpz_t());
            }
        }else if( bitk1==1 && bitk2==1){
            if(!p12_empty){    
                if(!p3_empty){
                    if(p3_x != p12_x){
                        addPointEc(p3_x, p3_y, p12_x, p12_y, p3_x, p3_y);
                    }else{
                        if(p3_y != p12_y){
                            p3_empty = true;
                        }else{
                            dblPointEc(p3_x, p3_y, p3_x, p3_y);
                        }
                    }
                }else{
                    p3_empty = false;
                    mpz_set(p3_x.get_mpz_t(), p12_x.get_mpz_t());
                    mpz_set(p3_y.get_mpz_t(), p12_y.get_mpz_t());
                }
            }
        }
        // double p3
        if(!p3_empty and i!=0){
            dblPointEc(p3_x, p3_y, p3_x, p3_y);
        }
        
    }
    mpz_clear(rawK1);
    mpz_clear(rawK2);

}

void addPointEc ( const mpz_class &p1_x, const mpz_class &p1_y, 
                  const mpz_class &p2_x, const mpz_class &p2_y, 
                  mpz_class &p3_x, mpz_class &p3_y){
    
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
    fec.sub(aux1, x1, x3);;
    fec.mul(aux1, aux1, s);
    fec.sub(y3, aux1, y1);

    fec.toMpz(p3_x.get_mpz_t(), x3);
    fec.toMpz(p3_y.get_mpz_t(), y3);
}

void dblPointEc ( const mpz_class &p1_x, const mpz_class &p1_y, 
                  mpz_class &p3_x, mpz_class &p3_y){

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
    fec.sub(aux1, x1, x3);;
    fec.mul(aux1, aux1, s);
    fec.sub(y3, aux1, y1);
    
    fec.toMpz(p3_x.get_mpz_t(), x3);
    fec.toMpz(p3_y.get_mpz_t(), y3);
}

void mulPointEcJacobian ( const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &p1_z, const mpz_class &k1, 
                          const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &p2_z, const mpz_class &k2, 
                          mpz_class &p3_x, mpz_class &p3_y, mpz_class &p3_z){

    
    RawFec::Element x1, y1, z1, x2, y2, z2;
    RawFec::Element x12, y12, z12, x3, y3, z3;

    fec.fromMpz(x1, p1_x.get_mpz_t());
    fec.fromMpz(y1, p1_y.get_mpz_t());
    fec.fromMpz(z1, p1_z.get_mpz_t());

    fec.fromMpz(x2, p2_x.get_mpz_t());
    fec.fromMpz(y2, p2_y.get_mpz_t());
    fec.fromMpz(z2, p2_z.get_mpz_t()); 

    fec.copy(x3, fec.zero());
    fec.copy(y3, fec.zero());
    fec.copy(z3, fec.zero());

    bool p12_empty;
    generalAddPointEcJacobian(x1, y1, z1, false, x2, y2, z2, false, x12, y12, z12, p12_empty);
    
    // start the loop
    mpz_t rawK1;
    mpz_t rawK2;

    mpz_init(rawK1);
    mpz_init(rawK2);
    mpz_set(rawK1, k1.get_mpz_t());
    mpz_set(rawK2, k2.get_mpz_t());

    bool p3_empty=true;

    for(int i=255; i>=0; --i){

        // take next bits
        int bitk1 = mpz_tstbit(rawK1, i);
        int bitk2 = mpz_tstbit(rawK2, i);
        // add contribution depending on bits
        if( bitk1==1 && bitk2==0){
            generalAddPointEcJacobian(x1, y1, z1, false, x3, y3, z3, p3_empty, x3, y3, z3,p3_empty);
        }else if( bitk1==0 && bitk2==1){
            generalAddPointEcJacobian(x2, y2, z2, false, x3, y3, z3, p3_empty, x3, y3, z3, p3_empty);
        }else if( bitk1==1 && bitk2==1){
            generalAddPointEcJacobian(x12, y12, z12, p12_empty, x3, y3, z3, p3_empty, x3, y3, z3, p3_empty);
        }
        // double p3
        if(!p3_empty and i!=0){
            dblPointEcJacobian(x3, y3, z3, x3, y3, z3);
        }
    }
    mpz_clear(rawK1);
    mpz_clear(rawK2);

    // save results
    fec.toMpz(p3_x.get_mpz_t(), x3);
    fec.toMpz(p3_y.get_mpz_t(), y3);
    fec.toMpz(p3_z.get_mpz_t(), z3);
}

// p3=p2+p1:  https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
void addPointEcJacobian ( const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &p1_z,
                          const mpz_class &p2_x, const mpz_class &p2_y, const mpz_class &p2_z,
                          mpz_class &p3_x, mpz_class &p3_y, mpz_class &p3_z){

    RawFec::Element x1, y1, z1, x2, y2, z2, x3, y3, z3;
    fec.fromMpz(x1, p1_x.get_mpz_t());
    fec.fromMpz(y1, p1_y.get_mpz_t());
    fec.fromMpz(z1, p1_z.get_mpz_t());
    fec.fromMpz(x2, p2_x.get_mpz_t());
    fec.fromMpz(y2, p2_y.get_mpz_t());
    fec.fromMpz(z2, p2_z.get_mpz_t());

    addPointEcJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3);

    // save results
    fec.toMpz(p3_x.get_mpz_t(), x3);
    fec.toMpz(p3_y.get_mpz_t(), y3);
    fec.toMpz(p3_z.get_mpz_t(), z3);

}

// p3=2*p1:  https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
void dblPointEcJacobian ( const mpz_class &p1_x, const mpz_class &p1_y, const mpz_class &p1_z,
                          mpz_class &p3_x, mpz_class &p3_y, mpz_class &p3_z){
    
    RawFec::Element x1, y1, z1, x3, y3, z3;
    fec.fromMpz(x1, p1_x.get_mpz_t());
    fec.fromMpz(y1, p1_y.get_mpz_t());
    fec.fromMpz(z1, p1_z.get_mpz_t());

    dblPointEcJacobian(x1, y1, z1, x3, y3, z3);

    //save results
    fec.toMpz(p3_x.get_mpz_t(), x3);
    fec.toMpz(p3_y.get_mpz_t(), y3);
    fec.toMpz(p3_z.get_mpz_t(), z3);
                            
}


// p3=p2+p1:  https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
void addPointEcJacobian ( const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                          const RawFec::Element &x2, const RawFec::Element &y2, const RawFec::Element &z2,
                          RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3){

    RawFec::Element z1z1, z2z2, u1, u2, s1, s2, h, i, j, r, v, rr, aux1, aux2;

    fec.square(z1z1, z1); //z1z1
    fec.square(z2z2, z2); //z2z2
    fec.mul(u1, x1, z2z2); //u1
    fec.mul(u2, x2, z1z1); //u2
    //s1
    fec.mul(s1, y1, z2);
    fec.mul(s1, s1, z2z2);
    //s2
    fec.mul(s2, y2, z1);
    fec.mul(s2, s2, z1z1);
    //h
    fec.sub(h, u2, u1);
    //i
    fec.add(i, h, h);
    fec.square(i, i);
    //j
    fec.mul(j, h, i);
    //r
    fec.sub(r, s2, s1);
    fec.add(r, r, r);
    //v
    fec.mul(v, u1, i);
    //x3
    fec.square(rr, r);
    fec.add(aux1, v, v);
    fec.sub(x3, rr, j);
    fec.sub(x3, x3, aux1);
    //y3
    fec.sub(aux1, v, x3);
    fec.mul(aux1, aux1, r);
    fec.mul(aux2, s1, j);
    fec.add(aux2, aux2, aux2);
    fec.sub(y3, aux1, aux2);
    //z3
    fec.add(aux1, z1, z2);
    fec.square(aux1, aux1);
    fec.sub(aux1, aux1, z1z1);
    fec.sub(aux1, aux1, z2z2);
    fec.mul(z3, aux1, h);

}

// p3=2*p1:  https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
void dblPointEcJacobian ( const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                          RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3){

    RawFec::Element a, b, c, d, e, f, aux1, aux2;
    RawFec::Element y1_; // avoid errors when y3 is the same as y1
    fec.copy(y1_, y1);

    fec.square(a, x1); //a
    fec.square(b, y1_); //b
    fec.square(c, b); //c
    //d
    fec.add(d, x1, b);
    fec.square(d, d);
    fec.sub(d, d, a);
    fec.sub(d, d, c);
    fec.add(d, d, d);
    //e
    fec.add(e, a, a);
    fec.add(e, e, a);
    //f
    fec.square(f, e);
    //x3
    fec.add(aux1, d, d);
    fec.sub(x3, f, aux1);
    //y3
    fec.sub(aux1, d, x3);   
    fec.mul(aux1, aux1, e);
    fec.fromString(aux2, "8",10);
    fec.mul(aux2, aux2, c);
    fec.sub(y3, aux1, aux2);
    //z3
    fec.mul(aux1, y1_, z1);
    fec.add(z3, aux1,aux1);
                            
}

 void generalAddPointEcJacobian(const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &z1,
                                const bool p1_empty,
                                const RawFec::Element &x2, const RawFec::Element &y2, const RawFec::Element &z2,
                                const bool p2_empty,
                                RawFec::Element &x3, RawFec::Element &y3, RawFec::Element &z3,
                                bool & p3_empty){

    if(p1_empty and p2_empty){
        p3_empty = true;
        return;
    } else {
        if(p1_empty){
            fec.copy(x3, x2);
            fec.copy(y3, y2);
            fec.copy(z3, z2);
            p3_empty = false;
            return;
        } else {
            if(p2_empty){
                fec.copy(x3, x1);
                fec.copy(y3, y1);
                fec.copy(z3, z1);
                p3_empty = false;
                return;
            } else {
                if(fec.eq(fec.mul(x1,z2),fec.mul(x2,z1))==0){
                    p3_empty = false;
                    addPointEcJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3);
                } else {
                    if(fec.eq(fec.mul(y1,z2),fec.mul(y2,z1))==0){
                        p3_empty = true;
                    } else {
                        p3_empty = false;
                        dblPointEcJacobian(x1, y1, z1, x3, y3, z3);
                    }
                }
            }
        }
    }   
}
    /*if (fec.eq(fec.mul(x1,z2),fec.mul(x2,z1))==0)
    {
        // p2.x != p1.x ==> p2 != p1
        p12_empty = false;
        addPointEcJacobian(x1, y1, z1, x2, y2, z2, x12, y12, z12);

    }    
    else if (fec.eq(fec.mul(y1,z2),fec.mul(y2,z1))==0)
    {
         // p2 == -p1
        p12_empty = true;
    }
    else
    {
        // p2 == p1
        p12_empty = false;
        dblPointEcJacobian(x1, y1, z1, x12, y12, z12);
       
    }*/