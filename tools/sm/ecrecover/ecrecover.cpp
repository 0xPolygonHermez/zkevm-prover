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
mpz_class PROVER_FORK_NAMESPACE::sqrtTonelliShanks ( const mpz_class &n, const mpz_class &p );

void mulPointEc ( const mpz_class &mulPointEc_p1_x,
                  const mpz_class &mulPointEc_p1_y,
                    const mpz_class &mulPointEc_k1,
                  const mpz_class &mulPointEc_p2_x, 
                  const mpz_class &mulPointEc_p2_y,
                    const mpz_class &mulPointEc_k2,
                        mpz_class &mulPointEc_p3_x,
                        mpz_class &mulPointEc_p3_y); 
void AddPointEc ( const mpz_class &mulPointEc_p1_x,
                  const mpz_class &mulPointEc_p1_y,
                  const mpz_class &mulPointEc_p2_x, 
                  const mpz_class &mulPointEc_p2_y,
                        mpz_class &mulPointEc_p3_x,
                        mpz_class &mulPointEc_p3_y);
void DblPointEc ( const mpz_class &mulPointEc_p1_x,
                  const mpz_class &mulPointEc_p1_y,
                        mpz_class &mulPointEc_p3_x,
                        mpz_class &mulPointEc_p3_y);

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
        ecrecover_y = 0; /*rick: check this case*/
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
    ecrecover_y = sqrtFpEc(aux3); //rick: expensive
    assert(ecrecover_y < FPEC);

    // pending: check indeed y^2 has an square root

    // parity:
    int bit0 = mpz_tstbit(ecrecover_y.get_mpz_t(), 0); 
    if (bit0 + ecrecover_v_parity -1 == 0)
    {
        ecrecover_y = FPEC - ecrecover_y;
    }

    // Calculate the point of the curve    
    mpz_class mulPointEc_k1 = FNEC - mulFnEc(signature, ecrecover_r_inv);
    mpz_class mulPointEc_k2 = mulFnEc(s, ecrecover_r_inv);
    
    mpz_class mulPointEc_p1_x = ECGX;
    mpz_class mulPointEc_p1_y = ECGY;

    mpz_class mulPointEc_p2_x = r;
    mpz_class mulPointEc_p2_y = ecrecover_y;

    mpz_class mulPointEc_p3_x;
    mpz_class mulPointEc_p3_y;

    mulPointEc(mulPointEc_p1_x, mulPointEc_p1_y, mulPointEc_k1, mulPointEc_p2_x, mulPointEc_p2_y, mulPointEc_k2, mulPointEc_p3_x, mulPointEc_p3_y);

    assert(mulPointEc_p3_x < FPEC);
    assert(mulPointEc_p3_y < FPEC);

    //generate keccak of public key to obtain ethereum address
    unsigned char outputHash[32];
    unsigned char inputHash[64];
    mpz_export(inputHash, nullptr, 0, 1, 0, 0, mulPointEc_p3_x.get_mpz_t());
    mpz_export(inputHash + 32, nullptr, 0, 1, 0, 0, mulPointEc_p3_y.get_mpz_t());
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
    RawFec::Element pfe = fec.negOne();
    mpz_class p;
    fec.toMpz(p.get_mpz_t(), pfe);
    p++;
    return PROVER_FORK_NAMESPACE::sqrtTonelliShanks(a, p);
}

mpz_class sqFpEc (const mpz_class &a)
{
    return (a*a)%FPEC;
}

void mulPointEc ( const mpz_class &mulPointEc_p1_x,
                  const mpz_class &mulPointEc_p1_y,
                    const mpz_class &mulPointEc_k1,
                  const mpz_class &mulPointEc_p2_x, 
                  const mpz_class &mulPointEc_p2_y,
                    const mpz_class &mulPointEc_k2,
                        mpz_class &mulPointEc_p3_x,
                        mpz_class &mulPointEc_p3_y)
{
    mulPointEc_p3_x = 0;
    mulPointEc_p3_y = 0;

    mpz_class mulPointEc_p12_x;
    mpz_class mulPointEc_p12_y;

    bool mulPointEc_p12_empty = 0;

    if (mulPointEc_p1_x != mulPointEc_p2_x)
    {
        // p2.x != p1.x ==> p2 != p1
        mulPointEc_p12_empty = false;
        AddPointEc(mulPointEc_p1_x, mulPointEc_p1_y, mulPointEc_p2_x, mulPointEc_p2_y, mulPointEc_p12_x, mulPointEc_p12_y);
    }    
    else if (mulPointEc_p1_y == mulPointEc_p2_y)
    {
        // p2 == p1
        mulPointEc_p12_empty = false;
        DblPointEc(mulPointEc_p1_x, mulPointEc_p1_y, mulPointEc_p12_x, mulPointEc_p12_y);
    }
    else
    {
        // p2 == -p1
        mulPointEc_p12_empty = true;
    }

    // start the loop
    mpz_t rawK1;
    mpz_t rawK2;

    mpz_init(rawK1);
    mpz_init(rawK2);

    mpz_set(rawK1, mulPointEc_k1.get_mpz_t());
    mpz_set(rawK2, mulPointEc_k2.get_mpz_t());


    bool mulPointEc_p3_empty = true;

    for(int i=255; i>=0; --i){
        
        // take next bits
        int bitk1 = mpz_tstbit(rawK1, i);
        int bitk2 = mpz_tstbit(rawK2, i);
        
        // add contribution depending on bits
        if( bitk1==1 && bitk2==0){
            if(!mulPointEc_p3_empty){
                if(mulPointEc_p3_x != mulPointEc_p1_x){
                    AddPointEc(mulPointEc_p3_x, mulPointEc_p3_y, mulPointEc_p1_x, mulPointEc_p1_y, mulPointEc_p3_x, mulPointEc_p3_y);
                }else{
                    if(mulPointEc_p3_y != mulPointEc_p1_y){
                        mulPointEc_p3_empty = true;
                    }else{
                        DblPointEc(mulPointEc_p3_x, mulPointEc_p3_y, mulPointEc_p3_x, mulPointEc_p3_y);
                    }
                }
            }else{
                mulPointEc_p3_empty = false;
                mpz_set(mulPointEc_p3_x.get_mpz_t(), mulPointEc_p1_x.get_mpz_t());
                mpz_set(mulPointEc_p3_y.get_mpz_t(), mulPointEc_p1_y.get_mpz_t());
            }
        }else if( bitk1==0 && bitk2==1){
            if(!mulPointEc_p3_empty){
                if(mulPointEc_p3_x != mulPointEc_p2_x){
                    AddPointEc(mulPointEc_p3_x, mulPointEc_p3_y, mulPointEc_p2_x, mulPointEc_p2_y, mulPointEc_p3_x, mulPointEc_p3_y);
                }else{
                    if(mulPointEc_p3_y != mulPointEc_p2_y){
                        mulPointEc_p3_empty = true;
                    }else{
                        DblPointEc(mulPointEc_p3_x, mulPointEc_p3_y, mulPointEc_p3_x, mulPointEc_p3_y);
                    }
                }
            }else{
                mulPointEc_p3_empty = false;
                mpz_set(mulPointEc_p3_x.get_mpz_t(), mulPointEc_p2_x.get_mpz_t());
                mpz_set(mulPointEc_p3_y.get_mpz_t(), mulPointEc_p2_y.get_mpz_t());
            }
        }else if( bitk1==1 && bitk2==1){
            if(!mulPointEc_p12_empty){    
                if(!mulPointEc_p3_empty){
                    if(mulPointEc_p3_x != mulPointEc_p12_x){
                        AddPointEc(mulPointEc_p3_x, mulPointEc_p3_y, mulPointEc_p12_x, mulPointEc_p12_y, mulPointEc_p3_x, mulPointEc_p3_y);
                    }else{
                        if(mulPointEc_p3_y != mulPointEc_p12_y){
                            mulPointEc_p3_empty = true;
                        }else{
                            DblPointEc(mulPointEc_p3_x, mulPointEc_p3_y, mulPointEc_p3_x, mulPointEc_p3_y);
                        }
                    }
                }else{
                    mulPointEc_p3_empty = false;
                    mpz_set(mulPointEc_p3_x.get_mpz_t(), mulPointEc_p12_x.get_mpz_t());
                    mpz_set(mulPointEc_p3_y.get_mpz_t(), mulPointEc_p12_y.get_mpz_t());
                }
            }
        }
        // double p3
        if(!mulPointEc_p3_empty and i!=0){
            DblPointEc(mulPointEc_p3_x, mulPointEc_p3_y, mulPointEc_p3_x, mulPointEc_p3_y);
        }
        
    }
    mpz_clear(rawK1);
    mpz_clear(rawK2);


}

void AddPointEc ( const mpz_class &mulPointEc_p1_x,
                  const mpz_class &mulPointEc_p1_y,
                  const mpz_class &mulPointEc_p2_x, 
                  const mpz_class &mulPointEc_p2_y,
                        mpz_class &mulPointEc_p3_x,
                        mpz_class &mulPointEc_p3_y)
{
    RawFec::Element x1, y1, x2, y2, x3, y3;
    fec.fromMpz(x1, mulPointEc_p1_x.get_mpz_t());
    fec.fromMpz(y1, mulPointEc_p1_y.get_mpz_t());
    fec.fromMpz(x2, mulPointEc_p2_x.get_mpz_t());
    fec.fromMpz(y2, mulPointEc_p2_y.get_mpz_t());

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

    fec.toMpz(mulPointEc_p3_x.get_mpz_t(), x3);
    fec.toMpz(mulPointEc_p3_y.get_mpz_t(), y3);
}

void DblPointEc ( const mpz_class &mulPointEc_p1_x,
                  const mpz_class &mulPointEc_p1_y,
                        mpz_class &mulPointEc_p3_x,
                        mpz_class &mulPointEc_p3_y)
{
    RawFec::Element x1, y1, x2, y2, x3, y3;
    fec.fromMpz(x1, mulPointEc_p1_x.get_mpz_t());
    fec.fromMpz(y1, mulPointEc_p1_y.get_mpz_t());
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
    
    fec.toMpz(mulPointEc_p3_x.get_mpz_t(), x3);
    fec.toMpz(mulPointEc_p3_y.get_mpz_t(), y3);
}