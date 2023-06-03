#include "ecrecover.hpp"
#include "zklog.hpp"
#include "ffiasm/fnec.hpp"
#include "ffiasm/fec.hpp"
#include "exit_process.hpp"
#include "definitions.hpp"
#include "main_sm/fork_5/main/eval_command.hpp"

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
    mpz_class ecrecover_v_parity;
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
    mpz_class ecrecover_y2;
    mpz_class aux1, aux2, aux3, aux4;
    aux1 = mulFpEc(r, r);
    aux2 = mulFpEc(aux1, r);
    aux1 = 7;
    aux3 = addFpEc(aux2, aux1);
    ecrecover_y2 = sqrtFpEc(aux3);

    // If it has root y ** (p-1)/2 = 1, if -1 => no root, not valid signature
    /*if (ecrecover_y2 == FPEC_NON_SQRT)

        %FPEC_NON_SQRT => A
        C => B
        $ => E      :EQ,JMPNC(ecrecover_has_sqrt)

        ; hasn't sqrt, now verify

        $ => C      :MLOAD(ecrecover_y2),CALL(checkSqrtFpEc)
        ; check must return on A register 1, because the root has no solution
        1           :ASSERT,JMP(ecrecover_not_exists_sqrt_of_y)

ecrecover_has_sqrt:
*/

//ecrecover_has_sqrt:
    //    ; (v == 1b) ecrecover_y_parity = 0x00
    //    ; (v == 1c) ecrecover_y_parity = 0x01

    //    ; C,B: y = sqrt(y^2)
    //    ; check B isn't an alias (B must be in [0, FPEC-1])

        //%FPEC_MINUS_ONE => A
        //0           :LT         ; assert to validate that B (y) isn't n alias.

        // C,B: y = sqrtFpEc(y^2)

        //0x01n => A
        //$ => A      :AND
        //$ => B      :MLOAD(ecrecover_v_parity)

        //; ecrevover_y xor ecrecover_y_parity => 0 same parity, 1 different parity
        //; ecrecover_y2  v parity
        //; parity (A)       (B)      A+B-1
        //;      0            0        -1     same parity
        //;      0            1         0     different parity
        //;      1            0         0     different parity
        //;      1            1         1     same parity

    mpz_class ecrecover_y;
    if ( (ecrecover_y2 + ecrecover_v_parity - 1) == 0 )
    {
        ecrecover_y = ecrecover_y2;
    }
    else
    {
        //; calculate neg(ecrecover_y) C = (A:FPEC) - (B:ecrecovery_y)
        ecrecover_y = FPEC - ecrecover_y2;
    }

//ecrecover_v_y2_same_parity:

    //; C = n - (hash * inv_r) % n
    mpz_class mulPointEc_k1 = FNEC - mulFnEc(hash, ecrecover_r_inv);

    //;   C = (s * inv_r) % n
    mpz_class mulPointEc_k2 = mulFnEc(s, ecrecover_r_inv);

    mpz_class mulPointEc_p1_x = ECGX;
    mpz_class mulPointEc_p1_y = ECGY;

    //; r isn't an alias because the range has been checked at beginning
    mpz_class mulPointEc_p2_x = r;

    //; y isn't an alias because was checked before
    //; (r,y) is a point of curve because it satisfacts the curve equation
    //$ => A      :MLOAD(ecrecover_y)
    //A           :MSTORE(mulPointEc_p2_y),CALL(mulPointEc)

    //mpz_class mulPointEc_p2_y = mulPointEc(ecrecover_y, mulPointEc_p2_x);


    //; generate keccak of public key to obtain ethereum address
    /*$ => E         :MLOAD(lastHashKIdUsed)
    E + 1 => E     :MSTORE(lastHashKIdUsed)
    0 => HASHPOS
    32 => D

    %FPEC => B
    $ => A         :MLOAD(mulPointEc_p3_x)
    1              :LT                  ; alias assert, mulPointEc_p3_x must be in [0, FPEC - 1]

    A              :HASHK(E)

    $ => A         :MLOAD(mulPointEc_p3_y)
    1              :LT                  ; alias assert, mulPointEc_p3_y must be in [0, FPEC - 1]

    A              :HASHK(E)

    64             :HASHKLEN(E)
    $ => A         :HASHKDIGEST(E)*/

    mpz_class keccakHash;

    //; for address take only last 20 bytes
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

    mpz_class mulPointEc_p12_empty = 0;

    if (mulPointEc_p1_x != mulPointEc_p2_x)
    {
        // p2.x != p1.x ==> p2 != p1
        mulPointEc_p12_empty = 0;
        AddPointEc(mulPointEc_p1_x, mulPointEc_p1_y, mulPointEc_p2_x, mulPointEc_p2_y, mulPointEc_p12_x, mulPointEc_p12_y);
    }    
    else if (mulPointEc_p1_y == mulPointEc_p2_y)
    {
        // p2 == p1
        mulPointEc_p12_empty = 0;
        DblPointEc(mulPointEc_p1_x, mulPointEc_p1_y, mulPointEc_p12_x, mulPointEc_p12_y);
    }
    else
    {
        // p2 == -p1
        mulPointEc_p12_empty = 1;
    }

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