#include "ecrecover.hpp"
#include "zklog.hpp"
#include "ffiasm/fnec.hpp"
#include "ffiasm/fec.hpp"
#include "exit_process.hpp"
#include "definitions.hpp"
#include "main_sm/fork_4/main/eval_command.hpp"

RawFnec fnec;
RawFec fec;

mpz_class FNEC("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
mpz_class FNEC_MINUS_ONE = FNEC - 1;
mpz_class FNEC_DIV_TWO("0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0");
mpz_class FPEC("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
mpz_class FPEC_NON_SQRT("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");

mpz_class invFnEc  (mpz_class &a);
mpz_class mulFpEc  (mpz_class &a, mpz_class &b);
mpz_class addFpEc  (mpz_class &a, mpz_class &b);
mpz_class sqrtFpEc (mpz_class &a);
mpz_class PROVER_FORK_NAMESPACE::sqrtTonelliShanks ( const mpz_class &n, const mpz_class &p );

bool ECRecover (mpz_class &hash, mpz_class &r, mpz_class &s, mpz_class &v)
{
    bool bPrecompiled = false;

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
        return false;
    }
    if (r > FNEC_MINUS_ONE)
    {
        zklog.error("ECRecover() found r>FNEC_MINUS_ONE r=" + r.get_str(16));
        return false;
    }

    // Check that s is in the range [1, ecrecover_s_upperlimit]
    if (s == 0)
    {
        zklog.error("ECRecover() found s=0");
        return false;
    }
    if (r > ecrecover_s_upperlimit)
    {
        zklog.error("ECRecover() found s>ecrecover_s_upperlimit s=" + s.get_str(16) + " ecrecover_s_upperlimit=" + ecrecover_s_upperlimit.get_str(16));
        return false;
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
        return false;
    }


    // Curve is y^2 = x^3 + 7  -->  Calculate y from x=r
    mpz_class ecrecover_y2;
    mpz_class aux1, aux2, aux3, aux4;
    aux1 = mulFpEc(r, r);
    aux2 = mulFpEc(aux1, r);
    aux1 = 7;
    aux3 = addFpEc(aux2, aux1);
    ecrecover_y2 = sqrtFpEc(aux3);
    /*

        ;; If has root y ** (p-1)/2 = 1, if -1 => no root, not valid signature

        %FPEC_NON_SQRT => A
        C => B
        $ => E      :EQ,JMPNC(ecrecover_has_sqrt)

        ; hasn't sqrt, now verify

        $ => C      :MLOAD(ecrecover_y2),CALL(checkSqrtFpEc)
        ; check must return on A register 1, because the root has no solution
        1           :ASSERT,JMP(ecrecover_not_exists_sqrt_of_y)

ecrecover_has_sqrt:
*/
    return true;
}

mpz_class invFnEc (mpz_class &value)
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

mpz_class mulFpEc  (mpz_class &a, mpz_class &b)
{
    return (a*b)%FPEC;
}

mpz_class addFpEc  (mpz_class &a, mpz_class &b)
{
    return (a+b)%FPEC;
}

mpz_class sqrtFpEc (mpz_class &a)
{
    RawFec::Element pfe = fec.negOne();
    mpz_class p;
    fec.toMpz(p.get_mpz_t(), pfe);
    p++;
    return PROVER_FORK_NAMESPACE::sqrtTonelliShanks(a, p);
}