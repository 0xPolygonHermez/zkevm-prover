#include <sstream>
#include "ff.hpp"

/* Extended Euclidean algorithm */
FieldElement FiniteField::inv(FieldElement a)
{
    if (a == 0)
    {
        cerr << "Error: FiniteField::inv() called with zero" << endl;
        exit(-1);
    }
    u_int64_t t = 0;
    u_int64_t r = p;
    u_int64_t newt = 1;
    u_int64_t newr = a % p;
    u_int64_t q;
    u_int64_t aux1;
    u_int64_t aux2;
    while (newr != 0)
    {
        q = r / newr;
        aux1 = t;
        aux2 = newt;
        t = aux2;
        newt = Goldilocks::gl_sub(aux1, Goldilocks::gl_mul(q, aux2));
        aux1 = r;
        aux2 = newr;
        r = aux2;
        newr = Goldilocks::gl_sub(aux1, Goldilocks::gl_mul(q, aux2));
    }

    return t;
}

string FiniteField::toString(FieldElement a, uint64_t radix)
{
    mpz_class aux = a;
    return aux.get_str(radix);
}

void FiniteField::fromString(FieldElement &a, const string &s, uint64_t radix)
{
    mpz_class aux(s, radix);
    if (aux >= 0)
    {
        if (aux >= p)
        {
            cerr << "Error: FiniteField::fromString() found invalid string value:" << s << endl;
            exit(-1);
        }
        a = aux.get_ui();
    }
    else
    {
        if (-aux >= p)
        {
            cerr << "Error: FiniteField::fromString() found invalid string value:" << s << endl;
            exit(-1);
        }
        aux = -aux;
        a = p - aux.get_ui();
    }
}

void FiniteField::check(bool condition)
{
    if (!condition)
    {
        cerr << "Error: FiniteFields::check() failed" << endl;
        exit(-1);
    }
}

void FiniteField::test(void)
{
    FieldElement a, b, r;

    a = 5;
    b = 10;
    r = add(a, b);
    check(r == 15);

    a = 10;
    b = p - 5;
    r = add(a, b);
    check(r % p == 5);

    a = 10;
    b = 5;
    r = sub(a, b);
    check(r % p == 5);

    a = 10;
    b = p - 5;
    r = sub(a, b);
    check(r % p == 15);

    a = 5;
    r = neg(a);
    check(r % p == (p - 5));

    a = p - 5;
    r = neg(a);
    check(r % p == 5);

    a = 5;
    b = 10;
    r = mul(a, b);
    check(r % p == 50);

    a = p - 5;
    b = p - 10;
    r = mul(a, b);
    check(r % p == 50);

    a = 5;
    b = inv(a);
    r = mul(a, b);
    check(r % p == 1);

    a = 50;
    b = 5;
    r = div(a, b);
    check(r % p == 10);

    fromString(a, "5");
    b = 10;
    r = add(a, b);
    check(r % p == 15);

    fromString(a, "-5");
    b = 10;
    r = add(a, b);
    check(r % p == 5);
}