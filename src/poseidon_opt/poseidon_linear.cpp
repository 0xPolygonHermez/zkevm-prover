#include "poseidon_linear.hpp"
#include "scalar.hpp"

using namespace std;

void PoseidonLinear (Goldilocks &fr, Poseidon_goldilocks &poseidon, vector<uint8_t> &bytes, mpz_class &hash)
{
    bytes.push_back(0x01);
    while ((bytes.size()%56)!=0)
    {
        bytes.push_back(0x00);
    }
    bytes[bytes.size()-1] |= 0x80;

    Goldilocks::Element st[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    for (uint64_t j=0; j<bytes.size(); j+=56)
    {
        // A[7] is left blank, since it is the highest byte of a field element, i.e. uncomplete
        Goldilocks::Element A[12] = {fr.zero(), fr.zero(), fr.zero(), fr.zero(), fr.zero(), fr.zero(), fr.zero(), fr.zero(), fr.zero(), fr.zero(), fr.zero(), fr.zero()};
        for (uint64_t k=0; k<56; k++)
        {
            uint64_t e = k/7;
            uint64_t pe = k%7;
            A[e] = fr.add( A[e], fr.fromU64(bytes[j+k] << (pe*8)) );
        }

        // Capacity is set to st
        A[8] = st[0];
        A[9] = st[1];
        A[10] = st[2];
        A[11] = st[3];

        // We call poseidon hash
        poseidon.hash(fr, A);

        // We retrieve the hash
        st[0] = A[0];
        st[1] = A[1];
        st[2] = A[2];
        st[3] = A[3];
    }
    
    mpz_class aux0(fr.toU64(st[0])), aux1(fr.toU64(st[1])), aux2(fr.toU64(st[2])), aux3(fr.toU64(st[3]));
    hash = aux0 + aux1*TwoTo64 + aux2*TwoTo128 + aux3*TwoTo192;
}