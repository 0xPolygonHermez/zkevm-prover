#include "poseidon_linear.hpp"
#include "scalar.hpp"

using namespace std;

void PoseidonLinear (Poseidon_goldilocks &poseidon, vector<uint8_t> &bytes, mpz_class &hash)
{
    bytes.push_back(0x01);
    while ((bytes.size()%56)!=0)
    {
        bytes.push_back(0x00);
    }
    bytes[bytes.size()-1] |= 0x80;

    uint64_t st[4] = {0, 0, 0, 0};
    for (uint64_t j=0; j<bytes.size(); j+=56)
    {
        // A[7] is left blank, since it is the highest byte of a field element, i.e. uncomplete
        uint64_t A[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (uint64_t k=0; k<56; k++)
        {
            uint64_t e = k/7;
            uint64_t pe = k%7;
            A[e] = A[e] + (bytes[j+k] << (pe*8));
        }

        // Capacity is set to st
        A[8] = st[0];
        A[9] = st[1];
        A[10] = st[2];
        A[11] = st[3];

        // We call poseidon hash
        poseidon.hash(A);

        // We retrieve the hash
        st[0] = A[0];
        st[1] = A[1];
        st[2] = A[2];
        st[3] = A[3];
    }
    
    mpz_class aux0(st[0]), aux1(st[1]), aux2(st[2]), aux3(st[3]);
    hash = aux0 + aux1*twoTo64 + aux2*twoTo128 + aux3*twoTo192;
}