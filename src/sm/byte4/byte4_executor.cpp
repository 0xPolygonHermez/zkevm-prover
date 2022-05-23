#include "byte4_executor.hpp"

void Byte4Executor::execute (map<uint32_t, bool> &input, Byte4CommitPols & pols)
{
    uint64_t N = pols.degree();

    // Based on the content of byte4[], fill the byte4_freeIn and byte4_out polynomials
    uint64_t p = 0;
    uint64_t last = 0;

    // Check that we have enough room in polynomials
    if (input.size()*2 > N)
    {
        cerr << "Error: Too many byte4 entries" << endl;
        exit(-1);
    }
    
    // Generate polynomials content out of byte4 content
    for (map<uint32_t,bool>::iterator it=input.begin(); it!=input.end(); it++)
    {
        uint32_t num = it->first;
        pols.freeIN[p] = num >> 16;
        pols.out[p] = last;
        p++;
        pols.freeIN[p] = num & 0xFFFF;
        pols.out[p] = num >> 16;
        p++;
        last = num;
    }
    pols.freeIN[p] = 0;
    pols.out[p] = last;
    p++;
    pols.freeIN[p] = 0;
    pols.out[p] = 0;
    p++;

    cout << "Byte4Executor successfully processed " << input.size() << " Byte4 actions" << endl;
}