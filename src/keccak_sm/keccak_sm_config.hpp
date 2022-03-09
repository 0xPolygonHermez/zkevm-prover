#ifndef KECCAK_SM_CONFIG
#define KECCAK_SM_CONFIG

/* Well-known positions:
0: zero bit value in input a, one bit value in input b
1...1600: Sin (1600 bits)
1601...3200: Sout (1600 bits)
3201...: available references for XOR and ANDP operations results
*/
#define ZeroRef      (0)
#define SinRef0      (1)
#define SoutRef0     (SinRef0+1600)
#define FirstNextRef (SoutRef0+1600)

/* Gets the 0...1599 position of the bit (x,y,z), as per Keccak spec */
#define Bit(x,y,z)   (64*(x) + 320*(y) + (z))

#define maxRefs 160000
#define MAX_CARRY_BITS 6

#define KeccakSM_Arity (23)
#define KeccakSM_PolLength (1<<(KeccakSM_Arity))

#define KeccakSM_Mask (uint64_t(1) + (uint64_t(1)<<7) + (uint64_t(1)<<14) + (uint64_t(1)<<21) + (uint64_t(1)<<28) + (uint64_t(1)<<35) + (uint64_t(1)<<42) + (uint64_t(1)<<49) + (uint64_t(1)<<56))
#define KeccakSM_NumberOfSlots (54)

#endif