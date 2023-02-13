#ifndef SHA256_SM_CONFIG
#define SHA256_SM_CONFIG

/* Well-known positions:
0: zero bit value in input a, one bit value in input b
1...1600: Sin (1600 bits)
1601...3200: Sout (1600 bits)
3201...: available references for XOR and ANDP operations results
*/
#define SHA256_ZeroRef      (0)
#define SHA256_SinRef0      (45)
#define SHA256_SoutRef0     (SHA256_SinRef0+(1600*44))
#define SHA256_FirstNextRef (1)

/* Gets the 0...1599 position of the bit (x,y,z), as per Keccak spec */
//#define Bit(x,y,z)   (64*(x) + 320*(y) + (z))

#define maxRefs 170000

/* Arity is the number of bits of the polynomials evaluations length */
//#define Keccak_Arity (23)
//#define Keccak_PolLength (1<<(Keccak_Arity))

/* Mask with the bits containing the pin values */
//#define Keccak_Mask  (0xFFFFFFFFFFF)

#define SHA256_SlotSize (162655)

//#define Keccak_NumberOfSlots ((Keccak_PolLength-1) / Keccak_SlotSize) // 54

#endif