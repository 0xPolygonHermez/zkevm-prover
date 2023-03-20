#ifndef SHA256_CONFIG_HPP
#define SHA256_CONFIG_HPP

#define SHA256_ZeroRef         (0)
#define SHA256_SlotSize        (160480)
#define SHA256_MaxRefs         (170000)
#define SHA256_FirstNextRef    (1)
#define SHA256_SinRef0         (45)
#define SHA256_SinRefNumber    (768)
#define SHA256_SinRefDistance  (44)
#define SHA256_SoutRef0        (SHA256_SinRef0+(SHA256_SinRefNumber*SHA256_SinRefDistance))
#define SHA256_SoutRefNumber   (256)
#define SHA256_SoutRefDistance (44)

/*

Sin:
    64x32 = 512 bits =   0..511 = data
    8x32  = 256 bits = 512..767 = hash state

Sout:
    8x32  = 256 bits =   0..255 = hash state

Counters:
    xors      = 60080 = 37.6177%
    ors       = 35520 = 22.24%
    andps     = 0 = 0%
    ands      = 64112 = 40.1423%
    nextRef-1 = 160480
*/

#endif