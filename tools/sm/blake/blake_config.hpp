#ifndef BLAKE_CONFIG_HPP
#define BLAKE_CONFIG_HPP

#define Blake_ZeroRef         (0)
#define Blake_SlotSize        (280129)
#define Blake_MaxRefs         (290000)
#define Blake_FirstNextRef    (1)
#define Blake_SinRef0         (45)
#define Blake_SinRefNumber    (1601)
#define Blake_SinRefDistance  (44)
#define Blake_SoutRef0        (Blake_SinRef0+(Blake_SinRefNumber*Blake_SinRefDistance))
#define Blake_SoutRefNumber   (512)
#define Blake_SoutRefDistance (44)

/*
Sin:
    8x128 = 1024 bits =    0..1023 = data chunk (chunk[128] --> m[16])
    8x64  =  512 bits = 1024..1535 = hash state (h[8] --> h64[8])
    1x64  =   64 bits = 1536..1599 = c bytes compressed
    1x1   =    1 bit  = 1600       = bLastChunk

Sout:
    8x64  =  512 bits =    0..511  = hash state (h[8] --> h64[8])

Counters:
    xors      = 99392 = 35.6847%
    ors       = 71424 = 25.6434%
    andps     = 0 = 0%
    ands      = 107712 = 38.6719%
    nextRef-1 = 280129

*/

#endif