#ifndef SHA256_SM_CONFIG
#define SHA256_SM_CONFIG

#define SHA256_ZeroRef         (0)
#define SHA256_SlotSize        (162655)
#define SHA256_MaxRefs         (170000)
#define SHA256_FirstNextRef    (1)
#define SHA256_SinRef0         (45)
#define SHA256_SinRefNumber    (1600)
#define SHA256_SinRefDistance  (44)
#define SHA256_SoutRef0        (SHA256_SinRef0+(SHA256_SinRefNumber*SHA256_SinRefDistance))
#define SHA256_SoutRefNumber   (1600)
#define SHA256_SoutRefDistance (44)

#endif