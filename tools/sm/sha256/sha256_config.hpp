#ifndef SHA256_CONFIG_HPP
#define SHA256_CONFIG_HPP

#define SHA256_ZeroRef         (0)
#define SHA256_SlotSize        (162656)
#define SHA256_MaxRefs         (170000)
#define SHA256_FirstNextRef    (1)
#define SHA256_SinRef0         (45)
#define SHA256_SinRefNumber    (1600)
#define SHA256_SinRefDistance  (44)
#define SHA256_SoutRef0        (SHA256_SinRef0+(SHA256_SinRefNumber*SHA256_SinRefDistance))
#define SHA256_SoutRefNumber   (1600)
#define SHA256_SoutRefDistance (44)

/*
xors=59824=37.5176%
ors=35520=22.2757%
andps=0=0%
ands=64112=40.2067%
nextRef-1=162656
*/

#endif