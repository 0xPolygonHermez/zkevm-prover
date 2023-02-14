#ifndef BLAKE_CONFIG_HPP
#define BLAKE_CONFIG_HPP

#define Blake_ZeroRef         (0)
#define Blake_SlotSize        (281216)
#define Blake_MaxRefs         (290000)
#define Blake_FirstNextRef    (1)
#define Blake_SinRef0         (45)
#define Blake_SinRefNumber    (1600)
#define Blake_SinRefDistance  (44)
#define Blake_SoutRef0        (Blake_SinRef0+(Blake_SinRefNumber*Blake_SinRefDistance))
#define Blake_SoutRefNumber   (1600)
#define Blake_SoutRefDistance (44)

/*
xors=98880=35.5663%
ors=71424=25.6906%
andps=0=0%
ands=107712=38.7431%
nextRef-1=281216
*/

#endif