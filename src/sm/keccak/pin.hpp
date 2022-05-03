#ifndef PIN_HPP
#define PIN_HPP

#include <vector>
#include <stdint.h>
#include <iostream>

using namespace std;

/*
    a -----||-\
           ||  )----- r
    b -----||-/
*/

enum PinId
{
    pin_a = 0, // Input a pin
    pin_b = 1, // Input b pin
    pin_r = 2  // Ouput bin, result, e.g. r = a^b
};

/* Describes how the bit value of that pin is established */
enum PinSource
{
    external = 0, /* A fixed value externally provided; an external signal */
    wired = 1, /* Connected to another pin */
    gated = 2 /* This pin is the output of a gate; only used with pin_r pins */
};

class Pin
{
private:
    PinId id;
public:
    PinSource source;
    uint64_t wiredRef;
    PinId wiredPinId;
    uint8_t value; /* Worst case scenario: max added value assuming all predecessors were 1 */
    uint64_t fanOut; /* Number of input pins connected to this one as an output */
    uint8_t bit; /* 0 or 1 */
    vector<uint64_t> connectionsToInputA;
    vector<uint64_t> connectionsToInputB;
    
    Pin (PinId id) : id(id) { reset(); };

    void reset (void)
    {
        switch (id)
        {
            case pin_a: 
                source = external;
                break;
            case pin_b: 
                source = external;
                break;
            case pin_r: 
                source = gated;
                break;
            default:
                cerr << "Error: Pin:reset() found invalid PinID:" << id << endl;
                exit(-1);
        }

        wiredRef = 0;
        wiredPinId = pin_r;
        value = 1;
        bit = 0;
        fanOut = 0;
        connectionsToInputA.clear();
        connectionsToInputB.clear();
    }
};

#endif