#include "pin.hpp"
#include "zklog.hpp"

string pin2string (PinId id)
{
    switch (id)
    {
        case PinId::pin_a: return "a";
        case PinId::pin_b: return "b";
        case PinId::pin_r: return "c";
        default:
            zklog.error("pin2string() got an invalid pin id=" + to_string(id));
            exitProcess();
    }
    return "";
}

PinId string2pin (string s)
{
    if (s=="a") return PinId::pin_a;
    if (s=="b") return PinId::pin_b;
    if (s=="c") return PinId::pin_r;

    zklog.error("string2pin() got an invalid pin id string=" + s);
    exitProcess();
    return PinId::pin_a;
}

void Pin::reset (void)
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
            zklog.error("Pin:reset() found invalid PinID:" + to_string(id));
            exitProcess();
    }

    wiredRef = 0;
    wiredPinId = pin_r;
    bit = 0;
    fanOut = 0;
    connectionsToInputA.clear();
    connectionsToInputB.clear();
}