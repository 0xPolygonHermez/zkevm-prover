#include "pin.hpp"

string pin2string (PinId id)
{
    switch (id)
    {
        case PinId::pin_a: return "a";
        case PinId::pin_b: return "b";
        case PinId::pin_r: return "c";
        default:
            cerr << "Error: pin2string() got an invalid pin id=" << id << endl;
            exitProcess();
    }
    return "";
}

PinId string2pin (string s)
{
    if (s=="a") return PinId::pin_a;
    if (s=="b") return PinId::pin_b;
    if (s=="c") return PinId::pin_r;

    cerr << "Error: string2pin() got an invalid pin id string=" << s << endl;
    exitProcess();
    return PinId::pin_a;
}