#ifndef TRACER_CONFIG_HPP
#define TRACER_CONFIG_HPP

class TraceConfig
{
public:
     // bEnabled = true if a trace configuration has been provided
    bool bEnabled;

    // Configuration parameters
    bool bDisableStorage;
    bool bDisableStack;
    bool bEnableMemory;
    bool bEnableReturnData;
    string txHashToGenerateFullTrace; // return full traces of this tx

    // Flags to enable/disable functionality
    bool bGenerateFullTrace;
    bool bGenerateStorage;
    bool bGenerateStack;
    bool bGenerateMemory;
    bool bGenerateReturnData;

    TraceConfig() :
        bEnabled(false),
        bDisableStorage(false),
        bDisableStack(false),
        bEnableMemory(false),
        bEnableReturnData(false),
        bGenerateFullTrace(true),
        bGenerateStorage(false),
        bGenerateStack(false),
        bGenerateMemory(false),
        bGenerateReturnData(false)
    {};

    // Call calculateFlags() once all configuration parameters have been set
    void calculateFlags (void)
    {
        bGenerateStorage      = bEnabled && !bDisableStorage;
        bGenerateStack        = bEnabled && !bDisableStack;
        bGenerateMemory       = bEnabled && bEnableMemory;
        bGenerateReturnData   = bEnabled && bEnableReturnData;
    } 

    bool operator==(TraceConfig &other)
    {
        return
            bEnabled == other.bEnabled &&
            bDisableStorage == other.bDisableStorage &&
            bDisableStack == other.bDisableStack &&
            bEnableMemory == other.bEnableMemory &&
            bEnableReturnData == other.bEnableReturnData &&
            txHashToGenerateFullTrace == other.txHashToGenerateFullTrace;
    };

    bool operator!=(TraceConfig &other) { return !(*this == other); };
    
    TraceConfig & operator=(const TraceConfig &other)
    {
        bEnabled = other.bEnabled;
        bDisableStorage = other.bDisableStorage;
        bDisableStack = other.bDisableStack;
        bEnableMemory = other.bEnableMemory;
        bEnableReturnData = other.bEnableReturnData;
        txHashToGenerateFullTrace = other.txHashToGenerateFullTrace;
        return *this;
    }

    string toString(void)
    {
        return
            "bEnabled=" + to_string(bEnabled) +
            ",bDisableStorage=" + to_string(bDisableStorage) +
            ",bDisableStack=" + to_string(bDisableStack) +
            ",bEnableMemory=" + to_string(bEnableMemory) +
            ",bEnableReturnData=" + to_string(bEnableReturnData) +
            ",txHashToGenerateFullTrace=" + txHashToGenerateFullTrace;
    }
};

#endif