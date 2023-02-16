#ifndef TRACER_CONFIG_HPP
#define TRACER_CONFIG_HPP

class TraceConfig
{
public:
    bool bEnabled;
    bool bDisableStorage;
    bool bDisableStack;
    bool bEnableMemory;
    bool bEnableReturnData;
    string txHashToGenerateExecuteTrace; // return execute traces of this tx
    string txHashToGenerateCallTrace; // return call traces of this tx

    TraceConfig() :
        bEnabled(false),
        bDisableStorage(false),
        bDisableStack(false),
        bEnableMemory(false),
        bEnableReturnData(false) {};

    bool generateCallTraces (void)
    {
        return bEnabled && ((txHashToGenerateExecuteTrace.size() > 0) || (txHashToGenerateCallTrace.size() > 0));
    }

    bool generateStorage (void)
    {
        return bEnabled && !bDisableStorage;
    }

    bool generateStack (void)
    {
        return bEnabled && !bDisableStack;
    }

    bool generateMemory (void)
    {
        return bEnabled && bEnableMemory;
    }

    bool generateReturnData (void)
    {
        return bEnabled && bEnableReturnData;
    }    

    bool operator==(TraceConfig &other)
    {
        return
            bEnabled == other.bEnabled &&
            bDisableStorage == other.bDisableStorage &&
            bDisableStack == other.bDisableStack &&
            bEnableMemory == other.bEnableMemory &&
            bEnableReturnData == other.bEnableReturnData &&
            txHashToGenerateExecuteTrace == other.txHashToGenerateExecuteTrace &&
            txHashToGenerateCallTrace == txHashToGenerateCallTrace;
    };

    bool operator!=(TraceConfig &other) { return !(*this == other); };
    
    TraceConfig & operator=(const TraceConfig &other)
    {
        bEnabled = other.bEnabled;
        bDisableStorage = other.bDisableStorage;
        bDisableStack = other.bDisableStack;
        bEnableMemory = other.bEnableMemory;
        bEnableReturnData = other.bEnableReturnData;
        txHashToGenerateExecuteTrace = other.txHashToGenerateExecuteTrace;
        txHashToGenerateCallTrace = other.txHashToGenerateCallTrace;
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
            ",txHashToGenerateExecuteTrace=" + txHashToGenerateExecuteTrace +
            ",txHashToGenerateCallTrace=" + txHashToGenerateCallTrace;
    }
};

#endif