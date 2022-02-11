#include <fstream>
#include <iostream>
#include <iomanip>
#include <uuid/uuid.h>
#include "utils.hpp"
#include "scalar.hpp"
#include "pols.hpp"
#include <openssl/md5.h>
#include "merkle.hpp"

using namespace std;

void printRegs(Context &ctx)
{
    cout << "Registers:" << endl;
    printU64(ctx, "A3", pol(A3)[ctx.step]);
    printU64(ctx, "A2", pol(A2)[ctx.step]);
    printU64(ctx, "A1", pol(A1)[ctx.step]);
    printReg(ctx, "A0", pol(A0)[ctx.step]);
    printU64(ctx, "B3", pol(B3)[ctx.step]);
    printU64(ctx, "B2", pol(B2)[ctx.step]);
    printU64(ctx, "B1", pol(B1)[ctx.step]);
    printReg(ctx, "B0", pol(B0)[ctx.step]);
    printU64(ctx, "C3", pol(C3)[ctx.step]);
    printU64(ctx, "C2", pol(C2)[ctx.step]);
    printU64(ctx, "C1", pol(C1)[ctx.step]);
    printReg(ctx, "C0", pol(C0)[ctx.step]);
    printU64(ctx, "D3", pol(D3)[ctx.step]);
    printU64(ctx, "D2", pol(D2)[ctx.step]);
    printU64(ctx, "D1", pol(D1)[ctx.step]);
    printReg(ctx, "D0", pol(D0)[ctx.step]);
    printU64(ctx, "E3", pol(E3)[ctx.step]);
    printU64(ctx, "E2", pol(E2)[ctx.step]);
    printU64(ctx, "E1", pol(E1)[ctx.step]);
    printReg(ctx, "E0", pol(E0)[ctx.step]);
    printReg(ctx, "SR", pol(SR)[ctx.step]);
    printU32(ctx, "CTX", pol(CTX)[ctx.step]);
    printU16(ctx, "SP", pol(SP)[ctx.step]);
    printU32(ctx, "PC", pol(PC)[ctx.step]);
    printU32(ctx, "MAXMEM", pol(MAXMEM)[ctx.step]);
    printU64(ctx, "GAS", pol(GAS)[ctx.step]);
    printU32(ctx, "zkPC", pol(zkPC)[ctx.step]);
    RawFr::Element step;
    ctx.fr.fromUI(step, ctx.step);
    printReg(ctx, "STEP", step, false, true);
#ifdef LOG_FILENAME
    cout << "File: " << ctx.fileName << " Line: " << ctx.line << endl;
#endif
}

void printVars(Context &ctx)
{
    cout << "Variables:" << endl;
    uint64_t i = 0;
    for (map<string, RawFr::Element>::iterator it = ctx.vars.begin(); it != ctx.vars.end(); it++)
    {
        cout << "i: " << i << " varName: " << it->first << " fe: " << fe2n(ctx.fr, ctx.prime, it->second) << endl;
        i++;
    }
}

string printFea(Context &ctx, Fea &fea)
{
    return "fe0:" + ctx.fr.toString(fea.fe0, 16) +
           " fe1:" + ctx.fr.toString(fea.fe1, 16) +
           " fe2:" + ctx.fr.toString(fea.fe2, 16) +
           " fe3:" + ctx.fr.toString(fea.fe3, 16);
}

void printMem(Context &ctx)
{
    cout << "Memory:" << endl;
    uint64_t i = 0;
    for (map<uint64_t, Fea>::iterator it = ctx.mem.begin(); it != ctx.mem.end(); it++)
    {
        mpz_class addr(it->first);
        cout << "i: " << i << " address:" << addr.get_str(16) << " ";
        cout << printFea(ctx, it->second);
        cout << endl;
        i++;
    }
}

#ifdef USE_LOCAL_STORAGE
void printStorage(Context &ctx)
{
    uint64_t i = 0;
    for (map<RawFr::Element, mpz_class, CompareFe>::iterator it = ctx.sto.begin(); it != ctx.sto.end(); it++)
    {
        RawFr::Element fe = it->first;
        mpz_class scalar = it->second;
        cout << "Storage: " << i << " fe: " << ctx.fr.toString(fe, 16) << " scalar: " << scalar.get_str(16) << endl;
    }
}
#endif

void printReg(Context &ctx, string name, RawFr::Element &fe, bool h, bool bShort)
{
    cout << "    Register: " << name << " Value: " << ctx.fr.toString(fe, 16) << endl;
}

void printDb(Context &ctx)
{
    ctx.db.print();
}

void printU64(Context &ctx, string name, uint64_t v)
{
    cout << "    U64: " << name << ":" << v << endl;
}

void printU32(Context &ctx, string name, uint32_t v)
{
    cout << "    U32: " << name << ":" << v << endl;
}

void printU16(Context &ctx, string name, uint16_t v)
{
    cout << "    U16: " << name << ":" << v << endl;
}

void printBa(uint8_t * pData, uint64_t dataSize, string name)
{
    cout << name << " = ";
    for (uint64_t k=0; k<dataSize; k++)
    {
        cout << byte2string(pData[k]) << ":";
    }
    cout << endl;
}

string rt2string(eReferenceType rt)
{
    switch (rt)
    {
    case rt_unknown:
        return "rt_unknown";
    case rt_pol:
        return "rt_pol";
    case rt_field:
        return "rt_field";
    case rt_treeGroup:
        return "rt_treeGroup";
    case rt_treeGroup_elementProof:
        return "rt_treeGroup_elementProof";
    case rt_treeGroup_groupProof:
        return "rt_treeGroup_groupProof";
    case rt_treeGroupMultipol:
        return "rt_treeGroupMultipol";
    case rt_treeGroupMultipol_groupProof:
        return "rt_treeGroupMultipol_groupProof";
    case rt_idxArray:
        return "rt_idxArray";
    case rt_int:
        return "rt_int";
    default:
        cerr << "rt2string() found unrecognized reference type: " << rt << endl;
        exit(-1);
    }
    enum eReferenceType
    {
        rt_unknown = 0,
        rt_pol = 1,
        rt_field = 2,
        rt_treeGroup = 3,
        rt_treeGroup_elementProof = 4,
        rt_treeGroup_groupProof = 5,
        rt_treeGroupMultipol = 6,
        rt_treeGroupMultipol_groupProof = 7,
        rt_idxArray = 8,
        rt_int = 9
    };
}

string calculateExecutionHash(RawFr &fr, Reference &ref, string prevHash)
{
    switch (ref.type)
    {
    case rt_pol:
    {
        unsigned char result[MD5_DIGEST_LENGTH];
        char tempHash[32];
        std::string currentHash;
        std::string newHash;
        string pol;
        for (uint64_t i = 0; i < ref.N; i++)
        {
            pol += fr.toString(ref.pPol[i], 16);
        }
        MD5((unsigned char *)pol.c_str(), pol.size(), result);
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            currentHash.append(tempHash);
        }
        string concatHashes = prevHash + currentHash;
        MD5((unsigned char *)concatHashes.c_str(), concatHashes.size(), result);

        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            newHash.append(tempHash);
        }
        return newHash;
    }
    case rt_field:
    {
        unsigned char result[MD5_DIGEST_LENGTH];
        char tempHash[32];
        std::string currentHash;
        std::string newHash;
        string pol = fr.toString(ref.fe, 16);

        MD5((unsigned char *)pol.c_str(), pol.size(), result);
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            currentHash.append(tempHash);
        }
        string concatHashes = prevHash + currentHash;
        MD5((unsigned char *)concatHashes.c_str(), concatHashes.size(), result);

        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            newHash.append(tempHash);
        }
        return newHash;
    }
    case rt_treeGroup:
    {
        unsigned char result[MD5_DIGEST_LENGTH];
        char tempHash[32];
        string currentHash;
        string newHash;
        string auxHash;
        string tempConcatHashes;
        string pol;
        Merkle M(MERKLE_ARITY);
        uint32_t groupSize = M.numHashes(ref.groupSize);
        uint64_t k = 0;
        for (; k < (ref.memSize / sizeof(RawFr::Element)) - groupSize * ref.nGroups; k++)
        {
            pol += fr.toString(ref.pTreeGroup[k], 16);
        }
        MD5((unsigned char *)pol.c_str(), pol.size(), result);

        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            auxHash.append(tempHash);
        }
        tempConcatHashes.append(auxHash);
        pol = "";
        auxHash = "";
        for (; k < (ref.memSize / sizeof(RawFr::Element)); k++)
        {
            pol += fr.toString(ref.pTreeGroup[k], 16);
        }
        MD5((unsigned char *)pol.c_str(), pol.size(), result);
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            auxHash.append(tempHash);
        }

        tempConcatHashes.append(auxHash);
        MD5((unsigned char *)tempConcatHashes.c_str(), tempConcatHashes.size(), result);
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            currentHash.append(tempHash);
        }
        string concatHashes = prevHash + currentHash;
        MD5((unsigned char *)concatHashes.c_str(), concatHashes.size(), result);

        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            newHash.append(tempHash);
        }
        return newHash;
    }

    case rt_treeGroup_elementProof:
    {
        unsigned char result[MD5_DIGEST_LENGTH];
        char tempHash[32];
        std::string currentHash;
        std::string newHash;
        string pol;
        for (uint64_t i = 0; i < ref.memSize / sizeof(RawFr::Element); i++)
        {
            pol += fr.toString(ref.pTreeGroup_elementProof[i], 16);
        }
        MD5((unsigned char *)pol.c_str(), pol.size(), result);
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            currentHash.append(tempHash);
        }
        string concatHashes = prevHash + currentHash;
        MD5((unsigned char *)concatHashes.c_str(), concatHashes.size(), result);

        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            newHash.append(tempHash);
        }
        return newHash;
    }
    case rt_treeGroup_groupProof:
    {
        unsigned char result[MD5_DIGEST_LENGTH];
        char tempHash[32];
        std::string currentHash;
        std::string newHash;
        string pol;
        for (uint64_t i = 0; i < ref.memSize / sizeof(RawFr::Element); i++)
        {
            pol += fr.toString(ref.pTreeGroup_groupProof[i], 16);
        }
        MD5((unsigned char *)pol.c_str(), pol.size(), result);
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            currentHash.append(tempHash);
        }
        string concatHashes = prevHash + currentHash;
        MD5((unsigned char *)concatHashes.c_str(), concatHashes.size(), result);

        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            newHash.append(tempHash);
        }
        return newHash;
    }
    case rt_treeGroupMultipol:
    {
        unsigned char result[MD5_DIGEST_LENGTH];
        char tempHash[32];
        std::string currentHash;
        std::string newHash;
        string pol;
        string auxHash;
        string tempConcatHashes;
        Merkle M(MERKLE_ARITY);
        uint32_t polProofSize = M.numHashes(ref.nPols);

        for (uint32_t k = 0; k < (ref.memSize / sizeof(RawFr::Element)); k++)
        {
            if (k % (polProofSize + ref.groupSize) == 0 && k <= (polProofSize + ref.groupSize) * ref.nGroups && k != 0)
            {
                MD5((unsigned char *)pol.c_str(), pol.size(), result);
                for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
                {
                    sprintf(tempHash, "%02x", result[i]);
                    auxHash.append(tempHash);
                }
                tempConcatHashes.append(auxHash);
                pol = "";
                auxHash = "";
            }
            pol.append(fr.toString(ref.pTreeGroupMultipol[k], 16));
        }
        MD5((unsigned char *)pol.c_str(), pol.size(), result);
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            auxHash.append(tempHash);
        }
        tempConcatHashes.append(auxHash);
        MD5((unsigned char *)tempConcatHashes.c_str(), tempConcatHashes.size(), result);

        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            currentHash.append(tempHash);
        }

        string concatHashes = prevHash.append(currentHash);
        MD5((unsigned char *)concatHashes.c_str(), concatHashes.size(), result);

        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            newHash.append(tempHash);
        }
        return newHash;
    }
    case rt_treeGroupMultipol_groupProof:
    {
        unsigned char result[MD5_DIGEST_LENGTH];
        char tempHash[32];
        std::string currentHash;
        std::string newHash;
        string pol;
        for (uint64_t i = 0; i < ref.memSize / sizeof(RawFr::Element); i++)
        {
            pol += fr.toString(ref.pTreeGroupMultipol_groupProof[i], 16);
        }
        MD5((unsigned char *)pol.c_str(), pol.size(), result);
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            currentHash.append(tempHash);
        }
        string concatHashes = prevHash + currentHash;
        MD5((unsigned char *)concatHashes.c_str(), concatHashes.size(), result);

        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            newHash.append(tempHash);
        }
        return newHash;
    }
    case rt_idxArray:
    {
        unsigned char result[MD5_DIGEST_LENGTH];
        char tempHash[32];

        string array;
        for (uint32_t k = 0; k < ref.N; k++)
        {
            array.append(std::to_string(ref.pIdxArray[k]));
        }
        MD5((unsigned char *)array.c_str(), array.size(), result);
        std::string currentHash;
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            currentHash.append(tempHash);
        }
        string concatHashes = prevHash + currentHash;
        MD5((unsigned char *)concatHashes.c_str(), concatHashes.size(), result);

        std::string newHash;
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            newHash.append(tempHash);
        }
        return newHash;
    }
    case rt_int:
    {
        unsigned char result[MD5_DIGEST_LENGTH];
        char tempHash[32];
        std::string currentHash;
        std::string newHash;
        string array = std::to_string(ref.integer);

        MD5((unsigned char *)array.c_str(), array.size(), result);
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            currentHash.append(tempHash);
        }
        string concatHashes = prevHash + currentHash;
        MD5((unsigned char *)concatHashes.c_str(), concatHashes.size(), result);

        for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        {
            sprintf(tempHash, "%02x", result[i]);
            newHash.append(tempHash);
        }
        return newHash;
    }
    default:
    {
        cerr << "  printReference() found unrecognized reference type: " << ref.type << endl;
        exit(-1);
    }
    }
}
void printReference(RawFr &fr, Reference &ref)
{
    cout << "  Reference of type: " << rt2string(ref.type) << endl;
    switch (ref.type)
    {
    case rt_pol:
    {
        cout << "  ref.N: " << ref.N << endl;
        cout << "  ref.elementType: " << et2string(ref.elementType) << endl;
        cout << "  ref.memSize: " << ref.memSize << endl;
        uint64_t printed = 0;
        for (uint64_t i = 0; i < ref.N; i++)
        {
            if (fr.isZero(ref.pPol[i]))
                continue;
            if (i > 5 && i < ref.N - 5)
                continue;
            if (printed < 10)
                cout << "  ref.pPol[" << i << "]: " << fr.toString(ref.pPol[i], 16) << endl;
            printed++;
        }
        cout << "  found " << printed << " non-zero elements" << endl;
        return;
    }
    case rt_field:
    {
        cout << "  ref.fe: " << fr.toString(ref.fe, 16) << endl;
        return;
    }
    case rt_treeGroup:
    {
        cout << "  ref.elementType: " << ref.elementType << endl;
        cout << "  ref.memSize: " << ref.memSize << endl;
        cout << "  ref.memSize: " << ref.nGroups << endl;
        cout << "  ref.memSize: " << ref.groupSize << endl;
        cout << "  ref.memSize: " << ref.nPols << endl;

        cout << "  ref.pTreeGroupMultipol[" << 0 << "][" << 0 << "]: " << fr.toString(ref.pTreeGroup[0], 16) << endl;
        cout << "  ref.pTreeGroupMultipol[" << 0 << "][" << 1 << "]: " << fr.toString(ref.pTreeGroup[1], 16) << endl;
        cout << "  ref.pTreeGroupMultipol[last - 1]: " << fr.toString(ref.pTreeGroup[(ref.memSize / sizeof(RawFr::Element)) - 2], 16) << endl;
        cout << "  ref.pTreeGroupMultipollast]: " << fr.toString(ref.pTreeGroup[(ref.memSize / sizeof(RawFr::Element)) - 1], 16) << endl;
        return;
    }

    case rt_treeGroup_elementProof:
    {
        cout << "  ref.elementType: " << ref.elementType << endl;
        cout << "  ref.memSize: " << ref.memSize << endl;
        cout << "  ref.memSize: " << ref.nGroups << endl;
        cout << "  ref.memSize: " << ref.groupSize << endl;

        cout << "  ref.pTreeGroup_elementProof[" << 0 << "][" << 0 << "]: " << fr.toString(ref.pTreeGroup_elementProof[0], 16) << endl;
        cout << "  ref.pTreeGroup_elementProof[" << 0 << "][" << 0 << "]: " << fr.toString(ref.pTreeGroup_elementProof[1], 16) << endl;
        cout << "  ref.pTreeGroup_elementProof[" << 0 << "][" << 0 << "]: " << fr.toString(ref.pTreeGroup_elementProof[(ref.memSize / sizeof(RawFr::Element)) - 2], 16) << endl;
        cout << "  ref.pTreeGroup_elementProof[" << 0 << "][" << 0 << "]: " << fr.toString(ref.pTreeGroup_elementProof[(ref.memSize / sizeof(RawFr::Element)) - 1], 16) << endl;

        return;
    }
    case rt_treeGroup_groupProof:
    {
        cout << "  ref.elementType: " << ref.elementType << endl;
        cout << "  ref.memSize: " << ref.memSize << endl;
        cout << "  ref.nGroups: " << ref.nGroups << endl;
        cout << "  ref.groupSize: " << ref.groupSize << endl;

        cout << "  ref.pTreeGroup_groupProof[" << 0 << "][" << 0 << "]: " << fr.toString(ref.pTreeGroup_groupProof[0], 16) << endl;
        cout << "  ref.pTreeGroup_groupProof[" << 0 << "][" << 1 << "]: " << fr.toString(ref.pTreeGroup_groupProof[1], 16) << endl;
        cout << "  ref.pTreeGroup_groupProof[0]: " << fr.toString(ref.pTreeGroup_groupProof[(ref.memSize / sizeof(RawFr::Element)) - 2], 16) << endl;
        cout << "  ref.pTreeGroup_groupProof[last]: " << fr.toString(ref.pTreeGroup_groupProof[(ref.memSize / sizeof(RawFr::Element)) - 1], 16) << endl;
        return;
    }
    case rt_treeGroupMultipol:
    {
        cout << "  ref.elementType: " << ref.elementType << endl;
        cout << "  ref.memSize: " << ref.memSize << endl;
        cout << "  ref.memSize: " << ref.nGroups << endl;
        cout << "  ref.memSize: " << ref.groupSize << endl;
        cout << "  ref.memSize: " << ref.nPols << endl;

        cout << "  ref.pTreeGroupMultipol[" << 0 << "][" << 0 << "]: " << fr.toString(ref.pTreeGroupMultipol[0], 16) << endl;
        cout << "  ref.pTreeGroupMultipol[" << 0 << "][" << 1 << "]: " << fr.toString(ref.pTreeGroupMultipol[1], 16) << endl;
        cout << "  ref.pTreeGroupMultipolMainTree[0]: " << fr.toString(ref.pTreeGroupMultipol[(ref.memSize / sizeof(RawFr::Element)) - 2], 16) << endl;
        cout << "  ref.pTreeGroupMultipolMainTree[last]: " << fr.toString(ref.pTreeGroupMultipol[(ref.memSize / sizeof(RawFr::Element)) - 1], 16) << endl;
        return;
    }
    case rt_treeGroupMultipol_groupProof:
    {
        cout << "  ref.elementType: " << ref.elementType << endl;
        cout << "  ref.memSize: " << ref.memSize << endl;
        cout << "  ref.memSize: " << ref.nGroups << endl;
        cout << "  ref.memSize: " << ref.groupSize << endl;
        cout << "  ref.memSize: " << ref.nPols << endl;

        cout << "  ref.pTreeGroupMultipol[" << 0 << "][" << 0 << "]: " << fr.toString(ref.pTreeGroupMultipol_groupProof[0], 16) << endl;
        cout << "  ref.pTreeGroupMultipol[" << 0 << "][" << 1 << "]: " << fr.toString(ref.pTreeGroupMultipol_groupProof[1], 16) << endl;
        cout << "  ref.pTreeGroupMultipolMainTree[0]: " << fr.toString(ref.pTreeGroupMultipol_groupProof[(ref.memSize / sizeof(RawFr::Element)) - 2], 16) << endl;
        cout << "  ref.pTreeGroupMultipolMainTree[last]: " << fr.toString(ref.pTreeGroupMultipol_groupProof[(ref.memSize / sizeof(RawFr::Element)) - 1], 16) << endl;
        return;
    }
    case rt_idxArray:
    {
        cout << "  ref.N: " << ref.N << endl;
        cout << "  ref.elementType: " << et2string(ref.elementType) << endl;
        cout << "  ref.memSize: " << ref.memSize << endl;
        uint64_t printed = 0;
        for (uint64_t i = 0; i < ref.N; i++)
        {
            if (ref.pIdxArray[i] == 0)
                continue;
            if (printed < 10)
                cout << "  ref.pIdxArray[" << i << "]: " << ref.pIdxArray[i] << endl;
            printed++;
        }
        cout << "  found " << printed << " non-zero elements" << endl;
        return;
    }
    case rt_int:
    {
        cout << "  ref.integer: " << ref.integer << endl;
        return;
    }
    default:
    {
        cerr << "  printReference() found unrecognized reference type: " << ref.type << endl;
        exit(-1);
    }
    }
}

uint64_t TimeDiff(const struct timeval &startTime, const struct timeval &endTime)
{
    struct timeval diff;

    // Calculate the time difference
    diff.tv_sec = endTime.tv_sec - startTime.tv_sec;
    if (endTime.tv_usec >= startTime.tv_usec)
        diff.tv_usec = endTime.tv_usec - startTime.tv_usec;
    else if (diff.tv_sec > 0)
    {
        diff.tv_usec = 1000000 + endTime.tv_usec - startTime.tv_usec;
        diff.tv_sec--;
    }
    else
    {
        cerr << "Error: TimeDiff() got startTime > endTime" << endl;
        exit(-1);
    }

    // Return the total number of us
    return diff.tv_usec + 1000000 * diff.tv_sec;
}

uint64_t TimeDiff(const struct timeval &startTime)
{
    struct timeval endTime;
    gettimeofday(&endTime, NULL);
    return TimeDiff(startTime, endTime);
}

string getTimestamp (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    char tmbuf[64], buf[256];
    strftime(tmbuf, sizeof tmbuf, "%Y%m%d_%H%M%S", gmtime(&tv.tv_sec));
    snprintf(buf, sizeof buf, "%s_%03ld", tmbuf, tv.tv_usec/1000);
    return buf;
}

string getUUID (void)
{
    char uuidString[37];
    uuid_t uuid;
    uuid_generate(uuid);
    uuid_unparse(uuid, uuidString);
    return uuidString;
}

void json2file(const json &j, const string &fileName)
{
    ofstream outputStream(fileName);
    if (!outputStream.good())
    {
        cerr << "Error: json2file() failed loading output JSON file " << fileName << endl;
        exit(-1);
    }
    outputStream << setw(4) << j << endl;
    outputStream.close();
}

void file2json(const string &fileName, json &j)
{
    std::ifstream inputStream(fileName);
    if (!inputStream.good())
    {
        cerr << "Error: file2json() failed loading input JSON file " << fileName << endl;
        exit(-1);
    }
    inputStream >> j;
    inputStream.close();
}


void inputProver2Input (RawFr &fr, const zkprover::InputProver &inputProver, Input &input)
{
    // Parse public inputs
    zkprover::PublicInputs publicInputs = inputProver.public_inputs();
    input.publicInputs.oldStateRoot = publicInputs.old_state_root();
    input.publicInputs.oldLocalExitRoot = publicInputs.old_local_exit_root();
    input.publicInputs.newStateRoot = publicInputs.new_state_root();
    input.publicInputs.newLocalExitRoot = publicInputs.new_local_exit_root();
    input.publicInputs.sequencerAddr = publicInputs.sequencer_addr();
    input.publicInputs.batchHashData = publicInputs.batch_hash_data();
    input.publicInputs.chainId = publicInputs.chain_id();
    input.publicInputs.batchNum = publicInputs.batch_num();
#ifdef LOG_RPC_INPUT
    cout << "input.publicInputs.oldStateRoot: " << input.publicInputs.oldStateRoot << endl;
    cout << "input.publicInputs.oldLocalExitRoot: " << input.publicInputs.oldLocalExitRoot << endl;
    cout << "input.publicInputs.newStateRoot: " << input.publicInputs.newStateRoot << endl;
    cout << "input.publicInputs.newLocalExitRoot: " << input.publicInputs.newLocalExitRoot << endl;
    cout << "input.publicInputs.sequencerAddr: " << input.publicInputs.sequencerAddr << endl;
    cout << "input.publicInputs.batchHashData: " << input.publicInputs.batchHashData << endl;
    cout << "input.publicInputs.chainId: " << to_string(input.publicInputs.chainId) << endl;
    cout << "input.publicInputs.batchNum: " << to_string(input.publicInputs.batchNum) << endl;
#endif

    // Parse global exit root
    input.globalExitRoot = inputProver.global_exit_root();
#ifdef LOG_RPC_INPUT
    cout << "input.globalExitRoot: " << input.globalExitRoot << endl;
#endif

    // Parse transactions list
    for (int i=0; i<inputProver.txs_size(); i++)
    {
        input.txs.push_back(inputProver.txs(i));
#ifdef LOG_RPC_INPUT
        cout << "input.txStrings[" << to_string(i) << "]: " << input.txs[i] << endl;
#endif
    }

    // Preprocess the transactions
    input.preprocessTxs();

    // Parse keys map
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> > db;
    db = inputProver.db();
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> >::iterator it;
    for (it=db.begin(); it!=db.end(); it++)
    {
        vector<RawFr::Element> dbValue;
        string concatenatedValues = it->second;
        if (concatenatedValues.size()%64!=0)
        {
            cerr << "Error: inputProver2Input() found invalid db value size: " << concatenatedValues.size() << endl;
            exit(-1); // TODO: return an error
        }
        for (uint64_t i=0; i<concatenatedValues.size(); i+=64)
        {
            RawFr::Element fe;
            string2fe(fr, concatenatedValues.substr(i, 64), fe);
            dbValue.push_back(fe);
        }
        RawFr::Element fe;
        string2fe(fr, it->first, fe);
        input.db[fe] = dbValue;
#ifdef LOG_RPC_INPUT
        cout << "input.keys[" << it->first << "]: " << input.keys[it->first] << endl;
#endif
    }
}

void input2InputProver (RawFr &fr, const Input &input, zkprover::InputProver &inputProver)
{
    // Parse public inputs
    zkprover::PublicInputs * pPublicInputs = new zkprover::PublicInputs();
    pPublicInputs->set_old_state_root(input.publicInputs.oldStateRoot);
    pPublicInputs->set_old_local_exit_root(input.publicInputs.oldStateRoot);
    pPublicInputs->set_new_state_root(input.publicInputs.newStateRoot);
    pPublicInputs->set_new_local_exit_root(input.publicInputs.newLocalExitRoot);
    pPublicInputs->set_sequencer_addr(input.publicInputs.sequencerAddr);
    pPublicInputs->set_batch_hash_data(input.publicInputs.batchHashData);
    pPublicInputs->set_chain_id(input.publicInputs.chainId);
    pPublicInputs->set_batch_num(input.publicInputs.batchNum);
    inputProver.set_allocated_public_inputs(pPublicInputs);

    // Parse global exit root
    inputProver.set_global_exit_root(input.globalExitRoot);

    // Parse transactions list
    for (uint64_t i=0; i<input.txs.size(); i++)
    {
        inputProver.add_txs(input.txs[i]);
    }

    // Parse keys map
    map< RawFr::Element, vector<RawFr::Element>, CompareFe >::const_iterator it;
    for (it=input.db.begin(); it!=input.db.end(); it++)
    {
        string key = NormalizeToNFormat(fr.toString(it->first, 16), 64);
        string value;
        vector<RawFr::Element> dbValue = it->second;
        for (uint64_t i=0; i<dbValue.size(); i++)
        {
            value += NormalizeToNFormat(fr.toString(dbValue[i], 16), 64);
        }
        (*inputProver.mutable_db())[key] = value;
    }
}

void proof2ProofProver (RawFr &fr, const Proof &proof, zkprover::Proof &proofProver)
{
    // Set proofA
    //google::protobuf::RepeatedPtrField<std::string>* pProofA = proofProver.mutable_proofa();
    //pProofA->Reserve(proof.proofA.size());
    for (uint64_t i=0; i<proof.proofA.size(); i++)
    {
        proofProver.add_proofa(proof.proofA[i]);
        //std::string aux = proof.proofA[i];
        //std::string * pAux = pProofA->Add();
        //*pAux = aux;
#ifdef LOG_RPC_OUTPUT
        cout << "RCP output proofA[" << i << "] = " << proof.proofA[i] << endl;
#endif
    }

    // Set proofB
    for (uint64_t i=0; i<proof.proofB.size(); i++)
    {
        zkprover::ProofB *pProofB = proofProver.add_proofb();
        for (uint64_t j=0; j<proof.proofB[i].proof.size(); j++)
        {
            pProofB->add_proofs(proof.proofB[i].proof[j]);
#ifdef LOG_RPC_OUTPUT
        cout << "RCP output proofB[" << i << "].proof[" << j << "] = " << proof.proofB[i].proof[j] << endl;
#endif            
        }
    }

    // Set proofC
    for (uint64_t i=0; i<proof.proofC.size(); i++)
    {
        proofProver.add_proofc(proof.proofC[i]);
#ifdef LOG_RPC_OUTPUT
        cout << "RCP output proofC[" << i << "] = " << proof.proofC[i] << endl;
#endif
    }
}