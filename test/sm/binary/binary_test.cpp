#include <vector>
#include "binary_test.hpp"
#include "binary_action.hpp"
#include "binary_executor.hpp"

using namespace std;

uint64_t BinarySMTest (Goldilocks &fr, const Config &config)
{
    uint64_t numberOfErrors = 0;

    BinaryExecutor binaryExecutor(fr, config);

    vector<BinaryAction> list;

    BinaryAction action;

    // Add: opcode=1
    action.opcode = 1;
    
    action.a.set_str("0FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.b.set_str("0FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.b.set_str("1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE", 16);
    list.push_back(action);

    action.a.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.b.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.c.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE", 16);
    list.push_back(action);

    action.a.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("b486e735789b55a76376c3478ae4bc588d0740184aa0873dd0386392daed8db5", 16);
    action.c.set_str("649b4c45bb034df66329b0c327023b1eec4d927d75c3ef2525820401441a42f4", 16);
    list.push_back(action);

    // Sub: opcode=2
    action.opcode = 2;

    action.a.set_str("2", 16);
    action.b.set_str("1", 16);
    action.c.set_str("1", 16);
    list.push_back(action);

    action.a.set_str("0", 16);
    action.b.set_str("1", 16);
    action.c.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    list.push_back(action);

    action.a.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.b.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("b486e735789b55a76376c3478ae4bc588d0740184aa0873dd0386392daed8db5", 16);
    action.b.set_str("a01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.c.set_str("1472822536335d5863c3d5cbeec73d922dc0edb31f7d1f567aeec32471c0d876", 16);
    list.push_back(action);

    action.a.set_str("a01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("b486e735789b55a76376c3478ae4bc588d0740184aa0873dd0386392daed8db5", 16);
    action.c.set_str("eb8d7ddac9cca2a79c3c2a341138c26dd23f124ce082e0a985113cdb8e3f278a", 16);
    list.push_back(action);

    // LT Less Than: opcode=3
    action.opcode = 3;

    action.a.set_str("0", 16);
    action.b.set_str("1", 16);
    action.c.set_str("1", 16);
    list.push_back(action);

    action.a.set_str("1", 16);
    action.b.set_str("0", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("0", 16);
    action.b.set_str("0", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("0", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("a01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.c.set_str("1", 16);
    list.push_back(action);

    action.a.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("a01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    // GT Greater Than: opcode=4
    action.opcode = 4;

    action.a.set_str("0", 16);
    action.b.set_str("1", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("1", 16);
    action.b.set_str("0", 16);
    action.c.set_str("1", 16);
    list.push_back(action);

    action.a.set_str("0", 16);
    action.b.set_str("0", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("0", 16);
    action.c.set_str("1", 16);
    list.push_back(action);

    action.a.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("a01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("a01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.c.set_str("1", 16);
    list.push_back(action);

    // SLT Signed Less Than: opcode=5
    action.opcode = 5;

    action.a.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.b.set_str("0", 16);
    action.c.set_str("1", 16);
    list.push_back(action);

    action.a.set_str("0", 16);
    action.b.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("0", 16);
    action.b.set_str("0", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("FFFFFF", 16);
    action.b.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF000000", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF000000", 16);
    action.b.set_str("FFFFFF", 16);
    action.c.set_str("1", 16);
    list.push_back(action);

    // SGT Signed Greater Than: opcode=6
    action.opcode = 6;

    action.a.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.b.set_str("0", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("0", 16);
    action.b.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.c.set_str("1", 16);
    list.push_back(action);

    action.a.set_str("0", 16);
    action.b.set_str("0", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    // EQ Equal: opcode=7
    action.opcode = 7;

    action.a.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("0", 16);
    action.b.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.c.set_str("1", 16);
    list.push_back(action);

    action.a.set_str("3f", 16);
    action.b.set_str("3f", 16);
    action.c.set_str("1", 16);
    list.push_back(action);

    // ISZERO: opcode=8
    action.opcode = 8;

    action.a.set_str("0", 16);
    action.b.set_str("0", 16);
    action.c.set_str("1", 16);
    list.push_back(action);

    action.a.set_str("5", 16);
    action.b.set_str("0", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("0", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    // AND: opcode=9
    action.opcode = 9;

    action.a.set_str("0F01", 16);
    action.b.set_str("0F01", 16);
    action.c.set_str("0F01", 16);
    list.push_back(action);

    action.a.set_str("0E0E", 16);
    action.b.set_str("0101", 16);
    action.c.set_str("0000", 16);
    list.push_back(action);

    action.a.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.b.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.c.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    list.push_back(action);

    action.a.set_str("0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F", 16);
    action.b.set_str("0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F", 16);
    action.c.set_str("0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F", 16);
    list.push_back(action);

    // OR: opcode=a
    action.opcode = 0xa;

    action.a.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("b486e735789b55a76376c3478ae4bc588d0740184aa0873dd0386392daed8db5", 16);
    action.c.set_str("b496e7357afffdeffff6ef7f9efdfededf47527d6ba3e7ffd579e3fefbedbdbf", 16);
    list.push_back(action);

    // XOR: opcode=b
    action.opcode = 0xb;

    action.a.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.b.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F", 16);
    action.b.set_str("F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0", 16);
    action.c.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    list.push_back(action);

    action.a.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("b486e735789b55a76376c3478ae4bc588d0740184aa0873dd0386392daed8db5", 16);
    action.c.set_str("49282253afcade99cc42e3c16f9c29ed241127d6183e0da8571c3fcb3c1388a", 16);
    list.push_back(action);

    // NOT: opcode=c
    action.opcode = 0xc;

    action.a.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    action.b.set_str("0", 16);
    action.c.set_str("0", 16);
    list.push_back(action);

    action.a.set_str("0F", 16);
    action.b.set_str("0", 16);
    action.c.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0", 16);
    list.push_back(action);

    action.a.set_str("b01465104267f84effb2ed7b9c1d7ec65f4652652b2367e75549a06e692cb53f", 16);
    action.b.set_str("0", 16);
    action.c.set_str("4feb9aefbd9807b1004d128463e28139a0b9ad9ad4dc9818aab65f9196d34ac0", 16);
    list.push_back(action);

    action.a.set_str("1", 16);
    action.b.set_str("0", 16);
    action.c.set_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE", 16);
    list.push_back(action);

    binaryExecutor.execute(list);

    return numberOfErrors;
}