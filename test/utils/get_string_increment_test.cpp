#include <vector>
#include "get_string_increment_test.hpp"
#include "utils.hpp"
#include "scalar.hpp"

using namespace std;

struct StringIncrementTest
{
    string oldString;
    string newString;
    uint64_t offset;
    uint64_t length;
};

vector<StringIncrementTest> stringIncrementTest =
{
    {"", "", 0, 0},
    {"", "00", 0, 0},
    {"00", "000100", 1, 1},
    {"01000100", "01000100000100", 5, 1},
    {"01000100000100", "01000000000100", 2, 1},
    {"01000100000100", "010001000001", 0, 6},
    {"01000100000100", "010001000001009821", 7, 2},
    {"01000100000100", "01000100000100009821", 8, 2},
    {"01000100000100", "0100010000010000982100", 8, 2},
    {"01000100000100", "01000000000100", 2, 1},
    {"01000100000100", "01000101000100", 3, 1}
};

uint64_t GetStringIncrementTest (void)
{
    uint64_t numberOfFailed = 0;

    for (uint64_t i=0; i<stringIncrementTest.size(); i++)
    {
        uint64_t offset, length;
        getStringIncrement(string2ba(stringIncrementTest[i].oldString), string2ba(stringIncrementTest[i].newString), offset, length);
        if ((offset != stringIncrementTest[i].offset) || (length != stringIncrementTest[i].length))
        {
            zklog.error("GetStringIncrementTest() failed i=" + to_string(i) + " offset=" + to_string(offset) + " expectedOffset=" + to_string(stringIncrementTest[i].offset) + " length=" + to_string(length) + " expectedLength=" + to_string(stringIncrementTest[i].length));
            numberOfFailed++;
        }
    }

    if (numberOfFailed != 0)
    {
        zklog.error("GetStringIncrementTest() failed " + to_string(numberOfFailed) + " tests");
    }
    else
    {
        zklog.info("GetStringIncrementTest() succeeded");
    }
    return numberOfFailed;
}
