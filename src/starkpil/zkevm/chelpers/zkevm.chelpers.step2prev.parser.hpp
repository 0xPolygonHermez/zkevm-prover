#define NOPS_ 2094
#define NARGS_ 6804
#define NTEMP1_ 39
#define NTEMP3_ 2


uint64_t op2prev[NOPS_] = { 79, 79, 59, 88, 82, 82, 59, 88, 82, 80, 80, 80, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 8, 8, 46, 0, 1, 56, 8, 46, 8, 46, 0, 8, 46, 0, 8, 46, 0, 8, 46, 0, 8, 46, 0, 8, 46, 0, 8, 46, 0, 8, 46, 0, 8, 46, 0, 8, 46, 0, 8, 46, 0, 8, 46, 0, 8, 46, 0, 8, 46, 0, 8, 46, 0, 1, 4, 1, 45, 0, 30, 45, 54, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 59, 88, 82, 82, 59, 88, 54, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 100, 82, 100, 54, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 100, 82, 100, 54, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 100, 82, 100, 8, 3, 54, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 54, 0, 59, 88, 82, 82, 59, 88, 79, 100, 82, 100, 79, 100, 82, 100, 79, 100, 82, 100, 81, 79, 79, 79, 79, 79, 53, 1, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 82, 82, 82, 82, 82, 82, 82, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 5, 79, 79, 79, 79, 79, 53, 1, 53, 0, 53, 0, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 82, 82, 82, 82, 82, 82, 82, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 79, 100, 82, 100, 50, 27, 1, 96, 1, 27, 56, 0, 53, 86, 54, 1, 78, 54, 1, 78, 54, 1, 78, 54, 1, 78, 54, 1, 78, 54, 1, 78, 54, 1, 78, 54, 1, 78, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 82, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 43, 61, 90, 82, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 44, 61, 89, 79, 79, 79, 79, 79, 79, 79, 79, 79, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 82, 82, 82, 82, 82, 82, 82, 82, 82, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 8, 3, 3, 3, 3, 3, 3, 8, 3, 3, 3, 3, 3, 3, 8, 3, 3, 3, 3, 3, 3, 46, 0, 46, 0, 53, 0, 53, 0, 53, 0, 53, 0, 78, 47, 79, 8, 3, 3, 3, 3, 3, 3, 37, 45, 1, 27, 57, 59, 12, 70, 12, 70, 12, 70, 88, 82, 82, 82, 82, 82, 59, 12, 70, 12, 70, 12, 70, 88, 53, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 78, 79, 79, 79, 79, 59, 12, 70, 12, 70, 12, 70, 88, 82, 82, 82, 82, 82, 59, 12, 70, 12, 70, 12, 70, 88, 82, 79, 79, 79, 59, 12, 70, 12, 70, 88, 82, 82, 82, 82, 59, 12, 70, 12, 70, 88, 82, 79, 79, 79, 59, 12, 70, 12, 70, 88, 82, 82, 82, 82, 59, 12, 70, 12, 70, 88, 82, 79, 79, 79, 59, 12, 70, 12, 70, 88, 82, 82, 82, 82, 59, 12, 70, 12, 70, 88, 82, 79, 79, 79, 59, 12, 70, 12, 70, 88, 82, 82, 82, 82, 59, 12, 70, 12, 70, 88, 50, 27, 1, 96, 1, 27, 56, 0, 53, 86, 79, 82, 79, 79, 82, 79, 59, 12, 70, 88, 82, 59, 12, 70, 12, 44, 61, 89, 79, 79, 79, 79, 79, 79, 79, 79, 82, 79, 79, 79, 79, 79, 79, 79, 79, 82, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 1, 96, 79, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 44, 61, 89, 79, 79, 79, 79, 79, 79, 79, 79, 79, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 82, 82, 82, 82, 82, 82, 82, 82, 82, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 35, 35, 21, 47, 35, 0, 10, 78, 100, 32, 78, 22, 45, 87, 82, 79, 79, 79, 79, 55, 59, 12, 70, 12, 70, 12, 70, 12, 70, 88, 82, 82, 82, 82, 82, 82, 59, 12, 70, 12, 70, 12, 70, 12, 70, 88, 50, 93, 4, 27, 56, 53, 0, 50, 86, 79, 82, 79, 79, 82, 79, 59, 12, 70, 88, 82, 59, 12, 70, 12, 44, 61, 89, 79, 79, 79, 79, 79, 79, 79, 79, 82, 79, 79, 79, 79, 79, 79, 79, 79, 82, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 54, 78, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 44, 61, 89, 79, 79, 79, 79, 79, 79, 79, 79, 79, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 82, 82, 82, 82, 82, 82, 82, 82, 82, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 79, 100, 82, 100, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 53, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 53, 0, 78, 79, 79, 79, 79, 79, 79, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 88, 50, 50, 0, 1, 50, 46, 53, 0, 50, 0, 53, 0, 0, 78, 79, 50, 1, 50, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 4, 47, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 54, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 1, 78, 50, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 1, 78, 50, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 1, 78, 50, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 1, 78, 50, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 1, 78, 50, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 1, 78, 50, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 1, 78, 50, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 1, 78, 79, 37, 22, 2, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 27, 96, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 43, 64, 90, 4, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 44, 61, 89, 78, 79, 50, 1, 78, 78, 78, 78, 78, 78, 78, 78, 79, 37, 22, 2, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 93, 99, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 43, 64, 90, 4, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 44, 61, 89, 78, 79, 50, 1, 78, 78, 78, 78, 78, 78, 78, 78, 79, 37, 22, 2, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 87, 79, 50, 27, 96, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 43, 64, 90, 4, 59, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 70, 12, 44, 61, 89};


uint64_t args2prev[NARGS_] = { 0, 1, 751, 1, 0, 751, 0, 0, 0, 11182014481, 389, 1, 0, 0, 3, 1, 2, 0, 0, 0, 11182014484, 389, 1, 0, 0, 46, 1, 54, 1, 8388608, 751, 2, 3, 1, 8388608, 751, 3, 4, 1, 8388608, 751, 4, 45, 751, 5, 2, 751, 6, 46, 751, 7, 47, 751, 8, 48, 751, 9, 49, 751, 10, 50, 751, 11, 51, 751, 12, 52, 751, 13, 53, 751, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 1, 8, 0, 0, 0, 1, 1, 9, 0, 0, 0, 1, 1, 10, 0, 0, 0, 1, 1, 11, 0, 0, 0, 1, 1, 12, 0, 0, 0, 1, 11182014487, 389, 13, 0, 0, 46, 1, 105, 2, 103, 3, 104, 4, 106, 5, 86, 6, 95, 7, 96, 8, 97, 9, 98, 10, 99, 11, 100, 12, 101, 13, 102, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 1, 8, 0, 0, 0, 1, 1, 9, 0, 0, 0, 1, 1, 10, 0, 0, 0, 1, 1, 11, 0, 0, 0, 1, 1, 12, 0, 0, 0, 1, 11182014490, 389, 13, 0, 1, 18, 34, 0, 20, 36, 2, 3ULL, 0, 0, 1, 2, 1, 0, 224, 751, 4, 1, 228, 751, 0, 5, 21, 1, 5ULL, 0, 0, 6, 22, 2, 7ULL, 0, 3, 1, 2, 0, 7, 23, 1, 9ULL, 0, 2, 3, 1, 0, 8, 24, 1, 11ULL, 0, 3, 2, 1, 0, 9, 25, 1, 13ULL, 0, 2, 3, 1, 0, 10, 26, 1, 15ULL, 0, 3, 2, 1, 0, 11, 27, 1, 17ULL, 0, 2, 3, 1, 0, 12, 28, 1, 19ULL, 0, 3, 2, 1, 0, 13, 29, 1, 21ULL, 0, 2, 3, 1, 0, 14, 30, 1, 23ULL, 0, 3, 2, 1, 0, 15, 31, 1, 25ULL, 0, 2, 3, 1, 0, 16, 32, 1, 27ULL, 0, 3, 2, 1, 0, 17, 33, 1, 29ULL, 0, 2, 3, 1, 0, 18, 34, 1, 31ULL, 0, 3, 2, 1, 0, 19, 35, 1, 33ULL, 0, 2, 3, 1, 0, 20, 36, 1, 35ULL, 0, 0, 2, 1, 1, 0, 224, 751, 0, 229, 751, 230, 751, 2, 0, 231, 751, 0, 1, 2, 1, 4, 0, 0, 1ULL, 223, 751, 3, 1, 0, 0, 137, 751, 5, 1, 136, 751, 6, 2, 0, 1, 0, 135, 751, 7, 1, 2, 0, 0, 134, 751, 8, 2, 1, 0, 0, 133, 751, 9, 1, 2, 0, 0, 132, 751, 10, 2, 1, 0, 0, 131, 751, 11, 1, 2, 0, 0, 130, 751, 12, 2, 1, 0, 0, 129, 751, 13, 1, 2, 0, 0, 128, 751, 14, 2, 1, 0, 0, 127, 751, 15, 1, 2, 0, 0, 126, 751, 16, 2, 1, 0, 0, 125, 751, 17, 1, 2, 0, 0, 124, 751, 18, 2, 1, 0, 0, 123, 751, 19, 1, 2, 0, 0, 122, 751, 20, 2, 1, 0, 0, 153, 751, 21, 1, 2, 0, 0, 152, 751, 22, 2, 1, 0, 0, 151, 751, 23, 1, 2, 0, 0, 150, 751, 24, 2, 1, 0, 0, 149, 751, 25, 1, 2, 0, 0, 148, 751, 26, 2, 1, 0, 0, 147, 751, 27, 1, 2, 0, 0, 146, 751, 28, 2, 1, 0, 0, 145, 751, 29, 1, 2, 0, 0, 144, 751, 30, 2, 1, 0, 0, 143, 751, 31, 1, 2, 0, 0, 142, 751, 32, 2, 1, 0, 0, 141, 751, 33, 1, 2, 0, 0, 140, 751, 34, 2, 1, 0, 0, 139, 751, 35, 1, 2, 0, 0, 138, 751, 36, 2, 1, 0, 0, 3, 0, 7709131022, 408, 2, 0, 0, 110, 1, 4, 0, 0, 0, 11182014493, 389, 1, 0, 0, 58, 751, 5, 1, 59, 751, 6, 2, 0, 1, 0, 60, 751, 7, 1, 2, 0, 0, 61, 751, 8, 2, 1, 0, 0, 62, 751, 9, 1, 2, 0, 0, 63, 751, 10, 2, 1, 0, 0, 64, 751, 11, 1, 2, 0, 0, 65, 751, 12, 2, 1, 0, 0, 66, 751, 13, 1, 2, 0, 0, 67, 751, 14, 2, 1, 0, 0, 68, 751, 15, 1, 2, 0, 0, 69, 751, 16, 2, 1, 0, 0, 70, 751, 17, 1, 2, 0, 0, 71, 751, 18, 2, 1, 0, 0, 72, 751, 19, 1, 2, 0, 0, 73, 751, 20, 2, 1, 0, 0, 74, 751, 21, 1, 2, 0, 0, 75, 751, 22, 2, 1, 0, 0, 76, 751, 23, 1, 2, 0, 0, 77, 751, 24, 2, 1, 0, 0, 78, 751, 25, 1, 2, 0, 0, 79, 751, 26, 2, 1, 0, 0, 80, 751, 27, 1, 2, 0, 0, 81, 751, 28, 2, 1, 0, 0, 82, 751, 29, 1, 2, 0, 0, 83, 751, 30, 2, 1, 0, 0, 84, 751, 31, 1, 2, 0, 0, 85, 751, 32, 2, 1, 0, 0, 86, 751, 33, 1, 2, 0, 0, 87, 751, 34, 2, 1, 0, 0, 88, 751, 35, 1, 2, 0, 0, 89, 751, 36, 2, 1, 0, 11182014464, 389, 2, 0, 4, 11182014465, 389, 0, 0, 90, 751, 5, 1, 91, 751, 6, 2, 0, 1, 0, 92, 751, 7, 1, 2, 0, 0, 93, 751, 8, 2, 1, 0, 0, 94, 751, 9, 1, 2, 0, 0, 95, 751, 10, 2, 1, 0, 0, 96, 751, 11, 1, 2, 0, 0, 97, 751, 12, 2, 1, 0, 0, 98, 751, 13, 1, 2, 0, 0, 99, 751, 14, 2, 1, 0, 0, 100, 751, 15, 1, 2, 0, 0, 101, 751, 16, 2, 1, 0, 0, 102, 751, 17, 1, 2, 0, 0, 103, 751, 18, 2, 1, 0, 0, 104, 751, 19, 1, 2, 0, 0, 105, 751, 20, 2, 1, 0, 0, 106, 751, 21, 1, 2, 0, 0, 107, 751, 22, 2, 1, 0, 0, 108, 751, 23, 1, 2, 0, 0, 109, 751, 24, 2, 1, 0, 0, 110, 751, 25, 1, 2, 0, 0, 111, 751, 26, 2, 1, 0, 0, 112, 751, 27, 1, 2, 0, 0, 113, 751, 28, 2, 1, 0, 0, 114, 751, 29, 1, 2, 0, 0, 115, 751, 30, 2, 1, 0, 0, 116, 751, 31, 1, 2, 0, 0, 117, 751, 32, 2, 1, 0, 0, 118, 751, 33, 1, 2, 0, 0, 119, 751, 34, 2, 1, 0, 0, 120, 751, 35, 1, 2, 0, 0, 121, 751, 36, 2, 1, 0, 11182014466, 389, 2, 0, 4, 11182014467, 389, 0, 0, 154, 751, 5, 1, 155, 751, 6, 2, 0, 1, 0, 156, 751, 7, 1, 2, 0, 0, 157, 751, 8, 2, 1, 0, 0, 158, 751, 9, 1, 2, 0, 0, 159, 751, 10, 2, 1, 0, 0, 160, 751, 11, 1, 2, 0, 0, 161, 751, 12, 2, 1, 0, 0, 162, 751, 13, 1, 2, 0, 0, 163, 751, 14, 2, 1, 0, 0, 164, 751, 15, 1, 2, 0, 0, 165, 751, 16, 2, 1, 0, 0, 166, 751, 17, 1, 2, 0, 0, 167, 751, 18, 2, 1, 0, 0, 168, 751, 19, 1, 2, 0, 0, 169, 751, 20, 2, 1, 0, 0, 170, 751, 21, 1, 2, 0, 0, 171, 751, 22, 2, 1, 0, 0, 172, 751, 23, 1, 2, 0, 0, 173, 751, 24, 2, 1, 0, 0, 174, 751, 25, 1, 2, 0, 0, 175, 751, 26, 2, 1, 0, 0, 176, 751, 27, 1, 2, 0, 0, 177, 751, 28, 2, 1, 0, 0, 178, 751, 29, 1, 2, 0, 0, 179, 751, 30, 2, 1, 0, 0, 180, 751, 31, 1, 2, 0, 0, 181, 751, 32, 2, 1, 0, 0, 182, 751, 33, 1, 2, 0, 0, 183, 751, 34, 2, 1, 0, 0, 184, 751, 35, 1, 2, 0, 0, 186, 751, 36, 2, 1, 0, 11182014468, 389, 2, 0, 4, 11182014469, 389, 0, 0, 34, 35, 3, 0, 36, 0, 187, 751, 5, 1, 188, 751, 6, 2, 0, 1, 0, 189, 751, 7, 1, 2, 0, 0, 190, 751, 8, 2, 1, 0, 0, 191, 751, 9, 1, 2, 0, 0, 192, 751, 10, 2, 1, 0, 0, 193, 751, 11, 1, 2, 0, 0, 194, 751, 12, 2, 1, 0, 0, 195, 751, 13, 1, 2, 0, 0, 196, 751, 14, 2, 1, 0, 0, 197, 751, 15, 1, 2, 0, 0, 198, 751, 16, 2, 1, 0, 0, 199, 751, 17, 1, 2, 0, 0, 200, 751, 18, 2, 1, 0, 0, 202, 751, 19, 1, 2, 0, 0, 203, 751, 20, 2, 1, 0, 0, 204, 751, 21, 1, 2, 0, 0, 205, 751, 22, 2, 1, 0, 0, 206, 751, 23, 1, 2, 0, 0, 207, 751, 24, 2, 1, 0, 0, 208, 751, 25, 1, 2, 0, 0, 209, 751, 26, 2, 1, 0, 0, 210, 751, 27, 1, 2, 0, 0, 211, 751, 28, 2, 1, 0, 0, 212, 751, 29, 1, 2, 0, 0, 213, 751, 30, 2, 1, 0, 0, 214, 751, 31, 1, 2, 0, 0, 215, 751, 32, 2, 1, 0, 0, 216, 751, 33, 1, 2, 0, 0, 185, 751, 34, 2, 1, 0, 0, 201, 751, 35, 1, 2, 0, 0, 217, 751, 36, 2, 1, 0, 0, 3, 0, 11182014496, 389, 2, 0, 0, 108, 1, 107, 0, 0, 0, 11182014499, 389, 1, 0, 0, 232, 751, 11182014470, 389, 0, 0, 109, 11182014471, 389, 0, 0, 233, 751, 11182014472, 389, 0, 0, 109, 11182014473, 389, 0, 0, 234, 751, 11182014474, 389, 0, 0, 109, 11182014475, 389, 0, 1, 0ULL, 2, 235, 751, 3, 260, 751, 4, 262, 751, 5, 266, 751, 6, 264, 751, 0, 8ULL, 273, 751, 7, 0, 267, 751, 0, 1, 0, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 11182014502, 389, 7, 0, 0, 113, 1, 111, 2, 3, 3, 2, 4, 112, 5, 114, 6, 115, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 11182014505, 389, 6, 0, 3, 276, 1, 8388608, 751, 275, 1, 8388608, 751, 4, 235, 751, 5, 261, 751, 6, 263, 751, 7, 267, 751, 8, 265, 751, 0, 2ULL, 274, 751, 1, 0, 268, 751, 0, 4ULL, 272, 751, 2, 1, 0, 0, 8ULL, 273, 751, 1, 2, 0, 0, 3, 0, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 1, 8, 0, 0, 0, 1, 11182014508, 389, 1, 0, 0, 113, 1, 111, 2, 3, 3, 2, 4, 112, 5, 114, 6, 115, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 11182014511, 389, 6, 0, 0, 304, 751, 11182014476, 389, 0, 0, 2, 11182014477, 389, 0, 0, 306, 751, 307, 751, 2, 1ULL, 0, 0, 2, 308, 751, 7709130799, 408, 0, 148, 0, 2, 308, 751, 1, 1ULL, 0, 0, 1, 304, 751, 1, 0, 2, 0, 128ULL, 7709130799, 408, 7709130800, 408, 1, 0, 0, 7709130800, 408, 140, 1, 0, 296, 751, 17, 1, 0, 7709130800, 408, 141, 1, 0, 297, 751, 18, 1, 0, 7709130800, 408, 142, 1, 0, 298, 751, 19, 1, 0, 7709130800, 408, 143, 1, 0, 299, 751, 20, 1, 0, 7709130800, 408, 144, 1, 0, 300, 751, 21, 1, 0, 7709130800, 408, 145, 1, 0, 301, 751, 22, 1, 0, 7709130800, 408, 146, 1, 0, 302, 751, 23, 1, 0, 7709130800, 408, 147, 1, 0, 303, 751, 24, 1, 25, 315, 751, 26, 316, 751, 27, 317, 751, 28, 318, 751, 29, 311, 751, 30, 312, 751, 31, 313, 751, 32, 314, 751, 0, 277, 751, 1, 278, 751, 2, 279, 751, 3, 280, 751, 4, 281, 751, 5, 282, 751, 6, 283, 751, 7, 284, 751, 8, 285, 751, 9, 286, 751, 10, 287, 751, 11, 288, 751, 12, 289, 751, 13, 290, 751, 14, 291, 751, 15, 292, 751, 16, 125, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 1, 8, 0, 0, 0, 1, 1, 9, 0, 0, 0, 1, 1, 10, 0, 0, 0, 1, 1, 11, 0, 0, 0, 1, 1, 12, 0, 0, 0, 1, 1, 13, 0, 0, 0, 1, 1, 14, 0, 0, 0, 1, 1, 15, 0, 0, 1, 1, 1, 16, 0, 11182014514, 389, 1, 1, 0, 148, 0, 17, 0, 1, 18, 0, 0, 0, 1, 1, 19, 0, 0, 0, 1, 1, 20, 0, 0, 0, 1, 1, 21, 0, 0, 0, 1, 1, 22, 0, 0, 0, 1, 1, 23, 0, 0, 0, 1, 1, 24, 0, 0, 0, 1, 1, 25, 0, 0, 0, 1, 1, 26, 0, 0, 0, 1, 1, 27, 0, 0, 0, 1, 1, 28, 0, 0, 0, 1, 1, 29, 0, 0, 0, 1, 1, 30, 0, 0, 0, 1, 1, 31, 0, 0, 0, 1, 1, 32, 0, 0, 1, 11182014514, 389, 1, 0, 0, 7709131025, 408, 11182014514, 389, 1, 0, 321, 751, 1, 324, 751, 2, 325, 751, 3, 326, 751, 4, 327, 751, 5, 328, 751, 6, 329, 751, 7, 330, 751, 8, 331, 751, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 11182014517, 389, 8, 0, 0, 46, 1, 37, 2, 38, 3, 39, 4, 40, 5, 41, 6, 42, 7, 43, 8, 44, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 11182014520, 389, 8, 0, 0, 6, 10, 1, 0, 14, 0, 1, 18, 1, 0, 22, 0, 1, 26, 1, 0, 30, 3, 1, 34, 0, 7, 11, 1, 0, 15, 0, 1, 19, 1, 0, 23, 0, 1, 27, 1, 0, 31, 2, 1, 35, 0, 8, 12, 1, 0, 16, 0, 1, 20, 1, 0, 24, 0, 1, 28, 1, 0, 32, 4, 1, 36, 0, 2ULL, 2, 1, 3, 0, 0, 3ULL, 4, 2, 1, 0, 0, 4ULL, 349, 751, 1, 2, 0, 0, 8ULL, 350, 751, 2, 1, 0, 0, 16ULL, 351, 751, 1, 2, 0, 0, 32ULL, 352, 751, 2, 1, 0, 3, 2, 5, 344, 751, 4, 6, 346, 751, 0, 5, 9, 1, 0, 13, 0, 1, 17, 1, 0, 21, 0, 1, 25, 1, 0, 29, 2, 1, 33, 0, 348, 751, 353, 751, 1, 2, 0, 2, 1, 353, 751, 0, 1ULL, 4, 1, 0, 353, 1, 8388608, 751, 0, 3, 0, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 11182014523, 389, 1, 0, 0, 151, 1, 152, 2, 153, 3, 154, 4, 155, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 11182014526, 389, 4, 0, 0, 1ULL, 419, 751, 1, 2ULL, 420, 751, 2, 0, 1, 0, 4ULL, 422, 751, 1, 2, 0, 0, 8ULL, 421, 751, 2, 1, 0, 0, 16ULL, 423, 751, 1, 2, 0, 0, 32ULL, 424, 751, 2, 1, 0, 0, 64ULL, 425, 751, 1, 2, 0, 0, 128ULL, 426, 751, 2, 1, 0, 0, 256ULL, 428, 751, 1, 2, 0, 0, 512ULL, 408, 751, 2, 1, 0, 0, 1024ULL, 409, 751, 1, 2, 0, 0, 2048ULL, 418, 751, 2, 1, 0, 0, 4096ULL, 411, 751, 1, 2, 0, 0, 8192ULL, 410, 751, 2, 1, 0, 0, 16384ULL, 415, 751, 1, 2, 0, 0, 32768ULL, 417, 751, 2, 1, 0, 0, 65536ULL, 416, 751, 1, 2, 0, 0, 131072ULL, 414, 751, 2, 1, 0, 0, 262144ULL, 413, 751, 1, 2, 0, 0, 524288ULL, 412, 751, 2, 1, 0, 0, 1048576ULL, 427, 751, 1, 2, 0, 0, 2097152ULL, 405, 751, 2, 1, 0, 0, 4194304ULL, 398, 751, 1, 2, 0, 0, 8388608ULL, 397, 751, 2, 1, 0, 0, 16777216ULL, 402, 751, 1, 2, 0, 0, 33554432ULL, 403, 751, 2, 1, 0, 0, 67108864ULL, 401, 751, 1, 2, 0, 0, 134217728ULL, 399, 751, 2, 1, 0, 0, 268435456ULL, 400, 751, 1, 2, 0, 0, 536870912ULL, 406, 751, 2, 1, 0, 0, 1073741824ULL, 407, 751, 1, 2, 0, 0, 1, 1, 429, 751, 2, 430, 751, 3, 396, 751, 4, 404, 751, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 11182014529, 389, 4, 0, 0, 156, 1, 157, 2, 158, 3, 159, 4, 160, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 11182014532, 389, 4, 0, 0, 164, 1, 433, 751, 2, 437, 751, 3, 441, 751, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 11182014535, 389, 3, 0, 0, 165, 1, 166, 2, 167, 3, 168, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 11182014538, 389, 3, 0, 0, 164, 1, 434, 751, 2, 438, 751, 3, 442, 751, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 11182014541, 389, 3, 0, 0, 165, 1, 166, 2, 167, 3, 168, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 11182014544, 389, 3, 0, 0, 164, 1, 435, 751, 2, 439, 751, 3, 443, 751, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 11182014547, 389, 3, 0, 0, 165, 1, 166, 2, 167, 3, 168, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 11182014550, 389, 3, 0, 0, 164, 1, 436, 751, 2, 440, 751, 3, 444, 751, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 11182014553, 389, 3, 0, 0, 165, 1, 166, 2, 167, 3, 168, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 11182014556, 389, 3, 0, 0, 462, 751, 463, 751, 9, 1ULL, 0, 0, 9, 464, 751, 7709130811, 408, 0, 189, 0, 9, 464, 751, 1, 1ULL, 0, 0, 1, 459, 751, 1, 0, 9, 0, 128ULL, 7709130811, 408, 7709130813, 408, 1, 0, 3, 7709130813, 408, 4, 188, 5, 460, 751, 0, 449, 751, 1, 171, 2, 450, 751, 0, 0, 0, 1, 1, 0, 0, 0, 1, 11182014562, 389, 2, 0, 0, 191, 0, 3, 0, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 1, 11182014562, 389, 1, 0, 0, 11182014559, 389, 11182014562, 389, 1, 10, 468, 751, 11, 469, 751, 12, 470, 751, 13, 471, 751, 14, 472, 751, 15, 473, 751, 16, 474, 751, 17, 475, 751, 18, 192, 0, 451, 751, 1, 452, 751, 2, 453, 751, 3, 454, 751, 4, 455, 751, 5, 456, 751, 6, 457, 751, 7, 458, 751, 8, 172, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 11182014568, 389, 8, 0, 0, 9, 464, 751, 7709130812, 408, 0, 190, 0, 7709130812, 408, 0, 10, 0, 1, 11, 0, 0, 0, 1, 1, 12, 0, 0, 0, 1, 1, 13, 0, 0, 0, 1, 1, 14, 0, 0, 0, 1, 1, 15, 0, 0, 0, 1, 1, 16, 0, 0, 0, 1, 1, 17, 0, 0, 0, 1, 1, 18, 0, 0, 1, 11182014568, 389, 1, 0, 0, 11182014565, 389, 11182014568, 389, 1, 0, 477, 751, 1, 480, 751, 2, 481, 751, 3, 482, 751, 4, 483, 751, 5, 484, 751, 6, 485, 751, 7, 486, 751, 8, 487, 751, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 11182014571, 389, 8, 0, 0, 46, 1, 37, 2, 38, 3, 39, 4, 40, 5, 41, 6, 42, 7, 43, 8, 44, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 11182014574, 389, 8, 0, 0, 496, 1, 8388608, 751, 496, 751, 1, 497, 1, 8388608, 751, 497, 751, 2, 0, 1, 0, 508, 751, 2, 1, 497, 1, 8388608, 751, 497, 751, 2, 0, 1, 0, 45, 1ULL, 1, 0, 11182014478, 389, 1, 0, 1ULL, 1, 1, 0, 0, 2, 11182014478, 389, 2, 0, 1, 7709130844, 408, 2, 11182014478, 389, 0, 204, 1, 509, 751, 2, 510, 751, 3, 511, 751, 4, 512, 751, 5, 511, 1, 8388608, 751, 205, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 11182014577, 389, 5, 0, 0, 194, 1, 195, 2, 196, 3, 197, 4, 198, 5, 199, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 11182014580, 389, 5, 0, 0, 530, 751, 531, 751, 7709130824, 408, 1ULL, 0, 0, 7709130824, 408, 532, 751, 1, 1ULL, 0, 0, 1, 527, 751, 1, 128ULL, 7709130824, 408, 2, 0, 1, 0, 533, 751, 527, 751, 7709130826, 408, 2, 0, 3, 7709130826, 408, 4, 226, 5, 528, 751, 0, 517, 751, 1, 208, 2, 518, 751, 0, 0, 0, 1, 1, 0, 0, 0, 1, 11182014586, 389, 2, 0, 0, 229, 0, 3, 0, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 1, 11182014586, 389, 1, 0, 0, 11182014583, 389, 11182014586, 389, 1, 9, 538, 751, 10, 539, 751, 11, 540, 751, 12, 541, 751, 13, 542, 751, 14, 543, 751, 15, 544, 751, 16, 545, 751, 17, 232, 0, 519, 751, 1, 520, 751, 2, 521, 751, 3, 522, 751, 4, 523, 751, 5, 524, 751, 6, 525, 751, 7, 526, 751, 8, 209, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 11182014589, 389, 8, 0, 0, 533, 751, 228, 1, 0, 0, 9, 0, 1, 10, 0, 0, 0, 1, 1, 11, 0, 0, 0, 1, 1, 12, 0, 0, 0, 1, 1, 13, 0, 0, 0, 1, 1, 14, 0, 0, 0, 1, 1, 15, 0, 0, 0, 1, 1, 16, 0, 0, 0, 1, 1, 17, 0, 0, 1, 11182014589, 389, 1, 1, 0, 7709131028, 408, 11182014589, 389, 1, 0, 547, 751, 1, 550, 751, 2, 551, 751, 3, 552, 751, 4, 553, 751, 5, 554, 751, 6, 555, 751, 7, 556, 751, 8, 557, 751, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 11182014592, 389, 8, 0, 0, 46, 1, 37, 2, 38, 3, 39, 4, 40, 5, 41, 6, 42, 7, 43, 8, 44, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 11182014595, 389, 8, 0, 0, 727, 751, 11182014479, 389, 0, 0, 45, 11182014480, 389, 0, 3, 629, 751, 4, 628, 751, 5, 627, 751, 6, 626, 751, 7, 625, 751, 8, 624, 751, 9, 623, 751, 10, 622, 751, 11, 638, 751, 12, 639, 751, 13, 640, 751, 14, 641, 751, 15, 642, 751, 16, 643, 751, 17, 644, 751, 18, 645, 751, 19, 646, 751, 20, 647, 751, 21, 648, 751, 22, 649, 751, 23, 650, 751, 24, 653, 751, 25, 651, 751, 26, 652, 751, 27, 654, 751, 28, 719, 751, 29, 720, 751, 30, 722, 751, 31, 723, 751, 32, 721, 751, 33, 725, 751, 34, 724, 751, 0, 1ULL, 684, 751, 1, 2ULL, 685, 751, 2, 0, 1, 0, 4ULL, 686, 751, 1, 2, 0, 0, 8ULL, 707, 751, 2, 1, 0, 0, 16ULL, 705, 751, 1, 2, 0, 0, 32ULL, 693, 751, 2, 1, 0, 0, 64ULL, 696, 751, 1, 2, 0, 0, 128ULL, 695, 751, 2, 1, 0, 0, 256ULL, 697, 751, 1, 2, 0, 0, 512ULL, 700, 751, 2, 1, 0, 0, 1024ULL, 699, 751, 1, 2, 0, 0, 2048ULL, 676, 751, 2, 1, 0, 0, 4096ULL, 677, 751, 1, 2, 0, 0, 8192ULL, 675, 751, 2, 1, 0, 0, 16384ULL, 674, 751, 1, 2, 0, 0, 32768ULL, 668, 751, 2, 1, 0, 0, 65536ULL, 670, 751, 1, 2, 0, 0, 131072ULL, 669, 751, 2, 1, 0, 0, 262144ULL, 690, 751, 1, 2, 0, 0, 524288ULL, 691, 751, 2, 1, 0, 0, 1048576ULL, 692, 751, 1, 2, 0, 0, 2097152ULL, 680, 751, 2, 1, 0, 0, 4194304ULL, 681, 751, 1, 2, 0, 0, 8388608ULL, 708, 751, 2, 1, 0, 0, 16777216ULL, 655, 751, 1, 2, 0, 0, 33554432ULL, 656, 751, 2, 1, 0, 0, 67108864ULL, 657, 751, 1, 2, 0, 0, 134217728ULL, 661, 751, 2, 1, 0, 0, 268435456ULL, 658, 751, 1, 2, 0, 0, 536870912ULL, 659, 751, 2, 1, 0, 0, 1073741824ULL, 664, 751, 1, 2, 0, 0, 2147483648ULL, 666, 751, 2, 1, 0, 0, 4294967296ULL, 663, 751, 1, 2, 0, 0, 8589934592ULL, 667, 751, 2, 1, 0, 0, 17179869184ULL, 665, 751, 1, 2, 0, 0, 34359738368ULL, 662, 751, 2, 1, 0, 0, 68719476736ULL, 660, 751, 1, 2, 0, 0, 137438953472ULL, 683, 751, 2, 1, 0, 0, 274877906944ULL, 682, 751, 1, 2, 0, 0, 549755813888ULL, 678, 751, 2, 1, 0, 0, 1099511627776ULL, 741, 751, 1, 2, 0, 0, 2199023255552ULL, 671, 751, 2, 1, 0, 0, 4398046511104ULL, 709, 751, 1, 2, 0, 0, 8796093022208ULL, 710, 751, 2, 1, 0, 0, 17592186044416ULL, 694, 751, 1, 2, 0, 0, 35184372088832ULL, 698, 751, 2, 1, 0, 0, 70368744177664ULL, 742, 751, 1, 2, 0, 0, 140737488355328ULL, 687, 751, 2, 1, 0, 0, 281474976710656ULL, 688, 751, 1, 2, 0, 0, 562949953421312ULL, 689, 751, 2, 1, 0, 0, 1125899906842624ULL, 701, 751, 1, 2, 0, 0, 2251799813685248ULL, 704, 751, 2, 1, 0, 0, 4503599627370496ULL, 703, 751, 1, 2, 0, 0, 9007199254740992ULL, 702, 751, 2, 1, 0, 0, 2, 1, 672, 751, 2, 673, 751, 35, 706, 751, 36, 739, 751, 37, 740, 751, 38, 618, 751, 0, 3, 0, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 1, 8, 0, 0, 0, 1, 1, 9, 0, 0, 0, 1, 1, 10, 0, 0, 0, 1, 1, 11, 0, 0, 0, 1, 1, 12, 0, 0, 0, 1, 1, 13, 0, 0, 0, 1, 1, 14, 0, 0, 0, 1, 1, 15, 0, 0, 0, 1, 1, 16, 0, 0, 0, 1, 1, 17, 0, 0, 0, 1, 1, 18, 0, 0, 0, 1, 1, 19, 0, 0, 0, 1, 1, 20, 0, 0, 0, 1, 1, 21, 0, 0, 0, 1, 1, 22, 0, 0, 0, 1, 1, 23, 0, 0, 0, 1, 1, 24, 0, 0, 0, 1, 1, 25, 0, 0, 0, 1, 1, 26, 0, 0, 0, 1, 1, 27, 0, 0, 0, 1, 1, 28, 0, 0, 0, 1, 1, 29, 0, 0, 0, 1, 1, 30, 0, 0, 0, 1, 1, 31, 0, 0, 0, 1, 1, 32, 0, 0, 0, 1, 1, 33, 0, 0, 0, 1, 1, 34, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 35, 0, 0, 0, 1, 1, 36, 0, 0, 0, 1, 1, 37, 0, 0, 0, 1, 11182014598, 389, 38, 0, 0, 47, 1, 48, 2, 49, 3, 50, 4, 51, 5, 52, 6, 53, 7, 54, 8, 56, 9, 57, 10, 58, 11, 59, 12, 60, 13, 61, 14, 62, 15, 63, 16, 64, 17, 65, 18, 66, 19, 67, 20, 68, 21, 69, 22, 70, 23, 71, 24, 72, 25, 73, 26, 74, 27, 75, 28, 79, 29, 76, 30, 77, 31, 78, 32, 85, 33, 55, 34, 80, 35, 81, 36, 82, 37, 83, 38, 84, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 1, 8, 0, 0, 0, 1, 1, 9, 0, 0, 0, 1, 1, 10, 0, 0, 0, 1, 1, 11, 0, 0, 0, 1, 1, 12, 0, 0, 0, 1, 1, 13, 0, 0, 0, 1, 1, 14, 0, 0, 0, 1, 1, 15, 0, 0, 0, 1, 1, 16, 0, 0, 0, 1, 1, 17, 0, 0, 0, 1, 1, 18, 0, 0, 0, 1, 1, 19, 0, 0, 0, 1, 1, 20, 0, 0, 0, 1, 1, 21, 0, 0, 0, 1, 1, 22, 0, 0, 0, 1, 1, 23, 0, 0, 0, 1, 1, 24, 0, 0, 0, 1, 1, 25, 0, 0, 0, 1, 1, 26, 0, 0, 0, 1, 1, 27, 0, 0, 0, 1, 1, 28, 0, 0, 0, 1, 1, 29, 0, 0, 0, 1, 1, 30, 0, 0, 0, 1, 1, 31, 0, 0, 0, 1, 1, 32, 0, 0, 0, 1, 1, 33, 0, 0, 0, 1, 1, 34, 0, 0, 0, 1, 1, 35, 0, 0, 0, 1, 1, 36, 0, 0, 0, 1, 1, 37, 0, 0, 0, 1, 11182014601, 389, 38, 0, 0, 676, 751, 605, 751, 1, 677, 751, 619, 751, 2, 0, 1, 3, 2, 672, 751, 0, 678, 751, 614, 751, 1, 262144ULL, 0, 0, 65536ULL, 674, 751, 2, 1, 0, 0, 674, 751, 615, 751, 1, 2, 0, 0, 131072ULL, 675, 751, 2, 1, 0, 24, 2, 3, 13, 24, 14, 620, 751, 0, 597, 751, 693, 751, 15, 0, 694, 751, 0, 573, 751, 638, 751, 1, 581, 751, 639, 751, 2, 0, 1, 0, 589, 751, 640, 751, 1, 2, 0, 0, 582, 751, 641, 751, 2, 1, 0, 0, 597, 751, 642, 751, 1, 2, 0, 0, 605, 751, 643, 751, 2, 1, 0, 0, 645, 751, 646, 751, 1, 637, 751, 0, 0, 2, 1, 1, 613, 751, 644, 751, 2, 0, 1, 0, 614, 751, 647, 751, 1, 2, 0, 0, 615, 751, 648, 751, 2, 1, 0, 0, 616, 751, 649, 751, 1, 2, 0, 0, 617, 751, 650, 751, 2, 1, 0, 0, 651, 751, 45, 1, 2, 0, 0, 619, 751, 652, 751, 2, 1, 0, 0, 620, 751, 653, 751, 1, 2, 0, 0, 712, 751, 719, 751, 2, 1, 0, 0, 713, 751, 720, 751, 1, 2, 0, 0, 714, 751, 721, 751, 2, 1, 0, 0, 715, 751, 722, 751, 1, 2, 0, 0, 716, 751, 723, 751, 2, 1, 0, 0, 717, 751, 724, 751, 1, 2, 0, 0, 718, 751, 725, 751, 2, 1, 0, 0, 621, 751, 654, 751, 1, 2, 0, 25, 1, 629, 751, 16, 25, 0, 572, 751, 638, 751, 1, 580, 751, 639, 751, 2, 0, 1, 0, 588, 751, 640, 751, 1, 2, 0, 0, 589, 751, 641, 751, 2, 1, 0, 0, 596, 751, 642, 751, 1, 2, 0, 0, 604, 751, 643, 751, 2, 1, 0, 0, 612, 751, 644, 751, 1, 2, 0, 0, 636, 751, 645, 751, 2, 1, 0, 26, 2, 628, 751, 17, 26, 0, 571, 751, 638, 751, 1, 579, 751, 639, 751, 2, 0, 1, 0, 587, 751, 640, 751, 1, 2, 0, 0, 588, 751, 641, 751, 2, 1, 0, 0, 595, 751, 642, 751, 1, 2, 0, 0, 603, 751, 643, 751, 2, 1, 0, 0, 611, 751, 644, 751, 1, 2, 0, 0, 635, 751, 645, 751, 2, 1, 0, 27, 2, 627, 751, 18, 27, 0, 570, 751, 638, 751, 1, 578, 751, 639, 751, 2, 0, 1, 0, 586, 751, 640, 751, 1, 2, 0, 0, 587, 751, 641, 751, 2, 1, 0, 0, 594, 751, 642, 751, 1, 2, 0, 0, 602, 751, 643, 751, 2, 1, 0, 0, 610, 751, 644, 751, 1, 2, 0, 0, 634, 751, 645, 751, 2, 1, 0, 28, 2, 626, 751, 19, 28, 0, 569, 751, 638, 751, 1, 577, 751, 639, 751, 2, 0, 1, 0, 585, 751, 640, 751, 1, 2, 0, 0, 586, 751, 641, 751, 2, 1, 0, 0, 593, 751, 642, 751, 1, 2, 0, 0, 601, 751, 643, 751, 2, 1, 0, 0, 609, 751, 644, 751, 1, 2, 0, 0, 633, 751, 645, 751, 2, 1, 0, 29, 2, 625, 751, 20, 29, 0, 568, 751, 638, 751, 1, 576, 751, 639, 751, 2, 0, 1, 0, 584, 751, 640, 751, 1, 2, 0, 0, 585, 751, 641, 751, 2, 1, 0, 0, 592, 751, 642, 751, 1, 2, 0, 0, 600, 751, 643, 751, 2, 1, 0, 0, 608, 751, 644, 751, 1, 2, 0, 0, 632, 751, 645, 751, 2, 1, 0, 30, 2, 624, 751, 21, 30, 0, 567, 751, 638, 751, 1, 575, 751, 639, 751, 2, 0, 1, 0, 583, 751, 640, 751, 1, 2, 0, 0, 584, 751, 641, 751, 2, 1, 0, 0, 591, 751, 642, 751, 1, 2, 0, 0, 599, 751, 643, 751, 2, 1, 0, 0, 607, 751, 644, 751, 1, 2, 0, 0, 631, 751, 645, 751, 2, 1, 0, 31, 2, 623, 751, 22, 31, 0, 566, 751, 638, 751, 1, 574, 751, 639, 751, 2, 0, 1, 0, 582, 751, 640, 751, 1, 2, 0, 0, 583, 751, 641, 751, 2, 1, 0, 0, 590, 751, 642, 751, 1, 2, 0, 0, 598, 751, 643, 751, 2, 1, 0, 0, 606, 751, 644, 751, 1, 2, 0, 0, 630, 751, 645, 751, 2, 1, 0, 32, 2, 622, 751, 23, 32, 2, 461, 751, 0, 467, 751, 462, 751, 1, 0, 478, 751, 3, 1, 1ULL, 4, 478, 751, 0, 480, 751, 7709130813, 408, 7709130814, 408, 0, 488, 751, 5, 7709130814, 408, 0, 481, 751, 7709130813, 408, 7709130815, 408, 0, 489, 751, 6, 7709130815, 408, 0, 482, 751, 7709130813, 408, 7709130816, 408, 0, 490, 751, 7, 7709130816, 408, 0, 483, 751, 7709130813, 408, 7709130817, 408, 0, 491, 751, 8, 7709130817, 408, 0, 484, 751, 7709130813, 408, 7709130818, 408, 0, 492, 751, 9, 7709130818, 408, 0, 485, 751, 7709130813, 408, 7709130819, 408, 0, 493, 751, 10, 7709130819, 408, 0, 486, 751, 7709130813, 408, 7709130820, 408, 0, 494, 751, 11, 7709130820, 408, 0, 487, 751, 7709130813, 408, 7709130821, 408, 0, 495, 751, 12, 7709130821, 408, 0, 477, 751, 479, 751, 1, 1ULL, 0, 7709130841, 408, 1, 191, 0, 2, 0, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 1, 8, 0, 0, 0, 1, 1, 9, 0, 0, 0, 1, 1, 10, 0, 0, 0, 1, 1, 11, 0, 0, 0, 1, 1, 12, 0, 0, 1, 1, 1, 7709130841, 408, 0, 11182014604, 389, 1, 1, 0, 693, 751, 694, 751, 0, 13, 0, 1, 14, 0, 0, 0, 1, 1, 15, 0, 0, 0, 1, 1, 16, 0, 0, 0, 1, 1, 17, 0, 0, 0, 1, 1, 18, 0, 0, 0, 1, 1, 19, 0, 0, 0, 1, 1, 20, 0, 0, 0, 1, 1, 21, 0, 0, 0, 1, 1, 22, 0, 0, 0, 1, 1, 23, 0, 0, 1, 11182014604, 389, 1, 0, 0, 7709131031, 408, 11182014604, 389, 1, 12, 24, 13, 620, 751, 0, 597, 751, 701, 751, 14, 0, 702, 751, 15, 25, 16, 26, 17, 27, 18, 28, 19, 29, 20, 30, 21, 31, 22, 32, 2, 529, 751, 0, 537, 751, 530, 751, 1, 0, 548, 751, 3, 1, 1ULL, 1, 548, 751, 0, 550, 751, 7709130826, 408, 7709130828, 408, 0, 558, 751, 4, 7709130828, 408, 0, 551, 751, 7709130826, 408, 7709130829, 408, 0, 559, 751, 5, 7709130829, 408, 0, 552, 751, 7709130826, 408, 7709130830, 408, 0, 560, 751, 6, 7709130830, 408, 0, 553, 751, 7709130826, 408, 7709130831, 408, 0, 561, 751, 7, 7709130831, 408, 0, 554, 751, 7709130826, 408, 7709130832, 408, 0, 562, 751, 8, 7709130832, 408, 0, 555, 751, 7709130826, 408, 7709130833, 408, 0, 563, 751, 9, 7709130833, 408, 0, 556, 751, 7709130826, 408, 7709130834, 408, 0, 564, 751, 10, 7709130834, 408, 0, 557, 751, 7709130826, 408, 7709130835, 408, 0, 565, 751, 11, 7709130835, 408, 0, 547, 751, 549, 751, 7709130827, 408, 1ULL, 0, 7709130842, 408, 7709130827, 408, 229, 0, 2, 0, 1, 3, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 1, 8, 0, 0, 0, 1, 1, 9, 0, 0, 0, 1, 1, 10, 0, 0, 0, 1, 1, 11, 0, 0, 1, 1, 1, 7709130842, 408, 0, 11182014607, 389, 1, 1, 0, 701, 751, 702, 751, 0, 12, 0, 1, 13, 0, 0, 0, 1, 1, 14, 0, 0, 0, 1, 1, 15, 0, 0, 0, 1, 1, 16, 0, 0, 0, 1, 1, 17, 0, 0, 0, 1, 1, 18, 0, 0, 0, 1, 1, 19, 0, 0, 0, 1, 1, 20, 0, 0, 0, 1, 1, 21, 0, 0, 0, 1, 1, 22, 0, 0, 1, 11182014607, 389, 1, 0, 0, 7709131034, 408, 11182014607, 389, 1, 13, 24, 14, 620, 751, 0, 597, 751, 697, 751, 15, 0, 698, 751, 16, 25, 17, 26, 18, 27, 19, 28, 20, 29, 21, 30, 22, 31, 23, 32, 2, 305, 751, 0, 320, 751, 306, 751, 1, 0, 322, 751, 3, 1, 1ULL, 4, 322, 751, 0, 324, 751, 7709130800, 408, 7709130801, 408, 0, 332, 751, 5, 7709130801, 408, 0, 325, 751, 7709130800, 408, 7709130802, 408, 0, 333, 751, 6, 7709130802, 408, 0, 326, 751, 7709130800, 408, 7709130803, 408, 0, 334, 751, 7, 7709130803, 408, 0, 327, 751, 7709130800, 408, 7709130804, 408, 0, 335, 751, 8, 7709130804, 408, 0, 328, 751, 7709130800, 408, 7709130805, 408, 0, 336, 751, 9, 7709130805, 408, 0, 329, 751, 7709130800, 408, 7709130806, 408, 0, 337, 751, 10, 7709130806, 408, 0, 330, 751, 7709130800, 408, 7709130807, 408, 0, 338, 751, 11, 7709130807, 408, 0, 331, 751, 7709130800, 408, 7709130808, 408, 0, 339, 751, 12, 7709130808, 408, 0, 321, 751, 323, 751, 1, 1ULL, 0, 7709130843, 408, 1, 149, 0, 2, 0, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 5, 0, 0, 0, 1, 1, 6, 0, 0, 0, 1, 1, 7, 0, 0, 0, 1, 1, 8, 0, 0, 0, 1, 1, 9, 0, 0, 0, 1, 1, 10, 0, 0, 0, 1, 1, 11, 0, 0, 0, 1, 1, 12, 0, 0, 1, 1, 1, 7709130843, 408, 0, 11182014610, 389, 1, 1, 0, 697, 751, 698, 751, 0, 13, 0, 1, 14, 0, 0, 0, 1, 1, 15, 0, 0, 0, 1, 1, 16, 0, 0, 0, 1, 1, 17, 0, 0, 0, 1, 1, 18, 0, 0, 0, 1, 1, 19, 0, 0, 0, 1, 1, 20, 0, 0, 0, 1, 1, 21, 0, 0, 0, 1, 1, 22, 0, 0, 0, 1, 1, 23, 0, 0, 1, 11182014610, 389, 1, 0, 0, 7709131037, 408, 11182014610, 389, 1};