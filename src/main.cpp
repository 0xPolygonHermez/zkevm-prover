#include <stdio.h>
#include "zkevm_api.hpp"
#include <string.h>

int main(int argc, char **argv)
{
    /* CONFIG */

    if (argc == 2)
    {
        if ((strcmp(argv[1], "-v") == 0) || (strcmp(argv[1], "--version") == 0))
        {
            // If requested to only print the version,   then exit the program
            return 0;
        }
    }

    // Parse the name of the configuration file
    char *configFile = (char *)"config/config.json";
    if (argc == 3)
    {
        if ((strcmp(argv[1], "-c") == 0) || (strcmp(argv[1], "--config") == 0))
        {
            configFile = argv[2];
        }
    }
    zkevm_main(configFile, NULL,NULL,NULL);

}