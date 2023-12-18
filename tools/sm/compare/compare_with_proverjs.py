#
# Script: verification_script.py
#
# This script is used to compare the commited polynomials evaluations of the proverjs and the proverc.
# 
# run: python3 verification_script.py
#
# notes:
#   - the script output are saved within the folder outputs_path, in the file script_output.txt
#   - the script saves the inputs processed within the folder outputs_path, in the file inputs_done.txt, so it can be resumed
#     to start from the beginning, just delete the file inputs_done.txt
#   - the script assumes that the proverjs and proverc are already compiled
#

import os
import subprocess
import json

#inputs:

# inputs_folder_path: path to the folder with the executor inputs
inputs_folder_path = '/path/to/inputs/folder/'
# proverjs repo path
proverjs_path = '/path/to/proverjs/folder/'
# Rom file path
rom_path = '/path/to/rom/file.json'
# proverc path
proverc_path = '/path/to/proverc/folder/'
# proverc config file path
proverc_config_path = '/path/to/proverc/config.json'
# outputs folder path
output_path = '/path/to/script/output/folder/'


def process_file(file_path, file_name):

    print(f"** {file_name}:")
    print(f"   Processing proverjs:")
    
    # Change the current working directory to the proverjs folder
    os.chdir(proverjs_path)

    # Run the proverjs
    command = f"node --max-old-space-size=65536 src/main_executor {file_path} -r {rom_path} -o tmp/commit.bin > runjs.log"
    try:
        print(f"   Running: {command}")
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error proverjs processing file {file_path}: {e}")

    # Modify the config file of the proverc to use the input file
    with open(proverc_config_path, 'r') as f:
        data = json.load(f)

    # Replace inputfile
    data["inputFile"] = file_path

    # Write the config_rick_2.json file
    with open(proverc_config_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    # Change the current working directory to the proverc folder
    os.chdir(proverc_path)
    
    command2 = f"./build/zkProver -c {proverc_config_path} > runc.log"
    print(f"   Processing proverc:")
    try:
        print(f"   Running: {command2}")
        subprocess.run(command2, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error proverc processing file {file_path}: {e}")

    command3 = f"cmp -l {proverc_path}runtime/zkevm.commit {proverjs_path}/tmp/commit.bin > kcommit.txt"
    print(f"   Processing cmp:")
    print(f"   Running: {command3}")
    subprocess.run(command3, check=True, shell=True)

    cmp_file_path = proverc_path + 'kcommit.txt'
    save=False
    if(os.path.getsize(cmp_file_path) == 0):
        print("   Commitment matches!")
    else:
        print("   ERROR!!!!!! Commitment does not match!")
        save=True

    
    os.chdir(output_path)
    if(save):
        mkdir_command = f"mkdir {file_name}"
        subprocess.run(mkdir_command, check=True, shell=True)
        mv_command = f"mv {proverjs_path}tmp/commit.bin {file_name}/commitjs.bin"
        subprocess.run(mv_command, check=True, shell=True)
        mv_command = f"mv {proverc_path}runtime/zkevm.commit {file_name}/commitc.bin"
        subprocess.run(mv_command, check=True, shell=True)
        mv_command = f"mv {proverjs_path}runjs.log tmp/{file_name}/runjs.log.txt"
        subprocess.run(mv_command, check=True, shell=True)
        mv_command = f"mv {proverc_path}runc.log tmp/{file_name}/runc.log.txt"
        subprocess.run(mv_command, check=True, shell=True)

    #print result in output file
    fileoutput_path = output_path + 'script_output.txt'
    with open(fileoutput_path, 'a+') as f:
        f.write(f"** {file_name}:\n")
        f.write(f"   Processing proverjs:\n")
        f.write(f"   Running: {command}\n")
        f.write(f"   Processing proverc:\n")
        f.write(f"   Running: {command2}\n")
        f.write(f"   Processing cmp:\n")
        f.write(f"   Running: {command3}\n")
        if(os.path.getsize(cmp_file_path) == 0):
            f.write("   Commitment matches!\n")
        else:
            f.write("   ERROR!!!!!! Commitment does not match!\n")
        f.write("\n")

def process_folder():
    
    #load done list
    done_list = []
    done_path = output_path + 'inputs_done.txt'
    if os.path.exists(done_path):
        with open(done_path, 'r') as f:
            done_list = f.read().splitlines()

    # Iterate  through each file in the folder
    for file_name in os.listdir(inputs_folder_path):
        file_path = os.path.join(inputs_folder_path, file_name)

        #if filenema is not in the done list:
        if file_name not in done_list:
            # Check if the path is a file (not a subfolder)
            if os.path.isfile(file_path):
                process_file(file_path, file_name)
            # Add at the end of the done list
            with open(done_path, 'a+') as f:
                f.write(f"{file_name}\n")
        else:
            print(f"** {file_name} is in the done list, skipping")        



# Call the function to process the folder
process_folder()
