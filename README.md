# zkproverc

sudo apt install libomp-dev
sudo apt install libgmp-dev
sudo apt install nlohmann-json3-dev
sudo apt install postgresql
sudo apt install libpqxx-dev libpqxx-doc
sudo apt install nasm
sudo apt install libsecp256k1-dev

The following files must be added manually.  Please check size and md5 checksum.

$ ll testvectors/constantstree.bin
-rw-rw-r-- 1 fractasy fractasy 268715104 ene 11 16:41 testvectors/constantstree.bin
$ ll testvectors/verifier.dat
-rw-rw-r-- 1 fractasy fractasy 297485608 ene 12 10:54 testvectors/verifier.dat
$ ll testvectors/starkverifier_0001.zkey
-rw-r--r-- 1 fractasy fractasy 16816778703 ene 12 18:23 testvectors/starkverifier_0001.zkey

$ md5sum testvectors/constantstree.bin
02dc0dfe47a7aaacca6a34486ad5f314  testvectors/constantstree.bin
$ md5sum testvectors/verifier.dat
771a7a09f419f5e6f28dd0cc5a94c621  testvectors/verifier.dat
$ md5sum testvectors/starkverifier_0001.zkey
e460d81646a3a0ce81a561bbbb871363  testvectors/starkverifier_0001.zkey
