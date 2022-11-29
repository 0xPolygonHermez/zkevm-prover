commit 0681f39afcb28f70a0849f54a7bab71a33150581
Author: Ricard Borrell <75077385+rickb80@users.noreply.github.com>
Date:   Tue Nov 22 14:28:23 2022 +0000

    Transpose optimization

commit 57faf3122a5c5ec68aa84b356055e150aeb4d159
Author: Ricard Borrell <75077385+rickb80@users.noreply.github.com>
Date:   Thu Nov 17 13:51:14 2022 +0000

    gpu integration

commit f6ec1ec9b94d376b57a368ed7ad594965cb35116
Author: Ricard Borrell <75077385+rickb80@users.noreply.github.com>
Date:   Thu Nov 17 13:49:50 2022 +0000

    gpu integration and optimizations

commit fdecb432deabc87eaee55e0c919d5e04643df1cd
Author: Ricard Borrell <75077385+rickb80@users.noreply.github.com>
Date:   Thu Nov 3 10:33:49 2022 +0000

    Full integration in first stark

commit 366d19625cf57807157c43897bac2d1ed4bdabf8
Author: Ricard Borrell <75077385+rickb80@users.noreply.github.com>
Date:   Wed Nov 2 13:48:54 2022 +0000

    First integration

commit f69bfa015d655b0b7de55999ba02e67a1c1d7e39
Merge: 402b47b 75a8602
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sun Oct 9 01:17:32 2022 +0200

    Merge pull request #133 from 0xPolygonHermez/develop
    
    v0.4.4.0

commit 75a8602c47bfda60bdf5b5c9b94bca8cac0e1779
Merge: 2152cef 67e3202
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sun Oct 9 01:17:07 2022 +0200

    Merge pull request #132 from 0xPolygonHermez/edu/fix_SLT_main_executor
    
    fix error condition on negative edge conditions

commit 67e32026bbc23af216ad0839e98f04dff24f95af
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Oct 8 23:14:51 2022 +0000

    bump version

commit 93af5d195311f025eb18a0cb6ab301bd948ade75
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Oct 8 23:11:58 2022 +0000

    fix error condition on negative edge conditions

commit 402b47bc7fdf6d68ed2302836cd8af871f4e48f0
Merge: 107a23d 2152cef
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sat Oct 8 17:14:16 2022 +0200

    Merge pull request #131 from 0xPolygonHermez/develop
    
    v0.4.3.0

commit 2152cef33e1597a49996a44bd4ea371c3eab6f13
Merge: 39e7988 4a29d8e
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sat Oct 8 17:13:53 2022 +0200

    Merge pull request #130 from 0xPolygonHermez/fractasy/more_performance_improvements
    
    Fix full tracer: txCount, and unnecessary onOpcode work when no count…

commit 4a29d8ed4dff400e4c9f554d82a583594a6143b6
Merge: cb39e26 39e7988
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Sat Oct 8 17:08:10 2022 +0200

    Merge branch 'develop' into fractasy/more_performance_improvements

commit cb39e269b49c9b9dabc9418dc3650a6fb19a8c4c
Author: fractasy <fractasy@gmail.com>
Date:   Sat Oct 8 15:03:48 2022 +0000

    Fix full tracer: txCount, and unnecessary onOpcode work when no counters are requested.

commit 107a23dce7e43d8b83e3e589704da23d77431a1f
Merge: 1d7abe6 39e7988
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Oct 7 20:30:47 2022 +0200

    Merge pull request #129 from 0xPolygonHermez/develop
    
    v0.4.2.0

commit 39e798834c794bbc8dd1005f4d3b2642fd244e73
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Oct 7 18:27:19 2022 +0000

    bump version

commit 7ac1e2a3267a86bb859b1ebe22f414c0f7284702
Merge: 7531876 470935a
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Fri Oct 7 20:02:30 2022 +0200

    Merge pull request #128 from 0xPolygonHermez/fractasy/more_performance_improvements
    
    Fractasy/more performance improvements

commit 470935a09c1d530c542816d2e87e548d5921b1cc
Author: fractasy <fractasy@gmail.com>
Date:   Fri Oct 7 18:00:51 2022 +0000

    Change config.maxExecutorThreads to 20

commit cced3c10b62400462336633f349cc8654ff53860
Author: fractasy <fractasy@gmail.com>
Date:   Fri Oct 7 17:58:02 2022 +0000

    Support no_counters in executor.proto

commit 1d7abe6ad76e93683cfae1caed63d15d065a67d3
Merge: cac74fd 7531876
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Oct 6 21:38:53 2022 +0200

    Merge pull request #126 from 0xPolygonHermez/develop
    
    v0.4.1.0

commit 75318768ff5dd48820710e46d27d2654c41d2cff
Merge: d6811a4 cac74fd
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Oct 6 21:38:27 2022 +0200

    Merge branch 'main' into develop

commit d6811a49eb00e7e136ccf493d3a77d664c9939d3
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Oct 6 19:37:10 2022 +0000

    Update version

commit e70fbc592a2bebdb9abe5e4bd757c1a6e426504d
Merge: 0c2f36b 98d224c
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Thu Oct 6 14:30:53 2022 -0500

    Merge pull request #125 from 0xPolygonHermez/fractasy/performance_improvements
    
    Multiple fixes during Uniswap integration tests

commit 98d224c97fe0e2da0147f38dd9ca604e50e38b71
Author: fractasy <fractasy@gmail.com>
Date:   Thu Oct 6 18:32:35 2022 +0000

    Multiple fixes during Uniswap integration tests

commit cac74fde8a64fc79aff4ad03b60e5009ada9a4ac
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Oct 6 10:07:45 2022 +0200

    Update version.hpp

commit ab27f037aa214a1aa7df5fcc297c85257f7b99c6
Merge: 121d691 0c2f36b
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Oct 6 10:06:54 2022 +0200

    Merge pull request #124 from 0xPolygonHermez/develop
    
    Develop

commit 0c2f36b3826c947064f41c1e159dcd82e9574e5d
Merge: 536c72b e9595d5
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Thu Oct 6 09:26:23 2022 +0200

    Merge pull request #122 from 0xPolygonHermez/fix-makefile-hpp
    
    Makefile is not automatically launching the build process when only hpp files has been modified

commit e9595d546995f2e54c88121c23ae69d6e19094fa
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Oct 5 18:03:33 2022 +0000

    fix missing download_dependencies target

commit f2c8ade0debe9f5b2c18d0f4277db2f73b069e88
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Oct 5 16:46:32 2022 +0000

    Makefile is not automatically launching the build process when only hpp files has been modified

commit 536c72b713e2425d6551c14173322a48371f492e
Merge: 80dbbd7 a5c8958
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Tue Oct 4 15:42:04 2022 +0200

    Merge pull request #121 from 0xPolygonHermez/fractasy/remove_batchL2Data_length_check
    
    Fractasy/remove batch l2 data length check

commit a5c895881b75391dc7dcd0cdfe5c533f5a9505f0
Author: fractasy <fractasy@gmail.com>
Date:   Tue Oct 4 13:40:54 2022 +0000

    Limit to 60000B

commit df5ea142fa18a315980572e22a0b149aa8e793af
Author: fractasy <fractasy@gmail.com>
Date:   Tue Oct 4 13:14:25 2022 +0000

    Remove batchL2Data length input check

commit 80dbbd7d247cfb845fee77170488b0f9e7c3bf5c
Merge: b0c2506 0619156
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Tue Oct 4 14:25:23 2022 +0200

    Merge pull request #118 from 0xPolygonHermez/fractasy/fix_main_exec_generated
    
    Fix main SM executor generated C code

commit b0c250647f1e5d56cf400beb86be3178705ab1e0
Merge: 79b3851 733eaeb
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Tue Oct 4 14:24:15 2022 +0200

    Merge pull request #119 from 0xPolygonHermez/fractasy/change_changeId_to_changeID_in_json_files
    
    Change changeId to changeID in input json files

commit 733eaeb817af4e86e3e4cd1a9eefa466e2a5c1cb
Author: fractasy <fractasy@gmail.com>
Date:   Tue Oct 4 10:26:44 2022 +0000

    Change changeId to changeID in input json files.  Fixed a bug in FullTracer::getFromMemory().

commit 06191565613ad2070ef8263c1a64e77de2d162df
Author: fractasy <fractasy@gmail.com>
Date:   Tue Oct 4 07:59:48 2022 +0000

    Fix main SM executor generated C code

commit 79b3851ac048cbb647f40f5e4c65c2f86c5950a3
Merge: b6a5ca0 fceb91f
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Tue Oct 4 09:49:06 2022 +0200

    Merge pull request #117 from 0xPolygonHermez/fix-vscode-launch
    
    Fix preLaunchTask errors in launch.json file

commit fceb91f7e909cb2288d2feddb1f9cc32b0c8c323
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Oct 3 23:59:57 2022 +0000

    fix preLaunchTask errors in launch.json file

commit b6a5ca072045c57e2ba04b18f2713b49dd9165f9
Merge: b60996c 11c5ff3
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Oct 3 22:24:56 2022 +0200

    Merge pull request #116 from 0xPolygonHermez/edu/makefile_download_dependencies
    
    Edu/makefile download dependencies

commit 11c5ff3759ed6ec2d4e2f750d84fd368653e5131
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Oct 3 22:03:28 2022 +0200

    Update download_dependencies.sh

commit 16b9aad8e3a3bc0fd3c6468fa86c6fea52e282d7
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Oct 3 22:02:44 2022 +0200

    Update version.hpp

commit 121d6913f8f58d91e90248b8d9d75942c1117cc7
Merge: e1e1580 5e39ea0
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Oct 3 22:01:35 2022 +0200

    Merge pull request #115 from 0xPolygonHermez/edu/makefile_download_dependencies
    
    add download script to makefile

commit 5e39ea0eef42bc41e67ba12574e6a9d9b7505af1
Author: eduadiez <eduadiez@gmail.com>
Date:   Mon Oct 3 19:52:43 2022 +0000

    add download script to makefile

commit b60996cc13812e7f07c810d7579f3d73c66fd643
Merge: 0900e2b 67c9c6f
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Mon Oct 3 08:42:43 2022 +0200

    Merge pull request #113 from 0xPolygonHermez/fractasy/remove_unneeded_makefile_flag
    
    Remove a C compilation flag from Makefile that is no longer needed, n…

commit 0900e2b6d97e61e07e15e6c320c212efcd0a34b3
Merge: bb15b0d 58a139e
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Mon Oct 3 08:42:24 2022 +0200

    Merge pull request #114 from 0xPolygonHermez/fractasy/fix_smt_hash_counter
    
    Fix SMT set hash counter

commit 58a139e51b0b859e0db15dbc28f7df6770a84e48
Author: fractasy <fractasy@gmail.com>
Date:   Sun Oct 2 10:13:22 2022 +0000

    Fix SMT set hash counter

commit 67c9c6fc306eba6df39cb3b4bdf7dc78aa0a8529
Author: fractasy <fractasy@gmail.com>
Date:   Sat Oct 1 17:59:52 2022 +0000

    Remove a C compilation flag from Makefile that is no longer needed, nor used

commit bb15b0d29275bd734ec5dce1c664228a4241b663
Merge: b7c5196 4f2e005
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sat Oct 1 19:46:13 2022 +0200

    Merge pull request #112 from 0xPolygonHermez/fractasy/add_litchi_generated_c_files
    
    Fractasy/add litchi generated c files

commit 4f2e0053ff17d2e7e2212386b4ac9ee75d16a833
Author: fractasy <fractasy@gmail.com>
Date:   Sat Oct 1 17:44:30 2022 +0000

    Add litchi c generated files

commit b7c519659727d2138d0e008488cda1794a2c8d97
Merge: 1786679 71deb01
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sat Oct 1 19:42:15 2022 +0200

    Merge pull request #111 from 0xPolygonHermez/edu/add_litchi_files
    
    Edu/add litchi files

commit 1786679f56f119814477e6cc422a7cc9cc031a3d
Merge: e7dc4e5 0a3fb1c
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sat Oct 1 19:22:14 2022 +0200

    Merge pull request #110 from 0xPolygonHermez/fractasy/remove_input_database_length_check
    
    Disable DB value check in Input load from json

commit 0a3fb1c16ff3204a26d5ca1372bcc929673a90c5
Author: fractasy <fractasy@gmail.com>
Date:   Sat Oct 1 17:17:22 2022 +0000

    Disable DB value check in Input load from json

commit 71deb01e984e468755ca7c6956db63dae057424a
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Oct 1 09:52:38 2022 +0000

    AFixzkevm.verifier.cpp

commit 295affdb75145a0c42f7028a10743336f068c026
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Oct 1 09:48:19 2022 +0000

    Adapt zkevm.verifier.cpp to goldilocks mi

commit 8657beeb337ef64fae1dd4982145897de37a05c7
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Oct 1 09:41:33 2022 +0000

    Adapt zkevm.verifier.cpp to goldilocks

commit f17da9c29aa775d7a7547c292c3e60ed11c9f3cb
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Oct 1 09:33:40 2022 +0000

    Adapt zkevm.c12a.verifier.cpp to goldilocks

commit 11f9918795c8a8fd431a01858f94c1f81b2da3c5
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Oct 1 09:27:03 2022 +0000

    Add zkevm c12a verifier

commit a96f7fb4767838274febc25aa7e8a969201bf6af
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Oct 1 09:22:34 2022 +0000

    Add zkevm c12b verifier

commit 3d52dc74ac731b3f62e2f17871343056d7529479
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Oct 1 09:14:08 2022 +0000

    Add zkevm cpp verifier

commit 6c8ee67cddbb44af8e356ffc02b1d2274e1e6fed
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Oct 1 09:04:09 2022 +0000

    add zkevm c helpers

commit c1b9dc980d95bff46480d5e157e6faad35fc9619
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Oct 1 09:03:56 2022 +0000

    add c12b helpers

commit 9fd9074b02ea29b3520a54e4635d067664f499b3
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Oct 1 09:03:44 2022 +0000

    add c12a herlpers

commit e1e1580573b0f7932a5f33e98008cf0924aa3a5d
Merge: e35cf68 73e2cfc
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sat Oct 1 10:48:13 2022 +0200

    Merge pull request #109 from 0xPolygonHermez/fix-makefile
    
    Fix makefile

commit 73e2cfcac339231e2003bc732025dc6e2691f87d
Merge: cd6fc45 e35cf68
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sat Oct 1 10:48:05 2022 +0200

    Merge branch 'main' into fix-makefile

commit e7dc4e5d4978d88098d2c51eb9f5c261fc72d7ab
Merge: 9d27c0d cd6fc45
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sat Oct 1 10:45:27 2022 +0200

    Merge pull request #108 from 0xPolygonHermez/fix-makefile
    
    Fix Makefile when building zkProver target

commit 9d27c0d38a8d1654358ec45af5a6fbe30af90aa9
Merge: 1ecbf38 3c3f83b
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Sat Oct 1 10:18:04 2022 +0200

    Merge pull request #107 from 0xPolygonHermez/fix_tx_hash_u64
    
    Chain ID support, new ROM, etc.  Getting Executor ready for Litchi.

commit cd6fc450b755894e4e401342cd2f5ff6d1d969b8
Author: agnusmor <agnusmor@gmail.com>
Date:   Sat Oct 1 03:11:09 2022 +0000

    Fix Makefile error when compiling zkProver target

commit 3c3f83bb57fd241f66452936bc7f64a447da0b43
Author: fractasy <fractasy@gmail.com>
Date:   Fri Sep 30 18:40:21 2022 +0000

    Chain ID support, new ROM, etc.  Getting Executor ready for Litchi.

commit e35cf685c26fe215a4b3a8a8796c262a13c52cba
Merge: 649e0bc 63eae7a
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Sep 30 18:37:21 2022 +0200

    Merge pull request #106 from 0xPolygonHermez/build-const-tree
    
    Build const tree

commit 1ecbf38b4ac25a7cfc0e6bd674f5ea7c07beb98a
Merge: 6f23ff7 63eae7a
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Sep 30 18:34:13 2022 +0200

    Merge pull request #91 from 0xPolygonHermez/build-const-tree
    
    Added implementation of the buildconsttree function in C++

commit 649e0bc626a6e56c04a979d898fb3b4338f40da7
Merge: f2fa60d 045b170
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Sep 30 12:35:25 2022 +0200

    Merge pull request #105 from 0xPolygonHermez/v0.3.1.0-rc-1
    
    v0.3.1.0-rc 1

commit 045b170e09ea60245cb774135618f0ff684a6be1
Merge: 8ad9f93 781a443
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Sep 30 12:29:32 2022 +0200

    Merge pull request #104 from 0xPolygonHermez/fix_gas_condition
    
    Fix gas used logic in Full Tracer: when used gas exceeds limit, repor…

commit 6f23ff7cbe09059e42936539f3e0ce5425965ebd
Merge: fd2629b 781a443
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Sep 30 12:27:41 2022 +0200

    Merge pull request #103 from 0xPolygonHermez/fix_gas_condition
    
    Fix gas used logic in Full Tracer: when used gas exceeds limit, repor…

commit 781a44301810634a040061887b80e16442e406d4
Author: fractasy <fractasy@gmail.com>
Date:   Fri Sep 30 10:21:27 2022 +0000

    Fix gas used logic in Full Tracer: when used gas exceeds limit, report only the limit

commit 8ad9f938c7dc8da0c6aab51813d08df38441c241
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Sep 30 10:11:51 2022 +0000

    Update version

commit f2fa60ddd0956e16fae416d04cc6cfa9ce633ea7
Merge: dabdea9 005fa61
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Sep 30 11:25:17 2022 +0200

    Merge pull request #102 from 0xPolygonHermez/v0.3.0.0-rc.1
    
    v0.3.0.0 rc.1

commit 005fa61f84939b60897e5b47441020d5757a9874
Merge: f30c25f dabdea9
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Sep 30 11:24:41 2022 +0200

    Merge branch 'main' into v0.3.0.0-rc.1

commit f30c25f7cdfc634d4c060a567acd464d6e38b502
Merge: 41df810 fd2629b
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Sep 30 09:59:11 2022 +0200

    Merge pull request #101 from 0xPolygonHermez/develop
    
    Develop

commit 63eae7a72e5c489c46b97ce862507e95e3d14f47
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Sep 29 16:05:45 2022 +0000

    fix -v argument in the bctree launch task

commit fd2629ba5397b9d752585d7bae784e90740bfa51
Merge: de74433 f1c9029
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Sep 29 17:17:14 2022 +0200

    Merge pull request #100 from 0xPolygonHermez/executor_client_multithread
    
    Executor client multithread improvements

commit f1c90295f45df3b417ea47e2e88bb898a82589b6
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 29 14:49:34 2022 +0000

    Add executor client multithread test.  Calculate the total executor throughput in gas/s, every second or more.

commit de74433f41f585e1d6f94c765dae00afbdba1f6c
Merge: 62c68f6 57559e1
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Sep 29 15:08:10 2022 +0200

    Merge pull request #99 from 0xPolygonHermez/log_time_statistic
    
    Create LOG_TIME_STATISTICS and fix bug in ctx.lastStep

commit 57559e107872cf7e38fd5820bdc4882f51b4bb75
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 29 12:57:16 2022 +0000

    Create LOG_TIME_STATISTICS and fix bug in ctx.lastStep

commit 62c68f678de98b7c0459428d2dbcc220adb55508
Merge: ee60f60 3a68335
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Sep 29 12:13:10 2022 +0200

    Merge pull request #98 from 0xPolygonHermez/move_keccak2_to_obsolete
    
    Fix last Ethereum tests error

commit 3a68335029db75e5b39fd684c3ff0f566eb8e6a1
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 29 10:05:34 2022 +0000

    Add LOG_SERVICE_EXECUTOR_INPUT/OUTPUT traces definition flags

commit e256c225e535c13fca9fb27db5d207ee33217a00
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 29 09:50:37 2022 +0000

    Prevent batch_num=0 in Executor and Prover services

commit 5244ab68d465748139214dd87d3e0001a28cc888
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 29 09:01:28 2022 +0000

    Add check in FullTracer to avoid pols gas being higher than gas left.  Initializa attributes in Full Tracer.

commit 4364a94b437f3d83e7822ac81f18b5564771f106
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 29 08:03:42 2022 +0000

    Use time counters only when LOG_TIMER is defined, in Executor service

commit 97f08c46b281ac039320213ef357235388aa5ca7
Author: fractasy <fractasy@gmail.com>
Date:   Wed Sep 28 18:08:09 2022 +0000

    Remove traces

commit 876ae9d4516243b4863f89abcebb3ac9f1ace8dd
Author: fractasy <fractasy@gmail.com>
Date:   Wed Sep 28 15:14:43 2022 +0000

    Fix last Ethereum tests error

commit 0f372a13c0ad354a9cfaac307821378d677b184d
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Sep 28 12:58:07 2022 +0000

    Use -v|--verkey as option to specfy the verkey file. Remove quotes in verkey values (GL MT)

commit ee60f600742cb281df070b9cfbcf24e29cf694d6
Merge: 9a5d5f8 f164ccb
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Wed Sep 28 12:50:11 2022 +0200

    Merge pull request #96 from 0xPolygonHermez/advance_data_size_check_to_main_sm_executor
    
    Advance batch L2 data size check to main sm executor input.preprocessTxs()

commit f164ccb120c825d5670661f6c4197ca897c3a31f
Author: fractasy <fractasy@gmail.com>
Date:   Wed Sep 28 10:45:11 2022 +0000

    Consider input.batchL2Data as a string, and compare it against 30000 bytes

commit a08eb1d5500032106f13fd3eff2eb5d6b1c9f650
Author: fractasy <fractasy@gmail.com>
Date:   Wed Sep 28 10:19:30 2022 +0000

    Do not allow batchL2Data bigger than 60000 bytes

commit c356dbf1ea08fb3aa1fb8ff46ff7074ebbe8a4bd
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue Sep 27 18:05:54 2022 +0000

    update bctree usage info

commit 9a5d5f8f267ce1c49f655da0774c7c7c48b93416
Merge: 0aab883 fbf10b8
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Tue Sep 27 17:33:23 2022 +0200

    Merge pull request #95 from 0xPolygonHermez/fix_tx_hash
    
    Fix tx hash

commit fbf10b8837c918b188073755fca60cfd97dcca38
Author: fractasy <fractasy@gmail.com>
Date:   Tue Sep 27 15:29:27 2022 +0000

    Fix tx hash

commit 0aab883435afd4f69c2516fc0bd1f4048eac057e
Merge: f4ec1ee fad5e21
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Tue Sep 27 16:33:23 2022 +0200

    Merge pull request #94 from 0xPolygonHermez/edu/enable_LOG_TX_HASH-1
    
    Update definitions.hpp

commit fad5e219616a31b63e6e5743bad23b62aa449829
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Tue Sep 27 16:32:40 2022 +0200

    Update definitions.hpp

commit 41df81019e6e3019651b3bfad582b6cdb5d0c667
Merge: 78ee835 f4ec1ee
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Tue Sep 27 11:53:53 2022 +0200

    Merge pull request #92 from 0xPolygonHermez/develop
    
    Develop

commit f4ec1ee405610b291aa8b96149e77a6a287ca947
Merge: fc52c19 b01d0a7
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Tue Sep 27 10:44:11 2022 +0200

    Merge pull request #90 from 0xPolygonHermez/fix_multithread_executor_issues
    
    Double-protect Database dbReadLogs with a mutex, in case config.saveD…

commit bc8395f541b5764215d9a93e72a31509630216f3
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue Sep 27 00:51:29 2022 +0000

    Updated default params for launch.json option "bctree"

commit 42158af8f777805dde1404e8375fd9f4a6ebe4f5
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue Sep 27 00:44:21 2022 +0000

    Added implementation of the buildconsttree function in C++

commit b01d0a704d08ff44e901e445b03bfc7b468ed93f
Author: fractasy <fractasy@gmail.com>
Date:   Mon Sep 26 16:12:19 2022 +0000

    Double-protect Database dbReadLogs with a mutex, in case config.saveDbReadsToFile=true and we have multithread clients.

commit fc52c19164fdfd65e31bcbe878696dc88675c616
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Sep 26 15:44:30 2022 +0200

    Update README.md

commit dabdea97f33138a73c9deb465d7f3cb0164f5853
Merge: f75eb70 8a44c02
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Sep 26 15:42:20 2022 +0200

    Merge pull request #89 from 0xPolygonHermez/edu/add_gha
    
    it adds gha to autogenerate docker images

commit 8a44c02761d57e69acbe5047b6cc25d1670ffac0
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Mon Sep 26 15:41:51 2022 +0200

    it adds gha to autogenerate docker images

commit 90e92f1633e2c6ea150b4564d4a1cb2b92c18b9d
Merge: d1ee72c de23202
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Sep 26 10:23:49 2022 +0200

    Merge pull request #87 from 0xPolygonHermez/fix_multithread_executor_issues
    
    Add runFileMultithread test.  Fix prover ignoring executor error.

commit de232022c71ef7c918044d9b77f9a1d08a70fab7
Author: fractasy <fractasy@gmail.com>
Date:   Mon Sep 26 08:21:46 2022 +0000

    Add runFileMultithread test.  Fix prover ignoring executor error.

commit d1ee72c599c66fbdc83f7f27bff821f27a5a9092
Merge: 2629160 30f0612
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Sep 26 09:19:57 2022 +0200

    Merge pull request #86 from 0xPolygonHermez/fix_multithread_executor_issues
    
    Add config.saveDbReadsToFile to control when database reads are store…

commit 30f06129a520823da05d5aa4173cb45bc6f9992c
Author: fractasy <fractasy@gmail.com>
Date:   Mon Sep 26 07:16:21 2022 +0000

    Update main SM executor generated code with latest changes

commit 4b039482ebf06a7472e6b4662340b46c8fab092f
Author: fractasy <fractasy@gmail.com>
Date:   Mon Sep 26 06:36:20 2022 +0000

    Add config.saveDbReadsToFile to control when database reads are stored into an xxx.out.input.json file

commit 78ee835777c7835e1112cec2ba1388a1b8c4a81f
Merge: 635d584 201b259
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sun Sep 25 18:54:27 2022 +0200

    Merge pull request #85 from 0xPolygonHermez/edu/hotfix_binary_sub
    
    Edu/hotfix binary sub

commit 2629160a2603fc828cba1c775d9c0a6a21e5e653
Merge: 99693d7 201b259
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sun Sep 25 18:52:18 2022 +0200

    Merge pull request #84 from 0xPolygonHermez/edu/hotfix_binary_sub
    
    Edu/hotfix binary sub

commit 201b259ee33010f0766a61ca84770b9880a02b35
Author: eduadiez <eduadiez@gmail.com>
Date:   Sun Sep 25 16:47:16 2022 +0000

    add uniswap input test

commit 795be9953d6264738ffe41cf1eab0fe6e1d7ae8c
Author: eduadiez <eduadiez@gmail.com>
Date:   Sun Sep 25 16:46:56 2022 +0000

    fix sub edge case binary

commit cce0335389e30556beb57ae8b30bfa5dc58c83b2
Author: eduadiez <eduadiez@gmail.com>
Date:   Sun Sep 25 16:45:30 2022 +0000

    Improve traces

commit 99693d72d6a691be9e9d0f9d1f4fee28be6faca2
Merge: 86012bb 1f76a66
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sun Sep 25 02:17:35 2022 +0200

    Merge pull request #83 from 0xPolygonHermez/edu/add_copy_runtime_files.sh
    
    Edu/add copy runtime files.sh

commit 635d58460e4f33167ff902ec21690000e34e06f7
Merge: beda39c 1f76a66
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sun Sep 25 02:16:27 2022 +0200

    Merge pull request #82 from 0xPolygonHermez/edu/add_copy_runtime_files.sh
    
    improve copy_runtime_files.sh

commit 1f76a660e4737ac5338f337dd4ec3904048f5394
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Sep 24 23:52:55 2022 +0000

    add tests to copy_runtime_files.sh

commit fe5d6c428cbaefb69720a0e06e0a58be3dcca21a
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Sep 24 23:48:04 2022 +0000

    fix typo

commit 98fe927cd1d65c5320c4882109505970a981e66f
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Sep 24 23:44:46 2022 +0000

    improve copy_runtime_files.sh

commit 86012bbdd1a46545fda445cb4bf9ca2c577bf9c1
Merge: 3089050 c4fc302
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sun Sep 25 00:57:56 2022 +0200

    Merge pull request #81 from 0xPolygonHermez/edu/fix_memory_delete_errors
    
    Edu/fix memory delete errors

commit beda39c7d8f65766cbd252def7d883d4e1116ad4
Merge: e8c0035 c4fc302
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sun Sep 25 00:57:18 2022 +0200

    Merge pull request #80 from 0xPolygonHermez/edu/fix_memory_delete_errors
    
    fix memory errors due to early deletion

commit c4fc302fac7ee08bcaabf56eaf977b8b46c60bd5
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Sep 24 22:52:52 2022 +0000

    fix memory errors due to early deletion

commit e8c003521aad5e6baf0c5abe32b8960c4114670c
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Sep 24 20:28:05 2022 +0000

    Update version

commit 3089050324970eda88c596f2d46f3859c213d4b6
Merge: 8380881 c2ed5b6
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sat Sep 24 22:19:13 2022 +0200

    Merge pull request #79 from 0xPolygonHermez/edu/adds_c12a_c12b_full
    
    Edu/adds c12a c12b full

commit c2ed5b6987515c260d819fb92c25903815ae3dba
Merge: fc577b1 8380881
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Sat Sep 24 22:17:08 2022 +0200

    Merge remote-tracking branch 'origin/develop' into edu/adds_c12a_c12b_full

commit fc577b12072bce243d35cefa79dc2fa4feae2b24
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Fri Sep 23 12:24:21 2022 +0200

    first implementation of n23 and c12a/b

commit 83808819f51e01163617670e22b0dfd491bad4fa
Merge: a5a73c6 f2db792
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Sat Sep 24 21:53:21 2022 +0200

    Merge pull request #78 from 0xPolygonHermez/gha
    
    Github Action to build docker image and run dependant node tests

commit f2db792892155ee9656931a6d78ee99556ff0f0d
Author: xavier-romero <xavier.romero@gmail.com>
Date:   Fri Sep 23 18:32:38 2022 +0200

    Github Action to build docker image and run dependant node tests

commit f855f497372171936f221fcb041a08134c9c924d
Merge: c8b3775 a5a73c6
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Fri Sep 23 12:20:49 2022 +0200

    Merge branch 'develop' into edu/adds_c12a_c12b_basic

commit c8b3775ee13fc8bd4d41e8756452dc509336b683
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Fri Sep 23 12:19:35 2022 +0200

    new config files

commit 75979016aa2bd96b4183dd1f531bb1e67662026d
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Fri Sep 23 12:17:53 2022 +0200

    new tests files for basic

commit ff1e83c3e24cb55d6e1653861d1fee9e284a2af1
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Fri Sep 23 12:15:08 2022 +0200

    new files config

commit 2db46c168c8d92ff6b7920ce0af98383c962b7bc
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Fri Sep 23 12:14:12 2022 +0200

    update goldilocks

commit a5a73c6542767ba468d8e7e6bc41d06d2f9ef1e6
Merge: b6d083c 42fed53
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Fri Sep 23 11:55:38 2022 +0200

    Merge pull request #76 from 0xPolygonHermez/cold_and_warm_addresses
    
    Cold and warm addresses

commit 42fed53c85206905fe923e26eeca9987e5615e37
Author: fractasy <fractasy@gmail.com>
Date:   Fri Sep 23 09:24:13 2022 +0000

    Fix full tracess to support invalid opcodes in onOpcode() call

commit b6d083c8553c6cae358094a66afa01b39891b007
Merge: ade1a30 c5323e0
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Fri Sep 23 10:07:46 2022 +0200

    Merge pull request #75 from 0xPolygonHermez/process_batch_memory_allocation_fix
    
    Fix process batch memory allocation: allocate only 1 evaluation, inst…

commit 78076cf6026df47318473afd8113d0617df719bd
Author: fractasy <fractasy@gmail.com>
Date:   Fri Sep 23 08:05:07 2022 +0000

    Fix ctx.accessStorage types: from uint32_t to mpz_class

commit c5323e0645956c8de43307be7b0a5b481cd76446
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 22 15:52:42 2022 +0000

    Fix process batch memory allocation: allocate only 1 evaluation, instead of N=degree evaluations

commit ade1a305779ebb4743443148b5f63f56332e02ea
Merge: 464605a 7900f7f
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Thu Sep 22 16:33:46 2022 +0200

    Merge pull request #74 from 0xPolygonHermez/remove-issetreplacingzero
    
    Remove funcName "isSetReplacingZero"

commit 464605ae4cf952cf162366b7d954da1305352e21
Merge: d839308 29c9b33
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Thu Sep 22 16:32:44 2022 +0200

    Merge pull request #70 from 0xPolygonHermez/proverjs_port
    
    Proverjs port

commit 7900f7fcfdbcf457a1aef5142cb22d9c632c2b61
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Sep 22 13:09:10 2022 +0000

    Remove funcName "isSetReplacingZero" as it is not used in the rom Storage. Update storage_sm_rom.json (fix typo "GetRkey")

commit 225f571d580be01b4577c049d1e6e189887ee533
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 22 09:12:14 2022 +0000

    Accessed storage addresses and keys

commit d839308e1aec64d5bf4494867e5e940cfdc5d507
Merge: 82ef846 110f213
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Fri Sep 16 15:53:11 2022 +0200

    Merge pull request #71 from 0xPolygonHermez/memory_compare
    
    Modify memory SM reorder function to be faster

commit 82ef84625987df5fb42cda7bdd463fe803b4cd5f
Merge: b14c8b8 2100025
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Fri Sep 16 15:52:21 2022 +0200

    Merge pull request #68 from 0xPolygonHermez/feature/mem_optimization
    
    reuse memory pointer

commit b14c8b858f7d200ab3fe5f6afb24b8b7408339ae
Merge: 96c218c bf1cb15
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Fri Sep 16 15:51:41 2022 +0200

    Merge pull request #69 from 0xPolygonHermez/hashk_op_length_check
    
    In main executor, check that hash length and data in op are consisten…

commit 96c218c81a26b82fdcb8b36bf48d50e3e48b7afb
Merge: 1c5595b 88c9fcd
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Fri Sep 16 15:50:58 2022 +0200

    Merge pull request #66 from 0xPolygonHermez/replace_zkassert
    
    Replace a zkassert in the 2 padding byte SMs by regular error code

commit 110f213394dbe77a9ce93669e9aaeec26ec54eae
Author: fractasy <fractasy@gmail.com>
Date:   Fri Sep 16 10:13:27 2022 +0000

    Modify memory SM reorder function to be faster

commit 210002595c4366717b07b2fd6f6f5c10d22bd073
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Thu Sep 15 19:54:48 2022 +0200

    reuse memory pointer

commit 1c5595b0b3f2f4cab67d1cf0f7b5437c2cbf4639
Merge: d27ec05 2cab22a
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Sep 15 18:43:54 2022 +0200

    Merge pull request #67 from 0xPolygonHermez/stark12_optimization
    
    Stark12 optimization

commit 2cab22a9068d6b0792d2cae0507797534d161409
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Thu Sep 15 18:37:23 2022 +0200

    delete tests

commit 9443cccf4164455989f390bb8c55fe9f55d97b37
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Thu Sep 15 18:34:46 2022 +0200

    fix starkC12 initialization

commit bf1cb15d8c44b8f66bd44d42ef4ddc12ca844598
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 15 15:50:14 2022 +0000

    In main executor, check that hash length and data in op are consistent, i.e. padding space is 0

commit 88c9fcd85048e3d8a88d101a98989a55d38e5396
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 15 15:04:51 2022 +0000

    Replace a zkassert in the 2 padding byte SMs by regular error code

commit b61a65289a7cc1692b743c691d68e5a80e6c3926
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Thu Sep 15 16:22:45 2022 +0200

    change config files

commit 29c9b3337caa4c31a06b8419a74534c169d53953
Merge: ffb4865 d27ec05
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 15 14:07:01 2022 +0000

    Merge remote-tracking branch 'origin/develop' into proverjs_port

commit ffb4865272a4e00e5ecd0c7201e7bde4e91dbe83
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 15 14:06:35 2022 +0000

    Port last changes from proverjs

commit d27ec05e2caffd1289c10b6ee722a719d0644f90
Merge: 5be4e14 b452d6f
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Thu Sep 15 16:03:11 2022 +0200

    Merge pull request #65 from 0xPolygonHermez/sm_overflow
    
    Sm overflow

commit c2d4daf20f40d849cecfe500c89602b818db3bfd
Merge: 8d9a59d 5be4e14
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Thu Sep 15 12:29:37 2022 +0200

    Merge remote-tracking branch 'origin/develop' into stark12_optimization

commit 8d9a59dcde354ee09ea2bae745be50f89f371627
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 15 10:05:30 2022 +0000

    Move StarkC12 out of the prove method to reuse its initialization

commit b452d6fc01fa4a0160c6f3465c71cd81bd1e004c
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 15 09:11:59 2022 +0000

    SM fixes: 1 in binary, 2 in Padding PG

commit 7c0610562c82391d4031e3cb3085c0687a1f2f73
Merge: 22b13ae 5be4e14
Author: fractasy <fractasy@gmail.com>
Date:   Wed Sep 14 15:43:15 2022 +0000

    Merge remote-tracking branch 'origin/develop' into sm_overflow

commit 22b13ae1f907e05ce05879e0453287fdb4432efe
Author: fractasy <fractasy@gmail.com>
Date:   Wed Sep 14 15:43:13 2022 +0000

    State machines evaluation overflow control code.  Also list all missing files at init.

commit 5be4e1482edaa4a02002a811dafef01c890f31f5
Merge: 0a97cd5 29c9d38
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Wed Sep 14 17:26:26 2022 +0200

    Merge pull request #64 from 0xPolygonHermez/store-db-input-file
    
    Store db reads in the output file generated that contains the input request data

commit 29c9d38c3ea36a15e4397d6dbba823daa05191d9
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Sep 14 14:51:19 2022 +0000

    Store db read operations in output (input) file

commit 0a97cd514886c82f60b39497aad32a54584c9c2f
Merge: 94b7820 5ea299d
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Wed Sep 14 09:10:52 2022 +0200

    Merge pull request #63 from 0xPolygonHermez/improve-runfile-implementarion
    
    Improve main.cpp runFile implementation to use a copy of config and pass it to ProverRequest::Init()

commit 94b7820a8e88643c912ad4529c6671fae6683954
Merge: c36970a 1e9daba
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Wed Sep 14 09:10:34 2022 +0200

    Merge pull request #62 from 0xPolygonHermez/add_date_and_time_to_time_traces
    
    Add date and time to TIMER traces, for debugging and execution conven…

commit c36970afd1b9c2619d99c480b4fffddd54b75afa
Merge: 43fbcbc 8801043
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Wed Sep 14 09:10:13 2022 +0200

    Merge pull request #61 from 0xPolygonHermez/bugfix/create-address-tracer
    
    Fix to value for create address

commit 43fbcbc535ad860a8f4d2bf07c6d26698b52a7df
Merge: 3897e24 45c3fdf
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Wed Sep 14 09:08:59 2022 +0200

    Merge pull request #58 from 0xPolygonHermez/config_improvements
    
    Config improvements

commit 5ea299dff646295a9e5a7397a845855569df8366
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue Sep 13 17:03:24 2022 +0000

    Issue #53: Improve main.cpp runFile implementation to use a copy of config and pass it to ProverRequest::Init()

commit 88010436bf4fe2bedeeaaa23e2633b3e93a5a863
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue Sep 13 13:10:27 2022 +0000

    Fix to value for create address

commit 1e9daba97d0c4036386197321d86eab46ae213a4
Author: fractasy <fractasy@gmail.com>
Date:   Tue Sep 13 12:11:42 2022 +0000

    Add date and time to TIMER traces, for debugging and execution convenience

commit 3897e242dbb888eaeff45e25527eb6dbfdf3687e
Merge: efc9ed6 69bfb2f
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Tue Sep 13 12:00:44 2022 +0200

    Merge pull request #59 from 0xPolygonHermez/EVMAP_C12_optimized
    
    C12 EVMAP optimized

commit f75eb707ea4f2fe9b6a35914605afa5f7fee57ca
Merge: eb0ab20 84f1fec
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Tue Sep 13 11:35:03 2022 +0200

    Merge pull request #60 from 0xPolygonHermez/release/v0.2.1
    
    Release/v0.2.1

commit 69bfb2f5541b6b1cc377e0e068111b9339e09d10
Author: rickb80 <rickborrell@gmail.com>
Date:   Mon Sep 12 13:24:49 2022 +0200

    C12 EVMAP optimized

commit 45c3fdfdd563dc073464cc2fc622051a8c447a66
Author: fractasy <fractasy@gmail.com>
Date:   Mon Sep 12 08:05:02 2022 +0000

    Config improvements

commit 84f1fec64a7a5222787d7db20b2e0edcf6b76259
Merge: 80a54bc efc9ed6
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Sep 9 18:17:31 2022 +0200

    Merge pull request #57 from 0xPolygonHermez/develop
    
    Develop

commit efc9ed6916042c0a1ff054a94bcde59494ff5502
Merge: eb0ab20 1e6396c
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Sep 9 17:20:11 2022 +0200

    Merge pull request #56 from 0xPolygonHermez/FRI_optimization
    
    Optimised FRIs

commit 80a54bcb5343f087712d1365323eeb3228964e50
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Fri Sep 9 17:15:02 2022 +0200

    update release verision

commit eb0ab20d1f7f8f6c58eafa721b19f567ba9dde80
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Fri Sep 9 17:11:27 2022 +0200

    Tag v0.2.0.0 (Mandarin)

commit 1e6396c897705495d026b2e8aa198dc1d8dfd800
Author: rickb80 <rickborrell@gmail.com>
Date:   Fri Sep 9 16:59:18 2022 +0200

    Optimized FRIs

commit 410530ba53b0ceda604b66f0a8e38c29c6f78e66
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Fri Sep 9 16:55:47 2022 +0200

    delete unnecessary files

commit 1e12361842c0d6b11818fcd450d8c210c8d0e054
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Fri Sep 9 15:49:16 2022 +0200

    update version

commit 90cae97021dc06977269dccf523817f598f44cee
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Fri Sep 9 15:12:31 2022 +0200

    mandarin ready changes

commit 63f67ddec404a8b8907f68e46d5546650dd9a3d0
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Fri Sep 9 13:59:55 2022 +0200

    complete basic test

commit 9ed34086f9c2e7a61500d9efd880297270ab5d28
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Thu Sep 8 20:57:23 2022 +0200

    add basic files to gitignore

commit b0681fa313aa9876d4f4428c49bf719a06761b8d
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Thu Sep 8 20:56:38 2022 +0200

    Improve basic tests

commit aefa7f59afaf744a780d406254898d31dff6bb79
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Thu Sep 8 20:55:19 2022 +0200

    new merkleTreeBN128

commit 183f40cd27f21c3390085e25a0eeb98fb65e0784
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 8 16:41:48 2022 +0000

    Add error trace to file2json()

commit 8533b1ff4d107642f939d05146cc9c3c0e2bf101
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 8 15:32:25 2022 +0000

    Modify FullTracer to filter the value of to when it is 0x0

commit 2dfa17f8ae0476d440d36979c2e21287c007faa1
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 8 14:46:13 2022 +0000

    Update executor from mandarin label in other repos files

commit 1861f34d5d90195f877d1273cdeba3e8f7a84431
Merge: cd660b0 4adaadf
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Sep 8 14:42:08 2022 +0200

    Merge pull request #52 from 0xPolygonHermez/v0.1.1
    
    V0.1.1

commit 4adaadf4adb0c033e295afc658b088884a425270
Merge: 427dc8e cd660b0
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Sep 8 14:41:29 2022 +0200

    Merge branch 'main' into v0.1.1

commit cd660b0e80b1412bba232b452eabc6919685e957
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Thu Sep 8 14:37:23 2022 +0200

    Integrate v0.1.1 changes

commit 83a29ecc33e2d6f1af1c237fa18ba4da229d3264
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Thu Sep 8 14:25:35 2022 +0200

    remove zkevm.starkinfo.json

commit 0b3042f750e5f7f2a7519e208dd41f19e8d27fce
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Thu Sep 8 14:23:53 2022 +0200

    update git ignore

commit 659fa2880cf35896c9f78386dc28ae1e149be91f
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 8 12:18:11 2022 +0000

    Add src/config/version.hpp, and report it when -v or --version

commit 427dc8e89607b87a3399e28c3f691e6b3d7edf1d
Merge: 5d0f731 85aab69
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Sep 8 14:15:32 2022 +0200

    Merge pull request #51 from 0xPolygonHermez/move-aggregator-addr
    
    move aggregatorAddress to publicInputs

commit 85aab69a8845a33e50be516e363eed5e187d328d
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Sep 8 11:59:08 2022 +0000

    move aggregatorAddress to publicInputs

commit 6a05efcdcc08f9ed3b4e823285199abdb58af88e
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 8 11:55:55 2022 +0000

    Update input and public input comments regarding aggregatorAddress field

commit 9cc99757450c9584fb6e2dc2ed8b8820755557ad
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 8 11:45:50 2022 +0000

    Remove memory and process traces from Prover::prove(), used for debugging memory leaks

commit 81fc5ae51ff2d3453e27de3af07b1c6cb271f773
Author: fractasy <fractasy@gmail.com>
Date:   Thu Sep 8 11:31:17 2022 +0000

    Fix README.md with git submodule update (without --remote)

commit 5d0f731999d070ffbf1c85c08e63cbfd8feae281
Merge: 9b564d9 2e7fe9e
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Sep 8 13:12:06 2022 +0200

    Merge pull request #50 from 0xPolygonHermez/add_opts
    
    Add opts

commit 2e7fe9e53f01465836a7d5e2e1c3bdb6edeb6ab1
Author: Eduardo Antuña <eduadiez@gmail.com>
Date:   Thu Sep 8 10:22:38 2022 +0200

    add avx2 flag

commit f89c1bf309658fdb789efa06b20ef15b52881512
Author: eduadiez <eduadiez@gmail.com>
Date:   Wed Sep 7 15:59:25 2022 +0000

    add new goldilocks

commit f4a1e0da0ce6dd9a18e8ba7c74d43415837f9bb7
Merge: 3b87d23 f3058d9
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Wed Sep 7 15:05:12 2022 +0200

    Merge pull request #49 from 0xPolygonHermez/fix-transaction-hash-data-0
    
    fix bug with data 0x00 (1 zero byte)

commit f3058d99ca134259f2d6d848dbc1da1964495053
Author: zkronos73 <zkronos73@gmail.com>
Date:   Wed Sep 7 14:31:02 2022 +0200

    fix bug with data 0x00 (1 zero byte)

commit 9b564d9e306a3619cfef067758c5cd8ec402387d
Merge: a820a99 97d39cf
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Tue Sep 6 17:52:39 2022 +0200

    Merge pull request #48 from 0xPolygonHermez/Opt-pilstark
    
    Opt pilstark

commit a820a99d0fdb899b7b318f743572776919bf087c
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Sep 5 09:54:45 2022 +0000

    fix memory leaks

commit 3b87d23a05105d0e622008e5dbf8970aa4966830
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Sep 5 08:43:04 2022 +0000

    update rom.json

commit 06abbb98cf08b0a72e8b8b307378599e5828f767
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Sep 5 07:30:34 2022 +0000

    move aggregatorAddress to publicInputs

commit 97d39cf9cd44f4691ca8e36b9f7d0c6eeedb430c
Author: rickb80 <rickborrell@gmail.com>
Date:   Wed Aug 31 13:18:47 2022 +0200

    CALCULATEH1H2 optimized

commit a6af21a67d0a17eed1966335299a677819e2f2e5
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Aug 31 11:14:10 2022 +0000

    Handle ooc as a response error

commit 2489de199677805554577932b92f4e28ee7a1c8b
Author: rickb80 <rickborrell@gmail.com>
Date:   Wed Aug 31 12:50:27 2022 +0200

    Minor change

commit 6fd73d59c69aa98037bfec1abff4e2700ebf5706
Author: rickb80 <rickborrell@gmail.com>
Date:   Wed Aug 31 12:21:55 2022 +0200

    Optimization CALCULATE_Z

commit a14f80183bfa85052a3781fa88948d3407a57435
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Aug 30 15:56:26 2022 +0000

    Debug second execution

commit 6887fb1aa09cad3a5c9b28a3a1c700945967a2ab
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue Aug 30 15:23:49 2022 +0000

    Remove expansion of ooc error. Don't load startVerifierFile if proof generation is not needed (runProverServer/runFile=false)

commit 26f91e94121522610fb8c79d7c1c86b3e284419c
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Aug 30 11:55:03 2022 +0000

    fix initialization

commit 6c83f6fbd455fa765897d92596929562feb7250c
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Aug 30 10:40:41 2022 +0000

    Initializate transcript on genProof

commit 88dfa0a6cbf267391f3db6a429904dda855d9e46
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Aug 30 10:05:56 2022 +0000

    Fix vars initalization

commit d9ab21248ba12e65d263ac615a8bf79f23fff049
Author: rickb80 <rickborrell@gmail.com>
Date:   Tue Aug 30 10:04:17 2022 +0200

    Optimized EVAMAP

commit 972377561006cdcc2ee2cccdccf49dae84732a7a
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Aug 29 14:34:46 2022 +0000

    pint meminfo and processinfo when starting and finishing the prove

commit f3f1e16ec05be3cdb8fb7d3bedc52e9a9f9598be
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Aug 29 14:32:23 2022 +0000

    pint meminfo and processinfo when starting and finishing the prove

commit 590b3a45494451d214493c40fc85c5bbc92510b1
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Aug 29 14:27:32 2022 +0000

    add debug info

commit b6d5675ea94b76f3178b03df6c79acdfc9e62ec1
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Aug 29 12:05:54 2022 +0000

    update debug traces groth16.c.hpp

commit 9389c911cc7e984403892c4c2cd0e17b7603889d
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Aug 29 11:59:50 2022 +0000

    add debug traces to groth16.c.hpp

commit 3716936263d36a3c9483f4f7e129cec6a6fc9552
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Aug 29 09:22:27 2022 +0000

    add debug traces to groth16.c.hpp

commit b4dadbaaa43667408169be7e7fd0d606d37e1045
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Aug 29 08:39:34 2022 +0000

    updated rom.json

commit c2dbdc8e775e1e5b5e85cbacede7b51ae55ca1c9
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Aug 25 15:55:19 2022 +0000

    update config.json default values

commit 0f9083d2369147096778b59855aa1d5ef87f2ace
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Aug 25 15:46:36 2022 +0000

    Add error at response level if no steps in trace

commit ec89ab31ac9151f230b82eb9e04853538d4adf30
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Thu Aug 25 15:35:16 2022 +0200

    Update README.md

commit d704a8730b2bac31321527c8089759f7d6b71d52
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Wed Aug 24 18:10:15 2022 +0200

    Update README.md

commit 190fba16f39e74b405140e48ebeed32380071227
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Wed Aug 24 17:34:44 2022 +0200

    Update README.md

commit 991559ce6e5d3ec9dc018f03683c758b40eff3d0
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Wed Aug 24 16:46:24 2022 +0200

    Update config.json

commit 5198ea93819cd8deaefe3153b114109f7c0602d2
Merge: 3bb60fe 6e858a6
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Aug 19 21:52:16 2022 +0200

    Merge pull request #46 from 0xPolygonHermez/pilstark-integration
    
    Pilstark integration

commit 6e858a6c34e28f49bd138c35c8c3e23808e12a95
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Aug 19 19:51:20 2022 +0000

    Working version

commit 06bf55eb8beb4d250080605abedc533b48b8a039
Merge: e7298ef 3bb60fe
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Aug 19 17:22:47 2022 +0000

    Merge remote-tracking branch 'origin/main' into pilstark-integration

commit e7298ef538793b0916de240d02dda3085fe175f5
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Aug 19 17:21:48 2022 +0000

    working version

commit 38c405b1104c76da5c27417f8729c1ae19741ae3
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Aug 19 17:21:00 2022 +0000

    fix zkin.proof.json

commit 3bb60fe69eab577bcef7ae881d9a58b9efeb7835
Author: fractasy <fractasy@gmail.com>
Date:   Fri Aug 19 14:19:52 2022 +0000

    Fix some compilation warnings and errors after the merge

commit 0f30c93d2db187aeaeaf708bcf49747f6752ee53
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Aug 19 10:20:09 2022 +0000

    add prover integration

commit bd8a73b969f4ccf62418404d599dbeb3d3491287
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Aug 19 10:19:22 2022 +0000

    Add C12

commit 579902991a41e31337e75109540a7f297d9fab45
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Aug 19 10:18:41 2022 +0000

    add sha256

commit 5af8773bca8f4187054560735a4648483b7dde7e
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Aug 19 10:18:30 2022 +0000

    add zkevm_c12_verifier_cpp

commit bf420540df45966c616ed0e2951246b4e7bb53a7
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Aug 19 10:17:43 2022 +0000

    Add calculateExps

commit 391d43dfe69b20f481ed4642517b8aa8f978f18a
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Aug 19 10:06:34 2022 +0000

    Add poseidon BN128

commit 3f1c8f64f08f6f544201cabf7cea4f8e339364a7
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 18 18:53:05 2022 +0200

    add verifier and witness computation

commit 412ac80767aa0217fcfbfc2ffd2f5cb2b609a725
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 18 16:58:43 2022 +0200

    remove all verifier

commit 2cd1c9cf174c778b7fdf48adb5ebb5ac4b2d2a1f
Merge: ab95e87 6e46b98
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 18 16:36:31 2022 +0200

    Merge remote-tracking branch 'origin/main' into pilstark-integration

commit ab95e87c1887d36f58c1399a0b94851585286e7a
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 18 16:29:49 2022 +0200

    update output files

commit 42b115473f2ed60bdaacf56311bda0f91ba5beb9
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 18 16:29:33 2022 +0200

    fix FRI

commit 7210acf7ac46bfeac0abdf6b4cebb1a166971efa
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 18 16:29:09 2022 +0200

    update basic calculate exps

commit 488a91d0e2430c36773ed2ea9f12534527ebe481
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 18 16:27:06 2022 +0200

    Update basic test

commit 6e46b985358bc540e5a519669be16566a2a124a4
Author: fractasy <fractasy@gmail.com>
Date:   Thu Aug 18 13:30:11 2022 +0000

    Add config.json parameters to configure prover mock timeout and enable executor ROM lines traces

commit 54bcd33bda25feb41d25be25e4597f41f7485ac7
Author: fractasy <fractasy@gmail.com>
Date:   Thu Aug 18 11:53:45 2022 +0000

    Update config.json file names to the new name convention

commit 5793b56630895bd3dddfe3b83872957927179eb4
Author: fractasy <fractasy@gmail.com>
Date:   Thu Aug 18 09:14:34 2022 +0000

    Fix src/starkpil/test/starkMock.cpp compilation after the big merge

commit a5010db25186cfabb8a6ff015cd0da4dfde628cb
Merge: 7e66a0e 2db108e
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Aug 18 10:46:29 2022 +0200

    Merge pull request #45 from 0xPolygonHermez/pilstark-integration
    
    Before step 5

commit 2db108e49ed45746dce58e70a52b4b5ae0446535
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 18 07:56:14 2022 +0000

    Before step 5

commit 7e66a0e9f0113d832a9440e53303ac94580558c4
Author: fractasy <fractasy@gmail.com>
Date:   Wed Aug 17 10:58:10 2022 +0000

    Update files from orange release tag

commit c16cd33ebfaeca7cafa06cf64b3871b06173a212
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 16 17:11:32 2022 +0000

    Load a default Input::aggregatorAddress when file does not contain it

commit aba6f2d3b21293771a5b392c4c407c9c18418eb1
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 16 17:05:43 2022 +0000

    Pass aggregator address in starg.genProof() call

commit 8e6b2211b2b58bfd30362b914722296af86629c9
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 16 16:43:57 2022 +0000

    Fix main SM executor generated code counters and rest of polynomials evaluations; now it matches JS and original C code

commit f5f86ca9921330e5730979b83c02c582608f7905
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 16 14:42:30 2022 +0000

    Remove unprocessed_transactions from FullTracer.  Update main SM executor generated code, now functional.

commit 3287fb1e233cf7ce7d98367c2cbae9dbaa39c6ae
Merge: 692ef73 73d0167
Author: fractasy <fractasy@gmail.com>
Date:   Thu Aug 11 10:14:40 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit 692ef7381686a757dd27a3e216acae8bbb0258bb
Author: fractasy <fractasy@gmail.com>
Date:   Thu Aug 11 10:14:37 2022 +0000

    Remove USE_LOCAL_STORAGE code, now unneeded. More main SM executor generated code changes.

commit 73d01679e3014242ea467869125ef4189e1b54db
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Aug 10 21:42:31 2022 +0000

    change to be optional the load of the params "form" and "aggregatorAddress" from the input file

commit 71737ac83bc8f5d3630cfc4af13aa8a47d8a68c9
Merge: 9302c94 d488a24
Author: fractasy <fractasy@gmail.com>
Date:   Wed Aug 10 16:57:17 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit 9302c94a6fadc3c2db4dbbd95f1a513a0a99b0bc
Author: fractasy <fractasy@gmail.com>
Date:   Wed Aug 10 16:57:07 2022 +0000

    More work on main SM executor generated code

commit d488a249f5eda5c0d31a2edffa58007b258713c7
Merge: 6c0d09c 38a33df
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Aug 10 09:59:25 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit 6c0d09c15b98c26e727b4e54b6d8f5dfc591ec5d
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Aug 10 09:53:17 2022 +0000

    Added "from" and "aggregatorAddress" properties to the input json file (save/load)

commit 38a33df47ecd0cd5cddd98f00f22725f90417683
Author: fractasy <fractasy@gmail.com>
Date:   Wed Aug 10 09:41:52 2022 +0000

    Fix (again) the FullTracer::onEvent() trace; which was not reached in the intrinsic invalid case

commit 5864ae76974ded88ef5d07143e34b75576e8a5b0
Author: fractasy <fractasy@gmail.com>
Date:   Wed Aug 10 09:12:23 2022 +0000

    Improve FullTracer::onError() trace

commit 04001f6bbe7098cabb59476abf5f9359c0fa0f32
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 9 17:05:43 2022 +0000

    Add missing selectors to main executor generated code

commit 09013c2f07be8eb820aac55f2958f6f54c1768ee
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 9 15:12:15 2022 +0000

    Update main generated code, and enable its compilation, still in -O0 mode

commit c3458e1de33c3ac6a794434d347079ecac84fea4
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 9 14:45:58 2022 +0000

    Add a tracer to FullTracer::onError() for the intrinsic_invalid case

commit aebc4521bd1f914ce849b3d2b47f7325ac63dc09
Merge: b96690a c17fd4e
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 9 06:48:04 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit b96690a5e492f6f4e1aabc33f73c6b9e2dd746eb
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 9 06:47:50 2022 +0000

    Add fake proof to Prover::proof()

commit c17fd4e8275336d4209dbc8e55375bb5aad872e4
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Aug 8 14:40:34 2022 +0000

    Added zeroTozero tests for empry tree and sibling=intermediate node. Fixed JMPZ to use only Op0

commit 61cca823da601178936fa96a67dca1107db24460
Author: fractasy <fractasy@gmail.com>
Date:   Mon Aug 8 10:37:47 2022 +0000

    Fix rlp_tx assignment in executor service

commit 8822de0bceb1693a7ad87ec1db98670541f57227
Author: fractasy <fractasy@gmail.com>
Date:   Mon Aug 8 10:14:51 2022 +0000

    Change executor.proto to have stack as a vector of mpz_class, instead of u64

commit ba30e698d5eccd36ce73590eb8141c013d9867ee
Author: fractasy <fractasy@gmail.com>
Date:   Mon Aug 8 09:07:55 2022 +0000

    Manage database errors at the evalCommand() level, and stop execution if they happen

commit a5c325f09ef9aff697db2e016fc874a6ce858127
Author: fractasy <fractasy@gmail.com>
Date:   Mon Aug 8 08:32:47 2022 +0000

    Return data and return_data in executor service

commit 93b8c3501dbdd52c67a3971e54f1ce4a7f9ebe6c
Author: fractasy <fractasy@gmail.com>
Date:   Mon Aug 8 08:08:05 2022 +0000

    Fix gas_cost implementation (pending validation with failing test)

commit b4ce6019ce0f2d4cccfe58e3e65f6912c7213435
Author: fractasy <fractasy@gmail.com>
Date:   Mon Aug 8 07:33:41 2022 +0000

    Return stack in executor service call_trace.steps

commit 87c202f128b845a336732d1f049f35a93433ba76
Author: fractasy <fractasy@gmail.com>
Date:   Mon Aug 8 07:17:19 2022 +0000

    Return an error in prover service when db value length is not a multiple of 16

commit c78c4b20b258a74d0efe923fce8f5ff17ced16e9
Author: fractasy <fractasy@gmail.com>
Date:   Mon Aug 8 06:55:11 2022 +0000

    Update storage_sm_rom.json

commit e8853f9326e73aef58427b17dcdc1d348fab4157
Author: agnusmor <agnusmor@gmail.com>
Date:   Fri Aug 5 19:38:48 2022 +0000

    fix initialization of inROTL_VH=false in StorageRomLine class

commit cbe438504f98862db25750cf4e6244c956b6e9cb
Author: fractasy <fractasy@gmail.com>
Date:   Fri Aug 5 17:51:26 2022 +0000

    Update rom.json

commit 696b9a5169ff8746b8d70bb17593bedafc21bc5c
Author: fractasy <fractasy@gmail.com>
Date:   Fri Aug 5 17:34:41 2022 +0000

    Update generated polynomials files. Fix executor error string instrinsic_invalid.

commit 4918aa717e1e5edc8883878b4851219a223d7ae5
Author: fractasy <fractasy@gmail.com>
Date:   Fri Aug 5 14:19:20 2022 +0000

    Update rom.json

commit d0b9b86a5269cba118f939f0a7600a5fcfaa6500
Merge: aa05dca 877a23a
Author: fractasy <fractasy@gmail.com>
Date:   Fri Aug 5 14:05:46 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit aa05dca72f0937218aa8fc110568c04edf856845
Author: fractasy <fractasy@gmail.com>
Date:   Fri Aug 5 14:04:48 2022 +0000

    Implement rlp_tx in executor service.  Fix bUnsignedTransaction assignment.

commit 877a23a07337228583d01f68c258e8ef39b636c4
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Fri Aug 5 11:42:02 2022 +0200

    Update README.md

commit 7e34e557a752aacffa7ca49b7095a24629c616f6
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Fri Aug 5 11:39:54 2022 +0200

    Update README.md

commit 43cf8ef476f6b8563d0067b03fed92b8add882f9
Merge: 774ac41 617eb73
Author: agnusmor <agnusmor@gmail.com>
Date:   Fri Aug 5 09:25:46 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit 774ac41f6115bf1e487fac02d6169565cca7870a
Author: agnusmor <agnusmor@gmail.com>
Date:   Fri Aug 5 09:25:21 2022 +0000

    fix thread-safe (lock) implementation of StateDB class

commit 617eb73e5361a93bcf2d71a7ee95fd7b701f0b0d
Author: fractasy <fractasy@gmail.com>
Date:   Fri Aug 5 08:44:59 2022 +0000

    Remove Main SM executor fast mode; using process batch mode instead

commit 25b8caf7bd3aedf9262cc4d2a8bca8fa6acf9a79
Author: fractasy <fractasy@gmail.com>
Date:   Fri Aug 5 08:12:20 2022 +0000

    Implement unsigned transactions with input.from.  Skip asserts.  Fix STARK compilation.

commit 9baeff1b785f3e5c0c2c43073370fcbdb29dca38
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Aug 5 08:01:46 2022 +0000

    remove _all files

commit 94d4471b198db265f1771d26e2cbd73b37e54466
Merge: 0fe461a 3ae3f3e
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Aug 4 21:19:17 2022 +0200

    Merge pull request #30 from 0xPolygonHermez/feature/starkpil
    
    Feature/starkpil

commit 3ae3f3e4a6c6cb1b7c05be3e417b8785d2f95f54
Merge: ff0c014 0fe461a
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Aug 4 21:18:19 2022 +0200

    Merge branch 'main' into feature/starkpil

commit ff0c01410f436218e0436a0854ca66670667c834
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 4 19:17:18 2022 +0000

    Add step 5

commit 3106ca156ce51b2e4d8c5ae3306f46bfef72501c
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 4 19:16:45 2022 +0000

    remove all example and add basic example

commit 1ae92cd729f1d475015f4a62356ad8a0a3c4468a
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 4 19:15:46 2022 +0000

    add basic.commit

commit 356d7a5133173a4025315fea5e6a6eb10bfb6adb
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 4 19:15:21 2022 +0000

    add lfs for basic.commit

commit 4bd2216ec8d6c0dbbf6397efe8b58a0f29ef2680
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 4 17:40:25 2022 +0000

    delete fibonacci files

commit 1a14a60e40e654a86d54723902ec1b130d1e8d6e
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Aug 4 17:38:06 2022 +0000

    Update goldilocks library

commit 0fe461a68a49f24d10dc69c0a0c6d0190424a3d7
Author: fractasy <fractasy@gmail.com>
Date:   Thu Aug 4 11:51:24 2022 +0000

    Fix carry in SLT

commit 97ef1b4ca3195b1e626cdad6e601fc1edfafbe08
Author: fractasy <fractasy@gmail.com>
Date:   Thu Aug 4 11:21:43 2022 +0000

    Fix Main Executor SLT

commit f3d952154b46ab0e250ac80cfd6ce47d319be7b8
Author: fractasy <fractasy@gmail.com>
Date:   Thu Aug 4 10:27:10 2022 +0000

    Fix memAlignWR8 in Main SM executor and eval command

commit 7907293ffd99cd9d7bb74b8b6cc93d402811f4e4
Author: fractasy <fractasy@gmail.com>
Date:   Thu Aug 4 09:43:51 2022 +0000

    Fix eval_getReg() of register GAS

commit 3381821780b9f372be1ebf5ec512dff5facfcd4a
Merge: 8c3e5e8 d5e81a0
Author: fractasy <fractasy@gmail.com>
Date:   Thu Aug 4 08:17:52 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit 8c3e5e8e2897435266315197746412ea339667fc
Author: fractasy <fractasy@gmail.com>
Date:   Thu Aug 4 08:17:42 2022 +0000

    Protect prover service code from input data that is too long

commit d5e81a08bcc7d437644dc25c7fe5e9226faabeaa
Author: eduadiez <eduadiez@gmail.com>
Date:   Wed Aug 3 15:30:13 2022 +0000

    Update ubuntu version in docker to LTS

commit 69c198a383610aaad13e0cb58863c4136cfd2f25
Author: eduadiez <eduadiez@gmail.com>
Date:   Wed Aug 3 15:29:03 2022 +0000

    Update ubuntu version in docker to LTS

commit 23095634ffeb7d3544ce239a20aa89aba39923d7
Author: fractasy <fractasy@gmail.com>
Date:   Wed Aug 3 14:40:39 2022 +0000

    Add intelligence to prover service mock.  Now, the proof will pretend to take 1 minute to generate, and after getProof will succeed.  During the computation, cancelProof will succeed.

commit 784e3c16e5658856ff0aae865bb8d7d40c290ce8
Author: fractasy <fractasy@gmail.com>
Date:   Wed Aug 3 14:15:35 2022 +0000

    Update executor.proto and zk_prover.proto.  Added aggregator address. Using ERROR_NO_ERROR in executor service.

commit ad9894da24fef781dfeae8c2064275975c001ff0
Author: fractasy <fractasy@gmail.com>
Date:   Wed Aug 3 10:21:50 2022 +0000

    Add testvectors/run script to execute the prover in the test servers

commit 2c1ad5c07ee88801e6e259b64301949293c0b04c
Author: fractasy <fractasy@gmail.com>
Date:   Wed Aug 3 09:44:02 2022 +0000

    Fix FullTracer::getTransactionHash() encoding length

commit 7941c8d31a362fdb3ecf3946c7a3f889059ce071
Author: fractasy <fractasy@gmail.com>
Date:   Wed Aug 3 09:01:45 2022 +0000

    Fix FullTracer calldata string when data is an empty array

commit 2950af4003d6c5b9e85b05ac9d1d9513a0e255c8
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Aug 3 08:24:58 2022 +0000

    added missed "endl"

commit ff42e9de60ce2dd21c7ea6b6fc2b733ac3c8dac5
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Aug 3 08:22:27 2022 +0000

    fix set "persitent" parameter in SetProgram GRPC request

commit 26a9b53b4cf4c1e6b3361eb4a88aabecc09c17e5
Author: fractasy <fractasy@gmail.com>
Date:   Wed Aug 3 07:46:14 2022 +0000

    Improve eval_getBytecode() error trace when calling database

commit 7682aa0d6ab75ebfc043efbd44bf71c6e590d0d6
Author: fractasy <fractasy@gmail.com>
Date:   Wed Aug 3 07:29:31 2022 +0000

    In eval_getBytecode(), when contract is not found locally, try to get it from the database

commit 0025b314baf105d87ed22ab76b931b8c8e76a661
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 2 15:44:00 2022 +0000

    Use Input::contractsBytecode in eval_command.cpp

commit 7d364d942ca6e3e31ddae592abaa4acc4cc00471
Merge: e06e0eb bf5a4d8
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue Aug 2 14:59:26 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit e06e0eb7fd9c11543f69466e4c71fd65406a448a
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue Aug 2 14:58:58 2022 +0000

    added testvectors/input_contracts.json test file

commit 490308f4aa29454287634edb4b5572d2ffc4abe4
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue Aug 2 14:58:24 2022 +0000

    Support contracts_bytecode in prover and executor services

commit bf5a4d82817b33c108ab612ffe8d7cab08524ed7
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 2 14:27:15 2022 +0000

    Refactor of FullTracer::getTransactionHash() to perform v transformation out of the method

commit 6d5395eb998ccdf3443dac666066c1781934e202
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Aug 2 13:49:54 2022 +0000

    config files

commit 60923884cfd7e7dd37b20af53b908d1e22fcc2f8
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Aug 2 13:49:19 2022 +0000

    After merkelizing 3

commit a2f56a2d4ce00cecd1643fbdbb379b3975a6d4fd
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 2 13:48:16 2022 +0000

    Update rom.json, remove tolerance to precompiled_* function names in rom commands

commit 14ced27799a2bc35dec5baf5b71e5ec87d52d6f5
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Aug 2 13:47:39 2022 +0000

    Add all calculate Expresions

commit 96bdd375f6515703de970e1e5369450e523e0780
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 2 08:51:35 2022 +0000

    Update rom.json: remove path from fileName, and fix lineStr to match the file

commit bad8f74566f2b057ca573f335c1755d1e2104e18
Author: fractasy <fractasy@gmail.com>
Date:   Tue Aug 2 06:18:47 2022 +0000

    Fix StateDBServiceImpl::Get/SetProgram() traces

commit d8feff3f1f8873623c9b34a72772ece1f3346dcc
Author: fractasy <fractasy@gmail.com>
Date:   Mon Aug 1 15:10:25 2022 +0000

    Fix eval_touchedStorageSlots() arguments type check

commit 6604278db0f74b60f84961bb522fb07e4aba5c19
Author: fractasy <fractasy@gmail.com>
Date:   Mon Aug 1 11:44:09 2022 +0000

    Fix Storage SM execution, mismatching old root

commit f73e3f6b9708145ecd19cdaffe1c454e2111a1a6
Author: fractasy <fractasy@gmail.com>
Date:   Mon Aug 1 09:16:50 2022 +0000

    Port changes from JS to Storage SM Executor: do not use insKey

commit def334b6fe44db0ed25dccc32e5d08f3f96c6d15
Merge: d696417 bfa43e1
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Aug 1 07:14:45 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit d696417b1c91dd85e0badfec26258d12aec2ffb9
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Aug 1 07:14:26 2022 +0000

    StateDB services log traces has been simplified. Modified SMT and Statedb service to return zkresult

commit bfa43e11fc386c3cd36eee3692244c912e889e38
Author: fractasy <fractasy@gmail.com>
Date:   Mon Aug 1 06:46:01 2022 +0000

    Protest FullTracer::onOpcode() from incorrect parameters

commit c20cab3c835ba159b5e67f66603c10d8dc0d4b99
Author: fractasy <fractasy@gmail.com>
Date:   Mon Aug 1 06:45:23 2022 +0000

    Add toString() to SMT get/set result, and SMT action

commit 73c24eae48b033cc635976bc69f41aae5084fa2e
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Jul 30 11:05:59 2022 +0000

    restore config.json

commit c95b38414566f360df89c85a407f6dfc0194a474
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Jul 30 10:50:47 2022 +0000

    remove unnecesary gitignore

commit dfc9dfefc347fe3258f3fcf918acdeaac7ae6ba6
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Jul 30 10:44:09 2022 +0000

    update submodule

commit ba800ec44fbe81e900f0be0f4513609b32e70e46
Merge: 2037608 3b22e4f
Author: eduadiez <eduadiez@gmail.com>
Date:   Sat Jul 30 10:22:06 2022 +0000

    Merge remote-tracking branch 'origin/main' into feature/starkpil

commit 3b22e4f3921978ad6a416a75344e53ff87c78930
Merge: 0d69b8e 86a6327
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 29 17:00:42 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit 0d69b8e1933139c96a3c1739e4a97d8ec74b2dc6
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 29 17:00:38 2022 +0000

    Fix FullTracer::onOpcode() protecting info size access

commit 86a63272b91c51c2c9dbb8f18b5f4600956b3693
Merge: 046d019 ccd6ba9
Author: agnusmor <agnusmor@gmail.com>
Date:   Fri Jul 29 16:57:06 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit ccd6ba99bc94d483038c1b535b834aee872b6f02
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Jul 29 16:12:23 2022 +0000

    adaptation to goldilocks as a submodule

commit 203760824c365014bfbe5aee928fe6168176d8f1
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Jul 29 13:19:43 2022 +0000

    updated my submodule

commit 046d0199620ec27401112f17b3ed22e2facb2542
Author: agnusmor <agnusmor@gmail.com>
Date:   Fri Jul 29 12:51:38 2022 +0000

    removed exitProcess call added for testing

commit 673f16d02b69f2e4521a47bee5508a04f5448c41
Author: agnusmor <agnusmor@gmail.com>
Date:   Fri Jul 29 12:44:47 2022 +0000

    changed evalCommand to use switch/enum instead of if/else string compare to improve performance. Renamed StateDB classes to better match reality. Added mem and process info to exitProcess function

commit 4b3ed0f4bf6bd9670abdfd81d66d9b96864d9675
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Jul 29 12:38:18 2022 +0000

    add goldilocks as a submodule

commit 7765e7f0330e54894b10f13c0fc2424d05e256b6
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Jul 29 11:31:14 2022 +0000

    remove old goldilocks files

commit 1f6983ffaf0f896b3fb3015303091d36a3c60136
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Jul 29 10:22:07 2022 +0000

    copyElement

commit 7be59f3286061d45aca84b30ce9e6f7f6f163f3e
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 29 10:11:21 2022 +0000

    Fix OOC condition in Main SM Executor: toS32() check

commit 8e88b1a57bbdc21be44fcc7ec5f418eacad6d697
Author: agnusmor <agnusmor@gmail.com>
Date:   Fri Jul 29 08:52:33 2022 +0000

    added proof_hash_counter property to set/get responses of the StateDB GRPC service

commit 7c44e298057150941270f015d8a85f93e818b390
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Jul 29 07:35:45 2022 +0000

    add calculateExps step2prev

commit cdc1c0bff2e9a5fc2cf68c4bef5c59c55cb22ed0
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 29 06:49:42 2022 +0000

    Undo Main SM Executor fix: use toS32 again, and investigate the OOC condition

commit b8b40e2783a448ded845e671f83581f6ae2d151e
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 28 16:26:54 2022 +0000

    Move calls to exit(-1) to exitProcess()

commit c74d9a9f2230a0af75bbbd06195f390986aa23fe
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 28 15:29:27 2022 +0000

    Undo clearing error in onProcessTx() and remove calls to exit(-1); use exitProcess() instead.

commit 6bba77cfd246ad2b0a6c78523c8bbbaed1b1b6b7
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 28 14:50:06 2022 +0000

    Protect executor service processBatch() from input data with invalid size

commit ea5ec88d505708ad41a9174e7b7a9a9dd8cc4c5b
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 28 14:28:29 2022 +0000

    Fix Main SM Executor JMPN to use toS64().  Avoid exit(-1) in Main SM Executor.

commit eec41b264b0f7903d735248a6f2608c868ddd68e
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 28 07:56:42 2022 +0000

    Update executor.proto with: from, tx_has_to_generate_xxx, Error

commit e835e8b79647e9b3b984059321f71a083154c6a5
Author: eduadiez <eduadiez@gmail.com>
Date:   Wed Jul 27 17:54:38 2022 +0000

    disabled cmPolsFile

commit 5a77b3c3436be65c6b1b90b2156aae39cd5c4d27
Author: eduadiez <eduadiez@gmail.com>
Date:   Wed Jul 27 16:34:13 2022 +0000

    Merge remote-tracking branch 'origin/main'

commit 4d629df0b5e7735d80cc38da79909256cdeda738
Author: eduadiez <eduadiez@gmail.com>
Date:   Wed Jul 27 16:31:22 2022 +0000

    all test re-implemented

commit 2fc8a6b479ffe4a3a340af196975bab22a4da918
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jul 27 15:31:59 2022 +0000

    Fix input batchL2Data in prover service

commit 840168c4a3f3511ee8a8be71f8df9d792626394d
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jul 27 14:07:56 2022 +0000

    Improve Main SM Executor print ROM line trace with file name and line

commit a282d52d49e6f83c8fe5727bcecb000e345ccd0e
Merge: c58abf5 361c547
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Jul 27 12:09:22 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit c58abf594202fb159e90bbe81d6b419de721fce3
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Jul 27 12:09:13 2022 +0000

    changed eval_cuntionCall to use switch/enum instead of if/else string compare to improve performance

commit 361c547ed4d5b58bcca55d614487cfa1a69972b4
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jul 27 09:15:43 2022 +0000

    Create exitProcess().  Remove Database dbNew and dbRemote.

commit 7a6c7f80dcdc4fd537c67b1ec30c319152dde042
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 26 10:17:32 2022 +0000

    Fix Main SM Executor assert trace

commit 5fcc6984826e6a5bd4e97634f54365aa79e18320
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 26 08:21:28 2022 +0000

    Add register RR to FullTracer::getRegFromCtx()

commit b2b3722d8126d80424848665f9ccad99c99bd719
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 26 07:50:04 2022 +0000

    Port evalCommand and FullTracer changes from JS, update rom.json.

commit 37da410ab67badea124fb094ce5ca00cbbe2a94b
Author: eduadiez <eduadiez@gmail.com>
Date:   Mon Jul 25 20:04:38 2022 +0000

    Add new testvectors

commit f66c32f24d98e2f1f64b655f150ce7ed6d3f0e0e
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 25 14:17:24 2022 +0000

    Port Storage SM changes regarding GET of value=0 2 sub-cases

commit d27c4a295a766b18ce66090cc397bb202bc7aa46
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 25 13:39:13 2022 +0000

    Add step and rom line string to Main SM Executor assert trace

commit 608e03cbc468fc048f7a3443eadbfbce57e3f995
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 25 12:55:06 2022 +0000

    Add file name to main SM executor assert trace

commit 49a8f03d33cd3afeea0963f87f44e0b7c1b51296
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 25 11:49:02 2022 +0000

    Code cleanup: delete commented code in input and main executor

commit 314bf6d49761292ae5893c5f7902df40b7302559
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 25 11:46:09 2022 +0000

    Add grpc request trace to prover genProof() method

commit 00bad448aae340adfe61c8b846110d2f4ae40c44
Author: fractasy <fractasy@gmail.com>
Date:   Sun Jul 24 12:41:19 2022 +0000

    Cleanup unused main ROM elements

commit be9070edb97c5c349dc46cc497cd56419fe9cf71
Author: fractasy <fractasy@gmail.com>
Date:   Sun Jul 24 11:58:34 2022 +0000

    Cleanup ecRecover code in Main SM Executor

commit 8964578f7cb745f20e72a6ccee9445d16488d6d4
Author: fractasy <fractasy@gmail.com>
Date:   Sun Jul 24 11:39:52 2022 +0000

    Cleanup some Prover code

commit 46072e0b0f5754c472fe8ae5670115449936287f
Merge: a0fb8be ce51377
Author: fractasy <fractasy@gmail.com>
Date:   Sun Jul 24 11:14:45 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit a0fb8beddee0fb9d62628b50eddd132b26746bb3
Author: fractasy <fractasy@gmail.com>
Date:   Sun Jul 24 11:14:34 2022 +0000

    Move fec and fnec to MainExecutor attributes list

commit ce513779722c8aee5eb2b24a7db7744328e59b8a
Author: zkronos73 <zkronos73@gmail.com>
Date:   Thu Jul 21 08:18:39 2022 +0200

    fix problem Dockerfile

commit e75f73cc33b19877051eb013848329dba6561c83
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Wed Jul 20 10:54:17 2022 +0200

    Update README.md

commit ecf265b5b613765b62a6638fff432fdb22698ea3
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Wed Jul 20 09:54:32 2022 +0200

    Update README.md

commit 89371d97d5be55f9aced8564977b6eed8ba13379
Merge: 365391d b387d5d
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Tue Jul 19 10:12:50 2022 +0200

    Merge pull request #15 from 0xPolygonHermez/fix-transaction-hash
    
    change how calculate v on getTransactionHash

commit 365391d8084296d0884fbb66f71fb54e452859d7
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Tue Jul 19 10:06:33 2022 +0200

    Update README.md

commit b387d5dbaab33059877eb951f02ac984e76e6764
Author: zkronos73 <zkronos73@gmail.com>
Date:   Tue Jul 19 09:14:17 2022 +0200

    change how calculate v on getTransactionHash

commit 3c79c99bf8f97c18bfb4b2b3bd1ef5626d8ca09c
Merge: 7853a5d 8e2c107
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 18 10:16:38 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit 7853a5dc57330aff2e2ff2a30b2ab30988245be3
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 18 10:16:28 2022 +0000

    Add gasProce to FullTracer::getTransactionHash() trace

commit 8e2c107faf47d54f7af33a273dccf9d3e48e6a5f
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Fri Jul 15 21:25:59 2022 +0200

    Update README.md

commit 863b40dfbd35f18128464182cffaa9d4973d14dc
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 15 18:21:51 2022 +0000

    Add Stark info to README.md

commit 798cdb78c030cbd84c2012f767019795ece04f36
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 15 18:09:38 2022 +0000

    Add general description to README.md

commit faec7d35ead10170c00341e6858ec2bffcfcfcd6
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Jul 15 18:48:47 2022 +0200

    start_test refactor

commit a7c6835a2f5839d197c0a27efec4957a9a7e0d8b
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Fri Jul 15 18:35:33 2022 +0200

    Update README.md

commit 154521977eae41e379694e7d0292b440bdb2ea8f
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 15 16:26:35 2022 +0000

    Use Database only through StateDB interface

commit 8d9907272667562824a39a3c9016daa3c8402e4c
Merge: 01dc03e 5d26599
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 15 15:32:04 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit 01dc03e2cd1247d7ba054b6ff6778e9e5d305f9e
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 15 15:31:27 2022 +0000

    Add traces to FullTraces::getTransactionHash()

commit 5d26599de90233d6a050c7d15f9613fed5508ccf
Author: agnusmor <agnusmor@gmail.com>
Date:   Fri Jul 15 14:21:03 2022 +0000

    Added delete_db.sh script to delete StateDB database objects. Fixed permission issue when creating schema/table in create_db.sh script

commit 145ab80e7d607a538b11ca2387ef4ba8af86e0b7
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Jul 15 11:47:45 2022 +0200

    remove output_tmp

commit ae640a73852838b49a723d560c927853c0026e60
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Jul 15 11:17:49 2022 +0200

    H1H2 computation

commit d32bde4ed090a7bf1ab6352866ae58d5a13542a5
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Jul 15 11:13:37 2022 +0200

    New configs

commit 6459472542c65aae7d93537fde78353b9b58d91b
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 15 08:37:57 2022 +0000

    Add -c option to zkProver to set the config file name

commit b34bf237c9a377d1daa3e412ccc398f40c28af9b
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 15 08:23:17 2022 +0000

    Rename Client to ProverClient.  Add ProverClient::waitForThread() method.  Extend test to 200 loops (1000 seconds).

commit 3037633f4da2d701a00e097bfd20009906facd51
Merge: 16fb364 1f25da2
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Jul 15 09:03:13 2022 +0200

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit 16fb3643e51d0d911f408d75db0941bec012a599
Author: eduadiez <eduadiez@gmail.com>
Date:   Fri Jul 15 09:01:38 2022 +0200

    Fix memory error NTT

commit 1f25da2ce69e8d2d67c0acca14de0ed347b41b25
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Thu Jul 14 22:02:22 2022 +0200

    Update README.md

commit 88157856e77a6e7801ed2a1042e84e1f98433ba9
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 14 19:24:06 2022 +0000

    Add config parameter proverClientHost to specify the IP address of the server to test

commit 48e2c0bdf0724a29c5464f3d4edf6156634008b9
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 14 18:49:27 2022 +0000

    Update zkprover.proto to latest version

commit 10714e94c6f82c9358ea57bd02161d492c7da2e3
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Thu Jul 14 19:44:52 2022 +0200

    Update README.md

commit 590024ae8b3d0eb315c8e8ce3ef1fe191c2f5f8b
Merge: 9351ccf 2f359f0
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Jul 14 17:37:06 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit 9351ccf8b55dde4acd7aabed4fb492f03a80309d
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Jul 14 17:36:54 2022 +0000

    updated usage help description

commit 2f359f00bffb1c58992a63042400e21841f7cf81
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Thu Jul 14 19:36:09 2022 +0200

    Update README.md

commit d6f3063f5346760be97ebaf32907fbce67d9b3fc
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Thu Jul 14 19:31:05 2022 +0200

    Update README.md

commit 1a0b7cc3371ff62561c0c7980d3c6c1325b66d30
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Thu Jul 14 19:22:20 2022 +0200

    Update README.md

commit a7d19115d0ba5290642a6b51eade6f3035c08d7f
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Thu Jul 14 19:19:38 2022 +0200

    Update README.md

commit aaf4d346608f2eced8cec23e3cba9f91a0a2aac5
Author: agnusmor <100322135+agnusmor@users.noreply.github.com>
Date:   Thu Jul 14 19:18:41 2022 +0200

    Update README.md

commit 5f8c612ba86549138700435910aaeea0557cd87e
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Jul 14 17:02:31 2022 +0000

    disable LOG_DB_READ/WRITE

commit 48d794a22804494eb7e85fd5741d449406388dc5
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Jul 14 17:01:01 2022 +0000

    write database.read/write log in a single line

commit e921f5a36fab4769f57ac162e68b43c03b0c3bdf
Merge: c17acd0 8991a50
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Jul 14 16:55:27 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit c17acd04c75737883f3ab5e153979b184e877184
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Jul 14 16:54:52 2022 +0000

    updated README.md file. Changed result_t to zkresult

commit 8991a5081bd9481d2145f8a562192b2eee77cb44
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Jul 14 18:34:10 2022 +0200

    Update last changes

commit 4db7614993845dfa6dc08e2d326ee9cfab7ac43f
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Jul 14 16:01:10 2022 +0000

    Improve mem copy with threading

commit 988dccfc344387a5293f0670f78d3934f943e40b
Merge: 3153395 55bb539
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 14 13:22:13 2022 +0000

    Fix compilation warning in Stark constructors order

commit 31533954c9ded47e1852d00283e5e6f9dd43a89a
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 14 13:15:54 2022 +0000

    Allow to copy constants.bin file into memory, instead of mapping, for performance resons

commit 330b54642bb8b6397883234fd8a8f0d2082752f8
Merge: 2ca5da2 55bb539
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Jul 14 13:00:43 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit 2ca5da2d70b63112be96b6babdd6cd06c949ed2f
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Jul 14 12:55:12 2022 +0000

    added result codes to database

commit 55bb5390f0d32cce68488f8cb55a8ecdf88f058d
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Jul 14 12:47:10 2022 +0200

    Fix zhInv

commit 0c8e16ac20275bf172e626ecc0775598bc7e7801
Merge: 4dd15a6 26bef25
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Jul 14 10:37:52 2022 +0000

    Merge branch 'main' of https://github.com/0xPolygonHermez/zkevm-prover

commit 4dd15a6f50d6cc439a6b27258f806095bcd1bc30
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Jul 14 10:34:47 2022 +0000

    Last changes

commit 058851ef46a76bc1f0c89441a867d7533020d845
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Jul 14 10:32:50 2022 +0000

    improve polinomial

commit 1ac1d12f2397d4cf13100352f43cc4a51208233e
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Jul 14 10:32:33 2022 +0000

    update poseidon

commit 6e418f6fd8c87e4f78c676368d22471cbe778df6
Author: eduadiez <eduadiez@gmail.com>
Date:   Thu Jul 14 10:32:08 2022 +0000

    Fix -O3 makefile

commit 26bef2535b4058a2b579b313eeaea127bf031152
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 14 10:30:34 2022 +0000

    Fix bug in Database::printTree(), when hash=0

commit ed54edfeb485169a146ba2793871d06d8f036fbc
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 14 09:55:30 2022 +0000

    Add Database::printTree() and call it from Smt::Set() if LOG_SMT_PRINT_TREE is defined

commit 6b5fc2d658e212c1230ddde5934597ab04f6113d
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 14 08:46:01 2022 +0000

    Split DB logs into read and write

commit 32438f59b84e00bf02558ca07d73931c2165f0bd
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Jul 13 16:27:22 2022 +0000

    added database read/write logs

commit 8c46267961e71869f882c43bdbb84b02cca34633
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jul 13 14:19:33 2022 +0000

    Fix config.json

commit 88c6f3afa37331692b0225802c1f5a70dd05da9c
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jul 13 13:40:26 2022 +0000

    Fix StateDB service and Database to use normalized strings as keys

commit 9df55c536a99e5939546045667252d5256b98d26
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 12 18:24:29 2022 +0000

    Add logs to response in executor service

commit 914d2d9f75a7c7d439a4fa7d6bc6c6274c8a8816
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 12 17:26:37 2022 +0000

    Fix ZhInv constructor when not generating proof

commit 69c92f84d01e951bf524237639b84c7b3b08ecc4
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Jul 12 16:59:49 2022 +0000

    First merkletree

commit d88c5c2e376a44660cf8524702b23817fcc76918
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Jul 12 16:59:39 2022 +0000

    Add polinomial class

commit ae30ef464a55af2e9039e2923d2a5b84bf4ab361
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Jul 12 16:59:21 2022 +0000

    remove nPublics

commit 0db3d24ff6b26450a34ed8c90b032052cec1ccd2
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Jul 12 16:58:28 2022 +0000

    Add goldilocks changes

commit c152e2244f4ef457f37adfc1fceac86bff963d11
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 12 15:16:26 2022 +0000

    Fix bugs in hashP and haskK

commit 22027a89e87cbf59cf361d29ee06aeda772b0102
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Jul 12 14:23:57 2022 +0000

    Parse nPublics

commit bcf5993673e49ebf317ad6f889b063ba5ebbc5e4
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 12 13:33:07 2022 +0000

    Fix hashPLen implementation

commit ee6db4eb44b79069f7ca54372d3b37764cf9e257
Merge: 8c4b9cc 00c3216
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 12 12:55:05 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit 00c3216d095fc9f4d8aa247fd9f65759d84e29a8
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Jul 12 12:54:16 2022 +0000

    Initial starkpil work

commit 8c4b9cc91f6bc25a2186947c981e183e19ba1034
Merge: d5b5263 ad534ec
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 12 11:57:57 2022 +0000

    Merge branch 'main' of github.com:0xPolygonHermez/zkevm-prover

commit d5b52631155f1465443f531c97a56870958ded15
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 12 11:56:49 2022 +0000

    Fix bugs in main SM executor: SP, hashP

commit a5763c4973b79c8be1929afcd4207e13fa603ed9
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Jul 12 11:44:59 2022 +0000

    Add zhInv

commit c8667505ff03d0b2a5d95c4473194b67e85556a3
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Jul 12 11:44:47 2022 +0000

    Add Transcript

commit cadd90e5c3dc5235f4197d29e6d1dafd2a7df805
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Jul 12 11:42:38 2022 +0000

    add test vectors

commit ad534ecde39dc249e8cf6d213b28848cb8c5a9fa
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Jul 12 11:37:21 2022 +0000

    Remove old files

commit a6d8604d74e7b9444fa3a49591c4bea6b6861b77
Author: eduadiez <eduadiez@gmail.com>
Date:   Tue Jul 12 11:36:03 2022 +0000

    Update goldilocks

commit f2ec7e0855be172272deb1dbc1ffa47d7d83c59a
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Jul 11 18:26:59 2022 +0000

    added create_db.sh script to create statedb database, user and table

commit 697a92656fbf6975ccc4dae56fefa4a084d04073
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 11 16:25:01 2022 +0000

    Add public inputs to stark genProof().  StarkInfo to use eSection enumerate instead of strings.

commit f0d0ac24f73774e20e3d6c959edd84e9086d1e24
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 11 11:51:33 2022 +0000

    Update rom.json with fix for HashP length

commit d2c5af8db682da4e5006506b67b88ea07b36f499
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 11 08:14:16 2022 +0000

    Return bytes executor.proto elements using string2ba()

commit c3a48d8876f59d6305ee75e680ac979fba6258cb
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 11 07:37:37 2022 +0000

    Remove ecrecover library

commit c3d22f956e5a068096ba91f54ebaa1046c680030
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 11 07:33:11 2022 +0000

    Code cleanup, specially Config

commit 4973ac83f55aece80b295e6e50393d76afa9e828
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 8 16:34:40 2022 +0000

    Files topology changes: create src/executor, src/prover, src/config, src/utils

commit 0f4612471138218930b51bf81b3ad08f83d6074e
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 8 14:53:37 2022 +0000

    Rename StarkInfo cm1_2ns to cm_2ns

commit 1fa8b3906b600e52c40b4c5f547c25c5fdab821d
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 8 14:39:51 2022 +0000

    Delete StarkInfo Expression and use a vector<uint64_t> instead with default value of 0

commit 47ca7761d80c9253066e0f4ca01f57d71e9800c0
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 8 14:23:24 2022 +0000

    Move keccak-f script generator code to /tools, and test to /test

commit cc72e56a962a88429de812374e31eacce57b805d
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 8 13:50:07 2022 +0000

    Move keccak-f test code to test directory

commit 79a1bb7106dc989665e7a577dcc1d130745819d7
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 8 13:16:11 2022 +0000

    Parse StarkInfo exps_2ns

commit 6600ec2cf05bff88dc067e27007d35e140b4c6b8
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 8 12:52:05 2022 +0000

    Parse StarkInfo::cm1_2ns

commit d64253d5f737016f49767c5c9f8c7ba7e4eab929
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 8 11:22:48 2022 +0000

    Fix full tracer calls to getVarFromCtx() with wrong true/false parameter

commit 929f6192f04bc857959dd4f52fdf5fe06ccfcfe8
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 8 10:31:48 2022 +0000

    Delete full_tracer_test.cpp

commit cf4f53677031a507ee880378e3d3a489734870d3
Merge: 44f523b 04a4701
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 8 10:05:29 2022 +0000

    Merge branch 'main' of github.com:hermeznetwork/zkproverc

commit 44f523b17e8d570471217919a7381f47eebcee9f
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 8 10:05:24 2022 +0000

    Fix SC with logs deployment

commit 04a47018fda8567860f004a5099af4ba14805410
Merge: 6ba1292 02f48d7
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Fri Jul 8 11:48:35 2022 +0200

    Merge pull request #14 from hermeznetwork/transaction-hash
    
    Transaction hash

commit 02f48d7fffe602270fd13078690b43c5ccd7ab9b
Author: zkronos73 <zkronos73@gmail.com>
Date:   Fri Jul 8 11:41:23 2022 +0200

    clean end of file

commit f3ef1bde2d5ccfb7c207375312b1ad3f4ce1d948
Author: zkronos73 <zkronos73@gmail.com>
Date:   Fri Jul 8 11:40:13 2022 +0200

    delete obsolete files

commit bccf03edf6e2342863d00492ba484734250d5814
Merge: 8ce79a7 6ba1292
Author: zkronos73 <zkronos73@gmail.com>
Date:   Fri Jul 8 11:37:03 2022 +0200

    Merge branch 'main' into transaction-hash

commit 8ce79a7419f2d2aa7ae12c01d4484bcfe7e2d273
Author: zkronos73 <zkronos73@gmail.com>
Date:   Fri Jul 8 11:32:29 2022 +0200

    reorganize files and clean code

commit 4c34354771d346ecbfd40022b77b4e89774ccb7e
Author: zkronos73 <zkronos73@gmail.com>
Date:   Fri Jul 8 11:18:32 2022 +0200

    clean and fix tests

commit a1bfd8b29cd6d8223fac53e5651abb9d65a1e266
Author: zkronos73 <zkronos73@gmail.com>
Date:   Fri Jul 8 09:18:32 2022 +0200

    clean tests getTransactionHash before integrate in full_tracer

commit 6ba12921ccc78069196babdb89d99f9eac710d8a
Merge: e002aed f7db171
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Jul 7 17:43:56 2022 +0000

    Merge branch 'main' of github.com:hermeznetwork/zkproverc

commit e002aedaed3618f641cb5f36252659cea42c454b
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Jul 7 17:43:18 2022 +0000

    test files for sm and service moved to /test folder

commit f7db1711530df539c2f7dc24a6f311851e5c5dc1
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 7 15:04:42 2022 +0000

    Update polinomials generated code with new getElement() method

commit 4c5788b4e34672ec857992b1d2f9aa612beb48a2
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 7 12:06:07 2022 +0000

    Update FullTracer from lastest changes from JS

commit 280e856496f584661737859969196422371c322a
Author: zkronos73 <zkronos73@gmail.com>
Date:   Thu Jul 7 13:33:26 2022 +0200

    first version of transaction_hash

commit 8d4e12d7b8949bf9ff6f6cbe966f21f7bbf91957
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 7 10:46:02 2022 +0000

    Update rom.json to fix new local state root in full tracer

commit 0a137637518b56b252899f5d5f6ffbae9ed21b0c
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 7 09:41:18 2022 +0000

    Avoid using FullTracer::labels and create Rom::memoryMap

commit 4e6a089332f93def7952708b18f434903a5e250d
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jul 7 08:07:43 2022 +0000

    Move constPols and constTreePols to Stark

commit 1f7544e04d86b2220f7a7b93701938d24b95d5e9
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jul 6 15:05:00 2022 +0000

    Add FullTracer traces.  Fix creation of 2 responses per TX.

commit fe0f8a512cba792938500dc0b7d8c6bf1e590faa
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jul 6 14:40:06 2022 +0000

    Avoid loading files during initialization that are not required if we are not going to generate a proof, for performance and file dependency reasons

commit a1dae043d20b672d9eb1311df41ab6920b7326d0
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jul 6 13:56:45 2022 +0000

    Fix new state root in FullTracer

commit 5b304c7b6bbe59c89944ad92dd7faf988a6f08b5
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jul 6 10:43:02 2022 +0000

    Implement eval_memAlignXxx() functions

commit 5d5903feba9a80170feecc537d3efc0485acd07d
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jul 6 09:26:09 2022 +0000

    Add timer traces to StarkInfo constructor

commit faf496692e7e3340ad5bbda3eecb408f97e860d5
Merge: 295a730 b585edb
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jul 6 08:57:20 2022 +0000

    Merge branch 'main' of github.com:hermeznetwork/zkproverc

commit 295a730b951f794e9b54b3c158e074dd2367a458
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jul 6 08:56:28 2022 +0000

    Implement eval_saveContractByteCode() and eval_getBytecode()

commit b585edb6c8c04bca56dc8c5ca8dffb14092d7827
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue Jul 5 15:26:40 2022 +0000

    moved LOG_STATEDB_SERVICE define to definitions.hpp file

commit 1a58f2b181ac1ad8a3eac5ceb8507c09986950ed
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue Jul 5 14:28:23 2022 +0000

    change config.statedbURL default value

commit 79da5dcaf84365d946be062cc1e7ae29b7d6c4ff
Merge: f24ef2a 59c0e3f
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue Jul 5 13:47:36 2022 +0000

    Merge branch 'main' of github.com:hermeznetwork/zkproverc

commit f24ef2a7a1607b4bc63a08a106ad31449bb9f5d3
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue Jul 5 13:39:05 2022 +0000

    minor changes to test files. Protect statedb services calls with try..catch

commit 59c0e3f3dd12ace7a5f7b0a40098ecb9f9b651da
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 5 10:07:47 2022 +0000

    Move StarkInfo enum parsing to class methods

commit 7484b25c2e5845c4cd78da6b5ad47dee123a48e4
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 5 09:58:36 2022 +0000

    Remove fake values in executor service and implement StarkInfo type long enum

commit deb2d90a689b78089da3d61c315e3fb6ebbea123
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 5 09:24:16 2022 +0000

    Change executor.proto booleans to u32, due to incompatibility issues with Go client

commit feefc5c5fbff3592fdcec6b9e32d4d0dfb5fe2bd
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 5 09:14:02 2022 +0000

    Use enumeration in StarkInfo EvMap type

commit 2d6ce6302c05f339dfe9aae9e244f0ce5cf8e36b
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 5 08:38:25 2022 +0000

    Convert StepOperation.op to an enum

commit f32d365185904e11b778e14b175fd62efd8d629b
Merge: fa444ad 35fc18f
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 5 07:25:10 2022 +0000

    Merge branch 'main' of github.com:hermeznetwork/zkproverc

commit fa444ad09fbf480cc4549b9eb214946a3e8c6887
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jul 5 07:19:01 2022 +0000

    Stark expressions parsing

commit 35fc18f3ca1183081a157eba8a231d6c111fde7b
Author: eduadiez <eduadiez@gmail.com>
Date:   Mon Jul 4 19:59:58 2022 +0000

    Update Goldilocks

commit f4e30feabdb89d18c22698a98439a2460706d1ca
Author: eduadiez <eduadiez@gmail.com>
Date:   Mon Jul 4 17:04:51 2022 +0000

    new pol extend

commit 28a5c459a105f9931b87f0d31428b95a6af00cad
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 4 16:00:43 2022 +0000

    StarkInfo third round of functionality

commit b1e8a28d1cb1cb2b5de2d1bd3278a05800b26a63
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 4 14:20:20 2022 +0000

    Fake some data in Executor GRPC service.  More on StarkInfo.

commit a31c7023f7a2c687ef94835e6c16c9d39ff2c08d
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 4 12:03:55 2022 +0000

    First StarkInfo version.  Fix steps Main SM Executor counter.

commit 025f7e693bbe6ce66c31dd28a997c64dc3548abf
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jul 4 09:53:51 2022 +0000

    Enable executor service response trace

commit 49a73b97f2dde6fe9604164359aead2f23dfdbac
Author: fractasy <fractasy@gmail.com>
Date:   Sun Jul 3 09:31:33 2022 +0000

    Fix database recovery in executor service

commit a2bb7c121aef8c8f4c7c5daa7aae2eca6702770e
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 1 17:47:58 2022 +0000

    Fix memory management in Executor::processBatch()

commit e94e09052b00c32d4d25a1118e8e0974ad2e1d3e
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 1 16:24:29 2022 +0000

    Add hashK traces

commit c610d4dc1604d0a708c25ba6ace7802c035dff51
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 1 14:40:46 2022 +0000

    Fix Arith SM Executor to use scalars in carry

commit 7c6293704aa5840abbdc702206d57fcf9038f41b
Merge: c20f962 e09cca5
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 1 07:58:31 2022 +0000

    Merge branch 'main' of github.com:hermeznetwork/zkproverc

commit c20f9626c9a53fdd919cdecf2cddafde8b371380
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jul 1 07:58:18 2022 +0000

    Integration fixes to match commit.bin from JS (part 1)

commit e09cca5c798ca07362fbdd51c4842ce46b087aaf
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Jun 30 19:37:02 2022 +0000

    debug log enabled. Fixed error in SetProgram method

commit f87fbcc81320d93c7d3a69fa048888d7309786c7
Merge: f60b121 69d75f1
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Jun 29 14:19:24 2022 +0000

    Merge branch 'main' of github.com:hermeznetwork/zkproverc

commit f60b12114b780acad70008590af25445c0650e9e
Author: agnusmor <agnusmor@gmail.com>
Date:   Wed Jun 29 14:16:03 2022 +0000

    added StateDB client factory. Reorganized StateDB test files

commit 69d75f1b9bbbefc97c4b16df6dd5ff5f61e023db
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jun 29 10:23:13 2022 +0000

    Print rom lines in Main SM executor

commit 5abb4864d8d59f107945f35f1d87c7ae4e32c2df
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jun 29 08:50:59 2022 +0000

    Replace Poseidon library by the one that is optimized for performance

commit 8032f28649b873c31f228f878390e2eb11d403d1
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jun 28 17:17:47 2022 +0000

    Basic Stark class that returns a fake proof to enable integration

commit f402ed0d8a5d7511d23d5ae5cd231be305a43b09
Merge: 951126b e726873
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jun 28 16:50:53 2022 +0000

    Merge branch 'main' of github.com:hermeznetwork/zkproverc

commit 951126b56885aac2e45a9af1692e7e53c4dfbfe3
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jun 28 16:50:27 2022 +0000

    Upgrade polynomials generated code in preparation for STARK integration

commit e72687322c90902d93011c4fefcde402d2373a3d
Merge: df6f6d8 9263f52
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Tue Jun 28 09:08:59 2022 +0200

    Merge pull request #12 from hermeznetwork/starkpil
    
    Starkpil

commit 9263f524e704f4a1dcd540645957e27d8830d408
Merge: ac2760d df6f6d8
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Tue Jun 28 08:57:49 2022 +0200

    Merge branch 'main' into starkpil

commit ac2760d7800eddef1a5f97258026e0bda7377329
Author: Eduardo Antuña <eduadiez@polygon.technology>
Date:   Tue Jun 28 06:54:53 2022 +0000

    Add transcript

commit 2364c981e95b466c5553bcf62d5c872c8f036bec
Merge: 4159323 ac1535c
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Jun 27 14:17:55 2022 +0200

    Merge pull request #11 from hermeznetwork/starkpil_v2
    
    Starkpil v2

commit ac1535c4a0a3af7ebc2480eb87648cb1e3300677
Author: Eduardo Antuña <eduadiez@polygon.technology>
Date:   Mon Jun 27 12:16:56 2022 +0000

    Merkelization working

commit fc8883ccf3f66d63b5e776c25c228d3ad7ac8a7a
Author: Eduardo Antuña <eduadiez@polygon.technology>
Date:   Mon Jun 27 11:53:55 2022 +0000

    Working tree

commit df6f6d8d41a0521da6f9bd2e184ed33e9e802ccb
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jun 27 11:44:32 2022 +0000

    Fix State DB compilation with old poseidon

commit 46efb0344502738253e12169dec102702ae2684f
Merge: 5387ed4 b63558a
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Jun 27 12:31:51 2022 +0200

    Merge pull request #10 from hermeznetwork/starkpil_v2
    
    Starkpil v2

commit b63558a190df6797211583e9484b1c9b988e421a
Author: Eduardo Antuña <eduadiez@polygon.technology>
Date:   Mon Jun 27 10:25:41 2022 +0000

    fixing MerkleTree WIP

commit 5387ed4cc339fbedacc3c177bbda0aba9fa1284c
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jun 27 10:03:36 2022 +0000

    Restore proper value of config.json

commit 66e3c2e04c04a61537eb17a639c2a29306f467c8
Merge: e9c53a1 4159323
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Jun 27 11:21:17 2022 +0200

    Merge pull request #9 from hermeznetwork/starkpil
    
    Starkpil

commit 4159323680a37905fcd3af49c6c9e878ed8c13c4
Merge: 0ea3a41 e9c53a1
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Jun 27 11:19:05 2022 +0200

    Merge branch 'main' into starkpil

commit ee95d8bb254575a74569883cbb638508c61a8731
Author: Eduardo Antuña <eduadiez@polygon.technology>
Date:   Mon Jun 27 09:17:52 2022 +0000

    Last changes after first merkle tree

commit 66dd62e8ad69e6d338ceaa30469df86f32727bfa
Author: Eduardo Antuña <eduadiez@polygon.technology>
Date:   Mon Jun 27 09:17:07 2022 +0000

    Update poseidon

commit 01c8f0e9f94448e681aa01edd9d5a2a18df37dde
Author: Eduardo Antuña <eduadiez@polygon.technology>
Date:   Mon Jun 27 09:16:51 2022 +0000

    Rename old goldilocks

commit e9c53a15194f56095ab699c7b1acaf212733edfa
Author: fractasy <fractasy@gmail.com>
Date:   Sat Jun 25 16:37:24 2022 +0000

    New iteration of executor.proto

commit 01cf7269bb49fbc3d5cd00bcde167a3742f1e325
Author: fractasy <fractasy@gmail.com>
Date:   Sat Jun 25 16:11:42 2022 +0000

    Fix last step recording in Main SM Executor

commit 07c54efb4d91f41bf6d146d0e8ff3aeaf3705227
Author: Eduardo Antuña <eduadiez@polygon.technology>
Date:   Fri Jun 24 14:08:32 2022 +0000

    Before Merkelizing 1....

commit 9b29fac5f9d04389be9b1805da35fdf3da8601d2
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jun 23 17:15:22 2022 +0000

    When in fast mode, exit the loop if rom calls beforeLast()

commit 965c4785343305c0d11957555819f0593257ffc2
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jun 23 15:29:47 2022 +0000

    Activate prover service

commit 94961614779317436cd90f1a48d4965df76f2695
Merge: 26e8351 e3ecf1f
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Jun 23 15:13:51 2022 +0000

    Merge branch 'main' of github.com:hermeznetwork/zkproverc

commit 26e8351d8f27d9d606e4918d8847b87dbe22071b
Author: agnusmor <agnusmor@gmail.com>
Date:   Thu Jun 23 15:06:30 2022 +0000

    added SET persistent parameter in main_executor. Changed db.create to db.write. Changed config.db parameters to a single databaseURL parameter

commit e3ecf1fb9418c533fb3ba0dbfc7d986446963dcb
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jun 23 10:26:59 2022 +0000

    Update zk-prover.proto, after removing execute and synchronize batch proposal methods

commit f0b5cdd146cccd9e400ed87900428e40251c11fc
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jun 23 10:11:09 2022 +0000

    Fix current (old) zkproverc prover service

commit 04ee72169ea1d0b2bf936ac466587717be6da8a3
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jun 23 08:24:32 2022 +0000

    Add config.executorClientHost configuration parameter to test external servers

commit 161e815f1d65707aef399e5234462ba5da9ff300
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jun 23 07:47:28 2022 +0000

    Add steps counters implementation to Main SM Executor

commit 4ed0f4f1dba70aaa87c1de2d9dfb9097b5c5651d
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jun 22 10:35:06 2022 +0000

    Counters implementation in SMs

commit 7557bc10609033103083e0b1038cac27481ff853
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jun 21 09:59:30 2022 +0000

    Fourth iteration of FullTracer porting from JS to C

commit 0ea3a410c7c4e25e60cdcc5353a32329996c1122
Author: Eduardo Antuña <eduadiez@polygon.technology>
Date:   Tue Jun 21 07:59:08 2022 +0000

    before pol_extend

commit 1a17d2a7126afed4baa7d8c850d936351dbffebc
Author: Eduardo Antuña <eduadiez@polygon.technology>
Date:   Mon Jun 20 15:47:12 2022 +0000

    Update readme

commit 47178dd52a41d9889bda3bc03387230bd8527c23
Author: Eduardo Antuña <eduadiez@polygon.technology>
Date:   Mon Jun 20 15:45:42 2022 +0000

    Update docker

commit 31e8e93bdf1b7cfc4db4ca7811c7c02c9b3012ec
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Jun 20 14:45:42 2022 +0000

    set GOLDILOCKS_DEBUG to 0

commit 98beb9ff866439c10b29a1eb4aee89bc3431ed08
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Jun 20 14:24:00 2022 +0000

    change config.json to default values

commit aa36a89d078ec1001222dde3c93d1ff2fc42fb70
Merge: e9e9e11 d4f82be
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Jun 20 14:23:10 2022 +0000

    config.json change to default

commit e9e9e11deacc1169e17eb6326499c2fae1992e6d
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon Jun 20 14:16:44 2022 +0000

    added zkstatedb service

commit d4f82bea492bf515b7d1296561f06aa2b1b11b0e
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jun 20 12:19:07 2022 +0000

    Third iteration of FullTracer from JS to C

commit 46a6dd782fbe50de0338e49a6d778f3dbb11cfe2
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jun 17 11:07:10 2022 +0000

    Merge last changes in FullTracer from JS

commit b90a220c712c53807b18e1d517cf488c46a23e21
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jun 16 17:47:29 2022 +0000

    Delete files used in old Stark implementation; making room for the new one

commit c63551e8412b4a41ac7545c6d6d699ad8cf5c795
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jun 16 17:38:09 2022 +0000

    Move ROM code to src/sm/main

commit 07fed28d0262dc8549d6030365b8d373d59ce587
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jun 16 15:31:07 2022 +0000

    Add FullTracer attribute to ProverRequest to share the same instance with the Executor

commit dc2c239af4b2fec56cfa68a16517b6b770fa7409
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jun 16 11:48:00 2022 +0000

    Prepare Executor service ProcessBatch response

commit a569886fd45c94c1ad840ed0c7516929b81422d4
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jun 16 09:28:00 2022 +0000

    Create processBatch() which ignores main SM rom assert. Update storage rom.

commit 46ad49a21536eb82e047d0fa2ce013d8600fb0f5
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jun 15 16:55:16 2022 +0000

    Add db to executor.proto to allow passing db initial values while testing Executor Service via GRPC

commit 256c01c3c28facb2b27f65fe5cbede12bc5b4aa2
Author: Eduardo Antuña <eduadiez@polygon.technology>
Date:   Wed Jun 15 16:36:36 2022 +0000

    Initial state

commit 0933496fb494d6f1372ce8604ffa9a3d5b91f736
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jun 15 16:13:30 2022 +0000

    First running version of the Executor Service, with partial funcionality

commit bce1950b15895f2e70e1e0010984e9641cf7cee1
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jun 15 10:22:42 2022 +0000

    Move zkprover service code to src/service/zkprover/ folder

commit f5dec9a9e509b1b04082411036edbc1c3bc226d0
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jun 15 09:10:47 2022 +0000

    First version of executor.proto

commit 211fcfcad21971af382f69e75e02d492041a6a50
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jun 15 08:27:05 2022 +0000

    LOG_STEPS_TO_FILE and update in Main SM Exec Generated

commit 3b43939745e4a06b3882f645146840d8db7c239a
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jun 14 11:33:57 2022 +0000

    Port latest change in Binary SM Executor from JS, fixing a corner case

commit c99437fadf5a5c79ab5e67a9bb38fde471b43838
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jun 14 10:02:08 2022 +0000

    Update commit_pols.hpp, and fix PaddingKKBit SM Executor connected polynomial

commit fb496e979848df6096aa028a62f0c7aefddd0fa7
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jun 14 09:09:51 2022 +0000

    Fix Storage SM Executor after Goldilocks migration

commit 86d4a55a00a5eb40c0daf25cc82b90a781c5e040
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jun 13 16:04:37 2022 +0000

    Fix main SM executor generated code

commit 7dff6e93348a8ab274bbd584440068ad61b28e1e
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jun 13 14:23:55 2022 +0000

    Fix Goldilocks inv() method, and avoid using .fe element directly

commit 5d0d0f8fae718d81c6adfa59d4c601a387a4541c
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jun 13 14:00:18 2022 +0000

    Fix Goldilocks library equal() method

commit 46417ce965a7ccc6614e8c9b8aafa33468b8a740
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jun 13 13:13:59 2022 +0000

    Fixes to main SM Execution

commit 383e2d1e34eb503c3973066149efda4e0e6795d1
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jun 13 12:08:59 2022 +0000

    Fix 2 bugs in Goldilocks library

commit f68a679cb46f439a89759b1e0267e3aeb38a4495
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jun 13 10:03:54 2022 +0000

    Massive port to new Goldilocks library

commit bfeecce9accfd003b5fd0c9a1e6a76d2855753d0
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jun 10 07:36:27 2022 +0000

    Add STARK PIL placeholder

commit 47f9740b7facb5c4ecf51dcacb68ebd500e980cf
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jun 9 15:29:22 2022 +0000

    Update latest Binary SM Executor changes

commit ea532336e233ebf804574cb62d0d0903b75f0757
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jun 9 07:00:00 2022 +0000

    Fix -O3 compilation of zkproverc by disabling it in generated code

commit 17a98327799210633db1402b0a338add63cb2fbe
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jun 8 15:06:55 2022 +0000

    Fix Storage SM Executor set zeroTozero case

commit 1902ba168f14cac892ee4f44a7ba5f5c7e19a251
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jun 8 10:08:03 2022 +0000

    Use different label names in fast and normal modes, in Main SM Exec Generated

commit 52b65668b075a9f5180ad48777fe4b019fe631d1
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jun 7 15:31:24 2022 +0000

    Update pols_generated with new version without types in PIL files

commit 86c3f68ae34f4c1dac7d43651910d0d5be7416b3
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jun 7 15:20:49 2022 +0000

    Add zkasm line string to generated code and to Main SM executor logs

commit 8597fcc7f256988803e50e296908bca4f7098fef
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jun 7 12:10:23 2022 +0000

    New GeneratedPols from PIL

commit d17e20154b648cfa874c620ded1a199b086967e8
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jun 6 15:32:17 2022 +0000

    Update main exec generator code

commit 92f67512a7391c63a5e9f0ce9987ceab73e7ed8f
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jun 6 06:48:43 2022 +0000

    Fix address offset limit in Main SM Executor

commit cbd527978728da7cb1cb5593ab58bbf2c15d5e09
Author: fractasy <fractasy@gmail.com>
Date:   Sat Jun 4 13:36:50 2022 +0000

    New version of Main SM Executor generated code

commit c5e45ab47f804e4686d89743ac63f42a1c7ee998
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Sat Jun 4 12:54:08 2022 +0000

    Port latest fixes in Binary SM Executor carry policy

commit 284b5d58e969708b9df59b3399656b400be0e607
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Fri Jun 3 14:33:32 2022 +0000

    Rename generated code directories

commit 026bb7620d86fb311aa7d8f44dd0d693aae77f81
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Fri Jun 3 13:17:00 2022 +0000

    Create timer.hpp, definitions.hpp and zkassert.hpp to unload config.hpp

commit c92d5bbd1f3faa435446db37d5585c1f247c6b9f
Author: jbaylina <jordi@baylina.cat>
Date:   Fri Jun 3 11:15:01 2022 +0200

    Fist draft version of Main SM Executor generated code

commit fc68edd674aa683c9b15dd16ca59d84ba0f76d74
Author: jbaylina <jordi@baylina.cat>
Date:   Wed Jun 1 18:39:50 2022 +0200

    First version of Main SM Executor events, to be tested when ROM uses them.

commit 2086fd1be66c1f29d2021963133df86829aaa97d
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 31 14:40:55 2022 +0200

    Fix Binary SM Executor l.351.  Improve performance when consuming remaining evaluations.

commit ce4b746d0e0aaf12f46c16556515a574680bcbca
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 31 11:47:42 2022 +0200

    Performance improvements in Main SM Executor: ctx.pStem and ctx.pZKPC

commit a5e5ba6e787219c99b05f08bac29ce4847f32143
Merge: 092dfcd 2f80b5c
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 31 11:08:21 2022 +0200

    Merge changes from main before pushing first Executor class implementation with parallel SM execution support

commit 092dfcd2470d6341d6a4908722aaae0a836f7767
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 31 08:42:01 2022 +0200

    Move management of SM Executors to the new class Executor, which supports executing SMs in parallel.  Performance improvements.

commit 2f80b5ce8a10425d586cbb2c5d5ee4c78455e2bb
Author: agnusmor <agnusmor@gmail.com>
Date:   Tue May 31 08:34:46 2022 +0200

    update rom file

commit 308b8b78db774a2c2ed91e29562605ae81f3a9c7
Merge: 3a588a6 bd1ca5d
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon May 30 11:13:32 2022 +0200

    Merge branch 'main' of github.com:hermeznetwork/zkproverc
    merge main changes into local branch

commit 3a588a66d34d5face065696f561d187e7e30f5d2
Author: agnusmor <agnusmor@gmail.com>
Date:   Mon May 30 11:11:50 2022 +0200

    mem_align SM implementation and integration with main

commit bd1ca5d6569447206fbf5aee098e4d133b4dbdde
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 30 08:30:12 2022 +0200

    Just some comments before a new clone

commit 8117bc62fd496243a6183c3eaa38bcf7aa23d932
Author: jbaylina <jordi@baylina.cat>
Date:   Fri May 27 17:41:02 2022 +0200

    SM common way to get, stora and use N, the number of evaluations, i.e. the polynomial degree

commit 7c218b57cc32d2412f4637abc9d4ad36fd674509
Author: jbaylina <jordi@baylina.cat>
Date:   Fri May 27 08:11:21 2022 +0200

    Fix Keccak-F SM Executor

commit 354dfa8e9ee084e5bad4e0880dcfa10430afa176
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 26 18:36:56 2022 +0200

    Fix Nine2One SM Executor

commit d8cc95cb1272b0c34d581abc2c3ad8467a588943
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 26 18:27:17 2022 +0200

    Fix Padding KK Bit SM Executor

commit 7c926bcd94df5f9d6e291ace9674e5df02a0f74d
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 26 15:44:51 2022 +0200

    Fix Binary SM Executor

commit e81d649147d1cd8a64c3bbf856b341f965269734
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 26 13:48:25 2022 +0200

    Fix PoseidonG SM Executor

commit 0361c00a79e3e3e396537ffe0dcbc9ae96d3c024
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 26 11:55:57 2022 +0200

    Fix Storage SM Executor

commit f32cf327ec1593d58fa2dae6948080a1c0570037
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 26 07:56:37 2022 +0200

    Fix Arith SM Executor

commit 4b86e1be7bf51e4752196d1d86902bed74702ad4
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 25 13:59:19 2022 +0200

    Fix Padding PG SM Executor

commit d4567d87857fc25ffc093800c6707836146a7bf7
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 25 10:28:20 2022 +0200

    Fix PaddingKK SM Executor and required data collection

commit 67b476228e0f92c36ffe652d809c6bd927e0be2d
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 25 08:21:34 2022 +0200

    Fix address offset values in Main SM Executor

commit ad7e7bb2f0f53edd710a0ab9a0a20acc9d129488
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 24 19:51:04 2022 +0200

    Fix indRR pol generation in Main SM Executor

commit 935376449da5a9c4f0c937e31d221daf9bb4707b
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 24 17:36:24 2022 +0200

    Remove binary contant polynomials code, since now we generate what we need at execution time

commit 13bda18408d2d2fc1bcbc745071bd9d3aa66cf1b
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 24 17:30:56 2022 +0200

    Update Main SM executor to meet the new Binary SM behavior, from JS

commit ea37ed387bdfedc6616f1bb0787fed9c7d46d545
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 24 14:48:27 2022 +0200

    Update Binary SM Executor from JS main

commit 6d5bfafb7f63efeb24872eb3021c31e14718a28c
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 24 08:40:47 2022 +0200

    Complete Poseidon G SM Executor implementation.  Collect Poseidon G data in Main SM.

commit 2d568fb368a2ad1c504c65f224d78b11f7c6ce17
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 23 21:49:03 2022 +0200

    Update commit_pols.hpp and constant_pols.hpp and fix Binary SM compilation, moving c0Temp from a polynomial to a local variable

commit 904623421cc976dd9be7ea97133529d8b0c3f70b
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 23 21:19:49 2022 +0200

    Integrate PoseidonG SM into Prover

commit d1d935905b8e4b85119839270b5f61b180e2c07d
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 23 14:34:17 2022 +0200

    Integrate Padding PG SM into Prover

commit b5f15241ee02d90226a822ebb29a9c63e5ba3fa4
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 23 10:09:40 2022 +0200

    Rename Keccak SM to KeccakF

commit 60dab9725ca4525240fe51692478cdf02c821474
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 23 09:56:32 2022 +0200

    Implement Byte4 SM and integrate it into the Prover

commit f365a1bf51ea97a92bd7d1e196275883b7027bea
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 23 09:32:17 2022 +0200

    Rename required SM fields

commit 16246be2c793a644fa002a4c55b385ac3667bcc7
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 23 09:18:19 2022 +0200

    Move Memory SM reorder() and print() to executor

commit 545ded1cdc4403bfc2a6f90892eac886ddf038b6
Author: jbaylina <jordi@baylina.cat>
Date:   Fri May 20 15:49:08 2022 +0200

    Integrate NormGate9 SM into Prover

commit b9c88c3a5356f65dc82a4a9296de1956ce73a64a
Author: jbaylina <jordi@baylina.cat>
Date:   Fri May 20 15:02:40 2022 +0200

    Integrate Keccak-F SM into Prover

commit f5ef47a824c8dbde1ede790d220e9c24c6581799
Author: jbaylina <jordi@baylina.cat>
Date:   Fri May 20 14:25:25 2022 +0200

    Integrate Nine2One SM into Prover

commit dab623f9797255883e623edbbeb89134169ce478
Author: jbaylina <jordi@baylina.cat>
Date:   Fri May 20 11:57:35 2022 +0200

    Integrate Paddking KK Bit SM into Prover

commit b6c1100bb629d0c43fe1bfdc62c778a44ba757e7
Author: jbaylina <jordi@baylina.cat>
Date:   Fri May 20 10:28:25 2022 +0200

    Integrate PaddingKKExecutor into Prover SM sequence

commit 6e99c7498c8d2bf06c8a46936f6886f55c7a2b92
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 19 20:31:48 2022 +0200

    First complete implementation -untested- of Arith SM

commit 352fb0acd7102642b989e58f2663825cbbd79ec4
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 19 15:15:47 2022 +0200

    Delete storage pols

commit 2c7feb38de1f484ca6ddecd49146b078a18376f6
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 19 15:12:33 2022 +0200

    Integrate Binary SM with Prover; call it after Main SM Executor has been called

commit 37de472245d1fa255ea1174ca72f23b54fed88bc
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 19 11:57:12 2022 +0200

    Implement eval_inverseFpEc, eval_inverseFnEc, eval_sqrtFpEc, eval_xAddPointEc, eval_yAddPointEc, eval_xDblPointEc, eval_yDblPointEc

commit e11aa08a3795af96a23d3dcfc3376317af4f45bc
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 19 09:45:14 2022 +0200

    Implement eval_copyTouchedAddress() and eval_touchedStorageSlots()

commit efb4429eada14ed5747cbadb483ee9f845bff2a1
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 19 09:15:22 2022 +0200

    Implement eval_bitwise() and eval_exp()

commit ba358accd88cae172810638342864539516a5ed0
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 18 18:30:27 2022 +0200

    Implement eval_comp and eval_storeLog

commit 035d771e82508ea505da04146aa624a7a5ff5fa6
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 18 18:01:43 2022 +0200

    Add JMPN to Main SM Executor

commit f89fb5287a14727c9b2e121273eb0b9868d28461
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 18 16:10:36 2022 +0200

    Make Prover::execute() specify the degree of the committed polynomials

commit 98940388e433ffede472fd31159af3301fed8754
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 18 16:03:47 2022 +0200

    Allocate 2-evaluations polynomials in Prover::execute()

commit f7780ce21520459e92069357fd89b1adc39e78aa
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 18 12:48:00 2022 +0200

    Update Storage SM changes from JS to pass PIL verification

commit cbc0ecab4f052f0fcb0bd995aa0e91c39cbb9c0d
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 18 11:11:46 2022 +0200

    Update polynomials header files

commit 840a59bdf0a2b417e715c3c740e204345058590e
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 18 11:01:23 2022 +0200

    Implement memAlign and memAlignWR related functionality in Main SM Executor

commit 6eda74e80c9a33767cf62678933808fa10dfed5c
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 17 18:47:12 2022 +0200

    Main SM Executor finishes, but a lot of functionality has not been called, not even implemented

commit db5156a09f632e402ee0f1cce2ec14ac1dec0344
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 17 18:09:19 2022 +0200

    Main SM Executor up to line 317

commit 7e68defbdab2bb654858fc4e382ba6371b7dbc48
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 17 17:16:17 2022 +0200

    Main SM Executor works up to evaluation 270

commit ebb88935b89d02a88e70a4395c224d7a16969716
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 17 14:23:29 2022 +0200

    Change Main SM Executor storage key generation, to match JS implementation

commit 4f787223c5932b59b9dfe791c3a63761dbc87219
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 17 13:39:46 2022 +0200

    Modify HashValue in Main SM Executor to have digest as a scalar, and bDigested

commit 4d933821af5cd301deae8e2cc3842d277b1954fc
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 17 08:47:55 2022 +0200

    Update main executor from zkproverjs main branch

commit 50ace09ce349be5731dff64582da0faec909b5be
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 16 18:24:38 2022 +0200

    First draft of Padding KK and Padding KK Bit SMs.  Update pols header files.

commit 82420383427b4e90d66fc7b48b82794b5303c1ee
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 16 09:12:18 2022 +0200

    Remove some SM files to name_executor.hpp

commit 6d66aaeefb824dcef8e41273915386653d7397d5
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 16 09:05:55 2022 +0200

    Rename Executor to MainExecutor, and define MainExecRequired

commit ce0e48b7120ae2a4059cb700788a32260fd20e82
Author: jbaylina <jordi@baylina.cat>
Date:   Fri May 13 18:27:42 2022 +0200

    First draft of NormGate9 SM

commit d0cc590a7040e0a08e4ebeadd5cc7699b1aab144
Author: jbaylina <jordi@baylina.cat>
Date:   Fri May 13 17:43:00 2022 +0200

    Integrate Keccak SM Executor with commit pols

commit c243cdc3ecf92b8fbe8536b6fe991e0aef0d35af
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 12 16:24:39 2022 +0200

    Fix Storage SM Executor: copy iHash data to pols.freeN, then to op

commit bd7a6dc37cfc661b29bf56056a0e3338d546d7ce
Author: jbaylina <jordi@baylina.cat>
Date:   Thu May 12 12:03:13 2022 +0200

    Change KeccakF SM to start Sin at gate 10.  Optimize Iota-Chi interaction.  Integrate Storage SM with cmPols.Storage.

commit 7a45f9a313bc0ed75dc1d5623cfdd08faa0fa58f
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 11 16:26:42 2022 +0200

    Update auto generated polynomial code files

commit c2a26c755c94f79fb02bc4e85197eab7d3c41c1e
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 11 14:57:31 2022 +0200

    Fix keccak_script.json generation: input in b pins

commit 69c0dc89cfbf2a62a201129de18cfbb2b3dd9b1b
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 11 14:10:23 2022 +0200

    Refactor Keccak-F executor to keep all gates in one single block

commit fcef0cf78b0a7943de952fb14c4118885c0a6262
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 11 11:31:47 2022 +0200

    Change format of keccak_script.json

commit 6b0a7dbe9c4873573dda06abef7bfae5d8d0f426
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 10 17:32:27 2022 +0200

    Move Sin XOR Rin out of Keccak-f loop

commit 166a497d0f60d5b12ad4d1de9a4f4b8ceee4ddba
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 10 15:26:59 2022 +0200

    Modify Keccak SM to store Sin and Sout in positions 9xN

commit 56d37151fb71af562f9173afa90056f8e5ae8be6
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 9 13:15:21 2022 +0200

    Integrate main committed polynomials with Main and Memory SM.  Add Fec and Fnec.  Add PoseidonLinear.  Add Databas::setProgram/getProgram.  Migrate bin and arith parts of Main Executor.

commit 223b9dd6e1863c5e8e9a016c9bfd495b0f93d8e4
Author: jbaylina <jordi@baylina.cat>
Date:   Wed May 4 16:04:15 2022 +0200

    Arith SM backup

commit 0154f59af180c8648f1f912aeb7e09b0c7989b1a
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 3 12:42:08 2022 +0200

    Refactor Keccak SM to avoid SM in names

commit e3353ae793e672910af090f3d9e0534e4a11b1f1
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 3 11:00:32 2022 +0200

    SM refactoring: moving all SM code under src/sm folder

commit c8150972a37ea102350ee4492c46dbb29622d227
Author: jbaylina <jordi@baylina.cat>
Date:   Tue May 3 10:48:01 2022 +0200

    Backup of Arith SM before SM folders refactory

commit 793299570ee1c37d627227890c554318cd7cf38a
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 2 19:18:19 2022 +0200

    Upgrade Memory SM to last version from JS

commit ceb7230aabfb0f639c3d2299b957b73329dd6a7f
Author: jbaylina <jordi@baylina.cat>
Date:   Mon May 2 16:02:40 2022 +0200

    Binary SM pol types; Storage SM op0Inv

commit ab46af6f0eb7c628fd0c1741ce57a47e55080f14
Author: jbaylina <jordi@baylina.cat>
Date:   Sat Apr 30 13:41:52 2022 +0200

    First version of Binary SM

commit 772f835ba877bdb1e24ff03d7a1adb31ce9629c8
Author: jbaylina <jordi@baylina.cat>
Date:   Fri Apr 29 11:49:54 2022 +0200

    Fix memory and storage SMs to match new PIL schemas

commit bc384c17559d10f6760b05bdf80b4add7abc3c1d
Author: jbaylina <jordi@baylina.cat>
Date:   Tue Apr 26 19:47:10 2022 +0200

    Add memory SM changes to Config

commit 9e47b804894897a4bb524ebed85e981bd6a3dc98
Author: jbaylina <jordi@baylina.cat>
Date:   Tue Apr 26 18:46:04 2022 +0200

    Backup memory state machine code

commit 9429089fe63de3d4d538f61f5a3b6b0cc739813c
Author: jbaylina <jordi@baylina.cat>
Date:   Thu Apr 14 12:32:00 2022 +0200

    Fix end of storage rom main loop, and improve Storage SM comments

commit 2724c0cebcdd03219680f8edd65a36eeaf1bc6a4
Author: jbaylina <jordi@baylina.cat>
Date:   Thu Apr 14 10:49:27 2022 +0200

    Update storage_sm_rom.json from a code cleanup

commit debb06a3caed482338e3fcc1e6ef96a5e5cfeba2
Author: jbaylina <jordi@baylina.cat>
Date:   Wed Apr 13 16:05:00 2022 +0200

    Fix Storage SM Get case (now unified) and fix long keys failing SMT unit test

commit 9647db38198afda8cf81e3d0166539c434d4b34c
Author: jbaylina <jordi@baylina.cat>
Date:   Tue Apr 12 16:51:37 2022 +0200

    Add input.batchL2Data and input.publicInputs.timestamp parsing to/from GRPC

commit 0e2872a082a666672711cd39c8ef5ccb7ed36989
Author: jbaylina <jordi@baylina.cat>
Date:   Tue Apr 12 12:08:23 2022 +0200

    Fix SMT Storage Executor zeroToZero case

commit 4b09727ef49a6a6c098a83e4d3ea98202c6be5cd
Author: jbaylina <jordi@baylina.cat>
Date:   Mon Apr 11 11:02:57 2022 +0200

    Fix SMT Set insertNotFound case

commit 067dc69421656db893307eac5abdbe285c230a19
Author: jbaylina <jordi@baylina.cat>
Date:   Mon Apr 11 09:59:48 2022 +0200

    Fix SMT Set deleteFound case in Storage SM Executor

commit c6ba8a8e5e6c0d5a0775b91f6a82cef96aff502a
Author: jbaylina <jordi@baylina.cat>
Date:   Fri Apr 8 14:17:12 2022 +0200

    Fix rest of SMT Set cases at StorageExecutor

commit fd5d3fc92695312b90021b40ce251e736830e226
Author: jbaylina <jordi@baylina.cat>
Date:   Wed Apr 6 18:54:29 2022 +0200

    Cleanup StorageExecutor traces

commit 013d599ed044ecacd1d3e15ca8984c5f98d1bc26
Author: jbaylina <jordi@baylina.cat>
Date:   Wed Apr 6 18:35:09 2022 +0200

    Fix SMT Set mode=insertFound case in StorageExecutor

commit 5b9d0369427f51550dd0db3795b1c8500eac5472
Author: jbaylina <jordi@baylina.cat>
Date:   Tue Apr 5 10:50:33 2022 +0200

    Control Storage Executor traces via config.hpp

commit 2c6b20d6102f05e9bf83b2a1bb3db8c474f25881
Author: jbaylina <jordi@baylina.cat>
Date:   Tue Apr 5 10:29:26 2022 +0200

    Fix Storage Executor SMT Set update case

commit 696b5f2c2250d4959a5a37f538c01a4b45895343
Author: jbaylina <jordi@baylina.cat>
Date:   Mon Apr 4 19:04:06 2022 +0200

    Fix SMT Get operation in Storage Executor

commit 20746036b30490e0c94c0fc9b36da047231ec22f
Author: jbaylina <jordi@baylina.cat>
Date:   Sat Apr 2 18:27:55 2022 +0200

    Add prime method to finite field library.  Use Poseidon singleton.

commit 81d21573a2bca76c09215e87c4d8ac78563261b3
Author: jbaylina <jordi@baylina.cat>
Date:   Fri Apr 1 12:31:39 2022 +0200

    Backup copy while working on Storage SM - Main SM integration

commit fab9b650eefe7c0c266a38df55815894436d0977
Author: jbaylina <jordi@baylina.cat>
Date:   Thu Mar 31 10:23:11 2022 +0200

    Enable mapping of constants.bin, now with right size

commit 083de38e16b7de3a2385328151b9fa4501e63a47
Author: jbaylina <jordi@baylina.cat>
Date:   Thu Mar 31 08:29:39 2022 +0200

    Migrate to new version of GRPC .proto file

commit 6e30367b1442525641d16ac93dbe72180cc2cfbb
Author: jbaylina <jordi@baylina.cat>
Date:   Wed Mar 30 13:36:46 2022 +0200

    New SMT based on a binary tree

commit 8511b97e2acab234310d230a8c04b7ace51fea68
Author: jbaylina <jordi@baylina.cat>
Date:   Mon Mar 28 15:51:11 2022 +0200

    First frop of zkprover migration to new components (new prime, new SMT, new proto...).  Only executor passes.

commit d823eec59da8eb49fb8e4fac433c0b7b21467e79
Author: fractasy <fractasy@gmail.com>
Date:   Wed Mar 23 10:38:34 2022 +0100

    Typedef FieldElement

commit 2fc51a889fdab4e48c816d6634467aab0cc14666
Author: fractasy <fractasy@gmail.com>
Date:   Tue Mar 22 17:23:08 2022 +0100

    Finite fields native library with golden prime

commit 436940ca79b2eda3d783bcc172946c3e304d692b
Author: fractasy <fractasy@gmail.com>
Date:   Tue Mar 22 10:45:01 2022 +0100

    Add SmtActionList and MemoryAccessList

commit b48fa4cdf5486c26ae1ed69a7ed928fd9d6a6ad6
Author: fractasy <fractasy@gmail.com>
Date:   Fri Mar 18 18:19:45 2022 +0100

    Fix storage SM memory allocation and program parsing

commit b88392a28678c669f54faa54571edd72fce01708
Author: fractasy <fractasy@gmail.com>
Date:   Fri Mar 18 17:51:22 2022 +0100

    Storage SM executor first version

commit c91b30490b51ba7ad11d002d46cfb9c3adbc89e2
Author: fractasy <fractasy@gmail.com>
Date:   Thu Mar 17 16:38:50 2022 +0100

    Delete storage SM zkasm and pil files, now locates in repository zkasmstorage

commit 7b93e551015a55353aa5f8fa4eff4f44e2693bb7
Author: fractasy <fractasy@gmail.com>
Date:   Wed Mar 16 13:04:19 2022 +0100

    Change Storage SM assembler files extension

commit b5f54297b3eda56e24e42f7e7ca491ce6d064053
Author: fractasy <fractasy@gmail.com>
Date:   Wed Mar 16 13:02:02 2022 +0100

    Change storage SM syntax to zkasm

commit d7cb3483e17f74d8713a7508a0b872a9cac48cd9
Author: fractasy <fractasy@gmail.com>
Date:   Wed Mar 16 09:54:55 2022 +0100

    Add iRotateLevel logic to storage_sm.pil

commit ea9deed748fb87741265b41b2ae91a21ce0b3b4f
Author: fractasy <fractasy@gmail.com>
Date:   Tue Mar 15 19:37:39 2022 +0100

    Make storage_sm.pil compile

commit 754a8323e522d2e5d57965bad60e28c2e64fb941
Author: fractasy <fractasy@gmail.com>
Date:   Tue Mar 15 19:10:15 2022 +0100

    Storage State Machine - first draft version

commit aab753a082e775ac11f2c88fbb447c6e5eb38cad
Author: fractasy <fractasy@gmail.com>
Date:   Fri Mar 11 11:14:24 2022 +0100

    Optimize Keccak SM performance

commit b0ce4ac4778bbca27955805ba8e6735b5592d99c
Author: fractasy <fractasy@gmail.com>
Date:   Wed Mar 9 14:15:10 2022 +0100

    Keccak SM separate executor test code in files

commit 351eed048ddc3cb25cc02ec6c39ee9f8b1f0511a
Author: fractasy <fractasy@gmail.com>
Date:   Wed Mar 9 10:50:17 2022 +0100

    Add a Keccak SM method to submit 54x9 slots at once

commit 31c4cae79f496484cd242fee0494ebb088146c70
Author: fractasy <fractasy@gmail.com>
Date:   Mon Mar 7 19:24:32 2022 +0100

    Add Keccak SM executor method to process 54 slots in parallel

commit 7674c2e1a8ce97c9cf815c8b95c91518fd6c155a
Author: fractasy <fractasy@gmail.com>
Date:   Mon Mar 7 16:01:04 2022 +0100

    Improve KeccakSM save pols comments and refactor; and use KeccakInstruction in KeccakSMExecutor

commit d82e561be302d99c4e982f746ac2f7e0285d1e76
Author: fractasy <fractasy@gmail.com>
Date:   Fri Mar 4 19:40:37 2022 +0100

    Add output and witness.wtns to .gitignore

commit 77e59580d6c0ce0ceae0ddda877023e0b44f6688
Author: fractasy <fractasy@gmail.com>
Date:   Fri Mar 4 19:38:01 2022 +0100

    Reuse keccak-f gates SinRef0+1088...1599 to make useful work

commit f5b3e266660bf4542215632018b0bb40db8742a6
Author: fractasy <fractasy@gmail.com>
Date:   Thu Mar 3 15:18:45 2022 +0100

    Rename evals to program in Keccak SM

commit 61ea86dd3ed6ab487c92c87baaa0d7214edf8152
Author: fractasy <fractasy@gmail.com>
Date:   Wed Mar 2 15:48:28 2022 +0100

    Allow connecting input pins to other input pins
    
    Arrange polynomials bits as ZeroRef, Sin0, Sout0, ... Sin52, Sout52, Intermediate0, ... Intermediate52
    
    Delete keccak_script.json since it is too large
    
    Git ignore keccak json files (too large)

commit 9b3b1a4004b80f2686e13deb264275afee54065e
Author: fractasy <fractasy@gmail.com>
Date:   Tue Mar 1 11:48:44 2022 +0100

    Centralize all gate operation in one single method OP

commit 493e46e9e738d8a7457090d22950512b9f4d70d1
Author: fractasy <fractasy@gmail.com>
Date:   Tue Mar 1 08:17:00 2022 +0100

    Rename a, b, c to refA, refB, refC

commit cfa2985aef79b2bd6b9425c09b1770cabf6bf280
Author: fractasy <fractasy@gmail.com>
Date:   Sat Feb 26 09:15:59 2022 +0100

    Refactor Keccak SM to use gates input A, input B and output pins

commit 6189cc9874a26ca97010e644441caf1d23947f35
Author: fractasy <fractasy@gmail.com>
Date:   Fri Feb 25 17:02:51 2022 +0100

    Avoid using SoutRef in Keccak script file and executor

commit f85aa346e353628beee1d3f32c8fd756b496f4f6
Author: fractasy <fractasy@gmail.com>
Date:   Fri Feb 25 13:42:51 2022 +0100

    Use polynomial identity constants in Keccak pols file generation

commit 00dd8b0ca9231be42b3f4f0fe3817553d0be6118
Author: fractasy <fractasy@gmail.com>
Date:   Thu Feb 24 19:45:34 2022 +0100

    Cleanup refactoring of Keccak SM

commit b9c6efd6293d4cdf19a20dd5a12af203131eeedb
Author: fractasy <fractasy@gmail.com>
Date:   Tue Feb 22 12:53:23 2022 +0100

    Store keccak polynomials in json file, rotated if connected

commit 239e2281cc2a5aef4103158fbbd6f98f60784af9
Author: fractasy <fractasy@gmail.com>
Date:   Tue Feb 22 11:28:15 2022 +0100

    Create class Gates to be used to model the Keccak SM circuit

commit dee03c4299bf8b26364c51154ce4106781ce2952
Author: fractasy <fractasy@gmail.com>
Date:   Mon Feb 21 17:27:05 2022 +0100

    Force final Sout refs to be SoutRef0+i

commit 37aa60a64974cc3923d18e181c28df52267aec09
Author: fractasy <fractasy@gmail.com>
Date:   Mon Feb 21 16:43:36 2022 +0100

    Add XORN function

commit 6f327aed3260d6c35f78ee2e108666785c281201
Author: fractasy <fractasy@gmail.com>
Date:   Mon Feb 21 12:00:52 2022 +0100

    Keccak SM Executor, finally working!

commit 022c31b9a347d0a232429701ec9e4e371b1e9245
Author: fractasy <fractasy@gmail.com>
Date:   Sat Feb 19 15:44:10 2022 +0100

    Fix previous commit

commit e3184a9152a1d0c65b4f6da95a153503506c9703
Author: fractasy <fractasy@gmail.com>
Date:   Sat Feb 19 15:40:01 2022 +0100

    Backup while implementing keccak executor

commit 7badcbf91a984b906436e5e2e3015c3317a8a837
Author: fractasy <fractasy@gmail.com>
Date:   Fri Feb 18 12:06:00 2022 +0100

    Improve keccak script generation

commit b9ab9bdaacc27c9610c24bf2eb5da3de42c5cdd1
Author: fractasy <fractasy@gmail.com>
Date:   Wed Feb 16 11:00:54 2022 +0100

    Rename Keccak2 and KeccakSM files

commit b754dca03250ad68f972c6529f42248bdf471065
Author: fractasy <fractasy@gmail.com>
Date:   Tue Feb 15 19:30:19 2022 +0100

    Use ANDP instead of AND in SM Keccak

commit d979178849f46051a1164f626d1fd3925cc43733
Author: fractasy <fractasy@gmail.com>
Date:   Tue Feb 15 18:25:18 2022 +0100

    Avoid using COPY in SM Keccak

commit fe7d43d929997d5043d2d97939fdfffaf7ee5997
Author: fractasy <fractasy@gmail.com>
Date:   Tue Feb 15 17:17:36 2022 +0100

    Backup SM Keccak changes

commit 081896fcaecdb9b25084cd67e0005f5bb94c2ec0
Author: fractasy <fractasy@gmail.com>
Date:   Tue Feb 15 16:53:29 2022 +0100

    First Keccak state machine version

commit 7d7bab87c9342669caf33d96f2248c3d8acb4f3e
Author: fractasy <fractasy@gmail.com>
Date:   Mon Feb 14 11:28:59 2022 +0100

    Add operation counters to keccak

commit bcbe9c8475dad4c100f8a31873f066e179f729c6
Author: fractasy <fractasy@gmail.com>
Date:   Fri Feb 11 12:52:42 2022 +0100

    Add const to keccak arguments

commit 00b49a89aa0922d88f14eb9956b296c6291f70f0
Author: fractasy <fractasy@gmail.com>
Date:   Fri Feb 11 10:27:35 2022 +0100

    Create a new version of Keccak, in preparation for the future Keccak state machine

commit 6265adbf765b5029194f70daa93e722eb8fa3cea
Author: fractasy <fractasy@gmail.com>
Date:   Wed Feb 9 12:01:52 2022 +0100

    Implement Executor::execute() fast mode to avoid storing all evaluations of committed polynomials

commit 2f075417078be24eba16f80a5a89042da5bfc570
Author: fractasy <fractasy@gmail.com>
Date:   Tue Feb 8 18:42:27 2022 +0100

    Server mock on port 50052 returns fake data, for testing

commit 0a8215f88db28a23ada5ba2540fec43c9e80af81
Author: fractasy <fractasy@gmail.com>
Date:   Tue Feb 8 13:07:13 2022 +0100

    Disable runClient in config.json

commit b28bf1339c1086fa52942e01044dea1c8544dfe9
Author: fractasy <fractasy@gmail.com>
Date:   Tue Feb 8 12:58:47 2022 +0100

    Update rom.json containing program and labels root elements

commit 3674b93a7d71c579c991cab85accdd617acd45c6
Merge: 0d4f079 940b98d
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Tue Feb 8 12:29:51 2022 +0100

    Merge pull request #8 from hermeznetwork/NewGRPCProtocol
    
    New grpc protocol

commit 940b98d9bd861b5cb0c3594f45927adeb60803b1
Merge: f39075a 0d4f079
Author: fractasy <89866610+fractasy@users.noreply.github.com>
Date:   Tue Feb 8 12:29:34 2022 +0100

    Merge branch 'main' into NewGRPCProtocol

commit 0d4f079f4e5f7f6704a295d0412c624b6b690842
Author: fractasy <fractasy@gmail.com>
Date:   Mon Feb 7 17:32:15 2022 +0100

    First implementation of ROM opcode changes

commit f39075a430304c15e915c6ee5bad27db52f3f1f6
Author: fractasy <fractasy@gmail.com>
Date:   Mon Feb 7 08:54:46 2022 +0100

    Implement a prover garbage collector to delete old completed requests

commit 40af8b481a11a71eeb68a1643a6fc84a8bf933e1
Author: fractasy <fractasy@gmail.com>
Date:   Sun Feb 6 12:10:58 2022 +0100

    Fix concatenated strings in db, and improve prober test

commit 7be763ea5623075a5902edfd63f06849e05ea2f7
Author: fractasy <fractasy@gmail.com>
Date:   Fri Feb 4 18:06:30 2022 +0100

    First implemenation of new gRPC protocol v0.0.1

commit 1c22bfc58615925e21c0576e8d22203d8458843c
Author: fractasy <fractasy@gmail.com>
Date:   Thu Feb 3 08:36:55 2022 +0100

    Fix call to system, and don't block prover thread if there are pending requests

commit 8ea53d9fd67ea9ba95477f1c97ec8d02abde6ac0
Author: fractasy <fractasy@gmail.com>
Date:   Thu Feb 3 08:04:24 2022 +0100

    Cleanup a couple of TODOs in eval_command.cpp

commit 148a8ea071eb41bbc50f5dbc6d7e1cdb6da3d7f6
Author: fractasy <fractasy@gmail.com>
Date:   Wed Feb 2 20:06:02 2022 +0100

    Cleanup TODOs, mostly pull-request related

commit e3707acb980d4aeed1594bf4bb0f5fc418207f5d
Author: fractasy <fractasy@gmail.com>
Date:   Tue Feb 1 13:10:50 2022 +0100

    Prover asynchronous API: submit a prover request, then wait for it to complete

commit 6c93dd57c8ca1252b36880d961442c1a13b7107f
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jan 31 18:20:35 2022 +0100

    Use ProveContext and config.outputPath

commit d23fd6b9fecef62201e8ce794e9bed1c64a29af2
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jan 28 22:38:40 2022 +0100

    Fix invalid TX corner case

commit 4b627e9d74f4659169f244cb6d1d756f1893b4f9
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jan 28 18:02:54 2022 +0100

    Use config.json to configure the zkProver

commit 75cec4628378396ac672b7aa60ce732c591b8bb6
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jan 28 11:20:09 2022 +0100

    Save timestamp.proof.json and timestamp.public.json to disk

commit 7bc449332f22a592f3ce66b87575927388b724f4
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jan 28 10:30:47 2022 +0100

    Add Prover timestamp and save input to timestamp.input.json

commit 4c009a857e44962e983af0f0a45534a4bbcb685f
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Tue Jan 25 16:46:00 2022 +0000

    Integration changes: GenProof returns State

commit fa9915884c4d8028ad65f1be7d1510a5e0af8b82
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Mon Jan 24 18:49:19 2022 +0000

    Change inXxx pols to be field elements, and eval commands

commit a108078440a1fe36583e6616754458f986372a10
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Mon Jan 24 07:43:18 2022 +0000

    Cleanup warnings

commit 367655be1b8fe4f8463ba6d569cb8fbc4e686339
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Sun Jan 23 10:16:41 2022 +0000

    Add public.json argument and remove rapidsnark_prover.cpp

commit 01f481ceba9be585ed4caa94bca1640251ab382a
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Fri Jan 21 19:34:18 2022 +0000

    Integration complete git add testvectors/stark.zkin.json!

commit e92f6f5a2a9d5811d98fae6534395d6a768f9283
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Fri Jan 21 17:32:48 2022 +0000

    Add parallelization to some BME operations

commit 29f6ad191fdf844c2b3989b9cf16ba6eda36654e
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Jan 21 17:36:02 2022 +0100

    Fix friReduce bug

commit bfcb2b5d55ea8ccd580e26b29e4bae0316be62d7
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Jan 21 16:24:09 2022 +0100

    Fix merge

commit a14f79f571100ae6ee99406be64d4e77b5f01dc5
Merge: 4c5b773 86ebb8b
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Jan 21 16:10:05 2022 +0100

    Merge pull request #7 from hermeznetwork/bug_fixing_execution_hash
    
    Bug fixing and add the execution hash

commit 86ebb8b0a27f17286cc8acebc3e2e5c6971a09c2
Merge: c52635e 4c5b773
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Jan 21 16:08:28 2022 +0100

    Merge branch 'main' into bug_fixing_execution_hash

commit c52635e5544e1a88157b3a2ae4ae4a4d8458af11
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Jan 21 16:01:56 2022 +0100

    add LOG_BME_HASH

commit 8d421e40e6dcc3d552c3ae5f7b73db9da2bd5957
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Jan 21 15:49:12 2022 +0100

    some fixes and i adds printExecutionHash

commit 4c5b7732e588eeee89a7385349d84e241c3d6448
Merge: f6b5411 a3d3ea5
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Fri Jan 21 14:48:38 2022 +0000

    Merge branch 'main' of https://github.com/hermeznetwork/zkproverc

commit f6b54116908faed637c37677845e431f21412314
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Fri Jan 21 14:48:12 2022 +0000

    Move constPols memory copy out of main loop

commit 6f6cb2cd6f54f69ebca4726361f498255657337a
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Jan 21 15:38:48 2022 +0100

    Remove constant trees

commit fadf3162069969449eb97b8773c1e2b88598a2a6
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Jan 21 15:14:32 2022 +0100

    add files to gitignore

commit a3d3ea522eb2b6965c3626da5a3a2c758ebbba29
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Jan 21 13:22:01 2022 +0100

    Update batchmachine_executor.cpp

commit 438321cccb09f52205f2471866f7b45ae2c2858e
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Fri Jan 21 12:10:28 2022 +0000

    Change BME time statistics format

commit 5d1895ad878ca427af4c2046a9d3b584e009195f
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Fri Jan 21 11:30:52 2022 +0000

    Add time statistics per operation to BME

commit f378bc5a02d358333800c6874b2c93bb9ea8f77e
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Fri Jan 21 08:56:03 2022 +0000

    Avoid copying constPols and cmPols of type et_field, since they are already mapped into memory

commit 7ad7463cae3314588d5aad84396561c8c8ebe047
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Thu Jan 20 18:38:07 2022 +0000

    Move zkey load out of main Prover loop.  Disable saving to file.

commit 69e39f485eacb3e54cee93b688f88e3e50baf867
Merge: 811e471 3d429c5
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Thu Jan 20 15:55:57 2022 +0000

    Merge branch 'main' of https://github.com/hermeznetwork/zkproverc

commit 811e471b4942cb5ededfebe147a804c824f004ff
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Thu Jan 20 15:55:13 2022 +0000

    Fix logger mutex attributes initialization, and avoid passing zkin.json as a file

commit 3d429c5e98c34b787c978c786ceff92076cbe97d
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Jan 20 09:57:42 2022 +0100

    Fix op_pol_rotate

commit be00707310b3a37965a086191f1a4c899f665d42
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Thu Jan 20 07:23:15 2022 +0000

    Configuration changes done during the first integration round with the Node

commit 0f2c650657c08d4181af953dbc95ae1fe67f1769
Merge: 4ad37d4 f77b257
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Wed Jan 19 11:47:54 2022 +0000

    Merge branch 'main' of https://github.com/hermeznetwork/zkproverc

commit 4ad37d46682ba165216ac999aa9041a9c46e5c46
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Wed Jan 19 11:46:56 2022 +0000

    ffiasm constant arguments and TODOs cleanup

commit f77b2574a6a6828ecfde1fd85730c2d05168969d
Author: eduadiez <eduardo@polygon.technology>
Date:   Tue Jan 18 19:50:56 2022 +0100

    fix op_pol_grandProduct

commit d622b601359b0c26ab34a14a51ce5e62e3a069db
Author: fractasy <fractasy@ns3175475.ip-51-210-116.eu>
Date:   Tue Jan 18 10:11:44 2022 +0000

    Traces and Makefile cleanup

commit 687758b565a04638641139d84e497743e1ba2aa8
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jan 18 08:26:47 2022 +0100

    Fix Rapid SNARK main method

commit b1c0817abc721c813bbe72a6c537e873ad2b8179
Merge: 49bb2cf 5a9399c
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jan 17 18:03:38 2022 +0100

    Merge branch 'main' of https://github.com/hermeznetwork/zkproverc

commit 49bb2cfadbe247196950284271735fd35778aff8
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jan 17 18:01:57 2022 +0100

    Fix Makefile not to include grpc_unsecure, and Rapid SNARK not to exit when complete

commit 5a9399cba757d67c542509dad08fbbca14be06dc
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Jan 17 12:07:15 2022 +0100

    Update Dockerfile

commit 6cabb59ddf2091c217a98808e840d9cd243baee5
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Jan 17 12:02:16 2022 +0100

    Add protobuf-compiler

commit 4da3c5fc7b92d9d0c4c1a0901b630440556d2fd4
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jan 17 11:09:13 2022 +0100

    Backup BME changes to make output JSON pass next stages

commit 470cd25eeccac6b3c2af418c9259f108d08cfa5e
Merge: f06233c 2859926
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Jan 14 19:01:45 2022 +0100

    Merge pull request #6 from hermeznetwork/fix_makefile
    
    fix Makefile

commit 285992642c3a71c47a7c9b3e364c7fd1aa68105f
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Jan 14 19:00:20 2022 +0100

    fix Makefile

commit f06233c2d7c51f0ea6b4dff9d9ad6b467e1bdfc3
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jan 14 15:41:18 2022 +0100

    Fix BME copy&paste, and proof2zkin

commit ff8cb6ebf107ebec02859f1491617984a7db6b52
Merge: 8535891 bc99e10
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Jan 14 13:13:20 2022 +0100

    Merge pull request #5 from hermeznetwork:fix_getGroupProofSize
    
    Merge pull request #4 from hermeznetwork/edu_additions

commit bc99e108be8615932d6d9843b216e6d4ab0ef04c
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Jan 14 13:12:15 2022 +0100

    Fix getGroupProofSize

commit 8535891fdfa2673a5eadc59d843bd0b1580c2433
Merge: e146439 8e87998
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Jan 14 12:55:04 2022 +0100

    Merge pull request #4 from hermeznetwork/edu_additions
    
    Edu additions

commit e1464392900c8444266b456f847c83d28e91533c
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jan 14 12:52:25 2022 +0100

    Configure database server via command line arguments

commit 8e879983db613df0da26e1f3151045a16beb5b16
Merge: e03e17e 8f20bc2
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Jan 14 12:41:24 2022 +0100

    Merge branch 'edu_additions' of https://github.com/hermeznetwork/zkproverc into edu_additions

commit e03e17edaac51599a4fe961bd5c105d0fcccb139
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Jan 14 12:41:21 2022 +0100

    remove breakpoint

commit 8f20bc2b54adcf4b1d5d93cdf3049d3550f6c033
Merge: ad818e9 b294ac5
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Fri Jan 14 12:38:54 2022 +0100

    Merge branch 'main' into edu_additions

commit ad818e97ba8c7d57ab270ed09fa129a07f60a6b4
Merge: 0961495 acf8565
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Jan 14 12:33:19 2022 +0100

    Merge branch 'refactor' into edu_additions

commit acf85656a9c78011c133e4b8b6d702f933a417f1
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Jan 14 12:30:26 2022 +0100

    Refactor group and element proof size

commit b294ac57de9a18d21c05f4645489d6776e40258c
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jan 14 09:39:09 2022 +0100

    If input.json argument passed, generate proof for this file; otherwise, run the gRPC server

commit c841fddeebfeebd07a9e561ea4fffea498b390dd
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Jan 14 09:22:45 2022 +0100

    all working

commit d6c3c904dc602a0bc497c52139e8eec931fdc771
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jan 13 18:29:20 2022 +0100

    Fix BME json generation code

commit 182a1328f30ba85729378ee647ceac28eee693a9
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jan 13 16:24:58 2022 +0100

    Fix bug in BME, removing fr first argument in self-methods

commit 5c2aab2f4d0623ba7afc9a6057163e7a3bc045b6
Merge: 05c7e3d 0961495
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Thu Jan 13 15:57:34 2022 +0100

    Merge pull request #3 from hermeznetwork/edu_additions
    
    valid up to the line: 3479

commit 0961495d90d02f3619ae22f42e31192c2d9c4b75
Author: eduadiez <eduardo@polygon.technology>
Date:   Thu Jan 13 15:42:15 2022 +0100

    valid up to the line: 3479

commit 05c7e3d82be5e0fb7b68077626fcbf0433bf30ef
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jan 13 14:08:43 2022 +0100

    Database refactoring to access remote server only to read keys not found in local database

commit fc1757f4fc514e961ac81d90bfd8be9ce87eb141
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jan 13 12:22:23 2022 +0100

    Modify timer traces to add starting...

commit bc5218bcaa732961f6b60cd9ead9cfb88bf2fbbf
Author: fractasy <fractasy@gmail.com>
Date:   Thu Jan 13 11:56:08 2022 +0100

    Rapid SNARK integration

commit 9e866be16d16d36e4c64470b8b5349dde5d1d17b
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jan 12 12:39:44 2022 +0100

    Fix verifier_cpp and enable Circom code

commit 2680797b4819670b4b4df5e61526d38c9d669bed
Author: fractasy <fractasy@gmail.com>
Date:   Wed Jan 12 09:04:50 2022 +0100

    Fix mem free crash and progress on Circom integration

commit 3730ed409275c1e90ca29bbf72c84db5047474be
Author: fractasy <fractasy@gmail.com>
Date:   Tue Jan 11 17:12:09 2022 +0100

    Update from zkproverjs changes in constant polynomials number, types and order

commit 44823c74586a148f843e72f4104e34435873ccca
Merge: dd3d0ad 5423a6c
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jan 10 09:46:00 2022 +0100

    Constant arguments in class methods, when possible (FF refactory pending)

commit dd3d0ad5ea31fbd780cd6bed7a040193a58f8842
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jan 10 09:30:34 2022 +0100

    Constant refactoring

commit 5423a6cde9e43539cf57c0393511c3e6485a3dbd
Merge: a35a7dd b122c7a
Author: Eduardo Antuña Díez <20141918+eduadiez@users.noreply.github.com>
Date:   Mon Jan 10 09:17:09 2022 +0100

    Merge pull request #2 from hermeznetwork/edu_additions
    
    Valid up to the line: 438

commit b122c7a6b4ca314267b2a1b7f2110f356a86e34f
Author: eduadiez <eduardo@polygon.technology>
Date:   Mon Jan 10 02:13:34 2022 +0100

    Valid up to the line: 438

commit a35a7dd79725bbac1fef65b9666e3ec01ff2e8ec
Author: fractasy <fractasy@gmail.com>
Date:   Fri Jan 7 09:43:10 2022 +0100

    Instantiate Prover in main.cpp and use it in both manual mode and server mode

commit 3927ed20c19a8747be67149a021f87c2dccd9257
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jan 3 19:21:59 2022 +0100

    Create Prover class to contain all the proving functionality, to be called by gRPC server later

commit 6b16f3a9aabb9f8f2dc57862111234afe8be33c9
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jan 3 17:59:04 2022 +0100

    Store reference to RawFr in Server/Service to init Input

commit 9c6114282e61d5c6a328be858733552e32582c7e
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jan 3 16:36:03 2022 +0100

    Decouple executor from json parsing code

commit 3bc61d0f9f256de6ec04c9cc2e9adb0ed41abc9d
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jan 3 12:18:19 2022 +0100

    Move ROM JSON parsing out of Executor

commit fb6dee20151ee83f2c19cc9b9faec7a81b6736cc
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jan 3 11:55:31 2022 +0100

    Add 2 missing changes in previous commit

commit 81c6fa87b53601dca6dc4d7beda47e980882d91b
Author: fractasy <fractasy@gmail.com>
Date:   Mon Jan 3 11:53:39 2022 +0100

    Add gRPC server and service

commit 890b79d5f4fbe718eee6ecc435af404c4d6081c2
Author: fractasy <fractasy@gmail.com>
Date:   Thu Dec 30 16:38:44 2021 +0100

    Use smt.get() in executor, if USE_LOCAL_STORAGE is not defined

commit e760cdef65a97441a3d6eee4691f5f54de1f8d80
Author: fractasy <fractasy@gmail.com>
Date:   Thu Dec 30 09:21:39 2021 +0100

    Backup

commit 3abcf8ed5df6271f4d5a61a25af0282cabde6827
Author: fractasy <fractasy@gmail.com>
Date:   Mon Dec 27 16:55:15 2021 +0100

    Migrate database to use BYTEA types hash and data

commit 88772087ba5f715468b6b9a3b829db12ae20e1c2
Author: fractasy <fractasy@gmail.com>
Date:   Mon Dec 27 08:54:27 2021 +0100

    Fix some Database dependencies via config.hpp

commit d003d43e33fd57bc960b8b7381521a6e26fd9d92
Author: fractasy <fractasy@gmail.com>
Date:   Fri Dec 24 19:32:34 2021 +0100

    Second version of Database, now working with localhost remote postgres server

commit 204ad3f40bed3ccbac9ce36af51da758fd782ba5
Author: fractasy <fractasy@gmail.com>
Date:   Wed Dec 22 17:13:39 2021 +0100

    Database wrapper, getting prepared for postgres remote database

commit 9b43ae1d2717f846ff346b800796d764391debbb
Author: fractasy <fractasy@gmail.com>
Date:   Wed Dec 22 12:29:45 2021 +0100

    Implement polMulAxi()

commit 87bfeaf44fd48c30e39d2f2dad64cc620b32d848
Author: fractasy <fractasy@gmail.com>
Date:   Wed Dec 22 12:03:26 2021 +0100

    Implementation of evalPol()

commit c54726aa8528112e367a3d782969447b3a65e8ff
Author: fractasy <fractasy@gmail.com>
Date:   Wed Dec 22 09:57:36 2021 +0100

    Integrate verifier_cpp

commit bbff65a070eb9d9c459c956d8688d81425b1c4bb
Author: fractasy <fractasy@gmail.com>
Date:   Wed Dec 22 08:37:59 2021 +0100

    Implement batchInverseTest()

commit ef96109fd9a3f97497cd90bce6c148fcd72dc18a
Author: fractasy <fractasy@gmail.com>
Date:   Tue Dec 21 18:52:51 2021 +0100

    First version of proof2zkin

commit 13b49dab7a8bf9c05cefe1b9dbd38ee87c28d1a6
Author: fractasy <fractasy@gmail.com>
Date:   Tue Dec 21 09:46:53 2021 +0100

    Implement batchInverse and update starkgen_bmscript.json

commit f5fc8a02273125a92d7095e43480403b4d931a3e
Author: fractasy <fractasy@gmail.com>
Date:   Tue Dec 21 08:24:27 2021 +0100

    Backup: working on STARK JSON generation

commit 436f18c6533c3a79ccec4ea1fae80a7e485e58a2
Author: fractasy <fractasy@gmail.com>
Date:   Sun Dec 19 20:41:15 2021 +0100

    Fix printReference()

commit 66d72ff81c90156f7101275426ed3f2f11f34740
Author: fractasy <fractasy@gmail.com>
Date:   Sun Dec 19 18:54:07 2021 +0100

    Use eOperation instead of strings, for performance reasons

commit 63a5d5a6f9f67f1af05efcddbb80311d0eccbaff
Author: fractasy <fractasy@gmail.com>
Date:   Sat Dec 18 20:56:26 2021 +0100

    Add printReference() to allow debugging of BME

commit 03f5fb52ae66cf483704a8fc52b93de1afc037fa
Author: fractasy <fractasy@gmail.com>
Date:   Sat Dec 18 18:50:30 2021 +0100

    First implementation of calculateH1H2, disabled by now

commit b02f4d408443fa9250e5d3449fa2384421434e90
Author: fractasy <fractasy@gmail.com>
Date:   Fri Dec 17 20:03:31 2021 +0100

    Implement idx operations in BME

commit fe038435157f4abe45b82480fc7a5b921cf68ffc
Author: fractasy <fractasy@gmail.com>
Date:   Fri Dec 17 19:05:44 2021 +0100

    Warnings cleanup

commit cf0cd641412ed82e4ff3f6cfa7978ab3d9fab538
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Dec 17 16:24:48 2021 +0100

    Revert to g++

commit 3ac285205e764aad58b583dda0a4778b6c96a6cd
Author: fractasy <fractasy@gmail.com>
Date:   Fri Dec 17 13:30:21 2021 +0100

    Fixes done during Edu's build system integration call

commit 4c822fe08cea61dac07d61f920d9990d932e8d4b
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Dec 17 12:09:15 2021 +0100

    add ASFLAGS

commit f968a8c541192b26a0c2e00f31b08add0c14e753
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Dec 17 12:03:34 2021 +0100

    add debug mode

commit f211a276a18e779f0735c9fb55ead02961051d81
Merge: 38bf06b 2905353
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Dec 17 11:46:35 2021 +0100

    Merge branch 'merkle'

commit 2905353f604062c745c63ab2a661f10a5eb8baad
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Dec 17 11:44:18 2021 +0100

    fix minor errors

commit 38bf06b6ddc062bffa2717b5a2f309d000304751
Author: fractasy <fractasy@gmail.com>
Date:   Fri Dec 17 11:05:23 2021 +0100

    First version of batch machine executor (BME)

commit ddb182e93ea3992713233107e502a58dd0aa2931
Author: eduadiez <eduardo@polygon.technology>
Date:   Fri Dec 17 08:59:53 2021 +0100

    add dockerfile

commit 2a342b4a5548b7f576622ead5760b5d8a3a4cce8
Author: eduadiez <eduardo@polygon.technology>
Date:   Thu Dec 16 18:55:51 2021 +0100

    Add merkle_group_multipol

commit 195b3dc224da12ff830ad02a4fee0c909ba3de8f
Author: eduadiez <eduardo@polygon.technology>
Date:   Thu Dec 16 18:55:22 2021 +0100

    Add merkle

commit 5aa4902b4639af412cc8e15c1057a6589d755416
Author: eduadiez <eduardo@polygon.technology>
Date:   Thu Dec 16 18:54:48 2021 +0100

    Add makefile and refactor tasks.json and launch.json

commit 5b4924dd51092bba5a377d371e3f6fd6d13145e5
Author: eduadiez <eduardo@polygon.technology>
Date:   Thu Dec 16 18:53:52 2021 +0100

    Fix typo

commit 114512844cd364236036d0d35b96f4b2d764f2af
Author: eduadiez <eduardo@polygon.technology>
Date:   Thu Dec 16 18:51:19 2021 +0100

    delete unnecessary files

commit c55647c2e5136e06478820d6acf0c8c6eb6badd8
Author: fractasy <fractasy@gmail.com>
Date:   Thu Dec 16 16:16:21 2021 +0100

    First version of script JSON parser

commit cde8c6af4263ab23af454e375fa5ad08d5577a78
Author: fractasy <fractasy@gmail.com>
Date:   Thu Dec 16 12:35:25 2021 +0100

    Backup copy, to cleanup later

commit a02e25a416610e1ba161f2c05278af49a87ba5d6
Author: fractasy <fractasy@gmail.com>
Date:   Thu Dec 16 11:20:35 2021 +0100

    Draft changes to StarkGen

commit f610986277757692918eebcddab1d02ccedaf3c7
Author: fractasy <fractasy@gmail.com>
Date:   Wed Dec 15 13:11:24 2021 +0100

    Parse PIL JSON file in a separate class, called at the main level

commit d12bcb2f768a6688be17e2895618ceb35d3e4479
Author: fractasy <fractasy@gmail.com>
Date:   Wed Dec 15 10:59:47 2021 +0100

    Create stark generator files

commit 18844943d22bf5dc4a117f467383231b20be22e3
Author: fractasy <fractasy@gmail.com>
Date:   Tue Dec 14 19:48:34 2021 +0100

    Parse and load pols out of executor

commit 60a44d0350b6d3901de4836a73940093a0b4555c
Author: fractasy <fractasy@gmail.com>
Date:   Tue Dec 14 08:51:26 2021 +0100

    Fix SMT to process 200 txs

commit a38014978757614b3644486511905645b2b6cded
Author: fractasy <fractasy@gmail.com>
Date:   Mon Dec 13 18:58:49 2021 +0100

    Fix RLP data size for r and s values

commit da9045c799f07ade79705b19cfa4c9801ae7134b
Author: fractasy <fractasy@gmail.com>
Date:   Mon Dec 13 18:16:30 2021 +0100

    Split Executor and Pols in init+loop

commit 7069067fd2ad4c705add0cdd437729dc3933c5f3
Author: fractasy <fractasy@gmail.com>
Date:   Mon Dec 13 12:01:52 2021 +0100

    Fix main arguments parsing

commit d86b1e4564303040259ed3b96f2284cc6db1edc3
Author: fractasy <fractasy@gmail.com>
Date:   Mon Dec 13 11:36:23 2021 +0100

    Avoid setting pol(x)[i]=0, since it is redundant

commit 0a69d24bf962b3734c953023174922881a2c9822
Author: fractasy <fractasy@gmail.com>
Date:   Mon Dec 13 08:22:52 2021 +0100

    Refactor timer traces

commit 9d13e28e3ffee093c11e86f8a0268faf85150d6e
Author: fractasy <fractasy@gmail.com>
Date:   Fri Dec 10 18:31:52 2021 +0100

    Improve time traces

commit 0b3e37d1429e709d7195e8a816bdd79d12325c40
Author: fractasy <fractasy@gmail.com>
Date:   Fri Dec 10 13:50:04 2021 +0100

    Add time logs for performance analysis

commit af4c2c71ef0a2d83a2c008082901093189160566
Author: fractasy <fractasy@gmail.com>
Date:   Fri Dec 10 11:33:02 2021 +0100

    Improve RLP value generation, and add comment to executor

commit c2cd85e0d792d7368e13c25f8a67bd1e027aa3c3
Author: fractasy <fractasy@gmail.com>
Date:   Thu Dec 9 16:33:22 2021 +0100

    Remove src/keccak-tiny/*

commit 76f49b2f419becda82476f6cefec5380151a81d6
Author: fractasy <fractasy@gmail.com>
Date:   Thu Dec 9 16:30:42 2021 +0100

    Fix RLP numeric value limitation

commit 368330640879f3737b86337f5381935e3e437684
Author: fractasy <fractasy@gmail.com>
Date:   Thu Dec 9 12:42:15 2021 +0100

    Cleanup

commit 6c50fa454ce4b71183f0cc72707eae50cce04153
Author: fractasy <fractasy@gmail.com>
Date:   Thu Dec 9 12:19:52 2021 +0100

    Refactoring of polynomials

commit 9fe55ac236f0e4bda93921d2452b0bd7133098fe
Author: fractasy <fractasy@gmail.com>
Date:   Thu Dec 9 11:06:18 2021 +0100

    Refactoring and cleanup

commit 2f507fefb1dfbe593ed9d27a6c2e07317dfc2975
Author: fractasy <fractasy@gmail.com>
Date:   Tue Dec 7 20:14:30 2021 +0100

    Add comments

commit 0fb3717433c0c464d77df462ece5311fe6619684
Author: fractasy <fractasy@gmail.com>
Date:   Tue Dec 7 19:23:07 2021 +0100

    Add execution time logs, add comments, and cleanup code

commit 60af33ed44abfb6f5de6c26c143e0451819c4828
Author: fractasy <fractasy@gmail.com>
Date:   Tue Dec 7 13:35:48 2021 +0100

    Fix ecrecover()

commit fa167645a946d0c22ab5840018ca424b1e28ebbe
Author: fractasy <fractasy@gmail.com>
Date:   Tue Dec 7 11:13:41 2021 +0100

    Integrate the correct, working version of keccak from XKCP, and fix RLP

commit 9f0deaea7d8ea42dec58c4ec5968c1b08f6b098e
Author: fractasy <fractasy@gmail.com>
Date:   Tue Dec 7 08:31:40 2021 +0100

    Fix byte4 and hardcoded hashes.  Now, all evaluations are run.

commit aec35177005aaf1466902fa1a2458820043e273c
Author: fractasy <fractasy@gmail.com>
Date:   Fri Dec 3 12:56:44 2021 +0100

    Integrate RLPValue and fix bugs to complete all evaluations

commit 7b01a92099b5886d1631399aea75e120546d8227
Author: fractasy <fractasy@gmail.com>
Date:   Tue Nov 30 16:04:49 2021 +0100

    Fix management of signed integers in CONST values

commit 4400e5477e8a0070076cf6e9f0a272f33118c5c6
Author: fractasy <fractasy@gmail.com>
Date:   Tue Nov 30 10:32:54 2021 +0100

    Integrate the call to keccak_256

commit 30e4c9b4b7ad3615cfe1cdaa4b375eaffd0e63b9
Author: fractasy <fractasy@gmail.com>
Date:   Mon Nov 29 19:22:25 2021 +0100

    Add prime, logs, ecrecover and keccak libraries

commit 544580ccf85aecbc04af4138c42d97cd42429301
Author: fractasy <fractasy@gmail.com>
Date:   Thu Nov 25 19:31:32 2021 +0100

    Fix bugs from JS vs. C parallel debugging

commit 3fbb489b6aa516aec8505a2aec923b642bb25615
Author: fractasy <fractasy@gmail.com>
Date:   Thu Nov 25 09:42:10 2021 +0100

    Update rom.json

commit 14706bccaca6010cdee6364bae3d9c61b37cd8bf
Author: fractasy <fractasy@gmail.com>
Date:   Thu Nov 25 09:38:15 2021 +0100

    Decouple SMT code from Executor code

commit 0f6530321ac679d03ec51ca0fa1aa4833572799a
Author: fractasy <fractasy@gmail.com>
Date:   Wed Nov 24 18:19:32 2021 +0100

    Pass arguments by reference in constructors to avoid storing pointers

commit b4bcb47fa23dad9a936afcbc3a3bdc34e511010e
Author: fractasy <fractasy@gmail.com>
Date:   Wed Nov 24 17:12:01 2021 +0100

    Implement SMT

commit 61ec7d387068a5b06e3bfc7d719fa4b1f068f8af
Author: fractasy <fractasy@gmail.com>
Date:   Tue Nov 23 19:28:39 2021 +0100

    Use CommandResult in eval_commands.cpp, and complete its implementation

commit c575cda3d6b8f29bbf4efbca60111b6345d1c1c3
Author: fractasy <fractasy@gmail.com>
Date:   Tue Nov 23 12:45:23 2021 +0100

    Use mpz_class instead of mpz_t

commit 02b9a0ea805dc540f396de4d90bbe450e3d00ccd
Author: fractasy <fractasy@gmail.com>
Date:   Mon Nov 22 18:17:13 2021 +0100

    Add a comparison operand class to create maps with key=field element.

commit df5d4792b8bd8bbc873e05bab8544d9646a79408
Author: fractasy <fractasy@gmail.com>
Date:   Mon Nov 22 17:44:21 2021 +0100

    Integrate Poseidon OPT.  Migrate some code in main executor loop.

commit 49e9d62bf51eb3565f3865a8497f861305d98af7
Author: fractasy <fractasy@gmail.com>
Date:   Fri Nov 19 19:33:38 2021 +0100

    Implement byte4 logic

commit 9fcee70dbc0cad135243b9f88a2de87990ce92ad
Author: fractasy <fractasy@gmail.com>
Date:   Fri Nov 19 18:59:55 2021 +0100

    Change ROM line structure to match polynomial element types, allowing direct assignment

commit fa101adfac0e7b4010b9c77b17b556cfcbea299c
Author: fractasy <fractasy@gmail.com>
Date:   Thu Nov 18 19:00:02 2021 +0100

    Refactor to use polynomial element sizes according to the element type

commit acf94b1a0de460ef4489bb9f600286ab9d95f992
Author: fractasy <fractasy@gmail.com>
Date:   Wed Nov 17 17:09:40 2021 +0100

    Final split in files

commit 09e0798969d142e2a921640ac6f44241b0bf4211
Author: fractasy <fractasy@gmail.com>
Date:   Wed Nov 17 12:03:04 2021 +0100

    Fix inc, dec, ind.  Implement nexti and checkFinalState()

commit f35f18d0726cee679cb9c38ca1dc838a69b6949f
Author: fractasy <fractasy@gmail.com>
Date:   Wed Nov 17 10:46:07 2021 +0100

    Split pols code in separate files

commit ffcfa534a4c84720a851c89c61da2a658f969746
Author: fractasy <fractasy@gmail.com>
Date:   Wed Nov 17 10:24:32 2021 +0100

    Split some code into individual files to improve project management

commit 4ee647be67111e145d805646b67b84ef9b0a39aa
Author: fractasy <fractasy@gmail.com>
Date:   Tue Nov 16 16:17:01 2021 +0100

    Update json files and integrate them

commit 957ed2da7f7311b77e9dbe7591118fc66ff06553
Author: fractasy <fractasy@gmail.com>
Date:   Tue Nov 16 12:12:08 2021 +0100

    Move cmdBefore and cmdAfter parsing out of the main loop

commit 7b65f076ad78f4b940cb8e5f40f7813b0790a283
Author: fractasy <fractasy@gmail.com>
Date:   Mon Nov 15 19:05:45 2021 +0100

    Move JSON out of the main executor loop; missing cmdBefore and cmdAfter

commit fcacb9d1ad27aef7ef889dc183e2f18643adf4ca
Author: fractasy <fractasy@gmail.com>
Date:   Sun Nov 14 12:35:21 2021 +0100

    Daily backup

commit de1e293d27395766c3bec02db3add9b86fcd038e
Author: fractasy <fractasy@gmail.com>
Date:   Sat Nov 13 18:43:21 2021 +0100

    Daily backup

commit 7ea12d2b7a9f763c21f65f13b6a33e0635ca2934
Author: fractasy <fractasy@gmail.com>
Date:   Sat Nov 13 11:25:19 2021 +0100

    Daily backup

commit 11fa2bf45cfb580116f04757cd84042bf3d7d12c
Author: fractasy <fractasy@gmail.com>
Date:   Thu Nov 11 16:45:36 2021 +0100

    Daily backup

commit 024adbd023e0122a634a2be9c3099e5b07885ed8
Author: fractasy <fractasy@gmail.com>
Date:   Wed Nov 10 16:47:36 2021 +0100

    Daily backup

commit 0dd39669da02bb48d96fc717abaf4bb230a686f9
Author: Jordi Baylina <jordi@baylina.cat>
Date:   Mon Nov 8 12:02:28 2021 +0100

    first commit
