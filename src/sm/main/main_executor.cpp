#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <gmpxx.h>
#include "config.hpp"
#include "main_executor.hpp"
#include "rom_line.hpp"
#include "rom_command.hpp"
#include "rom.hpp"
#include "context.hpp"
#include "input.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "eval_command.hpp"
#include "statedb_factory.hpp"
#include "goldilocks/goldilocks_base_field.hpp"
#include "ffiasm/fec.hpp"
#include "ffiasm/fnec.hpp"
#include "timer.hpp"
#include "eth_opcodes.hpp"
#include "opcode_address.hpp"

using namespace std;
using json = nlohmann::json;

#define MEM_OFFSET 0x30000
#define STACK_OFFSET 0x20000
#define CODE_OFFSET 0x10000
#define CTX_OFFSET 0x40000

MainExecutor::MainExecutor (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config) :
    fr(fr),
    N(MainCommitPols::pilDegree()),
    poseidon(poseidon),
    config(config)
{
    /* Get a StateDBClient interface, according to the configuration */
    pStateDB = StateDBClientFactory::createStateDBClient(fr, config);
    if (pStateDB == NULL)
    {
        cerr << "Error: MainExecutor::MainExecutor() failed calling StateDBClientFactory::createStateDBClient()" << endl;
        exit(-1);
    }

    /* Load and parse ROM JSON file */

    TimerStart(ROM_LOAD);

    // Check rom file name
    if (config.romFile.size()==0)
    {
        cerr << "Error: ROM file name is empty" << endl;
        exit(-1);
    }

    // Load file contents into a json instance
    json romJson;
    file2json(config.romFile, romJson);

    // Load program array in Rom instance
    if (!romJson.contains("program") ||
        !romJson["program"].is_array() )
    {
        cerr << "Error: ROM file does not contain a program array at root level" << endl;
        exit(-1);
    }
    //Rom romData;
    rom.load(fr, romJson["program"]);

    // Initialize the Ethereum opcode list: opcode=array position, operation=position content
    ethOpcodeInit();

    // Use the rom labels object to map every opcode to a ROM address
    if (!romJson.contains("labels") ||
        !romJson["labels"].is_object() )
    {
        cerr << "Error: ROM file does not contain a labels object at root level" << endl;
        exit(-1);
    }
    opcodeAddressInit(romJson["labels"]);

    TimerStopAndLog(ROM_LOAD);
};

MainExecutor::~MainExecutor ()
{
    delete pStateDB;
}

void MainExecutor::execute (ProverRequest &proverRequest, MainCommitPols &pols, MainExecRequired &required)
{
    TimerStart(EXECUTE_INITIALIZATION);
    
#ifdef LOG_TIME
    uint64_t poseidonTime=0, poseidonTimes=0;
    uint64_t smtTime=0, smtTimes=0;
    uint64_t keccakTime=0, keccakTimes=0;
#endif

    bool &bFastMode(proverRequest.bFastMode);
    bool &bProcessBatch(proverRequest.bProcessBatch);

    // Create context and store a finite field reference in it
    Context ctx(fr, fec, fnec, pols, rom, proverRequest);

    /* Sets first evaluation of all polynomials to zero */
    //initState(ctx);

#ifdef LOG_COMPLETED_STEPS_TO_FILE
    remove("c.txt");
#endif

#ifdef USE_LOCAL_STORAGE
    /* Copy input storage content into context storage */
    map< Goldilocks::Element, mpz_class, CompareFe>::iterator itsto;
    for (itsto=input.sto.begin(); itsto!=input.sto.end(); itsto++)
    {
        fe = itsto->first;
        ctx.sto[fe] = itsto->second;
    }
#endif

    if (proverRequest.input.db.size() > 0)
    {
        Database * pDatabase = pStateDB->getDatabase();
        if (pDatabase != NULL)
        {
            /* Copy input database content into context database */
            map< string, vector<Goldilocks::Element> >::const_iterator it;
            for (it=proverRequest.input.db.begin(); it!=proverRequest.input.db.end(); it++)
            {
                pDatabase->write(it->first, it->second, false);
            }
        }
    }

    // opN are local, uncommitted polynomials
    Goldilocks::Element op0, op1, op2, op3, op4, op5, op6, op7;

    uint64_t zkPC = 0; // Zero-knowledge program counter
    uint64_t step = 0; // Step, number of polynomial evaluation
    uint64_t i; // Step, as it is used internally, set to 0 in fast mode to reuse the same evaluation all the time
    uint64_t nexti; // Next step, as it is used internally, set to 0 in fast mode to reuse the same evaluation all the time
    ctx.N = N;
    ctx.pStep = &i; // ctx.pStep is used inside evaluateCommand() to find the current value of the registers, e.g. pols(A0)[ctx.step]
    ctx.pZKPC = &zkPC;

    TimerStopAndLog(EXECUTE_INITIALIZATION);

    TimerStart(EXECUTE_LOOP);

    for (step=0; step<N; step++)
    {
        if (bFastMode)
        {
            i = 0;
            nexti = 0;
            pols.FREE0[i] = fr.zero();
            pols.FREE1[i] = fr.zero();
            pols.FREE2[i] = fr.zero();
            pols.FREE3[i] = fr.zero();
            pols.FREE4[i] = fr.zero();
            pols.FREE5[i] = fr.zero();
            pols.FREE6[i] = fr.zero();
            pols.FREE7[i] = fr.zero();
        }
        else
        {
            i = step;
            // Calculate nexti to write the next evaluation register values according to setX
            // The registers of the evaluation 0 will be overwritten with the values from the last evaluation, closing the evaluation circle
            nexti = (i+1)%N;
        }
        zkPC = fr.toU64(pols.zkPC[i]); // This is the read line of ZK code

        uint64_t incHashPos = 0;
        uint64_t incCounter = 0;

#ifdef LOG_START_STEPS
        cout << "--> Starting step=" << step << " zkPC=" << zkPC << " zkasm=" << rom.line[zkPC].lineStr << endl;
#endif
#ifdef LOG_PRINT_ROM_LINES
        cout << "step=" << step << " rom.line[" << zkPC << "] =" << rom.line[zkPC].toString(fr) << endl;
#endif
#ifdef LOG_START_STEPS_TO_FILE
        {
        std::ofstream outfile;
        outfile.open("c.txt", std::ios_base::app); // append instead of overwrite
        outfile << "--> Starting step=" << step << " zkPC=" << zkPC << " zkasm=" << rom.line[zkPC].lineStr << endl;
        outfile.close();
        }
#endif

#ifdef LOG_FILENAME
        // Store fileName and line
        ctx.fileName = rom.line[zkPC].fileName;
        ctx.line = rom.line[zkPC].line;
#endif

        // Evaluate the list cmdBefore commands, and any children command, recursively
        for (uint64_t j=0; j<rom.line[zkPC].cmdBefore.size(); j++)
        {
            CommandResult cr;
            evalCommand(ctx, *rom.line[zkPC].cmdBefore[j], cr);
        }

        // Initialize the local registers to zero
        op0 = fr.zero();
        op1 = fr.zero();
        op2 = fr.zero();
        op3 = fr.zero();
        op4 = fr.zero();
        op5 = fr.zero();
        op6 = fr.zero();
        op7 = fr.zero();

        /*************/
        /* SELECTORS */
        /*************/

        // inX adds the corresponding register values to the op local register set, multiplied by inX
        // In case several inXs are set to !=0, those values will be added together to opN
        // e.g. op0 = inX*X0 + inY*Y0 + inZ*Z0 +...

        // If inA, op = op + inA*A
        if (!fr.isZero(rom.line[zkPC].inA))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inA, pols.A0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inA, pols.A1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inA, pols.A2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inA, pols.A3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inA, pols.A4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inA, pols.A5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inA, pols.A6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inA, pols.A7[i]));

            pols.inA[i] = rom.line[zkPC].inA;

#ifdef LOG_INX
            cout << "inA op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inB, op = op + inB*B
        if (!fr.isZero(rom.line[zkPC].inB))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inB, pols.B0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inB, pols.B1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inB, pols.B2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inB, pols.B3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inB, pols.B4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inB, pols.B5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inB, pols.B6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inB, pols.B7[i]));

            pols.inB[i] = rom.line[zkPC].inB;
            
#ifdef LOG_INX
            cout << "inB op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inA, op = op + inA*A
        if (!fr.isZero(rom.line[zkPC].inC))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inC, pols.C0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inC, pols.C1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inC, pols.C2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inC, pols.C3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inC, pols.C4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inC, pols.C5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inC, pols.C6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inC, pols.C7[i]));

            pols.inC[i] = rom.line[zkPC].inC;
            
#ifdef LOG_INX
            cout << "inC op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inD, op = op + inD*D
        if (!fr.isZero(rom.line[zkPC].inD))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inD, pols.D0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inD, pols.D1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inD, pols.D2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inD, pols.D3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inD, pols.D4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inD, pols.D5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inD, pols.D6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inD, pols.D7[i]));

            pols.inD[i] = rom.line[zkPC].inD;
            
#ifdef LOG_INX
            cout << "inD op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inE, op = op + inE*E
        if (!fr.isZero(rom.line[zkPC].inE))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inE, pols.E0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inE, pols.E1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inE, pols.E2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inE, pols.E3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inE, pols.E4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inE, pols.E5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inE, pols.E6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inE, pols.E7[i]));

            pols.inE[i] = rom.line[zkPC].inE;
            
#ifdef LOG_INX
            cout << "inE op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inSR, op = op + inSR*SR
        if (!fr.isZero(rom.line[zkPC].inSR))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inSR, pols.SR0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inSR, pols.SR1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inSR, pols.SR2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inSR, pols.SR3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inSR, pols.SR4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inSR, pols.SR5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inSR, pols.SR6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inSR, pols.SR7[i]));

            pols.inSR[i] = rom.line[zkPC].inSR;
            
#ifdef LOG_INX
            cout << "inSR op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inCTX, op = op + inCTX*CTX
        if (!fr.isZero(rom.line[zkPC].inCTX))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCTX, pols.CTX[i]));
            pols.inCTX[i] = rom.line[zkPC].inCTX;
#ifdef LOG_INX
            cout << "inCTX op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inSP, op = op + inSP*SP
        if (!fr.isZero(rom.line[zkPC].inSP))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inSP, pols.SP[i]));
            pols.inSP[i] = rom.line[zkPC].inSP;
#ifdef LOG_INX
            cout << "inSP op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inPC, op = op + inPC*PC
        if (!fr.isZero(rom.line[zkPC].inPC))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inPC, pols.PC[i]));
            pols.inPC[i] = rom.line[zkPC].inPC;
#ifdef LOG_INX
            cout << "inPC op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inGAS, op = op + inGAS*GAS
        if (!fr.isZero(rom.line[zkPC].inGAS))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inGAS, pols.GAS[i]));
            pols.inGAS[i] = rom.line[zkPC].inGAS;
#ifdef LOG_INX
            cout << "inGAS op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inMAXMEM, op = op + inMAXMEM*MAXMEM
        if (!fr.isZero(rom.line[zkPC].inMAXMEM))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inMAXMEM, pols.MAXMEM[i]));
            pols.inMAXMEM[i] = rom.line[zkPC].inMAXMEM;
#ifdef LOG_INX
            cout << "inMAXMEM op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inSTEP, op = op + inSTEP*STEP
        if (!fr.isZero(rom.line[zkPC].inSTEP))
        {
            op0 = fr.add(op0, fr.mul( rom.line[zkPC].inSTEP, fr.fromU64(step) ));
            pols.inSTEP[i] = rom.line[zkPC].inSTEP;
#ifdef LOG_INX
            cout << "inSTEP op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inRR, op = op + inRR*RR
        if (!fr.isZero(rom.line[zkPC].inRR))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inRR, pols.RR[i]));
            pols.inRR[i] = rom.line[zkPC].inRR;
#ifdef LOG_INX
            cout << "inRR op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inHASHPOS, op = op + inHASHPOS*HASHPOS
        if (!fr.isZero(rom.line[zkPC].inHASHPOS))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inHASHPOS, pols.HASHPOS[i]));
            pols.inHASHPOS[i] = rom.line[zkPC].inHASHPOS;
#ifdef LOG_INX
            cout << "inHASHPOS op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inCntArith, op = op + inCntArith*cntArith
        if (!fr.isZero(rom.line[zkPC].inCntArith))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCntArith, pols.cntArith[i]));
            pols.inCntArith[i] = rom.line[zkPC].inCntArith;
#ifdef LOG_INX
            cout << "inCntArith op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inCntBinary, op = op + inCntBinary*cntBinary
        if (!fr.isZero(rom.line[zkPC].inCntBinary))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCntBinary, pols.cntBinary[i]));
            pols.inCntBinary[i] = rom.line[zkPC].inCntBinary;
#ifdef LOG_INX
            cout << "inCntBinary op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inCntMemAlign, op = op + inCntMemAlign*cntMemAlign
        if (!fr.isZero(rom.line[zkPC].inCntMemAlign))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCntMemAlign, pols.cntMemAlign[i]));
            pols.inCntMemAlign[i] = rom.line[zkPC].inCntMemAlign;
#ifdef LOG_INX
            cout << "inCntMemAlign op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inCntKeccakF, op = op + inCntKeccakF*cntKeccakF
        if (!fr.isZero(rom.line[zkPC].inCntKeccakF))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCntKeccakF, pols.cntKeccakF[i]));
            pols.inCntKeccakF[i] = rom.line[zkPC].inCntKeccakF;
#ifdef LOG_INX
            cout << "inCntKeccakF op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inCntPoseidonG, op = op + inCntPoseidonG*cntPoseidonG
        if (!fr.isZero(rom.line[zkPC].inCntPoseidonG))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCntPoseidonG, pols.cntPoseidonG[i]));
            pols.inCntPoseidonG[i] = rom.line[zkPC].inCntPoseidonG;
#ifdef LOG_INX
            cout << "inCntPoseidonG op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inCntPaddingPG, op = op + inCntPaddingPG*cntPaddingPG
        if (!fr.isZero(rom.line[zkPC].inCntPaddingPG))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCntPaddingPG, pols.cntPaddingPG[i]));
            pols.inCntPaddingPG[i] = rom.line[zkPC].inCntPaddingPG;
#ifdef LOG_INX
            cout << "inCntPaddingPG op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inROTL_C, op = C rotated left
        if (!fr.isZero(rom.line[zkPC].inROTL_C))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inROTL_C, pols.C7[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inROTL_C, pols.C0[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inROTL_C, pols.C1[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inROTL_C, pols.C2[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inROTL_C, pols.C3[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inROTL_C, pols.C4[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inROTL_C, pols.C5[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inROTL_C, pols.C6[i]));
            
            pols.inROTL_C[i] = rom.line[zkPC].inROTL_C;
        }

        // If inCONST, op = op + CONST
        if (rom.line[zkPC].bConstLPresent)
        {
            scalar2fea(fr, rom.line[zkPC].CONSTL, op0, op1, op2, op3, op4, op5, op6, op7);
            pols.CONST0[i] = op0;
            pols.CONST1[i] = op1;
            pols.CONST2[i] = op2;
            pols.CONST3[i] = op3;
            pols.CONST4[i] = op4;
            pols.CONST5[i] = op5;
            pols.CONST6[i] = op6;
            pols.CONST7[i] = op7;
#ifdef LOG_INX
            cout << "CONSTL op=" << rom.line[zkPC].CONSTL.get_str(16) << endl;
#endif
        }
        else if (rom.line[zkPC].bConstPresent)
        {
            op0 = fr.add(op0, rom.line[zkPC].CONST);
            pols.CONST0[i] = rom.line[zkPC].CONST;
#ifdef LOG_INX
            cout << "CONST op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // Relative and absolute address auxiliary variables
        uint32_t addrRel = 0;
        uint64_t addr = 0;

        // If address is involved, load offset into addr
        if (rom.line[zkPC].mOp==1 || rom.line[zkPC].mWR==1 || rom.line[zkPC].hashK==1 || rom.line[zkPC].hashKLen==1 || rom.line[zkPC].hashKDigest==1 || rom.line[zkPC].hashP==1 || rom.line[zkPC].hashPLen==1 || rom.line[zkPC].hashPDigest==1 || rom.line[zkPC].JMP==1 || rom.line[zkPC].JMPN==1 || rom.line[zkPC].JMPC==1) {
            if (rom.line[zkPC].ind == 1)
            {
                addrRel = fr.toS32(pols.E0[i]);
            }
            if (rom.line[zkPC].indRR == 1)
            {
                addrRel = fr.toS32(pols.RR[i]);
            }
            if (rom.line[zkPC].bOffsetPresent && rom.line[zkPC].offset!=0)
            {
                // If offset is possitive, and the sum is too big, fail
                if (rom.line[zkPC].offset>0 && (uint64_t(addrRel)+uint64_t(rom.line[zkPC].offset))>=0x10000)
                {
                    cerr << "Error: addrRel >= 0x10000 ln: " << zkPC << endl;
                    exit(-1);                  
                }
                // If offset is negative, and its modulo is bigger than addrRel, fail
                if (rom.line[zkPC].offset<0 && (-rom.line[zkPC].offset)>addrRel)
                {
                    cerr << "Error: addrRel < 0 ln: " << zkPC << endl;
                    exit(-1);
                }
                addrRel += rom.line[zkPC].offset;
            }
            addr = addrRel;
#ifdef LOG_ADDR
            cout << "Any addr=" << addr << endl;
#endif
        }

        // If useCTX, addr = addr + CTX*CTX_OFFSET
        if (rom.line[zkPC].useCTX == 1) {
            addr += fr.toU64(pols.CTX[i])*CTX_OFFSET;
            pols.useCTX[i] = fr.one();
#ifdef LOG_ADDR
            cout << "useCTX addr=" << addr << endl;
#endif
        }

        // If isCode, addr = addr + CODE_OFFSET
        if (rom.line[zkPC].isCode == 1) {
            addr += CODE_OFFSET;
            pols.isCode[i] = fr.one();
#ifdef LOG_ADDR
            cout << "isCode addr=" << addr << endl;
#endif
        }

        // If isStack, addr = addr + STACK_OFFSET
        if (rom.line[zkPC].isStack == 1) {
            addr += STACK_OFFSET;
            addr += fr.toU64(pols.SP[i]);
            pols.isStack[i] = fr.one();
#ifdef LOG_ADDR
            cout << "isStack addr=" << addr << endl;
#endif
        }

        // If isMem, addr = addr + MEM_OFFSET
        if (rom.line[zkPC].isMem == 1) {
            addr += MEM_OFFSET;
            pols.isMem[i] = fr.one();
#ifdef LOG_ADDR
            cout << "isMem addr=" << addr << endl;
#endif
        }

        // Copy ROM flags into the polynomials
        if (rom.line[zkPC].incCode != 0)
        {
            pols.incCode[i] = fr.fromS32(rom.line[zkPC].incCode);
        }
        if (rom.line[zkPC].incStack != 0)
        {
            pols.incStack[i] = fr.fromS32(rom.line[zkPC].incStack);
        }
        if (rom.line[zkPC].ind == 1)
        {
            pols.ind[i] = fr.one();
        }
        if (rom.line[zkPC].indRR == 1)
        {
            pols.indRR[i] = fr.one();
        }

        // If offset, record it the committed polynomial
        if (rom.line[zkPC].bOffsetPresent && (rom.line[zkPC].offset!=0))
        {
            pols.offset[i] = fr.fromU64(rom.line[zkPC].offset);
        }

        /**************/
        /* FREE INPUT */
        /**************/

        // If inFREE, calculate the free input value, and add it to op
        if (!fr.isZero(rom.line[zkPC].inFREE))
        {
            // freeInTag must be present
            if (rom.line[zkPC].freeInTag.isPresent == false) {
                cerr << "Error: Instruction with freeIn without freeInTag: zkPC=" << zkPC << endl;
                exit(-1);
            }

            // Store free value here, and add it to op later
            Goldilocks::Element fi0;
            Goldilocks::Element fi1;
            Goldilocks::Element fi2;
            Goldilocks::Element fi3;
            Goldilocks::Element fi4;
            Goldilocks::Element fi5;
            Goldilocks::Element fi6;
            Goldilocks::Element fi7;

            // If there is no operation specified in freeInTag.op, then get the free value directly from the corresponding source
            if (rom.line[zkPC].freeInTag.op == "") {
                uint64_t nHits = 0;

                // If mRD (memory read) get fi=mem[addr], if it exsists
                if ( (rom.line[zkPC].mOp==1) && (rom.line[zkPC].mWR==0) )
                {
                    if (ctx.mem.find(addr) != ctx.mem.end()) {
#ifdef LOG_MEMORY
                        cout << "Memory read mRD: addr:" << addr << " " << printFea(ctx, ctx.mem[addr]) << endl;
#endif
                        fi0 = ctx.mem[addr].fe0;
                        fi1 = ctx.mem[addr].fe1;
                        fi2 = ctx.mem[addr].fe2;
                        fi3 = ctx.mem[addr].fe3;
                        fi4 = ctx.mem[addr].fe4;
                        fi5 = ctx.mem[addr].fe5;
                        fi6 = ctx.mem[addr].fe6;
                        fi7 = ctx.mem[addr].fe7;

                        /*MemoryAccess memoryAccess;
                        memoryAccess.bIsWrite = false;
                        memoryAccess.address = addr;
                        memoryAccess.pc = i;
                        memoryAccess.fe0 = fi0;
                        memoryAccess.fe1 = fi1;
                        memoryAccess.fe2 = fi2;
                        memoryAccess.fe3 = fi3;
                        memoryAccess.fe4 = fi4;
                        memoryAccess.fe5 = fi5;
                        memoryAccess.fe6 = fi6;
                        memoryAccess.fe7 = fi7;
                        mainExecRequired.memoryAccessList.access.push_back(memoryAccess);*/

                    } else {
                        fi0 = fr.zero();
                        fi1 = fr.zero();
                        fi2 = fr.zero();
                        fi3 = fr.zero();
                        fi4 = fr.zero();
                        fi5 = fr.zero();
                        fi6 = fr.zero();
                        fi7 = fr.zero();
                    }
                    nHits++;
                }

                // If sRD (storage read) get a poseidon hash, and read fi=sto[hash]
                if (rom.line[zkPC].sRD == 1)
                {
                    Goldilocks::Element Kin0[12];
                    Kin0[0] = pols.C0[i];
                    Kin0[1] = pols.C1[i];
                    Kin0[2] = pols.C2[i];
                    Kin0[3] = pols.C3[i];
                    Kin0[4] = pols.C4[i];
                    Kin0[5] = pols.C5[i];
                    Kin0[6] = pols.C6[i];
                    Kin0[7] = pols.C7[i];
                    Kin0[8] = fr.zero();
                    Kin0[9] = fr.zero();
                    Kin0[10] = fr.zero();
                    Kin0[11] = fr.zero();

                    Goldilocks::Element Kin1[12];
                    Kin1[0] = pols.A0[i];
                    Kin1[1] = pols.A1[i];
                    Kin1[2] = pols.A2[i];
                    Kin1[3] = pols.A3[i];
                    Kin1[4] = pols.A4[i];
                    Kin1[5] = pols.A5[i];
                    Kin1[6] = pols.B0[i];
                    Kin1[7] = pols.B1[i];

#ifdef LOG_TIME
                    struct timeval t;
                    gettimeofday(&t, NULL);
#endif
                    // Prepare PoseidonG required data
                    array<Goldilocks::Element,16> pg;
                    if (!bFastMode) for (uint64_t j=0; j<12; j++) pg[j] = Kin0[j];

                    // Call poseidon and get the hash key
                    Goldilocks::Element Kin0Hash[4];
                    poseidon.hash(Kin0Hash, Kin0);

                    // Complete PoseidonG required data
                    if (!bFastMode)
                    {
                        pg[12] = Kin0Hash[0];
                        pg[13] = Kin0Hash[1];
                        pg[14] = Kin0Hash[2];
                        pg[15] = Kin0Hash[3];
                        required.PoseidonG.push_back(pg);
                    }
                    
                    // Reinject the first resulting hash as the capacity for the next poseidon hash
                    Kin1[8] = Kin0Hash[0];
                    Kin1[9] = Kin0Hash[1];
                    Kin1[10] = Kin0Hash[2];
                    Kin1[11] = Kin0Hash[3];

                    // Prepare PoseidonG required data
                    if (!bFastMode) for (uint64_t j=0; j<12; j++) pg[j] = Kin1[j];

                    // Call poseidon hash
                    Goldilocks::Element Kin1Hash[4];
                    poseidon.hash(Kin1Hash, Kin1);

                    // Complete PoseidonG required data
                    if (!bFastMode)
                    {
                        pg[12] = Kin1Hash[0];
                        pg[13] = Kin1Hash[1];
                        pg[14] = Kin1Hash[2];
                        pg[15] = Kin1Hash[3];
                        required.PoseidonG.push_back(pg);
                    }

                    Goldilocks::Element key[4];
                    key[0] = Kin1Hash[0];
                    key[1] = Kin1Hash[1];
                    key[2] = Kin1Hash[2];
                    key[3] = Kin1Hash[3];
#ifdef LOG_TIME
                    poseidonTime += TimeDiff(t);
                    poseidonTimes+=3;
#endif

#ifdef LOG_STORAGE
                    cout << "Storage read sRD got poseidon key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << endl;
#endif 

#ifdef USE_LOCAL_STORAGE
                    //printStorage(ctx);
                    // Check that storage entry exists
                    if (ctx.sto.find(ctx.lastSWrite.key) == ctx.sto.end())
                    {
                        cerr << "Error: Storage not initialized, key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << " line: " << zkPC << " step: " << ctx.step << endl;
                        exit(-1);
                    }

                    //cout << "STORAGE1 i:" << i << " hash:" << fr.toString(ctx.lastSWrite.key, 16) << " value:" << ctx.sto[ctx.lastSWrite.key].get_str(16) << endl;

                    //SmtGetResult smtGetResult;
                    //smt.get(ctx.fr, ctx.db, pols.SR[i], ctx.lastSWrite.key, smtGetResult);
                    //cout << "STORAGE2 i:" << i << " hash:" << fr.toString(ctx.lastSWrite.key, 16) << " value:" << smtGetResult.value.get_str(16) << endl;

                    // Read the value from storage, and store it in fin
                    scalar2fea(fr, ctx.sto[ctx.lastSWrite.key], fi0, fi1, fi2, fi3);
#else
                    Goldilocks::Element oldRoot[4];
                    sr8to4(fr, pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);
                    
                    SmtGetResult smtGetResult;
                    mpz_class value;
                    pStateDB->get(oldRoot, key, value, &smtGetResult);
                    incCounter = smtGetResult.proofHashCounter + 2;
                    //cout << "smt.get() returns value=" << smtGetResult.value.get_str(16) << endl;
                    
                    scalar2fea(fr, smtGetResult.value, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
#endif
                    nHits++;
#ifdef LOG_STORAGE
                    cout << "Storage read sRD read from key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << " value:" << fr.toString(fi3, 16) << ":" << fr.toString(fi2, 16) << ":" << fr.toString(fi1, 16) << ":" << fr.toString(fi0, 16) << endl;
#endif 
                }

                // If sWR (storage write) calculate the poseidon hash key, check its entry exists in storage, and update new root hash
                if (rom.line[zkPC].sWR == 1)
                {
                    // reset lastSWrite
                    ctx.lastSWrite.reset();
                    Goldilocks::Element Kin0[12];
                    Kin0[0] = pols.C0[i];
                    Kin0[1] = pols.C1[i];
                    Kin0[2] = pols.C2[i];
                    Kin0[3] = pols.C3[i];
                    Kin0[4] = pols.C4[i];
                    Kin0[5] = pols.C5[i];
                    Kin0[6] = pols.C6[i];
                    Kin0[7] = pols.C7[i];
                    Kin0[8] = fr.zero();
                    Kin0[9] = fr.zero();
                    Kin0[10] = fr.zero();
                    Kin0[11] = fr.zero();

                    Goldilocks::Element Kin1[12];
                    Kin1[0] = pols.A0[i];
                    Kin1[1] = pols.A1[i];
                    Kin1[2] = pols.A2[i];
                    Kin1[3] = pols.A3[i];
                    Kin1[4] = pols.A4[i];
                    Kin1[5] = pols.A5[i];
                    Kin1[6] = pols.B0[i];
                    Kin1[7] = pols.B1[i];

#ifdef LOG_TIME
                    struct timeval t;
                    gettimeofday(&t, NULL);
#endif

                    // Prepare PoseidonG required data
                    array<Goldilocks::Element,16> pg;
                    if (!bFastMode) for (uint64_t j=0; j<12; j++) pg[j] = Kin0[j];

                    // Call poseidon and get the hash key
                    Goldilocks::Element Kin0Hash[4];
                    poseidon.hash(Kin0Hash, Kin0);

                    // Complete PoseidonG required data
                    if (!bFastMode)
                    {
                        pg[12] = Kin0Hash[0];
                        pg[13] = Kin0Hash[1];
                        pg[14] = Kin0Hash[2];
                        pg[15] = Kin0Hash[3];
                        required.PoseidonG.push_back(pg);
                    }
                    
                    Kin1[8] = Kin0Hash[0];
                    Kin1[9] = Kin0Hash[1];
                    Kin1[10] = Kin0Hash[2];
                    Kin1[11] = Kin0Hash[3];

                    ctx.lastSWrite.keyI[0] = Kin0Hash[0];
                    ctx.lastSWrite.keyI[1] = Kin0Hash[1];
                    ctx.lastSWrite.keyI[2] = Kin0Hash[2];
                    ctx.lastSWrite.keyI[3] = Kin0Hash[3];

                    // Prepare PoseidonG required data
                    if (!bFastMode) for (uint64_t j=0; j<12; j++) pg[j] = Kin1[j];

                    // Call poseidon hash
                    Goldilocks::Element Kin1Hash[4];
                    poseidon.hash(Kin1Hash, Kin1);

                    // Complete PoseidonG required data
                    if (!bFastMode)
                    {
                        pg[12] = Kin1Hash[0];
                        pg[13] = Kin1Hash[1];
                        pg[14] = Kin1Hash[2];
                        pg[15] = Kin1Hash[3];
                        required.PoseidonG.push_back(pg);
                    }

                    ctx.lastSWrite.key[0] = Kin1Hash[0];
                    ctx.lastSWrite.key[1] = Kin1Hash[1];
                    ctx.lastSWrite.key[2] = Kin1Hash[2];
                    ctx.lastSWrite.key[3] = Kin1Hash[3];
#ifdef LOG_TIME
                    poseidonTime += TimeDiff(t);
                    poseidonTimes++;
#endif

#ifdef LOG_STORAGE
                    cout << "Storage write sWR got poseidon key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << endl;
#endif                    
#ifdef USE_LOCAL_STORAGE
                    // Check that storage entry exists
                    if (ctx.sto.find(ctx.lastSWrite.key) == ctx.sto.end())
                    {
                        cerr << "Error: Storage write sWR not initialized key: " << fr.toString(ctx.lastSWrite.key, 16) << " line: " << zkPC << endl;
                        exit(-1);
                    }
#endif

                    // Call SMT to get the new Merkel Tree root hash
                    mpz_class scalarD;
                    fea2scalar(fr, scalarD, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]);
#ifdef LOG_TIME
                    gettimeofday(&t, NULL);
#endif
                    Goldilocks::Element oldRoot[4];
                    sr8to4(fr, pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);
                    
                    pStateDB->set(oldRoot, ctx.lastSWrite.key, scalarD, proverRequest.bUpdateMerkleTree, ctx.lastSWrite.newRoot, &ctx.lastSWrite.res);
                    incCounter = ctx.lastSWrite.res.proofHashCounter + 2;
#ifdef LOG_TIME
                    smtTime += TimeDiff(t);
                    smtTimes++;
#endif
                    ctx.lastSWrite.step = i;

                    sr4to8(fr, ctx.lastSWrite.newRoot[0], ctx.lastSWrite.newRoot[1], ctx.lastSWrite.newRoot[2], ctx.lastSWrite.newRoot[3], fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                    nHits++;
#ifdef LOG_STORAGE
                    cout << "Storage write sWR stored at key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << " newRoot: " << fr.toString(res.newRoot, 16) << endl;
#endif
                }

                if (rom.line[zkPC].hashK == 1)
                {
                    // If there is no entry in the hash database for this address, then create a new one
                    if (ctx.hashK.find(addr) == ctx.hashK.end())
                    {
                        HashValue hashValue;
                        ctx.hashK[addr] = hashValue;
                    }
                    
                    // Get the size of the hash from D0
                    int64_t iSize = fr.toS32(pols.D0[i]);
                    if ((iSize<0) || (iSize>32)) {
                        cerr << "Error: Invalid size for hashK:  Size:" << iSize << " Line:" << zkPC << endl;
                        exit(-1);
                    }
                    uint64_t size = iSize;

                    // Get the positon of the hash from HASHPOS
                    int64_t iPos = fr.toS32(pols.HASHPOS[i]);
                    if (iPos < 0)
                    {
                        cerr << "Error: invalid pos for HashK: pos:" << iPos << " Line:" << zkPC << endl;
                        exit(-1);
                    }
                    uint64_t pos = iPos;

                    // Check that pos+size do not exceed data size
                    if ( (pos+size) > ctx.hashK[addr].data.size())
                    {
                        cerr << "Error: hashK invalid size of hash: pos=" << pos << " size=" << size << " data.size=" << ctx.hashK[addr].data.size() << endl;
                        exit(-1);
                    }

                    // Copy data into fi
                    mpz_class s;
                    for (uint64_t j=0; j<size; j++)
                    {
                        uint8_t data = ctx.hashK[addr].data[pos+j];
                        s = (s<<uint64_t(8)) + mpz_class(data);
                    }
                    scalar2fea(fr, s, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;

#ifdef LOG_HASHK
                    cout << "hashK i=" << i << " zkPC=" << zkPC << " addr=" << addr << " pos=" << pos << " size=" << size << " data=" << s.get_str(16) << endl;
#endif
                }

                if (rom.line[zkPC].hashKDigest == 1)
                {
                    // If there is no entry in the hash database for this address, this is an error
                    if (ctx.hashK.find(addr) == ctx.hashK.end())
                    {
                        cerr << "Error: hashKDigest: digest not defined for addr=" << addr << endl;
                        exit(-1);
                    }

                    // If digest was not calculated, this is an error
                    if (!ctx.hashK[addr].bDigested)
                    {
                        cerr << "Error: hashKDigest: digest not calculated for addr=" << addr << ".  Call hashKLen to finish digest." << endl;
                        exit(-1);
                    }

                    // Copy digest into fi
                    scalar2fea(fr, ctx.hashK[addr].digest, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;

#ifdef LOG_HASHK
                    cout << "hashKDigest i=" << i << " zkPC=" << zkPC << " addr=" << addr << " digest=" << ctx.hashK[addr].digest.get_str(16) << endl;
#endif
                }

                if (rom.line[zkPC].hashP == 1)
                {
                    // If there is no entry in the hash database for this address, then create a new one
                    if (ctx.hashP.find(addr) == ctx.hashP.end())
                    {
                        HashValue hashValue;
                        ctx.hashP[addr] = hashValue;
                    }
                    
                    // Get the size of the hash from D0
                    int64_t iSize = fr.toS32(pols.D0[i]);
                    if ((iSize<0) || (iSize>32)) {
                        cerr << "Error: Invalid size for hashP:  Size:" << iSize << " Line:" << zkPC << endl;
                        exit(-1);
                    }
                    uint64_t size = iSize;

                    // Get the positon of the hash from HASHPOS
                    int64_t iPos = fr.toS32(pols.HASHPOS[i]);
                    if (iPos < 0)
                    {
                        cerr << "Error: invalid pos for HashP: pos:" << iPos << " Line:" << zkPC << endl;
                        exit(-1);
                    }
                    uint64_t pos = iPos;

                    // Check that pos+size do not exceed data size
                    if ( (pos+size) > ctx.hashP[addr].data.size())
                    {
                        cerr << "Error: hashP invalid size of hash: pos=" << pos << " size=" << size << " data.size=" << ctx.hashP[addr].data.size() << endl;
                        exit(-1);
                    }

                    // Copy data into fi
                    mpz_class s;
                    for (uint64_t j=0; j<size; j++)
                    {
                        uint8_t data = ctx.hashP[addr].data[pos+j];
                        s = (s<<uint64_t(8)) + mpz_class(data);
                    }
                    scalar2fea(fr, s, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;
                }

                if (rom.line[zkPC].hashPDigest == 1)
                {
                    // If there is no entry in the hash database for this address, this is an error
                    if (ctx.hashP.find(addr) == ctx.hashP.end())
                    {
                        cerr << "Error: hashPDigest: digest not defined" << endl;
                        exit(-1);
                    }

                    // If digest was not calculated, this is an error
                    if (!ctx.hashP[addr].bDigested)
                    {
                        cerr << "Error: hashPDigest: digest not calculated.  Call hashPLen to finish digest." << endl;
                        exit(-1);
                    }

                    // Copy digest into fi
                    scalar2fea(fr, ctx.hashP[addr].digest, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;
                }

                if (rom.line[zkPC].bin == 1)
                {
                    if (rom.line[zkPC].binOpcode == 0) // ADD
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a + b) & Mask256;
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 1) // SUB
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a - b + TwoTo256) & Mask256;
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 2) // LT
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a < b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 3) // SLT
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        if (a >= TwoTo255) a = a - TwoTo256;
                        if (b >= TwoTo255) b = b - TwoTo256;
                        c = (a < b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 4) // EQ
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a == b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 5) // AND
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a & b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 6) // OR
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a | b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 7) // XOR
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a ^ b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else
                    {
                        cerr << "Error: Invalid binary operation: opcode=" << rom.line[zkPC].binOpcode << endl;
                        exit(-1);
                    }
                }
#if 0
                // If shl, shift A, D bytes to the left, and discard highest bits
                if (rom.line[zkPC].shl == 1)
                {
                    // Read a=A
                    mpz_class a;
                    fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);

                    // Read s=D
                    uint64_t s = fe2n(fr, pols.D0[i]);
                    if ((s>32) || (s<0)) {
                        cerr << "Error: SHL too big: " << zkPC << endl;
                        exit(-1);
                    }

                    // Calculate b = shift a, s bytes to the left
                    mpz_class band("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
                    mpz_class b;
                    b = (a << s*8) & band;

                    // Copy fi=b
                    scalar2fea(fr, b, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                    nHits++;
                }

                // If shr, shift A, D bytes to the right
                if (rom.line[zkPC].shr == 1)
                {
                    // Read a=A
                    mpz_class a;
                    fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);

                    // Read s=D
                    uint64_t s = fe2n(fr, pols.D0[i]);
                    if ((s>32) || (s<0)) {
                        cerr << "Error: SHR too big: " << zkPC << endl;
                        exit(-1);
                    }

                    // Calculate b = shift a, s bytes to the right
                    mpz_class b = a >> s*8;

                    // Copy fi=b
                    scalar2fea(fr, b, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                    nHits++;
                }
#endif
                if (rom.line[zkPC].memAlign==1 && rom.line[zkPC].memAlignWR==0)
                {
                    mpz_class m0;
                    fea2scalar(fr, m0, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                    mpz_class m1;
                    fea2scalar(fr, m1, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                    mpz_class offsetScalar;
                    fea2scalar(fr, offsetScalar, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]);
                    if (offsetScalar<0 || offsetScalar>32)
                    {
                        cerr << "Error: MemAlign out of range offset=" << offsetScalar.get_str() << endl;
                        exit(-1);
                    }
                    uint64_t offset = offsetScalar.get_ui();
                    mpz_class leftV;
                    leftV = (m0 << (offset*8)) & Mask256;
                    mpz_class rightV;
                    rightV = (m1 >> (256 - offset*8)) & (Mask256 >> (256 - offset*8));
                    mpz_class _V;
                    _V = leftV | rightV;
                    scalar2fea(fr, _V, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                    nHits++;
                }

                // Check that one and only one instruction has been requested
                if (nHits == 0) {
                    cerr << "Error: Empty freeIn without a valid instruction: " << zkPC << endl;
                    exit(-1);
                }
                if (nHits > 1) {
                    cerr << "Error: Only one instruction that requires freeIn is alllowed: " << zkPC << endl;
                }
            }
            // If freeInTag.op!="", then evaluate the requested command (recursively)
            else
            {
                // Call evalCommand()
                CommandResult cr;
                evalCommand(ctx, rom.line[zkPC].freeInTag, cr);

                // Copy fi=command result, depending on its type 
                if (cr.type == crt_fea) {
                    fi0 = cr.fea0;
                    fi1 = cr.fea1;
                    fi2 = cr.fea2;
                    fi3 = cr.fea3;
                    fi4 = cr.fea4;
                    fi5 = cr.fea5;
                    fi6 = cr.fea6;
                    fi7 = cr.fea7;
                } else if (cr.type == crt_fe) {
                    fi0 = cr.fe;
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                    fi4 = fr.zero();
                    fi5 = fr.zero();
                    fi6 = fr.zero();
                    fi7 = fr.zero();
                } else if (cr.type == crt_scalar) {
                    scalar2fea(fr, cr.scalar, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                } else if (cr.type == crt_u16) {
                    fi0 = fr.fromU64(cr.u16);
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                    fi4 = fr.zero();
                    fi5 = fr.zero();
                    fi6 = fr.zero();
                    fi7 = fr.zero();
                } else if (cr.type == crt_u32) {
                    fi0 = fr.fromU64(cr.u32);
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                    fi4 = fr.zero();
                    fi5 = fr.zero();
                    fi6 = fr.zero();
                    fi7 = fr.zero();
                } else if (cr.type == crt_u64) {
                    fi0 = fr.fromU64(cr.u64);
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                    fi4 = fr.zero();
                    fi5 = fr.zero();
                    fi6 = fr.zero();
                    fi7 = fr.zero();
                } else {
                    cerr << "Error: unexpected command result type: " << cr.type << endl;
                    exit(-1);
                }
                // If we are in fast mode and we are consuming the last evaluations, exit the loop
                if (cr.beforeLast)
                {
                    if (ctx.lastStep == 0)
                    {
                        ctx.lastStep = step;
                    }
                    if (bFastMode)
                    {
                        break;
                    }
                }
            }

            // Store polynomial FREE=fi
            pols.FREE0[i] = fi0;
            pols.FREE1[i] = fi1;
            pols.FREE2[i] = fi2;
            pols.FREE3[i] = fi3;
            pols.FREE4[i] = fi4;
            pols.FREE5[i] = fi5;
            pols.FREE6[i] = fi6;
            pols.FREE7[i] = fi7;

            // op = op + inFREE*fi
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inFREE, fi0));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inFREE, fi1));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inFREE, fi2));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inFREE, fi3));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inFREE, fi4));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inFREE, fi5));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inFREE, fi6));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inFREE, fi7));

            // Copy ROM flags into the polynomials
            pols.inFREE[i] = rom.line[zkPC].inFREE;
        }

        /****************/
        /* INSTRUCTIONS */
        /****************/

        // If assert, check that A=op
        if (rom.line[zkPC].assert == 1 && !bProcessBatch)
        {
            if ( (!fr.equal(pols.A0[i], op0)) ||
                 (!fr.equal(pols.A1[i], op1)) ||
                 (!fr.equal(pols.A2[i], op2)) ||
                 (!fr.equal(pols.A3[i], op3)) ||
                 (!fr.equal(pols.A4[i], op4)) ||
                 (!fr.equal(pols.A5[i], op5)) ||
                 (!fr.equal(pols.A6[i], op6)) ||
                 (!fr.equal(pols.A7[i], op7)) )
            {
                cerr << "Error: ROM assert failed: AN!=opN ln: " << zkPC << endl;
                cout << "A: " << fr.toString(pols.A7[i], 16) << ":" << fr.toString(pols.A6[i], 16) << ":" << fr.toString(pols.A5[i], 16) << ":" << fr.toString(pols.A4[i], 16) << ":" << fr.toString(pols.A3[i], 16) << ":" << fr.toString(pols.A2[i], 16) << ":" << fr.toString(pols.A1[i], 16) << ":" << fr.toString(pols.A0[i], 16) << endl;
                cout << "OP:" << fr.toString(op7, 16) << ":" << fr.toString(op6, 16) << ":" << fr.toString(op5, 16) << ":" << fr.toString(op4,16) << ":" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0,16) << endl;
                exit(-1);
            }
            pols.assert_pol[i] = fr.one();
#ifdef LOG_ASSERT
            cout << "assert" << endl;
#endif
        }

        // Copy ROM flags into the polynomials
        if (rom.line[zkPC].mOp == 1)
        {
            pols.mOp[i] = fr.one();

            // If mWR, mem[addr]=op
            if (rom.line[zkPC].mWR == 1)
            {
                pols.mWR[i] = fr.one();

                ctx.mem[addr].fe0 = op0;
                ctx.mem[addr].fe1 = op1;
                ctx.mem[addr].fe2 = op2;
                ctx.mem[addr].fe3 = op3;
                ctx.mem[addr].fe4 = op4;
                ctx.mem[addr].fe5 = op5;
                ctx.mem[addr].fe6 = op6;
                ctx.mem[addr].fe7 = op7;

                if (!bFastMode)
                {
                    MemoryAccess memoryAccess;
                    memoryAccess.bIsWrite = true;
                    memoryAccess.address = addr;
                    memoryAccess.pc = i;
                    memoryAccess.fe0 = op0;
                    memoryAccess.fe1 = op1;
                    memoryAccess.fe2 = op2;
                    memoryAccess.fe3 = op3;
                    memoryAccess.fe4 = op4;
                    memoryAccess.fe5 = op5;
                    memoryAccess.fe6 = op6;
                    memoryAccess.fe7 = op7;
                    required.Memory.push_back(memoryAccess);
                }

#ifdef LOG_MEMORY
                cout << "Memory write mWR: addr:" << addr << " " << printFea(ctx, ctx.mem[addr]) << endl;
#endif
            }
            else
            {
                if (!bFastMode)
                {
                    MemoryAccess memoryAccess;
                    memoryAccess.bIsWrite = false;
                    memoryAccess.address = addr;
                    memoryAccess.pc = i;
                    memoryAccess.fe0 = op0;
                    memoryAccess.fe1 = op1;
                    memoryAccess.fe2 = op2;
                    memoryAccess.fe3 = op3;
                    memoryAccess.fe4 = op4;
                    memoryAccess.fe5 = op5;
                    memoryAccess.fe6 = op6;
                    memoryAccess.fe7 = op7;
                    required.Memory.push_back(memoryAccess);
                }

                if (ctx.mem.find(addr) != ctx.mem.end())
                {
                    if ( (!fr.equal(ctx.mem[addr].fe0, op0)) ||
                         (!fr.equal(ctx.mem[addr].fe1, op1)) ||
                         (!fr.equal(ctx.mem[addr].fe2, op2)) ||
                         (!fr.equal(ctx.mem[addr].fe3, op3)) ||
                         (!fr.equal(ctx.mem[addr].fe4, op4)) ||
                         (!fr.equal(ctx.mem[addr].fe5, op5)) ||
                         (!fr.equal(ctx.mem[addr].fe6, op6)) ||
                         (!fr.equal(ctx.mem[addr].fe7, op7)) )
                    {
                        cerr << "Error: Memory Read does not match" << endl;
                        exit(-1);
                    }
                }
                else
                {
                    if ( (!fr.isZero(op0)) ||
                         (!fr.isZero(op1)) ||
                         (!fr.isZero(op2)) ||
                         (!fr.isZero(op3)) ||
                         (!fr.isZero(op4)) ||
                         (!fr.isZero(op5)) ||
                         (!fr.isZero(op6)) ||
                         (!fr.isZero(op7)) )
                    {
                        cerr << "Error: Memory Read does not match (op!=0)" << endl;
                        exit(-1);
                    }
                }
            }
        }

        // Copy ROM flags into the polynomials
        if (rom.line[zkPC].sRD == 1)
        {
            pols.sRD[i] = fr.one();

            Goldilocks::Element Kin0[12];
            Kin0[0] = pols.C0[i];
            Kin0[1] = pols.C1[i];
            Kin0[2] = pols.C2[i];
            Kin0[3] = pols.C3[i];
            Kin0[4] = pols.C4[i];
            Kin0[5] = pols.C5[i];
            Kin0[6] = pols.C6[i];
            Kin0[7] = pols.C7[i];
            Kin0[8] = fr.zero();
            Kin0[9] = fr.zero();
            Kin0[10] = fr.zero();
            Kin0[11] = fr.zero();

            Goldilocks::Element Kin1[12];
            Kin1[0] = pols.A0[i];
            Kin1[1] = pols.A1[i];
            Kin1[2] = pols.A2[i];
            Kin1[3] = pols.A3[i];
            Kin1[4] = pols.A4[i];
            Kin1[5] = pols.A5[i];
            Kin1[6] = pols.B0[i];
            Kin1[7] = pols.B1[i];

#ifdef LOG_TIME
            struct timeval t;
            gettimeofday(&t, NULL);
#endif
            // Call poseidon and get the hash key
            Goldilocks::Element Kin0Hash[4];
            poseidon.hash(Kin0Hash, Kin0);
                    
            Goldilocks::Element keyI[4];
            keyI[0] = Kin0Hash[0];
            keyI[1] = Kin0Hash[1];
            keyI[2] = Kin0Hash[2];
            keyI[3] = Kin0Hash[3];

            Kin1[8] = Kin0Hash[0];
            Kin1[9] = Kin0Hash[1];
            Kin1[10] = Kin0Hash[2];
            Kin1[11] = Kin0Hash[3];

            Goldilocks::Element Kin1Hash[4];
            poseidon.hash(Kin1Hash, Kin1);

            Goldilocks::Element key[4];
            key[0] = Kin1Hash[0];
            key[1] = Kin1Hash[1];
            key[2] = Kin1Hash[2];
            key[3] = Kin1Hash[3];

#ifdef LOG_TIME
            poseidonTime += TimeDiff(t);
            poseidonTimes+=3;
#endif

#ifdef LOG_STORAGE
            cout << "Storage read sRD got poseidon key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << endl;
#endif 

#ifdef USE_LOCAL_STORAGE
            //printStorage(ctx);
            // Check that storage entry exists
            if (ctx.sto.find(ctx.lastSWrite.key) == ctx.sto.end())
            {
                cerr << "Error: Storage not initialized, key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << " line: " << zkPC << " step: " << ctx.step << endl;
                exit(-1);
            }

            //cout << "STORAGE1 i:" << i << " hash:" << fr.toString(ctx.lastSWrite.key, 16) << " value:" << ctx.sto[ctx.lastSWrite.key].get_str(16) << endl;

            //SmtGetResult smtGetResult;
            //smt.get(ctx.fr, ctx.db, pols.SR[i], ctx.lastSWrite.key, smtGetResult);
            //cout << "STORAGE2 i:" << i << " hash:" << fr.toString(ctx.lastSWrite.key, 16) << " value:" << smtGetResult.value.get_str(16) << endl;

            // Read the value from storage, and store it in fin
            scalar2fea(fr, ctx.sto[ctx.lastSWrite.key], fi0, fi1, fi2, fi3);
#else
            Goldilocks::Element oldRoot[4];
            sr8to4(fr, pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);
            
            SmtGetResult smtGetResult;
            mpz_class value;
            pStateDB->get(oldRoot, key, value, &smtGetResult);
            incCounter = smtGetResult.proofHashCounter + 2;
            //cout << "smt.get() returns value=" << smtGetResult.value.get_str(16) << endl;

            if (!bFastMode)
            {
                SmtAction smtAction;
                smtAction.bIsSet = false;
                smtAction.getResult = smtGetResult;
                required.Storage.push_back(smtAction);
            }
#endif

#ifdef LOG_STORAGE
            cout << "Storage read sRD read from key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << " value:" << fr.toString(fi3, 16) << ":" << fr.toString(fi2, 16) << ":" << fr.toString(fi1, 16) << ":" << fr.toString(fi0, 16) << endl;
#endif
            mpz_class opScalar;
            fea2scalar(fr, opScalar, op0, op1, op2, op3, op4, op5, op6, op7);
            if (smtGetResult.value != opScalar)
            {
                cerr << "Error: Storage read does not match: smtGetResult.value=" << smtGetResult.value.get_str() << " opScalar=" << opScalar.get_str() << endl;
                exit(-1);
            }

            for (uint64_t k=0; k<4; k++)
            {
                pols.sKeyI[k][i] = keyI[k];
                pols.sKey[k][i] = key[k];
            }
        }

        if (rom.line[zkPC].sWR == 1)
        {
            // Copy ROM flags into the polynomials
            pols.sWR[i] = fr.one();

            if ( (ctx.lastSWrite.step == 0) || (ctx.lastSWrite.step != i) )
            {
                // Reset lastSWrite
                ctx.lastSWrite.reset();

                Goldilocks::Element Kin0[12];
                Kin0[0] = pols.C0[i];
                Kin0[1] = pols.C1[i];
                Kin0[2] = pols.C2[i];
                Kin0[3] = pols.C3[i];
                Kin0[4] = pols.C4[i];
                Kin0[5] = pols.C5[i];
                Kin0[6] = pols.C6[i];
                Kin0[7] = pols.C7[i];
                Kin0[8] = fr.zero();
                Kin0[9] = fr.zero();
                Kin0[10] = fr.zero();
                Kin0[11] = fr.zero();

                Goldilocks::Element Kin1[12];
                Kin1[0] = pols.A0[i];
                Kin1[1] = pols.A1[i];
                Kin1[2] = pols.A2[i];
                Kin1[3] = pols.A3[i];
                Kin1[4] = pols.A4[i];
                Kin1[5] = pols.A5[i];
                Kin1[6] = pols.B0[i];
                Kin1[7] = pols.B1[i];

#ifdef LOG_TIME
                struct timeval t;
                gettimeofday(&t, NULL);
#endif
                // Call poseidon and get the hash key
                Goldilocks::Element Kin0Hash[4];
                poseidon.hash(Kin0Hash, Kin0);
                        
                ctx.lastSWrite.keyI[0] = Kin0Hash[0];
                ctx.lastSWrite.keyI[1] = Kin0Hash[1];
                ctx.lastSWrite.keyI[2] = Kin0Hash[2];
                ctx.lastSWrite.keyI[3] = Kin0Hash[3];

                Kin1[8] = Kin0Hash[0];
                Kin1[9] = Kin0Hash[1];
                Kin1[10] = Kin0Hash[2];
                Kin1[11] = Kin0Hash[3];

                Goldilocks::Element Kin1Hash[4];
                poseidon.hash(Kin1Hash, Kin1);

                ctx.lastSWrite.key[0] = Kin1Hash[0];
                ctx.lastSWrite.key[1] = Kin1Hash[1];
                ctx.lastSWrite.key[2] = Kin1Hash[2];
                ctx.lastSWrite.key[3] = Kin1Hash[3];
                
#ifdef LOG_TIME
                poseidonTime += TimeDiff(t);
                poseidonTimes++;
#endif

#ifdef USE_LOCAL_STORAGE
                // Check that storage entry exists
                if (ctx.sto.find(ctx.lastSWrite.key) == ctx.sto.end())
                {
                    cerr << "Error: Storage not initialized key: " << fr.toString(ctx.lastSWrite.key, 16) << " line: " << zkPC << endl;
                    exit(-1);
                }
#endif

                // Call SMT to get the new Merkel Tree root hash
                SmtSetResult res;
                mpz_class scalarD;
                fea2scalar(fr, scalarD, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]);
#ifdef LOG_TIME
                gettimeofday(&t, NULL);
#endif
                Goldilocks::Element oldRoot[4];
                sr8to4(fr, pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

                pStateDB->set(oldRoot, ctx.lastSWrite.key, scalarD, proverRequest.bUpdateMerkleTree, ctx.lastSWrite.newRoot, &ctx.lastSWrite.res);
                incCounter = res.proofHashCounter + 2;
#ifdef LOG_TIME
                smtTime += TimeDiff(t);
                smtTimes++;
#endif
                ctx.lastSWrite.step = i;
            }

            if (!bFastMode)
            {
                SmtAction smtAction;
                smtAction.bIsSet = true;
                smtAction.setResult = ctx.lastSWrite.res;
                required.Storage.push_back(smtAction);
            }

            // Check that the new root hash equals op0
            Goldilocks::Element oldRoot[4];
            sr8to4(fr, op0, op1, op2, op3, op4, op5, op6, op7, oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

            if ( !fr.equal(ctx.lastSWrite.newRoot[0], oldRoot[0]) ||
                 !fr.equal(ctx.lastSWrite.newRoot[1], oldRoot[1]) ||
                 !fr.equal(ctx.lastSWrite.newRoot[2], oldRoot[2]) ||
                 !fr.equal(ctx.lastSWrite.newRoot[3], oldRoot[3]) )
            {
                cerr << "Error: Storage write does not match; i: " << i << " zkPC: " << zkPC << 
                    " ctx.lastSWrite.newRoot: " << fr.toString(ctx.lastSWrite.newRoot[3], 16) << ":" << fr.toString(ctx.lastSWrite.newRoot[2], 16) << ":" << fr.toString(ctx.lastSWrite.newRoot[1], 16) << ":" << fr.toString(ctx.lastSWrite.newRoot[0], 16) <<
                    " oldRoot: " << fr.toString(oldRoot[3], 16) << ":" << fr.toString(oldRoot[2], 16) << ":" << fr.toString(oldRoot[1], 16) << ":" << fr.toString(oldRoot[0], 16) << endl;
                exit(-1);
            }

#ifdef USE_LOCAL_STORAGE
            // Store sto[poseidon_hash]=D
            mpz_class auxScalar;
            fea2scalar(fr, auxScalar, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i]);
            ctx.sto[ctx.lastSWrite.key] = auxScalar;
#endif
            Goldilocks::Element fea[4];
            sr8to4(fr, op0, op1, op2, op3, op4, op5, op6, op7, fea[0], fea[1], fea[2], fea[3]);
            if ( !fr.equal(ctx.lastSWrite.newRoot[0], fea[0]) ||
                 !fr.equal(ctx.lastSWrite.newRoot[1], fea[1]) ||
                 !fr.equal(ctx.lastSWrite.newRoot[2], fea[2]) ||
                 !fr.equal(ctx.lastSWrite.newRoot[3], fea[3]) )
            {
                cerr << "Error: Storage write does not match: ctx.lastSWrite.newRoot=" << fea2string(fr, ctx.lastSWrite.newRoot) << " op=" << fea << endl;
                exit(-1);
            }

            for (uint64_t k=0; k<4; k++)
            {
                pols.sKeyI[k][i] =  ctx.lastSWrite.keyI[k];
                pols.sKey[k][i] = ctx.lastSWrite.key[k];
            }
        }

        if (rom.line[zkPC].hashK == 1)
        {
            pols.hashK[i] = fr.one();

            // If there is no entry in the hash database for this address, then create a new one
            if (ctx.hashK.find(addr) == ctx.hashK.end())
            {
                HashValue hashValue;
                ctx.hashK[addr] = hashValue;
            }
            
            // Get the size of the hash from D0
            int64_t iSize = fr.toS32(pols.D0[i]);
            if ((iSize<0) || (iSize>32)) {
                cerr << "Error: Invalid size for hashK:  Size:" << iSize << " Line:" << zkPC << endl;
                exit(-1);
            }
            uint64_t size = iSize;

            // Get the positon of the hash from HASHPOS
            int64_t iPos = fr.toS32(pols.HASHPOS[i]);
            if (iPos < 0)
            {
                cerr << "Error: invalid pos for HashK: pos:" << iPos << " Line:" << zkPC << endl;
                exit(-1);
            }
            uint64_t pos = iPos;

            // Get contents of opN into a
            mpz_class a;
            fea2scalar(fr, a, op0, op1, op2, op3, op4, op5, op6, op7);

            // Fill the hash data vector with chunks of the scalar value
            mpz_class result;
            for (uint64_t j=0; j<size; j++) {
                result = (a >> (size-j-1)*8) & Mask8;
                uint8_t bm = result.get_ui();
                if (ctx.hashK[addr].data.size() == (pos+j))
                {
                    ctx.hashK[addr].data.push_back(bm);
                }
                else if (ctx.hashK[addr].data.size() < (pos+j))
                {
                    cerr << "Error: hashK: trying to insert data in a position:" << (pos+j) << " higher than current data size:" << ctx.hashK[addr].data.size() << endl;
                    exit(-1);
                }
                else
                {
                    uint8_t bh;
                    bh = ctx.hashK[addr].data[pos+j];
                    if (bm != bh)
                    {
                        cerr << "Error: HashK bytes do not match: addr=" << addr << " pos+j=" << pos+j << " is bm=" << bm << " and it should be bh=" << bh << endl;
                        exit(-1);
                    }
                }
            }

            // Record the read operation
            if ( (ctx.hashK[addr].reads.find(pos) != ctx.hashK[addr].reads.end()) &&
                 (ctx.hashK[addr].reads[pos] != size) )
            {
                cerr << "Error: HashK different read sizes in the same position addr=" << addr << " pos=" << pos << " ctx.hashK[addr].reads[pos]=" << ctx.hashK[addr].reads[pos] << " size=" << size << endl;
                exit(-1);
            }
            ctx.hashK[addr].reads[pos] = size;

            // Store the size
            incHashPos = size;

#ifdef LOG_HASHK
            cout << "hashK 2 i=" << i << " zkPC=" << zkPC << " addr=" << addr << " pos=" << pos << " size=" << size << " data=" << a.get_str(16) << endl;
#endif
        }

        if (rom.line[zkPC].hashKLen == 1)
        {
            pols.hashKLen[i] = fr.one();

            uint64_t lm = fr.toU64(op0);
            uint64_t lh = ctx.hashK[addr].data.size();
            if (lm != lh)
            {
                cerr << "Error: HashK length does not match match addr=" << addr << " is lm=" << lm << " and it should be lh=" << lh << endl;
                exit(-1);
            }
            if (!ctx.hashK[addr].bDigested)
            {
#ifdef LOG_TIME
                struct timeval t;
                gettimeofday(&t, NULL);
#endif
                string digestString = keccak256(ctx.hashK[addr].data.data(), ctx.hashK[addr].data.size());
                ctx.hashK[addr].digest.set_str(Remove0xIfPresent(digestString),16);
                ctx.hashK[addr].bDigested = true;
#ifdef LOG_TIME
                keccakTime += TimeDiff(t);
                keccakTimes++;
#endif

#ifdef LOG_HASHK
                cout << "hashKLen 2 calculate hashKLen: addr:" << addr << " hash:" << ctx.hashK[addr].digest.get_str(16) << " size:" << ctx.hashK[addr].data.size() << " data:";
                for (uint64_t k=0; k<ctx.hashK[addr].data.size(); k++) cout << byte2string(ctx.hashK[addr].data[k]) << ":";
                cout << endl;
#endif   
            }

#ifdef LOG_HASHK
            cout << "hashKLen 2 i=" << i << " zkPC=" << zkPC << " addr=" << addr << endl;
#endif
        }

        if (rom.line[zkPC].hashKDigest == 1)
        {
            pols.hashKDigest[i] = fr.one();

            // Get contents of op into dg
            mpz_class dg;
            fea2scalar(fr, dg, op0, op1, op2, op3, op4, op5, op6, op7);

            // Check the digest has been calculated
            if (!ctx.hashK[addr].bDigested)
            {
                cerr << "Error: hashKDigest: Cannot load keccak from DB" << endl;
                exit(-1);
            }
            
            if (dg != ctx.hashK[addr].digest)
            {
                cerr << "Error: hashKDigest: Digest does not match op" << endl;
                exit(-1);
            }
            incCounter = ceil((double(ctx.hashK[addr].data.size()) + double(1)) / double(136));

#ifdef LOG_HASHK
            cout << "hashKDigest i=" << i << " zkPC=" << zkPC << " addr=" << addr << " digest=" << ctx.hashK[addr].digest.get_str(16) << endl;
#endif
        }
            
        if (rom.line[zkPC].hashP == 1)
        {
            pols.hashP[i] = fr.one();

            // If there is no entry in the hash database for this address, then create a new one
            if (ctx.hashP.find(addr) == ctx.hashP.end())
            {
                HashValue hashValue;
                ctx.hashP[addr] = hashValue;
            }
            
            // Get the size of the hash from D0
            int64_t iSize = fr.toS32(pols.D0[i]);
            if ((iSize<0) || (iSize>32)) {
                cerr << "Error: Invalid size for hashP:  Size:" << iSize << " Line:" << zkPC << endl;
                exit(-1);
            }
            uint64_t size = iSize;

            // Get the positon of the hash from HASHPOS
            int64_t iPos = fr.toS32(pols.HASHPOS[i]);
            if (iPos < 0)
            {
                cerr << "Error: invalid pos for HashP: pos:" << iPos << " Line:" << zkPC << endl;
                exit(-1);
            }
            uint64_t pos = iPos;

            // Get contents of opN into a
            mpz_class a;
            fea2scalar(fr, a, op0, op1, op2, op3, op4, op5, op6, op7);

            // Fill the hash data vector with chunks of the scalar value
            mpz_class result;
            for (uint64_t j=0; j<size; j++) {
                result = (a >> (size-j-1)*8) & Mask8;
                uint8_t bm = result.get_ui();
                if (ctx.hashP[addr].data.size() == (pos+j))
                {
                    ctx.hashP[addr].data.push_back(bm);
                }
                else if (ctx.hashP[addr].data.size() < (pos+j))
                {
                    cerr << "Error: hashP: trying to insert data in a position:" << (pos+j) << " higher than current data size:" << ctx.hashP[addr].data.size() << endl;
                    exit(-1);
                }
                else
                {
                    uint8_t bh;
                    bh = ctx.hashP[addr].data[pos+j];
                    if (bm != bh)
                    {
                        cerr << "Error: HashP bytes do not match: addr=" << addr << " pos+j=" << pos+j << " is bm=" << bm << " and it should be bh=" << bh << endl;
                        exit(-1);
                    }
                }
            }

            // Record the read operation
            if ( (ctx.hashP[addr].reads.find(pos) != ctx.hashP[addr].reads.end()) &&
                 (ctx.hashP[addr].reads[pos] != size) )
            {
                cerr << "Error: HashP diferent read sizes in the same position addr=" << addr << " pos=" << pos << endl;
                exit(-1);
            }
            ctx.hashP[addr].reads[pos] = size;

            // Store the size
            incHashPos = size;
        }

        if (rom.line[zkPC].hashPLen == 1)
        {
            pols.hashPLen[i] = fr.one();

            uint64_t lm = fr.toU64(op0);
            uint64_t lh = ctx.hashP[addr].data.size();
            if (lm != lh)
            {
                cerr << "Error: HashP length does not match match addr=" << addr << " is lm=" << lm << " and it should be lh=" << lh << endl;
                exit(-1);
            }
            if (!ctx.hashP[addr].bDigested)
            {
#ifdef LOG_TIME
                struct timeval t;
                gettimeofday(&t, NULL);
#endif
                if (ctx.hashP[addr].data.size() == 0)
                {
                    cerr << "Error: HashP length found data empty" << endl;
                    exit(-1);
                }

                // Get a local copy of the bytes vector
                vector<uint8_t> data = ctx.hashP[addr].data;

                // Add padding = 0b1000...00001  up to a length of 56xN (7x8xN)
                data.push_back(0x01);
                while((data.size() % 56) != 0) data.push_back(0);
                data[data.size()-1] |= 0x80;

                // Create a FE buffer to store the transformed bytes into fe
                uint64_t bufferSize = data.size()/7;
                Goldilocks::Element * pBuffer = new Goldilocks::Element[bufferSize];
                if (pBuffer == NULL)
                {
                    cerr << "Error: HashP length failed allocating memory of " << bufferSize << " field elements" << endl;
                    exit(-1);
                }
                for (uint64_t j=0; j<bufferSize; j++) pBuffer[j] = fr.zero();

                // Copy the bytes into the fe lower 7 sections
                for (uint64_t j=0; j<data.size(); j++)
                {
                    uint64_t fePos = j/7;
                    uint64_t shifted = uint64_t(data[j]) << ((j%7)*8);
                    pBuffer[fePos] = fr.add(pBuffer[fePos], fr.fromU64(shifted));
                    //cout << "fePos=" << fePos << " data=" << to_string(data[j]) << " shifted=" << shifted << " fe=" << fr.toString(pBuffer[fePos],16) << endl;
                }

                Goldilocks::Element result[4];
                poseidon.linear_hash(result, pBuffer, bufferSize);
                fea2scalar(fr, ctx.hashP[addr].digest, result);
                //cout << "ctx.hashP[" << addr << "].digest=" << ctx.hashP[addr].digest.get_str(16) << endl;
                delete[] pBuffer;
                ctx.hashP[addr].bDigested = true;

                pStateDB->setProgram(result, ctx.hashP[addr].data, proverRequest.bUpdateMerkleTree);
#ifdef LOG_TIME
                poseidonTime += TimeDiff(t);
                poseidonTimes++;
#endif

#ifdef LOG_HASH
                cout << "Hash calculate hashPLen: addr:" << addr << " hash:" << ctx.hashP[addr].digest.get_str(16) << " size:" << ctx.hashP[addr].data.size() << " data:";
                for (uint64_t k=0; k<ctx.hashP[addr].data.size(); k++) cout << byte2string(ctx.hashP[addr].data[k]) << ":";
                cout << endl;
#endif   
            }
        }

        if (rom.line[zkPC].hashPDigest == 1)
        {
            pols.hashPDigest[i] = fr.one();

            // Get contents of op into dg
            mpz_class dg;
            fea2scalar(fr, dg, op0, op1, op2, op3, op4, op5, op6, op7);

            if (ctx.hashP.find(addr) == ctx.hashP.end())
            {
                HashValue hashValue;
                hashValue.digest = dg;
                hashValue.bDigested = true;
                Goldilocks::Element aux[4];
                scalar2fea(fr, dg, aux);
                pStateDB->getProgram(aux, hashValue.data);
                ctx.hashP[addr] = hashValue;
            }

            incCounter = ceil((double(ctx.hashP[addr].data.size()) + double(1)) / double(56));

            // Check that digest equals op
            if (dg != ctx.hashP[addr].digest)
            {
                cerr << "Error: hashPDigest: ctx.hashP[addr].digest=" << ctx.hashP[addr].digest.get_str(16) << " does not match op=" << dg.get_str(16) << endl;
                exit(-1);
            }
        }

        if (rom.line[zkPC].hashPDigest || rom.line[zkPC].sWR)
        {
            mpz_class op;
            fea2scalar(fr, op, op0, op1, op2, op3, op4, op5, op6, op7);

            // Store the binary action to execute it later with the binary SM
            if (!bFastMode)
            {
                BinaryAction binaryAction;
                binaryAction.a = op;
                binaryAction.b = 0;
                binaryAction.c = op;
                binaryAction.opcode = 1;
                required.Binary.push_back(binaryAction);
            }
        }

        // If arith, check that A*B + C = D<<256 + op, using scalars (result can be a big number)
        if (rom.line[zkPC].arith == 1)
        {
            if (rom.line[zkPC].arithEq0==1 && rom.line[zkPC].arithEq1==0 && rom.line[zkPC].arithEq2==0 && rom.line[zkPC].arithEq3==0)
            {            
                // Convert to scalar
                mpz_class A, B, C, D, op;
                fea2scalar(fr, A, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, B, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, C, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]);
                fea2scalar(fr, D, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]);
                fea2scalar(fr, op, op0, op1, op2, op3, op4, op5, op6, op7);

                // Check the condition
                if ( (A*B) + C != (D<<256) + op ) {
                    cerr << "Error: Arithmetic does not match: zkPC=" << zkPC << endl;
                    mpz_class left = (A*B) + C;
                    mpz_class right = (D<<256) + op;
                    cerr << "(A*B) + C = " << left.get_str(16) << endl;
                    cerr << "(D<<256) + op = " << right.get_str(16) << endl;
                    exit(-1);
                }

                // Copy ROM flags into the polynomials
                pols.arith[i] = fr.one();
                pols.arithEq0[i] = fr.one();

                // Store the arith action to execute it later with the arith SM
                if (!bFastMode)
                {
                    ArithAction arithAction;
                    arithAction.x1 = A;
                    arithAction.y1 = B;
                    arithAction.x2 = C;
                    arithAction.y2 = D;
                    arithAction.x3 = 0;
                    arithAction.y3 = op;
                    arithAction.selEq0 = 1;
                    arithAction.selEq1 = 0;
                    arithAction.selEq2 = 0;
                    arithAction.selEq3 = 0;
                    required.Arith.push_back(arithAction);
                }
            }
            else
            {
                // Convert to scalar
                mpz_class x1, y1, x2, y2, x3, y3;
                fea2scalar(fr, x1, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, y1, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, x2, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]);
                fea2scalar(fr, y2, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]);
                fea2scalar(fr, x3, pols.E0[i], pols.E1[i], pols.E2[i], pols.E3[i], pols.E4[i], pols.E5[i], pols.E6[i], pols.E7[i]);
                fea2scalar(fr, y3, op0, op1, op2, op3, op4, op5, op6, op7);

                // Convert to RawFec::Element
                RawFec::Element fecX1, fecY1, fecX2, fecY2, fecX3;
                fec.fromString(fecX1, x1.get_str());
                fec.fromString(fecY1, y1.get_str());
                fec.fromString(fecX2, x2.get_str());
                fec.fromString(fecY2, y2.get_str());
                fec.fromString(fecX3, x3.get_str());

                bool dbl = false;
                if (rom.line[zkPC].arithEq0==0 && rom.line[zkPC].arithEq1==1 && rom.line[zkPC].arithEq2==0 && rom.line[zkPC].arithEq3==1)
                {
                    dbl = false;
                }
                else if (rom.line[zkPC].arithEq0==0 && rom.line[zkPC].arithEq1==0 && rom.line[zkPC].arithEq2==1 && rom.line[zkPC].arithEq3==1)
                {
                    dbl = true;
                }
                else
                {
                    cerr << "Error: Invalid arithmetic op" << endl;
                    exit(-1);
                }

                RawFec::Element s;
                if (dbl)
                {
                    // s = 3*(x1^2)/(2*y1)
                    RawFec::Element numerator, denominator;

                    // numerator = 3*(x1^2)
                    fec.mul(numerator, fecX1, fecX1);
                    fec.fromUI(denominator, 3);
                    fec.mul(numerator, numerator, denominator);

                    // denominator = 2*y1 = y1+y1
                    fec.add(denominator, fecY1, fecY1);

                    // s = numerator/denominator
                    fec.div(s, numerator, denominator);

                    // TODO: y1 == 0 => division by zero ==> how manage? Feli
                }
                else
                {
                    // s = (y2-y1)/(x2-x1)
                    RawFec::Element numerator, denominator;

                    // numerator = y2-y1
                    fec.sub(numerator, fecY2, fecY1);

                    // denominator = x2-x1
                    fec.sub(denominator, fecX2, fecX1);

                    // s = numerator/denominator
                    fec.div(s, numerator, denominator);

                    // TODO: x2-x1 == 0 => division by zero ==> how manage? Feli
                }

                RawFec::Element fecS, minuend, subtrahend;
                mpz_class _x3, _y3;
                
                // Calculate _x3 = s*s - x1 +(x1 if dbl, x2 otherwise)
                fec.mul(minuend, s, s);
                fec.add(subtrahend, fecX1, dbl ? fecX1 : fecX2 );
                fec.sub(fecS, minuend, subtrahend);
                _x3.set_str(fec.toString(fecS), 10);

                // Calculate _y3 = s*(x1-x3) - y1
                fec.sub(subtrahend, fecX1, fecX3);
                fec.mul(minuend, s, subtrahend);
                fec.fromString(subtrahend, y1.get_str());
                fec.sub(fecS, minuend, subtrahend);
                _y3.set_str(fec.toString(fecS), 10);

                // Compare
                bool x3eq = (x3 == _x3);
                bool y3eq = (y3 == _y3);

                if (!x3eq || !y3eq)
                {
                    cerr << "Error: Arithmetic curve " << (dbl?"dbl":"add") << " point does not match" << endl;
                    cerr << " x1=" << x1.get_str() << endl;
                    cerr << " y1=" << y1.get_str() << endl;
                    cerr << " x2=" << x2.get_str() << endl;
                    cerr << " y2=" << y2.get_str() << endl;
                    cerr << " x3=" << x3.get_str() << endl;
                    cerr << " y3=" << y3.get_str() << endl;
                    cerr << "_x3=" << _x3.get_str() << endl;
                    cerr << "_y3=" << _y3.get_str() << endl;
                    exit(-1);
                }

                pols.arith[i] = fr.one();
                pols.arithEq0[i] = fr.fromU64(rom.line[zkPC].arithEq0);
                pols.arithEq1[i] = fr.fromU64(rom.line[zkPC].arithEq1);
                pols.arithEq2[i] = fr.fromU64(rom.line[zkPC].arithEq2);
                pols.arithEq3[i] = fr.fromU64(rom.line[zkPC].arithEq3);

                // Store the arith action to execute it later with the arith SM
                if (!bFastMode)
                {
                    ArithAction arithAction;
                    arithAction.x1 = x1;
                    arithAction.y1 = y1;
                    arithAction.x2 = dbl ? x1 : x2;
                    arithAction.y2 = dbl ? y1 : y2;
                    arithAction.x3 = x3;
                    arithAction.y3 = y3;
                    arithAction.selEq0 = 0;
                    arithAction.selEq1 = dbl ? 0 : 1;
                    arithAction.selEq2 = dbl ? 1 : 0;
                    arithAction.selEq3 = 1;
                    required.Arith.push_back(arithAction);
                }
            }
        }

        // Copy ROM flags into the polynomials
        //if (rom.line[zkPC].shl == 1) pols.shl[i] = 1; TODO: Check if this is correct
        //if (rom.line[zkPC].shr == 1) pols.shr[i] = 1;

        if (rom.line[zkPC].bin == 1)
        {
            if (rom.line[zkPC].binOpcode == 0) // ADD
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a + b) & Mask256;
                if (c != expectedC)
                {
                    cerr << "Error: Binary ADD operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = fr.zero();
                pols.carry[i] = fr.fromU64(((a + b) >> 256) > 0);

                // Store the binary action to execute it later with the binary SM
                if (!bFastMode)
                {
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 0;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 1) // SUB
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a - b + TwoTo256) & Mask256;
                if (c != expectedC)
                {
                    cerr << "Error: Binary SUB operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = fr.one();
                pols.carry[i] = fr.fromU64((a - b) < 0);

                // Store the binary action to execute it later with the binary SM
                if (!bFastMode)
                {
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 1;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 2) // LT
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a < b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary LT operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = fr.fromU64(2);
                pols.carry[i] = fr.fromU64(a < b);

                // Store the binary action to execute it later with the binary SM
                if (!bFastMode)
                {
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 2;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 3) // SLT
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);
                if (a >= TwoTo255) a = a - TwoTo256;
                if (b >= TwoTo255) b = b - TwoTo256;


                mpz_class expectedC;
                expectedC = (a < b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary SLT operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = fr.fromU64(3);
                mpz_class sa = ( (a >> 255) != 0 ) ? ((One<<256) - a) : a;
                mpz_class sb = ( (b >> 255) != 0 ) ? ((One<<256) - b) : b;
                pols.carry[i] = fr.fromU64(sa < sb);

                // Store the binary action to execute it later with the binary SM
                if (!bFastMode)
                {
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 3;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 4) // EQ
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a == b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary EQ operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = fr.fromU64(4);
                pols.carry[i] = fr.fromU64((a == b));
                
                // Store the binary action to execute it later with the binary SM
                if (!bFastMode)
                {
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 4;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 5) // AND
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a & b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary AND operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = fr.fromU64(5);
                
                // Store the binary action to execute it later with the binary SM
                if (!bFastMode)
                {
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 5;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 6) // OR
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a | b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary OR operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = fr.fromU64(6);
                
                // Store the binary action to execute it later with the binary SM
                if (!bFastMode)
                {
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 6;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 7) // XOR
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a ^ b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary XOR operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = fr.fromU64(7);
                
                // Store the binary action to execute it later with the binary SM
                if (!bFastMode)
                {
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 7;
                    required.Binary.push_back(binaryAction);
                }
            }
            else
            {
                cerr << "Error: Invalid binary operation opcode" << rom.line[zkPC].binOpcode <<  endl;
                exit(-1);
            }
            pols.bin[i] = fr.one();
        }

        if (rom.line[zkPC].memAlign==1)
        {
            pols.memAlign[i] = fr.one();

            mpz_class m0;
            fea2scalar(fr, m0, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
            mpz_class m1;
            fea2scalar(fr, m1, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
            mpz_class v;
            fea2scalar(fr, v, op0, op1, op2, op3, op4, op5, op6, op7);
            mpz_class offsetScalar;
            fea2scalar(fr, offsetScalar, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]);
            if (offsetScalar<0 || offsetScalar>32)
            {
                cerr << "Error: MemAlign out of range offset=" << offsetScalar.get_str() << endl;
                exit(-1);
            }
            uint64_t offset = offsetScalar.get_ui();

            if (rom.line[zkPC].memAlignWR==1 && rom.line[zkPC].memAlignWR8==0)
            {
                pols.memAlignWR[i] = fr.one();
                //pols.memAlignWR8[i] = fr.zero();

                mpz_class w0;
                fea2scalar(fr, w0, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]);
                mpz_class w1;
                fea2scalar(fr, w1, pols.E0[i], pols.E1[i], pols.E2[i], pols.E3[i], pols.E4[i], pols.E5[i], pols.E6[i], pols.E7[i]);
                mpz_class _W0;
                _W0 = (m0 & (TwoTo256 - (One << (256-offset*8)))) | (v >> offset*8);
                mpz_class _W1;
                _W1 = (m1 & (Mask256 >> offset*8)) | ((v << (256 - offset*8)) & Mask256);
                if ( (w0 != _W0) || (w1 != _W1) )
                {
                    cerr << "Error: MemAlign w0, w1 invalid: w0=" << w0.get_str(16) << " w1=" << w1.get_str(16) << " _W0=" << _W0.get_str(16) << " _W1=" << _W1.get_str(16) << " m0=" << m0.get_str(16) << " m1=" << m1.get_str(16) << " offset=" << offset << " v=" << v.get_str(16) << endl;
                    exit(-1);
                }

                if (!bFastMode)
                {
                    MemAlignAction memAlignAction;
                    memAlignAction.m0 = m0;
                    memAlignAction.m1 = m1;
                    memAlignAction.w0 = w0;
                    memAlignAction.w1 = w1;
                    memAlignAction.v = v;
                    memAlignAction.offset = offset;
                    memAlignAction.wr256 = 1;    
                    memAlignAction.wr8 = 0;    
                    required.MemAlign.push_back(memAlignAction);
                }
            }
            else if (rom.line[zkPC].memAlignWR==0 && rom.line[zkPC].memAlignWR8==1)
            {
                //pols.memAlignWR[i] = fr.zero();
                pols.memAlignWR8[i] = fr.one();

                mpz_class w0;
                fea2scalar(fr, w0, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]);
                mpz_class _W0;
//                mpz_class byteMaskOn256a("FF00000000000000000000000000000000000000000000000000000000000000", 16);
//                _W0 = (m0 & ~(byteMaskOn256a >> offset*8)) | (v & 0xFF << (31-offset)*8);
                mpz_class byteMaskOn256("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);              
                _W0 = (m0 & (byteMaskOn256 >> offset*8)) | (v & 0xFF << (31-offset)*8);
                if (w0 != _W0)
                {
                    cerr << "Error: MemAlign w0 invalid: w0=" << w0.get_str(16) << " _W0=" << _W0.get_str(16) << " m0=" << m0.get_str(16) << " offset=" << offset << " v=" << v.get_str(16) << endl;
                    exit(-1);
                }

                if (!bFastMode)
                {
                    MemAlignAction memAlignAction;
                    memAlignAction.m0 = m0;
                    memAlignAction.m1 = 0;
                    memAlignAction.w0 = w0;
                    memAlignAction.w1 = 0;
                    memAlignAction.v = v;
                    memAlignAction.offset = offset;
                    memAlignAction.wr256 = 0;    
                    memAlignAction.wr8 = 1;    
                    required.MemAlign.push_back(memAlignAction);     
                }            
            }
            else if (rom.line[zkPC].memAlignWR==0 && rom.line[zkPC].memAlignWR8==0)
            {
                //pols.memAlignWR[i] = fr.zero(); // TODO: Should we comment this out?
                //pols.memAlignWR8[i] = fr.zero(); // TODO: Should we comment this out?

                mpz_class leftV;
                leftV = (m0 << offset*8) & Mask256;
                mpz_class rightV;
                rightV = (m1 >> (256 - offset*8)) & (Mask256 >> (256 - offset*8));
                mpz_class _V;
                _V = leftV | rightV;
                if (v != _V)
                {
                    cerr << "Error: MemAlign v invalid: v=" << v.get_str(16) << " _V=" << _V.get_str(16) << " m0=" << m0.get_str(16) << " m1=" << m1.get_str(16) << " offset=" << offset << endl;
                    exit(-1);
                }

                if (!bFastMode)
                {
                    MemAlignAction memAlignAction;
                    memAlignAction.m0 = m0;
                    memAlignAction.m1 = m1;
                    memAlignAction.w0 = 0;
                    memAlignAction.w1 = 0;
                    memAlignAction.v = v;
                    memAlignAction.offset = offset;
                    memAlignAction.wr256 = 0;    
                    memAlignAction.wr8 = 0;    
                    required.MemAlign.push_back(memAlignAction);       
                }         
            }
        }

        //if (rom.line[zkPC].comparator == 1) pols.comparator[i] = 1; TODO: Check if this is correct
        //if (rom.line[zkPC].opcodeRomMap == 1) pols.opcodeRomMap[i] = 1;

        /***********/
        /* SETTERS */
        /***********/

        // If setA, A'=op
        if (rom.line[zkPC].setA == 1) {
            pols.A0[nexti] = op0;
            pols.A1[nexti] = op1;
            pols.A2[nexti] = op2;
            pols.A3[nexti] = op3;
            pols.A4[nexti] = op4;
            pols.A5[nexti] = op5;
            pols.A6[nexti] = op6;
            pols.A7[nexti] = op7;
            pols.setA[i] = fr.one();
#ifdef LOG_SETX
            cout << "setA A[nexti]=" << pols.A3[nexti] << ":" << pols.A2[nexti] << ":" << pols.A1[nexti] << ":" << fr.toString(pols.A0[nexti], 16) << endl;
#endif
        } else {
            pols.A0[nexti] = pols.A0[i];
            pols.A1[nexti] = pols.A1[i];
            pols.A2[nexti] = pols.A2[i];
            pols.A3[nexti] = pols.A3[i];
            pols.A4[nexti] = pols.A4[i];
            pols.A5[nexti] = pols.A5[i];
            pols.A6[nexti] = pols.A6[i];
            pols.A7[nexti] = pols.A7[i];
        }

        // If setB, B'=op
        if (rom.line[zkPC].setB == 1) {
            pols.B0[nexti] = op0;
            pols.B1[nexti] = op1;
            pols.B2[nexti] = op2;
            pols.B3[nexti] = op3;
            pols.B4[nexti] = op4;
            pols.B5[nexti] = op5;
            pols.B6[nexti] = op6;
            pols.B7[nexti] = op7;
            pols.setB[i] = fr.one();
#ifdef LOG_SETX
            cout << "setB B[nexti]=" << pols.B3[nexti] << ":" << pols.B2[nexti] << ":" << pols.B1[nexti] << ":" << fr.toString(pols.B0[nexti], 16) << endl;
#endif
        } else {
            pols.B0[nexti] = pols.B0[i];
            pols.B1[nexti] = pols.B1[i];
            pols.B2[nexti] = pols.B2[i];
            pols.B3[nexti] = pols.B3[i];
            pols.B4[nexti] = pols.B4[i];
            pols.B5[nexti] = pols.B5[i];
            pols.B6[nexti] = pols.B6[i];
            pols.B7[nexti] = pols.B7[i];
        }

        // If setC, C'=op
        if (rom.line[zkPC].setC == 1) {
            pols.C0[nexti] = op0;
            pols.C1[nexti] = op1;
            pols.C2[nexti] = op2;
            pols.C3[nexti] = op3;
            pols.C4[nexti] = op4;
            pols.C5[nexti] = op5;
            pols.C6[nexti] = op6;
            pols.C7[nexti] = op7;
            pols.setC[i] = fr.one();
#ifdef LOG_SETX
            cout << "setC C[nexti]=" << pols.C3[nexti] << ":" << pols.C2[nexti] << ":" << pols.C1[nexti] << ":" << fr.toString(pols.C0[nexti], 16) << endl;
#endif
        } else {
            pols.C0[nexti] = pols.C0[i];
            pols.C1[nexti] = pols.C1[i];
            pols.C2[nexti] = pols.C2[i];
            pols.C3[nexti] = pols.C3[i];
            pols.C4[nexti] = pols.C4[i];
            pols.C5[nexti] = pols.C5[i];
            pols.C6[nexti] = pols.C6[i];
            pols.C7[nexti] = pols.C7[i];
        }

        // If setD, D'=op
        if (rom.line[zkPC].setD == 1) {
            pols.D0[nexti] = op0;
            pols.D1[nexti] = op1;
            pols.D2[nexti] = op2;
            pols.D3[nexti] = op3;
            pols.D4[nexti] = op4;
            pols.D5[nexti] = op5;
            pols.D6[nexti] = op6;
            pols.D7[nexti] = op7;
            pols.setD[i] = fr.one();
#ifdef LOG_SETX
            cout << "setD D[nexti]=" << pols.D3[nexti] << ":" << pols.D2[nexti] << ":" << pols.D1[nexti] << ":" << fr.toString(pols.D0[nexti], 16) << endl;
#endif
        } else {
            pols.D0[nexti] = pols.D0[i];
            pols.D1[nexti] = pols.D1[i];
            pols.D2[nexti] = pols.D2[i];
            pols.D3[nexti] = pols.D3[i];
            pols.D4[nexti] = pols.D4[i];
            pols.D5[nexti] = pols.D5[i];
            pols.D6[nexti] = pols.D6[i];
            pols.D7[nexti] = pols.D7[i];
        }
        
        // If setE, E'=op
        if (rom.line[zkPC].setE == 1) {
            pols.E0[nexti] = op0;
            pols.E1[nexti] = op1;
            pols.E2[nexti] = op2;
            pols.E3[nexti] = op3;
            pols.E4[nexti] = op4;
            pols.E5[nexti] = op5;
            pols.E6[nexti] = op6;
            pols.E7[nexti] = op7;
            pols.setE[i] = fr.one();
#ifdef LOG_SETX
            cout << "setE E[nexti]=" << pols.E3[nexti] << ":" << pols.E2[nexti] << ":" << pols.E1[nexti] << ":" << fr.toString(pols.E0[nexti] ,16) << endl;
#endif
        } else {
            pols.E0[nexti] = pols.E0[i];
            pols.E1[nexti] = pols.E1[i];
            pols.E2[nexti] = pols.E2[i];
            pols.E3[nexti] = pols.E3[i];
            pols.E4[nexti] = pols.E4[i];
            pols.E5[nexti] = pols.E5[i];
            pols.E6[nexti] = pols.E6[i];
            pols.E7[nexti] = pols.E7[i];
        }

        // If setSR, SR'=op
        if (rom.line[zkPC].setSR == 1) {
            pols.SR0[nexti] = op0;
            pols.SR1[nexti] = op1;
            pols.SR2[nexti] = op2;
            pols.SR3[nexti] = op3;
            pols.SR4[nexti] = op4;
            pols.SR5[nexti] = op5;
            pols.SR6[nexti] = op6;
            pols.SR7[nexti] = op7;
            pols.setSR[i] = fr.one();
#ifdef LOG_SETX
            cout << "setSR SR[nexti]=" << fr.toString(pols.SR[nexti], 16) << endl;
#endif
        } else {
            pols.SR0[nexti] = pols.SR0[i];
            pols.SR1[nexti] = pols.SR1[i];
            pols.SR2[nexti] = pols.SR2[i];
            pols.SR3[nexti] = pols.SR3[i];
            pols.SR4[nexti] = pols.SR4[i];
            pols.SR5[nexti] = pols.SR5[i];
            pols.SR6[nexti] = pols.SR6[i];
            pols.SR7[nexti] = pols.SR7[i];
        }

        // If setCTX, CTX'=op
        if (rom.line[zkPC].setCTX == 1) {
            pols.CTX[nexti] = op0;
            pols.setCTX[i] = fr.one();
#ifdef LOG_SETX
            cout << "setCTX CTX[nexti]=" << pols.CTX[nexti] << endl;
#endif
        } else {
            pols.CTX[nexti] = pols.CTX[i];
        }

        // If setSP, SP'=op
        if (rom.line[zkPC].setSP == 1) {
            pols.SP[nexti] = op0;
            pols.setSP[i] = fr.one();
#ifdef LOG_SETX
            cout << "setSP SP[nexti]=" << pols.SP[nexti] << endl;
#endif
        } else {
            // SP' = SP + incStack
            pols.SP[nexti] = fr.add(pols.SP[i], fr.fromS32(rom.line[zkPC].incStack));
        }

        // If setPC, PC'=op
        if (rom.line[zkPC].setPC == 1) {
            pols.PC[nexti] = op0;
            pols.setPC[i] = fr.one();
#ifdef LOG_SETX
            cout << "setPC PC[nexti]=" << pols.PC[nexti] << endl;
#endif
        } else {
            // PC' = PC + incCode
            if (rom.line[zkPC].incCode<0 || rom.line[zkPC].incCode>0xFFFF)
            {
                cerr << "Error: incCode cannot be added to an u16 polynomial: " << rom.line[zkPC].incCode << endl;
                exit(-1);
            }
            pols.PC[nexti] = fr.add(pols.PC[i], fr.fromS32(rom.line[zkPC].incCode));
        }

        // If setRR, RR'=op0
        if (rom.line[zkPC].setRR == 1) {
            pols.RR[nexti] = op0;
            pols.setRR[i] = fr.one();
        } else {
            pols.RR[nexti] = pols.RR[i];
        }

        // If arith, increment pols.cntArith
        if (rom.line[zkPC].arith) {
            pols.cntArith[nexti] = fr.add(pols.cntArith[i], fr.one());
        } else {
            pols.cntArith[nexti] = pols.cntArith[i];
        }

        // If bin, increment pols.cntBinary
        if (rom.line[zkPC].bin) {
            pols.cntBinary[nexti] = fr.add(pols.cntBinary[i], fr.one());
        } else {
            pols.cntBinary[nexti] = pols.cntBinary[i];
        }

        // If memAlign, increment pols.cntMemAlign
        if (rom.line[zkPC].memAlign) {
            pols.cntMemAlign[nexti] = fr.add(pols.cntMemAlign[i], fr.one());
        } else {
            pols.cntMemAlign[nexti] = pols.cntMemAlign[i];
        }

        /*********/
        /* JUMPS */
        /*********/

        // If JMPN, jump conditionally if op0<0
        if (rom.line[zkPC].JMPN == 1)
        {
#ifdef LOG_JMP
            cout << "JMPN: op0=" << fr.toString(op0) << endl;
#endif
            int64_t o = fr.toS32(op0);
#ifdef LOG_JMP
            cout << "JMPN: o=" << o << endl;
#endif
            // If op<0, jump to addr: zkPC'=addr
            if (o < 0) {
                pols.isNeg[i] = fr.one();
                pols.zkPC[nexti] = fr.fromU64(addr);
                if (!bFastMode) required.Byte4[0x100000000 + o] = true;
#ifdef LOG_JMP
               cout << "JMPN next zkPC(1)=" << pols.zkPC[nexti] << endl;
#endif
            }
            // If op>=0, simply increase zkPC'=zkPC+1
            else
            {
                pols.zkPC[nexti] = fr.add(pols.zkPC[i], fr.one());
#ifdef LOG_JMP
                cout << "JMPN next zkPC(2)=" << pols.zkPC[nexti] << endl;
#endif
                if (!bFastMode) required.Byte4[o] = true;
            }
            pols.JMPN[i] = fr.one();
        }
        // If JMPC, jump conditionally if carry
        else if (rom.line[zkPC].JMPC == 1)
        {
            // If carry, jump to addr: zkPC'=addr
            if (!fr.isZero(pols.carry[i]))
            {
                pols.zkPC[nexti] = fr.fromU64(addr);
#ifdef LOG_JMP
               cout << "JMPC next zkPC(3)=" << pols.zkPC[nexti] << endl;
#endif
            }
            // If not carry, simply increase zkPC'=zkPC+1
            else
            {
                pols.zkPC[nexti] = fr.add(pols.zkPC[i], fr.one());
#ifdef LOG_JMP
                cout << "JMPC next zkPC(4)=" << pols.zkPC[nexti] << endl;
#endif
            }
            pols.JMPC[i] = fr.one();
        }
        // If JMP, directly jump zkPC'=addr
        else if (rom.line[zkPC].JMP == 1)
        {
            pols.zkPC[nexti] = fr.fromU64(addr);
#ifdef LOG_JMP
            cout << "JMP next zkPC(5)=" << pols.zkPC[nexti] << endl;
#endif
            pols.JMP[i] = fr.one();
        }
        // Else, simply increase zkPC'=zkPC+1
        else
        {
            pols.zkPC[nexti] = fr.add(pols.zkPC[i], fr.one());
        }

        // Calculate the new max mem address, if any
        uint32_t maxMemCalculated = 0;
        uint32_t mm = fr.toU64(pols.MAXMEM[i]);
        if (rom.line[zkPC].isMem==1)
        {
            if (addrRel>mm) {
                pols.isMaxMem[i] = fr.one();
                maxMemCalculated = addrRel;
                if (!bFastMode) required.Byte4[maxMemCalculated - mm] = true;
            } else {
                maxMemCalculated = mm;
                if (!bFastMode) required.Byte4[0] = true;
            }
        } else {
            maxMemCalculated = mm;
        }

        // If setMAXMEM, MAXMEM'=op
        if (rom.line[zkPC].setMAXMEM == 1) {
            pols.MAXMEM[nexti] = op0;
            pols.setMAXMEM[i] = fr.one();
#ifdef LOG_SETX
            cout << "setMAXMEM MAXMEM[nexti]=" << pols.MAXMEM[nexti] << endl;
#endif
        } else {
            pols.MAXMEM[nexti] = fr.fromU64(maxMemCalculated);
        }

        // If setGAS, GAS'=op
        if (rom.line[zkPC].setGAS == 1) {
            pols.GAS[nexti] = op0;
            pols.setGAS[i] = fr.one();
#ifdef LOG_SETX
            cout << "setGAS GAS[nexti]=" << pols.GAS[nexti] << endl;
#endif
        } else {
            pols.GAS[nexti] = pols.GAS[i];
        }

        // If setHASHPOS, HASHPOS' = op0 + incHashPos
        if (rom.line[zkPC].setHASHPOS == 1) {
            pols.HASHPOS[nexti] = fr.fromU64(fr.toS32(op0) + incHashPos);
            pols.setHASHPOS[i] = fr.one();
        } else {
            pols.HASHPOS[nexti] = fr.add( pols.HASHPOS[i], fr.fromU64(incHashPos) );
        }

        if (rom.line[zkPC].sRD || rom.line[zkPC].sWR || rom.line[zkPC].hashKDigest || rom.line[zkPC].hashPDigest)
        {
            pols.incCounter[i] = fr.fromU64(incCounter);
        }

        if (rom.line[zkPC].hashKDigest)
        {
            pols.cntKeccakF[nexti] = fr.add(pols.cntKeccakF[i], fr.fromU64(incCounter));
        }
        else
        {
            pols.cntKeccakF[nexti] = pols.cntKeccakF[i];
        }

        if (rom.line[zkPC].hashPDigest)
        {
            pols.cntPaddingPG[nexti] = fr.add(pols.cntPaddingPG[i], fr.fromU64(incCounter));
        }
        else
        {
            pols.cntPaddingPG[nexti] = pols.cntPaddingPG[i];
        }

        if (rom.line[zkPC].sRD || rom.line[zkPC].sWR || rom.line[zkPC].hashPDigest)
        {
            pols.cntPoseidonG[nexti] = fr.add(pols.cntPoseidonG[i], fr.fromU64(incCounter));
        }
        else
        {
            pols.cntPoseidonG[nexti] = pols.cntPoseidonG[i];
        }

        // Evaluate the list cmdAfter commands, and any children command, recursively
        for (uint64_t j=0; j<rom.line[zkPC].cmdAfter.size(); j++)
        {
            CommandResult cr;
            evalCommand(ctx, *rom.line[zkPC].cmdAfter[j], cr);
        }

#ifdef LOG_COMPLETED_STEPS
        cout << "<-- Completed step: " << step << " zkPC: " << zkPC << " op0: " << fr.toString(op0,16) << " A0: " << fr.toString(pols.A0[i],16) << " FREE0: " << fr.toString(pols.FREE0[i],16) << " FREE7: " << fr.toString(pols.FREE7[i],16) << endl;
#endif
#ifdef LOG_COMPLETED_STEPS_TO_FILE
        std::ofstream outfile;
        outfile.open("c.txt", std::ios_base::app); // append instead of overwrite
        outfile << "<-- Completed step: " << step << " zkPC: " << zkPC << " op0: " << fr.toString(op0,16) << " A0: " << fr.toString(pols.A0[i],16) << " FREE0: " << fr.toString(pols.FREE0[i],16) << " FREE7: " << fr.toString(pols.FREE7[i],16) << endl;
        outfile.close();
        //if (i==1000) break;
#endif

    } // End of main executor loop, for all evaluations

    // Copy the counters
    proverRequest.counters.arith = fr.toU64(pols.cntArith[0]);
    proverRequest.counters.binary = fr.toU64(pols.cntBinary[0]);
    proverRequest.counters.keccakF = fr.toU64(pols.cntKeccakF[0]);
    proverRequest.counters.memAlign = fr.toU64(pols.cntMemAlign[0]);
    proverRequest.counters.paddingPG = fr.toU64(pols.cntPaddingPG[0]);
    proverRequest.counters.poseidonG = fr.toU64(pols.cntPoseidonG[0]);
    proverRequest.counters.steps = ctx.lastStep;

    TimerStopAndLog(EXECUTE_LOOP);

    TimerStart(EXECUTE_CLEANUP);

    //printRegs(ctx);
    //printVars(ctx);
    //printMem(ctx);
    //printStorage(ctx);
    //printDb(ctx);

    // Check that all registers are set to 0
    if (!bFastMode) // In fast mode, last nexti was not 0 but 1, and pols have only 2 evaluations
    {
        checkFinalState(ctx);

        // Generate Padding KK required data
        for (uint64_t i=0; i<ctx.hashK.size(); i++)
        {
            PaddingKKExecutorInput h;
            h.dataBytes = ctx.hashK[i].data;
            uint64_t p = 0;
            while (p<ctx.hashK[i].data.size())
            {
                if (ctx.hashK[i].reads[p] != 0)
                {
                    h.reads.push_back(ctx.hashK[i].reads[p]);
                    p += ctx.hashK[i].reads[p];
                }
                else
                {
                    h.reads.push_back(1);
                    p++;
                }
            }
            if (p != ctx.hashK[i].data.size())
            {
                cerr << "Error: Main SM Executor: Reading hashK out of limits: i=" << i << " p=" << p << " ctx.hashK[i].data.size()=" << ctx.hashK[i].data.size() << endl;
                exit(-1);
            }
            required.PaddingKK.push_back(h);
        }

        // Generate Padding PG required data
        for (uint64_t i=0; i<ctx.hashP.size(); i++)
        {
            PaddingPGExecutorInput h;
            h.dataBytes = ctx.hashP[i].data;
            uint64_t p = 0;
            while (p<ctx.hashP[i].data.size())
            {
                if (ctx.hashP[i].reads[p] != 0)
                {
                    h.reads.push_back(ctx.hashP[i].reads[p]);
                    p += ctx.hashP[i].reads[p];
                }
                else
                {
                    h.reads.push_back(1);
                    p++;
                }
            }
            if (p != ctx.hashP[i].data.size())
            {
                cerr << "Error: Main SM Executor: Reading hashP out of limits: i=" << i << " p=" << p << " ctx.hashK[i].data.size()=" << ctx.hashK[i].data.size() << endl;
                exit(-1);
            }
            required.PaddingPG.push_back(h);
        }
    }

    TimerStopAndLog(EXECUTE_CLEANUP);

#ifdef LOG_TIME
    cout << "TIMER STATISTICS: Poseidon time: " << double(poseidonTime)/1000 << " ms, called " << poseidonTimes << " times, so " << poseidonTime/zkmax(poseidonTimes,(uint64_t)1) << " us/time" << endl;
    cout << "TIMER STATISTICS: SMT time: " << double(smtTime)/1000 << " ms, called " << smtTimes << " times, so " << smtTime/zkmax(smtTimes,(uint64_t)1) << " us/time" << endl;
    cout << "TIMER STATISTICS: Keccak time: " << double(keccakTime) << " ms, called " << keccakTimes << " times, so " << keccakTime/zkmax(keccakTimes,(uint64_t)1) << " us/time" << endl;
#endif
}

/* Sets first evaluation of all polynomials to zero */
void MainExecutor::initState(Context &ctx) // TODO: Should we delete this function? Default is already 0
{
    // Register value initial parameters
    ctx.pols.A0[0] = fr.zero();
    ctx.pols.A1[0] = fr.zero();
    ctx.pols.A2[0] = fr.zero();
    ctx.pols.A3[0] = fr.zero();
    ctx.pols.A4[0] = fr.zero();
    ctx.pols.A5[0] = fr.zero();
    ctx.pols.A6[0] = fr.zero();
    ctx.pols.B0[0] = fr.zero();
    ctx.pols.B1[0] = fr.zero();
    ctx.pols.B2[0] = fr.zero();
    ctx.pols.B3[0] = fr.zero();
    ctx.pols.B4[0] = fr.zero();
    ctx.pols.B5[0] = fr.zero();
    ctx.pols.B6[0] = fr.zero();
    ctx.pols.B7[0] = fr.zero();
    ctx.pols.C0[0] = fr.zero();
    ctx.pols.C1[0] = fr.zero();
    ctx.pols.C2[0] = fr.zero();
    ctx.pols.C3[0] = fr.zero();
    ctx.pols.C4[0] = fr.zero();
    ctx.pols.C5[0] = fr.zero();
    ctx.pols.C6[0] = fr.zero();
    ctx.pols.C7[0] = fr.zero();
    ctx.pols.D0[0] = fr.zero();
    ctx.pols.D1[0] = fr.zero();
    ctx.pols.D2[0] = fr.zero();
    ctx.pols.D3[0] = fr.zero();
    ctx.pols.D4[0] = fr.zero();
    ctx.pols.D5[0] = fr.zero();
    ctx.pols.D6[0] = fr.zero();
    ctx.pols.D7[0] = fr.zero();
    ctx.pols.E0[0] = fr.zero();
    ctx.pols.E1[0] = fr.zero();
    ctx.pols.E2[0] = fr.zero();
    ctx.pols.E3[0] = fr.zero();
    ctx.pols.E4[0] = fr.zero();
    ctx.pols.E5[0] = fr.zero();
    ctx.pols.E6[0] = fr.zero();
    ctx.pols.E7[0] = fr.zero();
    ctx.pols.SR0[0] = fr.zero();
    ctx.pols.SR1[0] = fr.zero();
    ctx.pols.SR2[0] = fr.zero();
    ctx.pols.SR3[0] = fr.zero();
    ctx.pols.SR4[0] = fr.zero();
    ctx.pols.SR5[0] = fr.zero();
    ctx.pols.SR6[0] = fr.zero();
    ctx.pols.SR7[0] = fr.zero();
    ctx.pols.CTX[0] = fr.zero();
    ctx.pols.SP[0] = fr.zero();
    ctx.pols.PC[0] = fr.zero();
    ctx.pols.MAXMEM[0] = fr.zero();
    ctx.pols.GAS[0] = fr.zero();
    ctx.pols.zkPC[0] = fr.zero();
    ctx.pols.cntArith[0] = fr.zero();
    ctx.pols.cntBinary[0] = fr.zero();
    ctx.pols.cntKeccakF[0] = fr.zero();
    ctx.pols.cntMemAlign[0] = fr.zero();
    ctx.pols.cntPaddingPG[0] = fr.zero();
    ctx.pols.cntPoseidonG[0] = fr.zero();
}

// Check that last evaluation (which is in fact the first one) is zero
void MainExecutor::checkFinalState(Context &ctx)
{
    if ( 
        (!fr.isZero(ctx.pols.A0[0])) ||
        (!fr.isZero(ctx.pols.A1[0])) ||
        (!fr.isZero(ctx.pols.A2[0])) ||
        (!fr.isZero(ctx.pols.A3[0])) ||
        (!fr.isZero(ctx.pols.A4[0])) ||
        (!fr.isZero(ctx.pols.A5[0])) ||
        (!fr.isZero(ctx.pols.A6[0])) ||
        (!fr.isZero(ctx.pols.A7[0])) ||
        (!fr.isZero(ctx.pols.B0[0])) ||
        (!fr.isZero(ctx.pols.B1[0])) ||
        (!fr.isZero(ctx.pols.B2[0])) ||
        (!fr.isZero(ctx.pols.B3[0])) ||
        (!fr.isZero(ctx.pols.B4[0])) ||
        (!fr.isZero(ctx.pols.B5[0])) ||
        (!fr.isZero(ctx.pols.B6[0])) ||
        (!fr.isZero(ctx.pols.B7[0])) ||
        (!fr.isZero(ctx.pols.C0[0])) ||
        (!fr.isZero(ctx.pols.C1[0])) ||
        (!fr.isZero(ctx.pols.C2[0])) ||
        (!fr.isZero(ctx.pols.C3[0])) ||
        (!fr.isZero(ctx.pols.C4[0])) ||
        (!fr.isZero(ctx.pols.C5[0])) ||
        (!fr.isZero(ctx.pols.C6[0])) ||
        (!fr.isZero(ctx.pols.C7[0])) ||
        (!fr.isZero(ctx.pols.D0[0])) ||
        (!fr.isZero(ctx.pols.D1[0])) ||
        (!fr.isZero(ctx.pols.D2[0])) ||
        (!fr.isZero(ctx.pols.D3[0])) ||
        (!fr.isZero(ctx.pols.D4[0])) ||
        (!fr.isZero(ctx.pols.D5[0])) ||
        (!fr.isZero(ctx.pols.D6[0])) ||
        (!fr.isZero(ctx.pols.D7[0])) ||
        (!fr.isZero(ctx.pols.E0[0])) ||
        (!fr.isZero(ctx.pols.E1[0])) ||
        (!fr.isZero(ctx.pols.E2[0])) ||
        (!fr.isZero(ctx.pols.E3[0])) ||
        (!fr.isZero(ctx.pols.E4[0])) ||
        (!fr.isZero(ctx.pols.E5[0])) ||
        (!fr.isZero(ctx.pols.E6[0])) ||
        (!fr.isZero(ctx.pols.E7[0])) ||
        (!fr.isZero(ctx.pols.SR0[0])) ||
        (!fr.isZero(ctx.pols.SR1[0])) ||
        (!fr.isZero(ctx.pols.SR2[0])) ||
        (!fr.isZero(ctx.pols.SR3[0])) ||
        (!fr.isZero(ctx.pols.SR4[0])) ||
        (!fr.isZero(ctx.pols.SR5[0])) ||
        (!fr.isZero(ctx.pols.SR6[0])) ||
        (!fr.isZero(ctx.pols.SR7[0])) ||
        (!fr.isZero(ctx.pols.CTX[0])) ||
        (!fr.isZero(ctx.pols.SP[0])) ||
        (!fr.isZero(ctx.pols.PC[0])) ||
        (!fr.isZero(ctx.pols.MAXMEM[0])) ||
        (!fr.isZero(ctx.pols.GAS[0])) ||
        (!fr.isZero(ctx.pols.zkPC[0]))
    ) {
        cerr << "Error: Program terminated with registers not set to zero" << endl;
        exit(-1);
    }
    else{
        //cout << "checkFinalState() succeeded" << endl;
    }
}
