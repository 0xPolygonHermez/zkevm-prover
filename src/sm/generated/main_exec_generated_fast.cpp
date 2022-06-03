#include "main_exec_generated_fast.hpp"

void MainExecGeneratedFast (FiniteField &fr, const Input &input, Database &db, Counters &counters){
    // opN are local, uncommitted polynomials
    FieldElement op0, op1, op2, op3, op4, op5, op6, op7;
    FieldElement A0, A1, A2, A3, A4, A5, A6, A7;
    A7 = A6 = A5 = A4 = A3 = A2 = A1 = A0 = fr.zero();
    FieldElement B0, B1, B2, B3, B4, B5, B6, B7;
    B7 = B6 = B5 = B4 = B3 = B2 = B1 = B0 = fr.zero();
    FieldElement C0, C1, C2, C3, C4, C5, C6, C7;
    C7 = C6 = C5 = C4 = C3 = C2 = C1 = C0 = fr.zero();
    FieldElement D0, D1, D2, D3, D4, D5, D6, D7;
    D7 = D6 = D5 = D4 = D3 = D2 = D1 = D0 = fr.zero();
    FieldElement E0, E1, E2, E3, E4, E5, E6, E7;
    E7 = E6 = E5 = E4 = E3 = E2 = E1 = E0 = fr.zero();
    FieldElement SR0, SR1, SR2, SR3, SR4, SR5, SR6, SR7;
    SR7 = SR6 = SR5 = SR4 = SR3 = SR2 = SR1 = SR0 = fr.zero();
    FieldElement HASHPOS, GAS, CTX, PC, SP, RR;
    HASHPOS = GAS = CTX = PC = SP = RR = fr.zero();
    uint64_t i=0; // Number of this evaluation
    uint64_t N=1<<23;

//RomLine0:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1:

    // op0 = op0 + inSTEP*STEP, where inSTEP=1
    op0 = i;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine4:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine5:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine6:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine7:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine8:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine9:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine10:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine11:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine12:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine13:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine14:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine15:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine16:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine17:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine18:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine19:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine20:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine21:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine22:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine23:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine24:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine25:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine26:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine27:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine28:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine29:

    // op0 = op0 + CONST
    op0 = 3;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine30:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine31:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine32:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine33:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine34:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine35:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine36:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine37:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine38:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine39:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine40:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine41:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine42:

    // op0 = op0 + CONST
    op0 = 3;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine43:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine44:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine45:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine46:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine47:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine48:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine49:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine50:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine51:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine52:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine53:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine54:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine55:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    // op = op + inC*C, where inC=1
    op0 = fr.add(op0, C0);
    op1 = fr.add(op1, C1);
    op2 = fr.add(op2, C2);
    op3 = fr.add(op3, C3);
    op4 = fr.add(op4, C4);
    op5 = fr.add(op5, C5);
    op6 = fr.add(op6, C6);
    op7 = fr.add(op7, C7);

    i++;
    if (i==N) return;

//RomLine56:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine57:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine58:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine59:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine60:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine61:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine62:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine63:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine64:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine65:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine66:

    // op0 = op0 + CONST
    op0 = 3;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine67:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine68:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine69:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine70:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine71:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine72:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine73:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine74:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine75:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine76:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine77:

    // op0 = op0 + CONST
    op0 = 8;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine78:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine79:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine80:

    // op0 = op0 + CONST
    op0 = 20;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine81:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine82:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine83:

    // op0 = op0 + CONST
    op0 = 8;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine84:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine85:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine86:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine87:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine88:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine89:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine90:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine91:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine92:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine93:

    // op = op + inSR*SR, where inSR=1
    op0 = SR0;
    op1 = SR1;
    op2 = SR2;
    op3 = SR3;
    op4 = SR4;
    op5 = SR5;
    op6 = SR6;
    op7 = SR7;

    i++;
    if (i==N) return;

//RomLine94:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine95:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine96:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine97:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine98:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine99:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine100:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine101:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine102:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine103:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine104:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine105:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine106:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine107:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine108:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine109:

    // op0 = op0 + CONST
    op0 = 110;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine110:

    // op0 = op0 + CONST
    op0 = 111;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine111:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(192));

    i++;
    if (i==N) return;

//RomLine112:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(248));

    i++;
    if (i==N) return;

//RomLine113:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(247));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine114:

    // op0 = op0 + CONST
    op0 = 115;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine115:

    // op0 = op0 + CONST
    op0 = 116;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine116:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine117:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(192));

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine118:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inC*C, where inC=1
    op0 = fr.add(op0, C0);
    op1 = fr.add(op1, C1);
    op2 = fr.add(op2, C2);
    op3 = fr.add(op3, C3);
    op4 = fr.add(op4, C4);
    op5 = fr.add(op5, C5);
    op6 = fr.add(op6, C6);
    op7 = fr.add(op7, C7);

    i++;
    if (i==N) return;

//RomLine119:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine120:

    // op0 = op0 + CONST
    op0 = 121;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine121:

    // op0 = op0 + CONST
    op0 = 122;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine122:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    i++;
    if (i==N) return;

//RomLine123:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(129));

    i++;
    if (i==N) return;

//RomLine124:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(137));

    i++;
    if (i==N) return;

//RomLine125:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine126:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine127:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine128:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine129:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine130:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine131:

    // op0 = op0 + CONST
    op0 = 132;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine132:

    // op0 = op0 + CONST
    op0 = 133;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine133:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine134:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine135:

    // op0 = op0 + CONST
    op0 = 136;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine136:

    // op0 = op0 + CONST
    op0 = 137;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine137:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    i++;
    if (i==N) return;

//RomLine138:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(129));

    i++;
    if (i==N) return;

//RomLine139:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(161));

    i++;
    if (i==N) return;

//RomLine140:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine141:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine142:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine143:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine144:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine145:

    // op0 = op0 + CONST
    op0 = 146;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine146:

    // op0 = op0 + CONST
    op0 = 147;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine147:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine148:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine149:

    // op0 = op0 + CONST
    op0 = 150;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine150:

    // op0 = op0 + CONST
    op0 = 151;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine151:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    i++;
    if (i==N) return;

//RomLine152:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(129));

    i++;
    if (i==N) return;

//RomLine153:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(161));

    i++;
    if (i==N) return;

//RomLine154:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine155:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine156:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine157:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine158:

    // op0 = op0 + CONST
    op0 = 159;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine159:

    // op0 = op0 + CONST
    op0 = 160;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine160:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine161:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine162:

    // op0 = op0 + CONST
    op0 = 163;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine163:

    // op0 = op0 + CONST
    op0 = 164;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine164:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    i++;
    if (i==N) return;

//RomLine165:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(129));

    i++;
    if (i==N) return;

//RomLine166:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(148));

    i++;
    if (i==N) return;

//RomLine167:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(149));

    i++;
    if (i==N) return;

//RomLine168:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine169:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine170:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine171:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine172:

    // op0 = op0 + CONST
    op0 = 173;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine173:

    // op0 = op0 + CONST
    op0 = 174;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine174:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine175:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine176:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine177:

    // op0 = op0 + CONST
    op0 = 178;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine178:

    // op0 = op0 + CONST
    op0 = 179;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine179:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    i++;
    if (i==N) return;

//RomLine180:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(129));

    i++;
    if (i==N) return;

//RomLine181:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(161));

    i++;
    if (i==N) return;

//RomLine182:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine183:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine184:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine185:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine186:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine187:

    // op0 = op0 + CONST
    op0 = 188;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine188:

    // op0 = op0 + CONST
    op0 = 189;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine189:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine190:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine191:

    // op0 = op0 + CONST
    op0 = 1024;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine192:

    // op0 = op0 + CONST
    op0 = 193;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine193:

    // op0 = op0 + CONST
    op0 = 194;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine194:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    i++;
    if (i==N) return;

//RomLine195:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(129));

    i++;
    if (i==N) return;

//RomLine196:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(184));

    i++;
    if (i==N) return;

//RomLine197:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(192));

    i++;
    if (i==N) return;

//RomLine198:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine199:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine200:

    // op0 = op0 + CONST
    op0 = 31;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine201:

    // op0 = op0 + CONST
    op0 = 202;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine202:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine203:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine204:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine205:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine206:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(183));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine207:

    // op0 = op0 + CONST
    op0 = 208;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine208:

    // op0 = op0 + CONST
    op0 = 209;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine209:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine210:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine211:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inD*D, where inD=-1
    op0 = fr.add(op0, fr.neg(D0));
    op1 = fr.add(op1, fr.neg(D1));
    op2 = fr.add(op2, fr.neg(D2));
    op3 = fr.add(op3, fr.neg(D3));
    op4 = fr.add(op4, fr.neg(D4));
    op5 = fr.add(op5, fr.neg(D5));
    op6 = fr.add(op6, fr.neg(D6));
    op7 = fr.add(op7, fr.neg(D7));

    i++;
    if (i==N) return;

//RomLine212:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inD*D, where inD=-1
    op0 = fr.add(op0, fr.neg(D0));
    op1 = fr.add(op1, fr.neg(D1));
    op2 = fr.add(op2, fr.neg(D2));
    op3 = fr.add(op3, fr.neg(D3));
    op4 = fr.add(op4, fr.neg(D4));
    op5 = fr.add(op5, fr.neg(D5));
    op6 = fr.add(op6, fr.neg(D6));
    op7 = fr.add(op7, fr.neg(D7));

    i++;
    if (i==N) return;

//RomLine213:

    // op0 = op0 + CONST
    op0 = 214;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine214:

    // op0 = op0 + CONST
    op0 = 215;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine215:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine216:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine217:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine218:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine219:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine220:

    // op0 = op0 + CONST
    op0 = 221;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine221:

    // op0 = op0 + CONST
    op0 = 222;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine222:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine223:

    // op0 = op0 + CONST
    op0 = 224;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine224:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine225:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine226:

    // op0 = op0 + CONST
    op0 = 227;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine227:

    // op0 = op0 + CONST
    op0 = 228;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine228:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    i++;
    if (i==N) return;

//RomLine229:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(129));

    i++;
    if (i==N) return;

//RomLine230:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(137));

    i++;
    if (i==N) return;

//RomLine231:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine232:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine233:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine234:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine235:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine236:

    // op0 = op0 + CONST
    op0 = 237;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine237:

    // op0 = op0 + CONST
    op0 = 238;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine238:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine239:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine240:

    // op0 = op0 + CONST
    op0 = 241;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine241:

    // op0 = op0 + CONST
    op0 = 242;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine242:

    // op0 = op0 + CONST
    op0 = 32896;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine243:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine244:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine245:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine246:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine247:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=-1
    op0 = fr.add(op0, fr.neg(B0));
    op1 = fr.add(op1, fr.neg(B1));
    op2 = fr.add(op2, fr.neg(B2));
    op3 = fr.add(op3, fr.neg(B3));
    op4 = fr.add(op4, fr.neg(B4));
    op5 = fr.add(op5, fr.neg(B5));
    op6 = fr.add(op6, fr.neg(B6));
    op7 = fr.add(op7, fr.neg(B7));

    i++;
    if (i==N) return;

//RomLine248:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    i++;
    if (i==N) return;

//RomLine249:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine250:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine251:

    // op0 = op0 + CONST
    op0 = 252;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine252:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine253:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inD*D, where inD=1
    op0 = fr.add(op0, D0);
    op1 = fr.add(op1, D1);
    op2 = fr.add(op2, D2);
    op3 = fr.add(op3, D3);
    op4 = fr.add(op4, D4);
    op5 = fr.add(op5, D5);
    op6 = fr.add(op6, D6);
    op7 = fr.add(op7, D7);

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine254:

    // op0 = op0 + CONST
    op0 = 255;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine255:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine256:

    // op0 = op0 + CONST
    op0 = 257;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine257:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine258:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inD*D, where inD=1
    op0 = fr.add(op0, D0);
    op1 = fr.add(op1, D1);
    op2 = fr.add(op2, D2);
    op3 = fr.add(op3, D3);
    op4 = fr.add(op4, D4);
    op5 = fr.add(op5, D5);
    op6 = fr.add(op6, D6);
    op7 = fr.add(op7, D7);

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine259:

    // op0 = op0 + CONST
    op0 = 260;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine260:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine261:

    // op0 = op0 + CONST
    op0 = 262;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine262:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine263:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inD*D, where inD=1
    op0 = fr.add(op0, D0);
    op1 = fr.add(op1, D1);
    op2 = fr.add(op2, D2);
    op3 = fr.add(op3, D3);
    op4 = fr.add(op4, D4);
    op5 = fr.add(op5, D5);
    op6 = fr.add(op6, D6);
    op7 = fr.add(op7, D7);

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine264:

    // op0 = op0 + CONST
    op0 = 265;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine265:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine266:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inC*C, where inC=1
    op0 = fr.add(op0, C0);
    op1 = fr.add(op1, C1);
    op2 = fr.add(op2, C2);
    op3 = fr.add(op3, C3);
    op4 = fr.add(op4, C4);
    op5 = fr.add(op5, C5);
    op6 = fr.add(op6, C6);
    op7 = fr.add(op7, C7);

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine267:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine268:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine269:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine270:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine271:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine272:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine273:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine274:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine275:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine276:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine277:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine278:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine279:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine280:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=-1
    op0 = fr.add(op0, fr.neg(B0));
    op1 = fr.add(op1, fr.neg(B1));
    op2 = fr.add(op2, fr.neg(B2));
    op3 = fr.add(op3, fr.neg(B3));
    op4 = fr.add(op4, fr.neg(B4));
    op5 = fr.add(op5, fr.neg(B5));
    op6 = fr.add(op6, fr.neg(B6));
    op7 = fr.add(op7, fr.neg(B7));

    // op = op + inC*C, where inC=-1
    op0 = fr.add(op0, fr.neg(C0));
    op1 = fr.add(op1, fr.neg(C1));
    op2 = fr.add(op2, fr.neg(C2));
    op3 = fr.add(op3, fr.neg(C3));
    op4 = fr.add(op4, fr.neg(C4));
    op5 = fr.add(op5, fr.neg(C5));
    op6 = fr.add(op6, fr.neg(C6));
    op7 = fr.add(op7, fr.neg(C7));

    // op = op + inD*D, where inD=-1
    op0 = fr.add(op0, fr.neg(D0));
    op1 = fr.add(op1, fr.neg(D1));
    op2 = fr.add(op2, fr.neg(D2));
    op3 = fr.add(op3, fr.neg(D3));
    op4 = fr.add(op4, fr.neg(D4));
    op5 = fr.add(op5, fr.neg(D5));
    op6 = fr.add(op6, fr.neg(D6));
    op7 = fr.add(op7, fr.neg(D7));

    i++;
    if (i==N) return;

//RomLine281:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine282:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine283:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine284:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine285:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=-1
    op0 = fr.add(op0, fr.neg(B0));
    op1 = fr.add(op1, fr.neg(B1));
    op2 = fr.add(op2, fr.neg(B2));
    op3 = fr.add(op3, fr.neg(B3));
    op4 = fr.add(op4, fr.neg(B4));
    op5 = fr.add(op5, fr.neg(B5));
    op6 = fr.add(op6, fr.neg(B6));
    op7 = fr.add(op7, fr.neg(B7));

    // op = op + inC*C, where inC=-1
    op0 = fr.add(op0, fr.neg(C0));
    op1 = fr.add(op1, fr.neg(C1));
    op2 = fr.add(op2, fr.neg(C2));
    op3 = fr.add(op3, fr.neg(C3));
    op4 = fr.add(op4, fr.neg(C4));
    op5 = fr.add(op5, fr.neg(C5));
    op6 = fr.add(op6, fr.neg(C6));
    op7 = fr.add(op7, fr.neg(C7));

    // op = op + inD*D, where inD=-1
    op0 = fr.add(op0, fr.neg(D0));
    op1 = fr.add(op1, fr.neg(D1));
    op2 = fr.add(op2, fr.neg(D2));
    op3 = fr.add(op3, fr.neg(D3));
    op4 = fr.add(op4, fr.neg(D4));
    op5 = fr.add(op5, fr.neg(D5));
    op6 = fr.add(op6, fr.neg(D6));
    op7 = fr.add(op7, fr.neg(D7));

    i++;
    if (i==N) return;

//RomLine286:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine287:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine288:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inD*D, where inD=1
    op0 = fr.add(op0, D0);
    op1 = fr.add(op1, D1);
    op2 = fr.add(op2, D2);
    op3 = fr.add(op3, D3);
    op4 = fr.add(op4, D4);
    op5 = fr.add(op5, D5);
    op6 = fr.add(op6, D6);
    op7 = fr.add(op7, D7);

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine289:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine290:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine291:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine292:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine293:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine294:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inC*C, where inC=-1
    op0 = fr.add(op0, fr.neg(C0));
    op1 = fr.add(op1, fr.neg(C1));
    op2 = fr.add(op2, fr.neg(C2));
    op3 = fr.add(op3, fr.neg(C3));
    op4 = fr.add(op4, fr.neg(C4));
    op5 = fr.add(op5, fr.neg(C5));
    op6 = fr.add(op6, fr.neg(C6));
    op7 = fr.add(op7, fr.neg(C7));

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine295:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine296:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine297:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine298:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inD*D, where inD=1
    op0 = fr.add(op0, D0);
    op1 = fr.add(op1, D1);
    op2 = fr.add(op2, D2);
    op3 = fr.add(op3, D3);
    op4 = fr.add(op4, D4);
    op5 = fr.add(op5, D5);
    op6 = fr.add(op6, D6);
    op7 = fr.add(op7, D7);

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine299:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine300:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inC*C, where inC=-1
    op0 = fr.add(op0, fr.neg(C0));
    op1 = fr.add(op1, fr.neg(C1));
    op2 = fr.add(op2, fr.neg(C2));
    op3 = fr.add(op3, fr.neg(C3));
    op4 = fr.add(op4, fr.neg(C4));
    op5 = fr.add(op5, fr.neg(C5));
    op6 = fr.add(op6, fr.neg(C6));
    op7 = fr.add(op7, fr.neg(C7));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine301:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine302:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine303:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine304:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inD*D, where inD=1
    op0 = fr.add(op0, D0);
    op1 = fr.add(op1, D1);
    op2 = fr.add(op2, D2);
    op3 = fr.add(op3, D3);
    op4 = fr.add(op4, D4);
    op5 = fr.add(op5, D5);
    op6 = fr.add(op6, D6);
    op7 = fr.add(op7, D7);

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine305:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine306:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine307:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine308:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine309:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine310:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine311:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine312:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine313:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine314:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine315:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine316:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine317:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine318:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine319:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine320:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine321:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine322:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine323:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine324:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine325:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine326:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine327:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine328:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine329:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine330:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine331:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine332:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine333:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine334:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine335:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine336:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine337:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine338:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine339:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine340:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine341:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine342:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine343:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine344:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine345:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine346:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine347:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine348:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine349:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine350:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine351:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine352:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine353:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine354:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine355:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine356:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine357:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine358:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine359:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine360:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine361:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine362:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine363:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine364:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine365:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine366:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine367:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine368:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine369:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine370:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine371:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine372:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine373:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine374:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine375:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine376:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine377:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine378:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine379:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine380:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine381:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine382:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine383:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine384:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine385:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine386:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine387:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine388:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine389:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine390:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine391:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine392:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine393:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine394:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine395:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine396:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine397:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine398:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine399:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine400:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine401:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine402:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine403:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine404:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine405:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine406:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine407:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine408:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine409:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine410:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine411:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine412:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine413:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine414:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine415:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine416:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine417:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine418:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine419:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine420:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine421:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine422:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine423:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine424:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine425:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine426:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine427:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine428:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine429:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine430:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine431:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine432:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine433:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine434:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine435:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine436:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine437:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine438:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine439:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine440:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine441:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine442:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine443:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine444:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine445:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine446:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine447:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine448:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine449:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine450:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine451:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine452:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine453:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine454:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine455:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine456:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine457:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine458:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine459:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine460:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine461:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine462:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine463:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine464:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine465:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine466:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine467:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine468:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine469:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine470:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine471:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine472:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine473:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine474:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine475:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine476:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine477:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine478:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine479:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine480:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine481:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine482:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine483:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine484:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine485:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine486:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine487:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine488:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine489:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine490:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine491:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine492:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine493:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine494:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine495:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine496:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine497:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine498:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine499:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine500:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine501:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine502:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine503:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine504:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine505:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine506:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine507:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine508:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine509:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine510:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine511:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine512:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine513:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine514:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine515:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine516:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine517:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine518:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine519:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine520:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine521:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine522:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine523:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine524:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine525:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine526:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine527:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine528:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine529:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine530:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine531:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine532:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine533:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine534:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine535:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine536:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine537:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine538:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine539:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine540:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine541:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine542:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine543:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine544:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine545:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine546:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine547:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine548:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine549:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine550:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine551:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine552:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine553:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine554:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine555:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine556:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine557:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine558:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine559:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine560:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine561:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine562:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine563:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3000));

    i++;
    if (i==N) return;

//RomLine564:

    // op0 = op0 + CONST
    op0 = 1024;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine565:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine566:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine567:

    // op0 = op0 + CONST
    op0 = 27;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine568:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine569:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine570:

    // op0 = op0 + CONST
    op0 = 28;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine571:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine572:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine573:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine574:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine575:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine576:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine577:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine578:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine579:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine580:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine581:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine582:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine583:

    // op0 = op0 + CONST
    op0 = 584;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine584:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine585:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine586:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine587:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine588:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(60));

    i++;
    if (i==N) return;

//RomLine589:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine590:

    // op0 = op0 + CONST
    op0 = 1024;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine591:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine592:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine593:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine594:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine595:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine596:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine597:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine598:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine599:

    // op0 = op0 + CONST
    op0 = 600;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine600:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine601:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine602:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine603:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine604:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine605:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine606:

    // op0 = op0 + CONST
    op0 = 607;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine607:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine608:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine609:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(600));

    i++;
    if (i==N) return;

//RomLine610:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine611:

    // op0 = op0 + CONST
    op0 = 1024;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine612:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine613:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine614:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine615:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine616:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine617:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine618:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine619:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine620:

    // op0 = op0 + CONST
    op0 = 621;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine621:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine622:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine623:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine624:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine625:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine626:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine627:

    // op0 = op0 + CONST
    op0 = 628;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine628:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine629:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(15));

    i++;
    if (i==N) return;

//RomLine630:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine631:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine632:

    // op = op + inA*A, where inA=-3
    op0 = fr.mul(-3, A0);
    op1 = fr.mul(-3, A1);
    op2 = fr.mul(-3, A2);
    op3 = fr.mul(-3, A3);
    op4 = fr.mul(-3, A4);
    op5 = fr.mul(-3, A5);
    op6 = fr.mul(-3, A6);
    op7 = fr.mul(-3, A7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine633:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine634:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine635:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine636:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine637:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine638:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine639:

    // op0 = op0 + CONST
    op0 = 640;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine640:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine641:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine642:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine643:

    // op0 = op0 + CONST
    op0 = 644;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine644:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine645:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine646:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine647:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine648:

    // op0 = op0 + CONST
    op0 = 649;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine649:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine650:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine651:

    // op0 = op0 + CONST
    op0 = 652;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine652:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine653:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine654:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine655:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine656:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine657:

    // op0 = op0 + CONST
    op0 = 658;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine658:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine659:

    // op0 = op0 + CONST
    op0 = 660;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine660:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine661:

    // op0 = op0 + CONST
    op0 = 662;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine662:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine663:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine664:

    // op0 = op0 + CONST
    op0 = 665;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine665:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine666:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine667:

    // op0 = op0 + CONST
    op0 = 668;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine668:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine669:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine670:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine671:

    // op0 = op0 + CONST
    op0 = 672;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine672:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine673:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine674:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine675:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine676:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine677:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine678:

    // op0 = op0 + CONST
    op0 = 679;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine679:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine680:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine681:

    // op0 = op0 + CONST
    op0 = 682;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine682:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine683:

    // op0 = op0 + CONST
    op0 = 684;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine684:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine685:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine686:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine687:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine688:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine689:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine690:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine691:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine692:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine693:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine694:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine695:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine696:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine697:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine698:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine699:

    // op = op + inC*C, where inC=8
    op0 = fr.mul(8, C0);
    op1 = fr.mul(8, C1);
    op2 = fr.mul(8, C2);
    op3 = fr.mul(8, C3);
    op4 = fr.mul(8, C4);
    op5 = fr.mul(8, C5);
    op6 = fr.mul(8, C6);
    op7 = fr.mul(8, C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine700:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine701:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine702:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine703:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine704:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine705:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine706:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine707:

    // op0 = op0 + CONST
    op0 = 200;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine708:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine709:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine710:

    // op0 = op0 + CONST
    op0 = 200;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine711:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine712:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine713:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine714:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine715:

    // op0 = op0 + CONST
    op0 = 128;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine716:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine717:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine718:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine719:

    // op0 = op0 + CONST
    op0 = 64;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine720:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine721:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine722:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine723:

    // op0 = op0 + CONST
    op0 = 724;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine724:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine725:

    // op0 = op0 + CONST
    op0 = 726;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine726:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine727:

    // op0 = op0 + CONST
    op0 = 728;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine728:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine729:

    // op0 = op0 + CONST
    op0 = 730;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine730:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine731:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine732:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine733:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine734:

    // op0 = op0 + CONST
    op0 = 735;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine735:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine736:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine737:

    // op0 = op0 + CONST
    op0 = 738;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine738:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(150));

    i++;
    if (i==N) return;

//RomLine739:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine740:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine741:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine742:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine743:

    // op0 = op0 + CONST
    op0 = 96;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine744:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine745:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine746:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine747:

    // op0 = op0 + CONST
    op0 = 64;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine748:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine749:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine750:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine751:

    // op0 = op0 + CONST
    op0 = 752;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine752:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine753:

    // op0 = op0 + CONST
    op0 = 754;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine754:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine755:

    // op0 = op0 + CONST
    op0 = 756;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine756:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine757:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine758:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine759:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine760:

    // op0 = op0 + CONST
    op0 = 761;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine761:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine762:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine763:

    // op0 = op0 + CONST
    op0 = 764;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine764:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(6000));

    i++;
    if (i==N) return;

//RomLine765:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine766:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine767:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine768:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine769:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine770:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine771:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine772:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine773:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine774:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine775:

    // op0 = op0 + CONST
    op0 = 776;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine776:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine777:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine778:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine779:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine780:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine781:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine782:

    // op0 = op0 + CONST
    op0 = 783;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine783:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine784:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine785:

    // op = op + inA*A, where inA=-34000
    op0 = fr.mul(-34000, A0);
    op1 = fr.mul(-34000, A1);
    op2 = fr.mul(-34000, A2);
    op3 = fr.mul(-34000, A3);
    op4 = fr.mul(-34000, A4);
    op5 = fr.mul(-34000, A5);
    op6 = fr.mul(-34000, A6);
    op7 = fr.mul(-34000, A7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(45000));

    i++;
    if (i==N) return;

//RomLine786:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine787:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine788:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine789:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine790:

    // op0 = op0 + CONST
    op0 = 213;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine791:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine792:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine793:

    // op0 = op0 + CONST
    op0 = 1024;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine794:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine795:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine796:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine797:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine798:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine799:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine800:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine801:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine802:

    // op0 = op0 + CONST
    op0 = 803;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine803:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine804:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine805:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine806:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine807:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine808:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine809:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine810:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine811:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine812:

    // op0 = op0 + CONST
    op0 = 813;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine813:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine814:

    // op0 = op0 + CONST
    op0 = 815;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine815:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine816:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine817:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine818:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine819:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine820:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine821:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine822:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine823:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine824:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine825:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine826:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine827:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine828:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine829:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine830:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine831:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine832:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine833:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine834:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(4));

    i++;
    if (i==N) return;

//RomLine835:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(5));

    i++;
    if (i==N) return;

//RomLine836:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(6));

    i++;
    if (i==N) return;

//RomLine837:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(7));

    i++;
    if (i==N) return;

//RomLine838:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(8));

    i++;
    if (i==N) return;

//RomLine839:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(9));

    i++;
    if (i==N) return;

//RomLine840:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(10));

    i++;
    if (i==N) return;

//RomLine841:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine842:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine843:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine844:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine845:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine846:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine847:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine848:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine849:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine850:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine851:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine852:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine853:

    // op = op + inSR*SR, where inSR=1
    op0 = SR0;
    op1 = SR1;
    op2 = SR2;
    op3 = SR3;
    op4 = SR4;
    op5 = SR5;
    op6 = SR6;
    op7 = SR7;

    i++;
    if (i==N) return;

//RomLine854:

    // op = op + inSR*SR, where inSR=1
    op0 = SR0;
    op1 = SR1;
    op2 = SR2;
    op3 = SR3;
    op4 = SR4;
    op5 = SR5;
    op6 = SR6;
    op7 = SR7;

    i++;
    if (i==N) return;

//RomLine855:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine856:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine857:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine858:

    // op0 = op0 + CONST
    op0 = 1000;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine859:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine860:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine861:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine862:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine863:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine864:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine865:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine866:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine867:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine868:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine869:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine870:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine871:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine872:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine873:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine874:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine875:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine876:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine877:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine878:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine879:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine880:

    // op0 = op0 + CONST
    op0 = 881;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine881:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine882:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine883:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine884:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine885:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine886:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine887:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine888:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine889:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine890:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine891:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine892:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine893:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine894:

    // op = op + inSR*SR, where inSR=1
    op0 = SR0;
    op1 = SR1;
    op2 = SR2;
    op3 = SR3;
    op4 = SR4;
    op5 = SR5;
    op6 = SR6;
    op7 = SR7;

    i++;
    if (i==N) return;

//RomLine895:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine896:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(21000));

    i++;
    if (i==N) return;

//RomLine897:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine898:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine899:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine900:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine901:

    // op0 = op0 + CONST
    op0 = fr.neg(1);
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine902:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine903:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine904:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1024);

    i++;
    if (i==N) return;

//RomLine905:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine906:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine907:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine908:

    // op0 = op0 + CONST
    op0 = 909;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine909:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine910:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=-1
    op0 = fr.add(op0, fr.neg(B0));
    op1 = fr.add(op1, fr.neg(B1));
    op2 = fr.add(op2, fr.neg(B2));
    op3 = fr.add(op3, fr.neg(B3));
    op4 = fr.add(op4, fr.neg(B4));
    op5 = fr.add(op5, fr.neg(B5));
    op6 = fr.add(op6, fr.neg(B6));
    op7 = fr.add(op7, fr.neg(B7));

    // op = op + inD*D, where inD=-1
    op0 = fr.add(op0, fr.neg(D0));
    op1 = fr.add(op1, fr.neg(D1));
    op2 = fr.add(op2, fr.neg(D2));
    op3 = fr.add(op3, fr.neg(D3));
    op4 = fr.add(op4, fr.neg(D4));
    op5 = fr.add(op5, fr.neg(D5));
    op6 = fr.add(op6, fr.neg(D6));
    op7 = fr.add(op7, fr.neg(D7));

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine911:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 31);

    i++;
    if (i==N) return;

//RomLine912:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine913:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine914:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine915:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine916:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine917:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(4));

    i++;
    if (i==N) return;

//RomLine918:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine919:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(16));

    i++;
    if (i==N) return;

//RomLine920:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine921:

    // op = op + inSR*SR, where inSR=1
    op0 = SR0;
    op1 = SR1;
    op2 = SR2;
    op3 = SR3;
    op4 = SR4;
    op5 = SR5;
    op6 = SR6;
    op7 = SR7;

    i++;
    if (i==N) return;

//RomLine922:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine923:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine924:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine925:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine926:

    // op0 = op0 + CONST
    op0 = 10;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine927:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine928:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine929:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine930:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine931:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine932:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine933:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine934:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine935:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine936:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine937:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine938:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine939:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine940:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine941:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(128));

    i++;
    if (i==N) return;

//RomLine942:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine943:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine944:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 214);

    i++;
    if (i==N) return;

//RomLine945:

    // op0 = op0 + CONST
    op0 = 148;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine946:

    // op0 = op0 + CONST
    op0 = 20;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine947:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine948:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine949:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 128);

    i++;
    if (i==N) return;

//RomLine950:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine951:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine952:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine953:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine954:

    // op0 = op0 + CONST
    op0 = 214;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine955:

    // op0 = op0 + CONST
    op0 = 148;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine956:

    // op0 = op0 + CONST
    op0 = 20;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine957:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine958:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine959:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine960:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine961:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine962:

    // op0 = op0 + CONST
    op0 = 128;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine963:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine964:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine965:

    // op0 = op0 + CONST
    op0 = 12;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine966:

    // op0 = op0 + CONST
    op0 = 967;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine967:

    // op0 = op0 + CONST
    op0 = 968;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine968:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine969:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine970:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine971:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine972:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine973:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine974:

    // op0 = op0 + CONST
    op0 = 975;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine975:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine976:

    // op = op + inA*A, where inA=-6
    op0 = fr.mul(-6, A0);
    op1 = fr.mul(-6, A1);
    op2 = fr.mul(-6, A2);
    op3 = fr.mul(-6, A3);
    op4 = fr.mul(-6, A4);
    op5 = fr.mul(-6, A5);
    op6 = fr.mul(-6, A6);
    op7 = fr.mul(-6, A7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine977:

    // op0 = op0 + CONST
    op0 = 1024;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine978:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine979:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine980:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine981:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine982:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine983:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine984:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine985:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine986:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine987:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine988:

    // op0 = op0 + CONST
    op0 = 989;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine989:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine990:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine991:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine992:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine993:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine994:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine995:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine996:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine997:

    // op0 = op0 + CONST
    op0 = 255;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine998:

    // op0 = op0 + CONST
    op0 = 20;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine999:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1000:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1001:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1002:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1003:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1004:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1005:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1006:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1007:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1008:

    // op0 = op0 + CONST
    op0 = 12;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1009:

    // op0 = op0 + CONST
    op0 = 1010;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1010:

    // op0 = op0 + CONST
    op0 = 1011;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1011:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1012:

    // op0 = op0 + CONST
    op0 = 1024;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1013:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1014:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1015:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1016:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32000));

    i++;
    if (i==N) return;

//RomLine1017:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1018:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1019:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1020:

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = PC;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1021:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1022:

    // op0 = op0 + CONST
    op0 = 1023;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1023:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1024:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1025:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = fr.add(op0, SP);

    i++;
    if (i==N) return;

//RomLine1026:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1027:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + inPC*PC, where inPC=-1
    op0 = fr.add(op0, fr.neg(PC));

    i++;
    if (i==N) return;

//RomLine1028:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1029:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1030:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1031:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1032:

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = PC;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine1033:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1034:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1035:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1036:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1037:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1038:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1039:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1040:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1041:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1042:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + inPC*PC, where inPC=-1
    op0 = fr.add(op0, fr.neg(PC));

    i++;
    if (i==N) return;

//RomLine1043:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1044:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1045:

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = PC;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine1046:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1047:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1048:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine1049:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1050:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1051:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine1052:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1053:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1054:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1055:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1056:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1057:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine1058:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1059:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1060:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1061:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1062:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine1063:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1064:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1065:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1066:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1067:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1068:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1069:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1070:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1071:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine1072:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1073:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1074:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1075:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1076:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1077:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1078:

    // op0 = op0 + CONST
    op0 = 1079;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1079:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1080:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1081:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1082:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine1083:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1084:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1085:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1086:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1087:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + inGAS*GAS, where inGAS=-1
    op0 = fr.add(op0, fr.neg(GAS));

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1088:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1089:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1090:

    // op0 = op0 + CONST
    op0 = 1091;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1091:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1092:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=-1
    op0 = fr.add(op0, fr.neg(B0));
    op1 = fr.add(op1, fr.neg(B1));
    op2 = fr.add(op2, fr.neg(B2));
    op3 = fr.add(op3, fr.neg(B3));
    op4 = fr.add(op4, fr.neg(B4));
    op5 = fr.add(op5, fr.neg(B5));
    op6 = fr.add(op6, fr.neg(B6));
    op7 = fr.add(op7, fr.neg(B7));

    i++;
    if (i==N) return;

//RomLine1093:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1094:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine1095:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1096:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1097:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1098:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1099:

    // op0 = op0 + CONST
    op0 = 1100;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1100:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1101:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1102:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1103:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1104:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1105:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1106:

    // op0 = op0 + CONST
    op0 = 1107;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1107:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1108:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1109:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1110:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine1111:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1112:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + inGAS*GAS, where inGAS=-1
    op0 = fr.add(op0, fr.neg(GAS));

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1113:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1114:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1115:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1116:

    // op0 = op0 + CONST
    op0 = 1117;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1117:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1118:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1119:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1120:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1121:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1122:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1123:

    // op0 = op0 + CONST
    op0 = 1124;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1124:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1125:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1126:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1127:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine1128:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1129:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine1130:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1131:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1132:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1133:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine1134:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1135:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1136:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1137:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1138:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1139:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1140:

    // op0 = op0 + inRR*RR, where inRR=1
    op0 = RR;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1141:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1142:

    // op0 = op0 + CONST
    op0 = 1024;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1143:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1144:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1145:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1146:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1147:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1148:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine1149:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1150:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1151:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1152:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1153:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1154:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1155:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1156:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1157:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1158:

    // op0 = op0 + CONST
    op0 = 1159;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1159:

    // op0 = op0 + CONST
    op0 = 1160;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1160:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1161:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1162:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1163:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1164:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine1165:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1166:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1167:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1168:

    // op0 = op0 + CONST
    op0 = 1169;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1169:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1170:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1171:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1172:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1173:

    // op0 = op0 + CONST
    op0 = 1174;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1174:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1175:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    i++;
    if (i==N) return;

//RomLine1176:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1177:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1178:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1179:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1180:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1181:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1182:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1183:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1184:

    // op0 = op0 + CONST
    op0 = 1185;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1185:

    // op0 = op0 + CONST
    op0 = 1186;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1186:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1187:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1188:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1189:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1190:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1191:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1192:

    // op0 = op0 + CONST
    op0 = 1193;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1193:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1194:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1195:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1196:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inD*D, where inD=1
    op0 = fr.add(op0, D0);
    op1 = fr.add(op1, D1);
    op2 = fr.add(op2, D2);
    op3 = fr.add(op3, D3);
    op4 = fr.add(op4, D4);
    op5 = fr.add(op5, D5);
    op6 = fr.add(op6, D6);
    op7 = fr.add(op7, D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1197:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1198:

    // op0 = op0 + CONST
    op0 = 1199;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1199:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1200:

    // op0 = op0 + CONST
    op0 = 1201;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1201:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1202:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    i++;
    if (i==N) return;

//RomLine1203:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1204:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1205:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1206:

    // op0 = op0 + inRR*RR, where inRR=1
    op0 = RR;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1207:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1208:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1209:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1210:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1211:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1212:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1213:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1214:

    // op0 = op0 + CONST
    op0 = 1215;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1215:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1216:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1217:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1218:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1219:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1220:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1221:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1222:

    // op0 = op0 + inRR*RR, where inRR=1
    op0 = RR;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1223:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1224:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1225:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1226:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1227:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1228:

    // op0 = op0 + CONST
    op0 = fr.neg(1);
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1229:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1230:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1231:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1232:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1233:

    // op0 = op0 + CONST
    op0 = 1234;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1234:

    // op0 = op0 + CONST
    op0 = 1235;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1235:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1236:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1237:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1238:

    // op0 = op0 + CONST
    op0 = 1239;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1239:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    i++;
    if (i==N) return;

//RomLine1240:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1241:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1242:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1243:

    // op0 = op0 + CONST
    op0 = 1244;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1244:

    // op0 = op0 + CONST
    op0 = 1245;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1245:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1246:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1247:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1248:

    // op0 = op0 + CONST
    op0 = 1249;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1249:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    i++;
    if (i==N) return;

//RomLine1250:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1251:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1252:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1253:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1254:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1255:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1256:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1257:

    // op0 = op0 + inRR*RR, where inRR=1
    op0 = RR;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1258:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1259:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1260:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1261:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1262:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1263:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    i++;
    if (i==N) return;

//RomLine1264:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1265:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1266:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1267:

    // op0 = op0 + CONST
    op0 = 1268;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1268:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1269:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1270:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1271:

    // op0 = op0 + CONST
    op0 = 1272;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1272:

    // op0 = op0 + CONST
    op0 = 1273;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1273:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    i++;
    if (i==N) return;

//RomLine1274:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1275:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1276:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1277:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1278:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1279:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1280:

    // op0 = op0 + CONST
    op0 = 1281;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1281:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1282:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1283:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1284:

    // op0 = op0 + CONST
    op0 = 1285;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1285:

    // op0 = op0 + CONST
    op0 = 1286;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1286:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    i++;
    if (i==N) return;

//RomLine1287:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1288:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1289:

    // op0 = op0 + CONST
    op0 = 1290;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1290:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1291:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1292:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inD*D, where inD=-1
    op0 = fr.add(op0, fr.neg(D0));
    op1 = fr.add(op1, fr.neg(D1));
    op2 = fr.add(op2, fr.neg(D2));
    op3 = fr.add(op3, fr.neg(D3));
    op4 = fr.add(op4, fr.neg(D4));
    op5 = fr.add(op5, fr.neg(D5));
    op6 = fr.add(op6, fr.neg(D6));
    op7 = fr.add(op7, fr.neg(D7));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1293:

    // op0 = op0 + CONST
    op0 = 1294;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1294:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    i++;
    if (i==N) return;

//RomLine1295:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1296:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1297:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1298:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1299:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1300:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1301:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1302:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1303:

    // op0 = op0 + inRR*RR, where inRR=1
    op0 = RR;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1304:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1305:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1306:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1307:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1308:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1309:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1310:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1311:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1312:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1313:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1314:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1315:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1316:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1317:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1318:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1319:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1320:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1321:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1322:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1323:

    // op0 = op0 + CONST
    op0 = 1324;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1324:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1325:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1326:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1327:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1328:

    // op0 = op0 + CONST
    op0 = 1329;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1329:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1330:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1331:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1332:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1333:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1334:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1335:

    // op0 = op0 + inRR*RR, where inRR=1
    op0 = RR;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1336:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1337:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1338:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1339:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1340:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1341:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1342:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1343:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1344:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1345:

    // op0 = op0 + CONST
    op0 = 1346;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1346:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1347:

    // op0 = op0 + CONST
    op0 = 1348;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1348:

    // op0 = op0 + CONST
    op0 = 1349;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1349:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inC*C, where inC=1
    op0 = fr.add(op0, C0);
    op1 = fr.add(op1, C1);
    op2 = fr.add(op2, C2);
    op3 = fr.add(op3, C3);
    op4 = fr.add(op4, C4);
    op5 = fr.add(op5, C5);
    op6 = fr.add(op6, C6);
    op7 = fr.add(op7, C7);

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1350:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1351:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1352:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1353:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1354:

    // op0 = op0 + CONST
    op0 = 1355;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1355:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1356:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1357:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1358:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inD*D, where inD=1
    op0 = fr.add(op0, D0);
    op1 = fr.add(op1, D1);
    op2 = fr.add(op2, D2);
    op3 = fr.add(op3, D3);
    op4 = fr.add(op4, D4);
    op5 = fr.add(op5, D5);
    op6 = fr.add(op6, D6);
    op7 = fr.add(op7, D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1359:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1360:

    // op0 = op0 + CONST
    op0 = 1361;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1361:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1362:

    // op0 = op0 + CONST
    op0 = 1363;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1363:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1364:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1365:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1366:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1367:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1368:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1369:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1370:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1371:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1372:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1373:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1374:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1375:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1376:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1377:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1378:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1379:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1380:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1381:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1382:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1383:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1384:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1385:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1386:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1387:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1388:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1389:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1390:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1391:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1392:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1393:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1394:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1395:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1396:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1397:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1398:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1399:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1400:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1401:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1402:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1403:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1404:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1405:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1406:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1407:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1408:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1409:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1410:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1411:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1412:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1413:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1414:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1415:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1416:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1417:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1418:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1419:

    // op = op + inA*A, where inA=-3
    op0 = fr.mul(-3, A0);
    op1 = fr.mul(-3, A1);
    op2 = fr.mul(-3, A2);
    op3 = fr.mul(-3, A3);
    op4 = fr.mul(-3, A4);
    op5 = fr.mul(-3, A5);
    op6 = fr.mul(-3, A6);
    op7 = fr.mul(-3, A7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine1420:

    // op = op + inB*B, where inB=3
    op0 = fr.mul(3, B0);
    op1 = fr.mul(3, B1);
    op2 = fr.mul(3, B2);
    op3 = fr.mul(3, B3);
    op4 = fr.mul(3, B4);
    op5 = fr.mul(3, B5);
    op6 = fr.mul(3, B6);
    op7 = fr.mul(3, B7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine1421:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1422:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1423:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1424:

    // op0 = op0 + inRR*RR, where inRR=1
    op0 = RR;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1425:

    // op0 = op0 + CONST
    op0 = 1426;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1426:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1427:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1428:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1429:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1430:

    // op0 = op0 + CONST
    op0 = 1431;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1431:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1432:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1433:

    // op0 = op0 + inRR*RR, where inRR=1
    op0 = RR;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1434:

    // op0 = op0 + CONST
    op0 = 1435;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1435:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1436:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1437:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1438:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1439:

    // op0 = op0 + CONST
    op0 = 1440;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1440:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1441:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1442:

    // op0 = op0 + inRR*RR, where inRR=1
    op0 = RR;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1443:

    // op0 = op0 + CONST
    op0 = 1444;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1444:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1445:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1446:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1447:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1448:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1449:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1450:

    // op0 = op0 + CONST
    op0 = 1451;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1451:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1452:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1453:

    // op0 = op0 + inRR*RR, where inRR=1
    op0 = RR;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1454:

    // op0 = op0 + CONST
    op0 = 1455;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1455:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1456:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1457:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1458:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1459:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1460:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1461:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1462:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1463:

    // op0 = op0 + CONST
    op0 = 1464;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1464:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1465:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1466:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1467:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1468:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1469:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1470:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1471:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1472:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1473:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1474:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1475:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1476:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1477:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1478:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1479:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1480:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1481:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1482:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1483:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1484:

    // op0 = op0 + CONST
    op0 = 8;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1485:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1486:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1487:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1488:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1489:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1490:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1491:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1492:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1493:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1494:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1495:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1496:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1497:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1498:

    // op0 = op0 + CONST
    op0 = 255;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1499:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1500:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1501:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1502:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1503:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1504:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1505:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1506:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1507:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1508:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1509:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1510:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1511:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1512:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1513:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1514:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1515:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1516:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1517:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1518:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1519:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1520:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1521:

    // op0 = op0 + CONST
    op0 = 8;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1522:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1523:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1524:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1525:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1526:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1527:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1528:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1529:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1530:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1531:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1532:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1533:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1534:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1535:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1536:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1537:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1538:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 256);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1539:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1540:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1541:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1542:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1543:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1544:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1545:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1546:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 256);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1547:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1548:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    i++;
    if (i==N) return;

//RomLine1549:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1550:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1551:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1552:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1553:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1554:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 256);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1555:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1556:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1557:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1558:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1559:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1560:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1561:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1562:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1563:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1564:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1565:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1566:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1567:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1568:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1569:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1570:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1571:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine1572:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1573:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1574:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1575:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1576:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1577:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1578:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1579:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1580:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1581:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1582:

    // op0 = op0 + CONST
    op0 = 1583;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1583:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1584:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1585:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1586:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1587:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1588:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1589:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1590:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1591:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1592:

    // op0 = op0 + CONST
    op0 = 1593;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1593:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1594:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1595:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(5));

    i++;
    if (i==N) return;

//RomLine1596:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1597:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1598:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1599:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1600:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1601:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1602:

    // op0 = op0 + CONST
    op0 = 1603;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1603:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1604:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1605:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1606:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1607:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1608:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1609:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1610:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1611:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1612:

    // op0 = op0 + CONST
    op0 = 1613;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1613:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1614:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1615:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(5));

    i++;
    if (i==N) return;

//RomLine1616:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1617:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1618:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1619:

    // op0 = op0 + CONST
    op0 = 1620;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1620:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1621:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1622:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1623:

    // op0 = op0 + CONST
    op0 = 1624;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1624:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1625:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1626:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1627:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1628:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1629:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1630:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1631:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1632:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1633:

    // op0 = op0 + CONST
    op0 = 1634;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1634:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1635:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1636:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(5));

    i++;
    if (i==N) return;

//RomLine1637:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1638:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1639:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1640:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1641:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1642:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1643:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1644:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1645:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(5));

    i++;
    if (i==N) return;

//RomLine1646:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1647:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1648:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1649:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1650:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1651:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1652:

    // op0 = op0 + CONST
    op0 = 1653;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1653:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1654:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1655:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(5));

    i++;
    if (i==N) return;

//RomLine1656:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1657:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1658:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1659:

    // op0 = op0 + CONST
    op0 = 1660;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1660:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1661:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1662:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1663:

    // op0 = op0 + CONST
    op0 = 1664;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1664:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1665:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1666:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1667:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1668:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1669:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1670:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1671:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1672:

    // op0 = op0 + CONST
    op0 = 1673;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1673:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1674:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1675:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(5));

    i++;
    if (i==N) return;

//RomLine1676:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1677:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1678:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1679:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1680:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1681:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1682:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1683:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1684:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1685:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(5));

    i++;
    if (i==N) return;

//RomLine1686:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1687:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1688:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1689:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1690:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1691:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1692:

    // op0 = op0 + CONST
    op0 = 1693;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1693:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1694:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1695:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1696:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1697:

    // op0 = op0 + CONST
    op0 = 1698;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1698:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1699:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1700:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(8));

    i++;
    if (i==N) return;

//RomLine1701:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1702:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1703:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1704:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1705:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1706:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1707:

    // op0 = op0 + CONST
    op0 = 1708;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1708:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1709:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1710:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1711:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1712:

    // op0 = op0 + CONST
    op0 = 1713;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1713:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1714:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1715:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(8));

    i++;
    if (i==N) return;

//RomLine1716:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1717:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1718:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1719:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1720:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1721:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1722:

    // op0 = op0 + CONST
    op0 = 1723;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1723:

    // op = op + inA*A, where inA=-50
    op0 = fr.mul(-50, A0);
    op1 = fr.mul(-50, A1);
    op2 = fr.mul(-50, A2);
    op3 = fr.mul(-50, A3);
    op4 = fr.mul(-50, A4);
    op5 = fr.mul(-50, A5);
    op6 = fr.mul(-50, A6);
    op7 = fr.mul(-50, A7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(10));

    i++;
    if (i==N) return;

//RomLine1724:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1725:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1726:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1727:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1728:

    // op0 = op0 + CONST
    op0 = 31;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1729:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1730:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1731:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1732:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1733:

    // op0 = op0 + CONST
    op0 = 7;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1734:

    // op0 = op0 + CONST
    op0 = 1735;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1735:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1736:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1737:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1738:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1739:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1740:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1741:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1742:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1743:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1744:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1745:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1746:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1747:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1748:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1749:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1750:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1751:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1752:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(5));

    i++;
    if (i==N) return;

//RomLine1753:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1754:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1755:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1756:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1757:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1758:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1759:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1760:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1761:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1762:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1763:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1764:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1765:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1766:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1767:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1768:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1769:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1770:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1771:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1772:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1773:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1774:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1775:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1776:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1777:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1778:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1779:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1780:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1781:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1782:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1783:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1784:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1785:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1786:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1787:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1788:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1789:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1790:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1791:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1792:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1793:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1794:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1795:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1796:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1797:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1798:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1799:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1800:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1801:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1802:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1803:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1804:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1805:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1806:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1807:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1808:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1809:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1810:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1811:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1812:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1813:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1814:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1815:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1816:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1817:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1818:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1819:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1820:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1821:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 31);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1822:

    // op0 = op0 + CONST
    op0 = 1823;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1823:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1824:

    // op0 = op0 + CONST
    op0 = 255;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1825:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1826:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1827:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1828:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1829:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1830:

    // op0 = op0 + CONST
    op0 = 1831;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1831:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1832:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1833:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1834:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1835:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1836:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1837:

    // op0 = op0 + CONST
    op0 = 1838;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1838:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1839:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1840:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1841:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1842:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1843:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1844:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1845:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1846:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1847:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1848:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1849:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1850:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1851:

    // op0 = op0 + CONST
    op0 = 1852;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1852:

    // op0 = op0 + CONST
    op0 = 1853;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1853:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1854:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1855:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1856:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1857:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1858:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1859:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1860:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1861:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1862:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1863:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1864:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1865:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1866:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1867:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1868:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1869:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1870:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1871:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1872:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1873:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine1874:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(30));

    i++;
    if (i==N) return;

//RomLine1875:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 31);

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1876:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1877:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1878:

    // op0 = op0 + CONST
    op0 = 1879;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1879:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1880:

    // op0 = op0 + CONST
    op0 = 6;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1881:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1882:

    // op0 = op0 + CONST
    op0 = 1883;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1883:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1884:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine1885:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1886:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1887:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1888:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1889:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1890:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine1891:

    // op0 = op0 + CONST
    op0 = 1892;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1892:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1893:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1894:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1895:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1896:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1897:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1898:

    // op0 = op0 + CONST
    op0 = 1899;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1899:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1900:

    // op0 = op0 + CONST
    op0 = 1901;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1901:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1902:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1903:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1904:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1905:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1906:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1907:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1908:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1909:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1910:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inE*E, where inE=-1
    op0 = fr.add(op0, fr.neg(E0));
    op1 = fr.add(op1, fr.neg(E1));
    op2 = fr.add(op2, fr.neg(E2));
    op3 = fr.add(op3, fr.neg(E3));
    op4 = fr.add(op4, fr.neg(E4));
    op5 = fr.add(op5, fr.neg(E5));
    op6 = fr.add(op6, fr.neg(E6));
    op7 = fr.add(op7, fr.neg(E7));

    i++;
    if (i==N) return;

//RomLine1911:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1912:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1913:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine1914:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1915:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1916:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1917:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1918:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine1919:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1920:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1921:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1922:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1923:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1924:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine1925:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1926:

    // op = op + inD*D, where inD=-2500
    op0 = fr.mul(-2500, D0);
    op1 = fr.mul(-2500, D1);
    op2 = fr.mul(-2500, D2);
    op3 = fr.mul(-2500, D3);
    op4 = fr.mul(-2500, D4);
    op5 = fr.mul(-2500, D5);
    op6 = fr.mul(-2500, D6);
    op7 = fr.mul(-2500, D7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(100));

    i++;
    if (i==N) return;

//RomLine1927:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1928:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1929:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1930:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine1931:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1932:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1933:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1934:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine1935:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1936:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1937:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine1938:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine1939:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1940:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1941:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1942:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine1943:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1944:

    // op0 = op0 + CONST
    op0 = 1945;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1945:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1946:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1947:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine1948:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1949:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1024);

    i++;
    if (i==N) return;

//RomLine1950:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1951:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1952:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1953:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1954:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1955:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1956:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1957:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1024);

    i++;
    if (i==N) return;

//RomLine1958:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1959:

    // op0 = op0 + CONST
    op0 = 1960;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1960:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1961:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1962:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1963:

    // op0 = op0 + CONST
    op0 = 1964;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1964:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine1965:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    i++;
    if (i==N) return;

//RomLine1966:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1967:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1968:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1969:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1970:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine1971:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1972:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1973:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine1974:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine1975:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1976:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine1977:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine1978:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1979:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1980:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine1981:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine1982:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine1983:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine1984:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1985:

    // op0 = op0 + CONST
    op0 = 1986;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1986:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1987:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1988:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1024);

    i++;
    if (i==N) return;

//RomLine1989:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1990:

    // op0 = op0 + CONST
    op0 = 1991;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1991:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine1992:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1993:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1025);

    i++;
    if (i==N) return;

//RomLine1994:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine1995:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine1996:

    // op0 = op0 + CONST
    op0 = 1997;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1997:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inC*C, where inC=1
    op0 = fr.add(op0, C0);
    op1 = fr.add(op1, C1);
    op2 = fr.add(op2, C2);
    op3 = fr.add(op3, C3);
    op4 = fr.add(op4, C4);
    op5 = fr.add(op5, C5);
    op6 = fr.add(op6, C6);
    op7 = fr.add(op7, C7);

    i++;
    if (i==N) return;

//RomLine1998:

    // op0 = op0 + CONST
    op0 = 1999;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine1999:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2000:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2001:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2002:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine2003:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2004:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2005:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2006:

    // op0 = op0 + CONST
    op0 = 1024;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2007:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2008:

    // op0 = op0 + CONST
    op0 = 2009;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2009:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2010:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2011:

    // op0 = op0 + CONST
    op0 = 1025;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2012:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inD*D, where inD=1
    op0 = fr.add(op0, D0);
    op1 = fr.add(op1, D1);
    op2 = fr.add(op2, D2);
    op3 = fr.add(op3, D3);
    op4 = fr.add(op4, D4);
    op5 = fr.add(op5, D5);
    op6 = fr.add(op6, D6);
    op7 = fr.add(op7, D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2013:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine2014:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2015:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2016:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2017:

    // op0 = op0 + CONST
    op0 = 2018;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2018:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2019:

    // op0 = op0 + CONST
    op0 = 2020;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2020:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2021:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2022:

    // op0 = op0 + CONST
    op0 = 2023;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2023:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2024:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2025:

    // op0 = op0 + CONST
    op0 = 2026;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2026:

    // op0 = op0 + CONST
    op0 = 2027;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2027:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2028:

    // op0 = op0 + CONST
    op0 = 2029;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2029:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2030:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2031:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2032:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inE*E, where inE=-1
    op0 = fr.add(op0, fr.neg(E0));
    op1 = fr.add(op1, fr.neg(E1));
    op2 = fr.add(op2, fr.neg(E2));
    op3 = fr.add(op3, fr.neg(E3));
    op4 = fr.add(op4, fr.neg(E4));
    op5 = fr.add(op5, fr.neg(E5));
    op6 = fr.add(op6, fr.neg(E6));
    op7 = fr.add(op7, fr.neg(E7));

    i++;
    if (i==N) return;

//RomLine2033:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2034:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2035:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine2036:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2037:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2038:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2039:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2040:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2041:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2042:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2043:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2044:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2045:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2046:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2047:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2048:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2049:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine2050:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2051:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2052:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2053:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2054:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2055:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2056:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2057:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2058:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine2059:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2060:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2061:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2062:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine2063:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2064:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine2065:

    // op0 = op0 + CONST
    op0 = 2066;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2066:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2067:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2068:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2069:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2070:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2071:

    // op0 = op0 + CONST
    op0 = 2072;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2072:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2073:

    // op0 = op0 + CONST
    op0 = 2074;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2074:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2075:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2076:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inE*E, where inE=-1
    op0 = fr.add(op0, fr.neg(E0));
    op1 = fr.add(op1, fr.neg(E1));
    op2 = fr.add(op2, fr.neg(E2));
    op3 = fr.add(op3, fr.neg(E3));
    op4 = fr.add(op4, fr.neg(E4));
    op5 = fr.add(op5, fr.neg(E5));
    op6 = fr.add(op6, fr.neg(E6));
    op7 = fr.add(op7, fr.neg(E7));

    i++;
    if (i==N) return;

//RomLine2077:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2078:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2079:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2080:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2081:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2082:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2083:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2084:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2085:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2086:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2087:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2088:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2089:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2090:

    // op = op + inD*D, where inD=-2500
    op0 = fr.mul(-2500, D0);
    op1 = fr.mul(-2500, D1);
    op2 = fr.mul(-2500, D2);
    op3 = fr.mul(-2500, D3);
    op4 = fr.mul(-2500, D4);
    op5 = fr.mul(-2500, D5);
    op6 = fr.mul(-2500, D6);
    op7 = fr.mul(-2500, D7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(100));

    i++;
    if (i==N) return;

//RomLine2091:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2092:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2093:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2094:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2095:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2096:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2097:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2098:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2099:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2100:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2101:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine2102:

    // op = op + inD*D, where inD=-2500
    op0 = fr.mul(-2500, D0);
    op1 = fr.mul(-2500, D1);
    op2 = fr.mul(-2500, D2);
    op3 = fr.mul(-2500, D3);
    op4 = fr.mul(-2500, D4);
    op5 = fr.mul(-2500, D5);
    op6 = fr.mul(-2500, D6);
    op7 = fr.mul(-2500, D7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(100));

    i++;
    if (i==N) return;

//RomLine2103:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2104:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2105:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine2106:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2107:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine2108:

    // op0 = op0 + CONST
    op0 = 2109;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2109:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2110:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2111:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2112:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2113:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2114:

    // op0 = op0 + CONST
    op0 = 2115;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2115:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2116:

    // op0 = op0 + CONST
    op0 = 2117;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2117:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2118:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2119:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inE*E, where inE=-1
    op0 = fr.add(op0, fr.neg(E0));
    op1 = fr.add(op1, fr.neg(E1));
    op2 = fr.add(op2, fr.neg(E2));
    op3 = fr.add(op3, fr.neg(E3));
    op4 = fr.add(op4, fr.neg(E4));
    op5 = fr.add(op5, fr.neg(E5));
    op6 = fr.add(op6, fr.neg(E6));
    op7 = fr.add(op7, fr.neg(E7));

    i++;
    if (i==N) return;

//RomLine2120:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2121:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2122:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2123:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2124:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2125:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2126:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2127:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2128:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2129:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inC*C, where inC=1
    op0 = fr.add(op0, C0);
    op1 = fr.add(op1, C1);
    op2 = fr.add(op2, C2);
    op3 = fr.add(op3, C3);
    op4 = fr.add(op4, C4);
    op5 = fr.add(op5, C5);
    op6 = fr.add(op6, C6);
    op7 = fr.add(op7, C7);

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2130:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2131:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2132:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 31);

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2133:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2134:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2135:

    // op0 = op0 + CONST
    op0 = 2136;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2136:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2137:

    // op0 = op0 + CONST
    op0 = 3;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2138:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2139:

    // op0 = op0 + CONST
    op0 = 2140;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2140:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2141:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine2142:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inC*C, where inC=1
    op0 = fr.add(op0, C0);
    op1 = fr.add(op1, C1);
    op2 = fr.add(op2, C2);
    op3 = fr.add(op3, C3);
    op4 = fr.add(op4, C4);
    op5 = fr.add(op5, C5);
    op6 = fr.add(op6, C6);
    op7 = fr.add(op7, C7);

    i++;
    if (i==N) return;

//RomLine2143:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2144:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine2145:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2146:

    // op0 = op0 + CONST
    op0 = 2147;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2147:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2148:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2149:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2150:

    // op0 = op0 + CONST
    op0 = 2151;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2151:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2152:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2153:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2154:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2155:

    // op0 = op0 + CONST
    op0 = 2156;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2156:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2157:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2158:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2159:

    // op0 = op0 + CONST
    op0 = 2160;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2160:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2161:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2162:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inE*E, where inE=-1
    op0 = fr.add(op0, fr.neg(E0));
    op1 = fr.add(op1, fr.neg(E1));
    op2 = fr.add(op2, fr.neg(E2));
    op3 = fr.add(op3, fr.neg(E3));
    op4 = fr.add(op4, fr.neg(E4));
    op5 = fr.add(op5, fr.neg(E5));
    op6 = fr.add(op6, fr.neg(E6));
    op7 = fr.add(op7, fr.neg(E7));

    i++;
    if (i==N) return;

//RomLine2163:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2164:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2165:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2166:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2167:

    // op = op + inD*D, where inD=-2500
    op0 = fr.mul(-2500, D0);
    op1 = fr.mul(-2500, D1);
    op2 = fr.mul(-2500, D2);
    op3 = fr.mul(-2500, D3);
    op4 = fr.mul(-2500, D4);
    op5 = fr.mul(-2500, D5);
    op6 = fr.mul(-2500, D6);
    op7 = fr.mul(-2500, D7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(100));

    i++;
    if (i==N) return;

//RomLine2168:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2169:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2170:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2171:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2172:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2173:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2174:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2175:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2176:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2177:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine2178:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2179:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2180:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inD*D, where inD=-1
    op0 = fr.add(op0, fr.neg(D0));
    op1 = fr.add(op1, fr.neg(D1));
    op2 = fr.add(op2, fr.neg(D2));
    op3 = fr.add(op3, fr.neg(D3));
    op4 = fr.add(op4, fr.neg(D4));
    op5 = fr.add(op5, fr.neg(D5));
    op6 = fr.add(op6, fr.neg(D6));
    op7 = fr.add(op7, fr.neg(D7));

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2181:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2182:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2183:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2184:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2185:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2186:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2187:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2188:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2189:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2190:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(20));

    i++;
    if (i==N) return;

//RomLine2191:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2192:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2193:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2194:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2195:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2196:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2197:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2198:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2199:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2200:

    // op0 = op0 + inHASHPOS*HASHPOS, where inHASHPOS=1
    op0 = HASHPOS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2201:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2202:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2203:

    // op0 = op0 + CONST
    op0 = 3;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2204:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2205:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine2206:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2207:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2208:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2209:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2210:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2211:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2212:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2213:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2214:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2215:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2216:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2217:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2218:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2219:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2220:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2221:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2222:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2223:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2224:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2225:

    // op0 = op0 + CONST
    op0 = 30000000;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2226:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2227:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2228:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2229:

    // op0 = op0 + CONST
    op0 = 1000;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2230:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2231:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2232:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2233:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2234:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2235:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2236:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine2237:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(5));

    i++;
    if (i==N) return;

//RomLine2238:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2239:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2240:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2241:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2242:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2243:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2244:

    // op0 = op0 + CONST
    op0 = 2245;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2245:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2246:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2247:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2248:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inE*E, where inE=-1
    op0 = fr.add(op0, fr.neg(E0));
    op1 = fr.add(op1, fr.neg(E1));
    op2 = fr.add(op2, fr.neg(E2));
    op3 = fr.add(op3, fr.neg(E3));
    op4 = fr.add(op4, fr.neg(E4));
    op5 = fr.add(op5, fr.neg(E5));
    op6 = fr.add(op6, fr.neg(E6));
    op7 = fr.add(op7, fr.neg(E7));

    i++;
    if (i==N) return;

//RomLine2249:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2250:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2251:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2252:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2253:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2254:

    // op0 = op0 + CONST
    op0 = 2255;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2255:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2256:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2257:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inE*E, where inE=-1
    op0 = fr.add(op0, fr.neg(E0));
    op1 = fr.add(op1, fr.neg(E1));
    op2 = fr.add(op2, fr.neg(E2));
    op3 = fr.add(op3, fr.neg(E3));
    op4 = fr.add(op4, fr.neg(E4));
    op5 = fr.add(op5, fr.neg(E5));
    op6 = fr.add(op6, fr.neg(E6));
    op7 = fr.add(op7, fr.neg(E7));

    i++;
    if (i==N) return;

//RomLine2258:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2259:

    // op0 = op0 + CONST
    op0 = 2260;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2260:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2261:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2262:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2263:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2264:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2265:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2266:

    // op0 = op0 + CONST
    op0 = 2267;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2267:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2268:

    // op0 = op0 + CONST
    op0 = 2269;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2269:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2270:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2271:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inE*E, where inE=-1
    op0 = fr.add(op0, fr.neg(E0));
    op1 = fr.add(op1, fr.neg(E1));
    op2 = fr.add(op2, fr.neg(E2));
    op3 = fr.add(op3, fr.neg(E3));
    op4 = fr.add(op4, fr.neg(E4));
    op5 = fr.add(op5, fr.neg(E5));
    op6 = fr.add(op6, fr.neg(E6));
    op7 = fr.add(op7, fr.neg(E7));

    i++;
    if (i==N) return;

//RomLine2272:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2273:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2274:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2275:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2276:

    // op0 = op0 + CONST
    op0 = 3;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2277:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2278:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine2279:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2280:

    // op = op + inA*A, where inA=-2000
    op0 = fr.mul(-2000, A0);
    op1 = fr.mul(-2000, A1);
    op2 = fr.mul(-2000, A2);
    op3 = fr.mul(-2000, A3);
    op4 = fr.mul(-2000, A4);
    op5 = fr.mul(-2000, A5);
    op6 = fr.mul(-2000, A6);
    op7 = fr.mul(-2000, A7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(100));

    i++;
    if (i==N) return;

//RomLine2281:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2282:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2283:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine2284:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2285:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2286:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine2287:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2288:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2289:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine2290:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2291:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2292:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2293:

    // op0 = op0 + CONST
    op0 = 3;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2294:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2295:

    // op = op + inSR*SR, where inSR=1
    op0 = SR0;
    op1 = SR1;
    op2 = SR2;
    op3 = SR3;
    op4 = SR4;
    op5 = SR5;
    op6 = SR6;
    op7 = SR7;

    i++;
    if (i==N) return;

//RomLine2296:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine2297:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2298:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine2299:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2300:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2301:

    // op = op + inA*A, where inA=-2100
    op0 = fr.mul(-2100, A0);
    op1 = fr.mul(-2100, A1);
    op2 = fr.mul(-2100, A2);
    op3 = fr.mul(-2100, A3);
    op4 = fr.mul(-2100, A4);
    op5 = fr.mul(-2100, A5);
    op6 = fr.mul(-2100, A6);
    op7 = fr.mul(-2100, A7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine2302:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2303:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2304:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2305:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2306:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(100));

    i++;
    if (i==N) return;

//RomLine2307:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2308:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2309:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2310:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2311:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2312:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2313:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2314:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2315:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2316:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(20000));

    i++;
    if (i==N) return;

//RomLine2317:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2318:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(100));

    i++;
    if (i==N) return;

//RomLine2319:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2320:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2321:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2322:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2323:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2324:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2325:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2326:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2327:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2328:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2329:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2330:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2331:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 19900);

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2332:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2333:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2334:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2335:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2336:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2337:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2338:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(15000));

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2339:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2340:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2341:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2342:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2343:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2344:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2345:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 15000);

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2346:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2347:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2348:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 2800);

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2349:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2350:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2900));

    i++;
    if (i==N) return;

//RomLine2351:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2352:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2353:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2354:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2355:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2356:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 15000);

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2357:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2358:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2359:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine2360:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2361:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2362:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2363:

    // op0 = op0 + CONST
    op0 = 3;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2364:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2365:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine2366:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2367:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2368:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2369:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine2370:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(8));

    i++;
    if (i==N) return;

//RomLine2371:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2372:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2373:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2374:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(10));

    i++;
    if (i==N) return;

//RomLine2375:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2376:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2377:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2378:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine2379:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2380:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine2381:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2382:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2383:

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = PC;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2384:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2385:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2386:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2387:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2388:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2389:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2390:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2391:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2392:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2393:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2394:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2395:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2396:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = fr.add(op0, PC);

    i++;
    if (i==N) return;

//RomLine2397:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2398:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2399:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2400:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2401:

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = PC;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2402:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2403:

    // op0 = op0 + CONST
    op0 = 2404;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2404:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2405:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2406:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = fr.add(op0, SP);

    i++;
    if (i==N) return;

//RomLine2407:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2408:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2409:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2410:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2411:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2412:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = fr.add(op0, PC);

    i++;
    if (i==N) return;

//RomLine2413:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2414:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2415:

    // op = op + inC*C, where inC=-1
    op0 = fr.neg(C0);
    op1 = fr.neg(C1);
    op2 = fr.neg(C2);
    op3 = fr.neg(C3);
    op4 = fr.neg(C4);
    op5 = fr.neg(C5);
    op6 = fr.neg(C6);
    op7 = fr.neg(C7);

    // op0 = op0 + CONST
    op0 = fr.add(op0, 32);

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2416:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2417:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2418:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2419:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = fr.add(op0, PC);

    i++;
    if (i==N) return;

//RomLine2420:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // op = op + inE*E, where inE=-1
    op0 = fr.add(op0, fr.neg(E0));
    op1 = fr.add(op1, fr.neg(E1));
    op2 = fr.add(op2, fr.neg(E2));
    op3 = fr.add(op3, fr.neg(E3));
    op4 = fr.add(op4, fr.neg(E4));
    op5 = fr.add(op5, fr.neg(E5));
    op6 = fr.add(op6, fr.neg(E6));
    op7 = fr.add(op7, fr.neg(E7));

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2421:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2422:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2423:

    // op0 = op0 + CONST
    op0 = 2424;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2424:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2425:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op = op + inB*B, where inB=1
    op0 = fr.add(op0, B0);
    op1 = fr.add(op1, B1);
    op2 = fr.add(op2, B2);
    op3 = fr.add(op3, B3);
    op4 = fr.add(op4, B4);
    op5 = fr.add(op5, B5);
    op6 = fr.add(op6, B6);
    op7 = fr.add(op7, B7);

    i++;
    if (i==N) return;

//RomLine2426:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2427:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2428:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2429:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2430:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2431:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2432:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2433:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2434:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2435:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2436:

    // op0 = op0 + CONST
    op0 = 3;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2437:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2438:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2439:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2440:

    // op0 = op0 + CONST
    op0 = 4;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2441:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2442:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2443:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2444:

    // op0 = op0 + CONST
    op0 = 5;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2445:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2446:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2447:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2448:

    // op0 = op0 + CONST
    op0 = 6;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2449:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2450:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2451:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2452:

    // op0 = op0 + CONST
    op0 = 7;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2453:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2454:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2455:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2456:

    // op0 = op0 + CONST
    op0 = 8;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2457:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2458:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2459:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2460:

    // op0 = op0 + CONST
    op0 = 9;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2461:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2462:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2463:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2464:

    // op0 = op0 + CONST
    op0 = 10;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2465:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2466:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2467:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2468:

    // op0 = op0 + CONST
    op0 = 11;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2469:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2470:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2471:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2472:

    // op0 = op0 + CONST
    op0 = 12;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2473:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2474:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2475:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2476:

    // op0 = op0 + CONST
    op0 = 13;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2477:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2478:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2479:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2480:

    // op0 = op0 + CONST
    op0 = 14;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2481:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2482:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2483:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2484:

    // op0 = op0 + CONST
    op0 = 15;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2485:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2486:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2487:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2488:

    // op0 = op0 + CONST
    op0 = 16;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2489:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2490:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2491:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2492:

    // op0 = op0 + CONST
    op0 = 17;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2493:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2494:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2495:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2496:

    // op0 = op0 + CONST
    op0 = 18;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2497:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2498:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2499:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2500:

    // op0 = op0 + CONST
    op0 = 19;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2501:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2502:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2503:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2504:

    // op0 = op0 + CONST
    op0 = 20;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2505:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2506:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2507:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2508:

    // op0 = op0 + CONST
    op0 = 21;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2509:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2510:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2511:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2512:

    // op0 = op0 + CONST
    op0 = 22;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2513:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2514:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2515:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2516:

    // op0 = op0 + CONST
    op0 = 23;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2517:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2518:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2519:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2520:

    // op0 = op0 + CONST
    op0 = 24;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2521:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2522:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2523:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2524:

    // op0 = op0 + CONST
    op0 = 25;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2525:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2526:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2527:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2528:

    // op0 = op0 + CONST
    op0 = 26;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2529:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2530:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2531:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2532:

    // op0 = op0 + CONST
    op0 = 27;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2533:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2534:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2535:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2536:

    // op0 = op0 + CONST
    op0 = 28;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2537:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2538:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2539:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2540:

    // op0 = op0 + CONST
    op0 = 29;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2541:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2542:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2543:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2544:

    // op0 = op0 + CONST
    op0 = 30;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2545:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2546:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2547:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2548:

    // op0 = op0 + CONST
    op0 = 31;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2549:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2550:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2551:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2552:

    // op0 = op0 + CONST
    op0 = 32;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2553:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2554:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine2555:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2556:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2557:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2558:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2559:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2560:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2561:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2562:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2563:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 2);

    i++;
    if (i==N) return;

//RomLine2564:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2565:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2566:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2567:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2568:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2569:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 3);

    i++;
    if (i==N) return;

//RomLine2570:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2571:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2572:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2573:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(4));

    i++;
    if (i==N) return;

//RomLine2574:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2575:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 4);

    i++;
    if (i==N) return;

//RomLine2576:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2577:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2578:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2579:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(5));

    i++;
    if (i==N) return;

//RomLine2580:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2581:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 5);

    i++;
    if (i==N) return;

//RomLine2582:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2583:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2584:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2585:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(6));

    i++;
    if (i==N) return;

//RomLine2586:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2587:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 6);

    i++;
    if (i==N) return;

//RomLine2588:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2589:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2590:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2591:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(7));

    i++;
    if (i==N) return;

//RomLine2592:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2593:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 7);

    i++;
    if (i==N) return;

//RomLine2594:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2595:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2596:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2597:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(8));

    i++;
    if (i==N) return;

//RomLine2598:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2599:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 8);

    i++;
    if (i==N) return;

//RomLine2600:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2601:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2602:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2603:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(9));

    i++;
    if (i==N) return;

//RomLine2604:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2605:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 9);

    i++;
    if (i==N) return;

//RomLine2606:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2607:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2608:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2609:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(10));

    i++;
    if (i==N) return;

//RomLine2610:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2611:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 10);

    i++;
    if (i==N) return;

//RomLine2612:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2613:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2614:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2615:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(11));

    i++;
    if (i==N) return;

//RomLine2616:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2617:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 11);

    i++;
    if (i==N) return;

//RomLine2618:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2619:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2620:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2621:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(12));

    i++;
    if (i==N) return;

//RomLine2622:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2623:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 12);

    i++;
    if (i==N) return;

//RomLine2624:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2625:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2626:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2627:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(13));

    i++;
    if (i==N) return;

//RomLine2628:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2629:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 13);

    i++;
    if (i==N) return;

//RomLine2630:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2631:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2632:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2633:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(14));

    i++;
    if (i==N) return;

//RomLine2634:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2635:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 14);

    i++;
    if (i==N) return;

//RomLine2636:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2637:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2638:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2639:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(15));

    i++;
    if (i==N) return;

//RomLine2640:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2641:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 15);

    i++;
    if (i==N) return;

//RomLine2642:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2643:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2644:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2645:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(16));

    i++;
    if (i==N) return;

//RomLine2646:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2647:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 16);

    i++;
    if (i==N) return;

//RomLine2648:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2649:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2650:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2651:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2652:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2653:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2654:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2655:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2656:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2657:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2658:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2659:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2660:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(2));

    i++;
    if (i==N) return;

//RomLine2661:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2662:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2663:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 2);

    i++;
    if (i==N) return;

//RomLine2664:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2665:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2666:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2667:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2668:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2669:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2670:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2671:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2672:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 3);

    i++;
    if (i==N) return;

//RomLine2673:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2674:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2675:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2676:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2677:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2678:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(4));

    i++;
    if (i==N) return;

//RomLine2679:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2680:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2681:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 4);

    i++;
    if (i==N) return;

//RomLine2682:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2683:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2684:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2685:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2686:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2687:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(5));

    i++;
    if (i==N) return;

//RomLine2688:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2689:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2690:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 5);

    i++;
    if (i==N) return;

//RomLine2691:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2692:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2693:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2694:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2695:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2696:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(6));

    i++;
    if (i==N) return;

//RomLine2697:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2698:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2699:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 6);

    i++;
    if (i==N) return;

//RomLine2700:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2701:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2702:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2703:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2704:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2705:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(7));

    i++;
    if (i==N) return;

//RomLine2706:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2707:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2708:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 7);

    i++;
    if (i==N) return;

//RomLine2709:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2710:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2711:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2712:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2713:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2714:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(8));

    i++;
    if (i==N) return;

//RomLine2715:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2716:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2717:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 8);

    i++;
    if (i==N) return;

//RomLine2718:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2719:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2720:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2721:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2722:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2723:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(9));

    i++;
    if (i==N) return;

//RomLine2724:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2725:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2726:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 9);

    i++;
    if (i==N) return;

//RomLine2727:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2728:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2729:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2730:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2731:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2732:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(10));

    i++;
    if (i==N) return;

//RomLine2733:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2734:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2735:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 10);

    i++;
    if (i==N) return;

//RomLine2736:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2737:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2738:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2739:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2740:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2741:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(11));

    i++;
    if (i==N) return;

//RomLine2742:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2743:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2744:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 11);

    i++;
    if (i==N) return;

//RomLine2745:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2746:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2747:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2748:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2749:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2750:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(12));

    i++;
    if (i==N) return;

//RomLine2751:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2752:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2753:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 12);

    i++;
    if (i==N) return;

//RomLine2754:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2755:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2756:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2757:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2758:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2759:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(13));

    i++;
    if (i==N) return;

//RomLine2760:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2761:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2762:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 13);

    i++;
    if (i==N) return;

//RomLine2763:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2764:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2765:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2766:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2767:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2768:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(14));

    i++;
    if (i==N) return;

//RomLine2769:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2770:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2771:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 14);

    i++;
    if (i==N) return;

//RomLine2772:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2773:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2774:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2775:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2776:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2777:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(15));

    i++;
    if (i==N) return;

//RomLine2778:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2779:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2780:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 15);

    i++;
    if (i==N) return;

//RomLine2781:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2782:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2783:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2784:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2785:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2786:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(16));

    i++;
    if (i==N) return;

//RomLine2787:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2788:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2789:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, 16);

    i++;
    if (i==N) return;

//RomLine2790:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2791:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(3));

    i++;
    if (i==N) return;

//RomLine2792:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2793:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2794:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine2795:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2796:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2797:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2798:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine2799:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2800:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(375));

    i++;
    if (i==N) return;

//RomLine2801:

    // op = op + inC*C, where inC=-8
    op0 = fr.mul(-8, C0);
    op1 = fr.mul(-8, C1);
    op2 = fr.mul(-8, C2);
    op3 = fr.mul(-8, C3);
    op4 = fr.mul(-8, C4);
    op5 = fr.mul(-8, C5);
    op6 = fr.mul(-8, C6);
    op7 = fr.mul(-8, C7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine2802:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2803:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine2804:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2805:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2806:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine2807:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2808:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2809:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2810:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine2811:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2812:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(375));

    i++;
    if (i==N) return;

//RomLine2813:

    // op = op + inC*C, where inC=-8
    op0 = fr.mul(-8, C0);
    op1 = fr.mul(-8, C1);
    op2 = fr.mul(-8, C2);
    op3 = fr.mul(-8, C3);
    op4 = fr.mul(-8, C4);
    op5 = fr.mul(-8, C5);
    op6 = fr.mul(-8, C6);
    op7 = fr.mul(-8, C7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine2814:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2815:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine2816:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2817:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2818:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine2819:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2820:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2821:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2822:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine2823:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2824:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(375));

    i++;
    if (i==N) return;

//RomLine2825:

    // op = op + inC*C, where inC=-8
    op0 = fr.mul(-8, C0);
    op1 = fr.mul(-8, C1);
    op2 = fr.mul(-8, C2);
    op3 = fr.mul(-8, C3);
    op4 = fr.mul(-8, C4);
    op5 = fr.mul(-8, C5);
    op6 = fr.mul(-8, C6);
    op7 = fr.mul(-8, C7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine2826:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2827:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine2828:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2829:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2830:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine2831:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2832:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2833:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2834:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine2835:

    // op0 = op0 + CONST
    op0 = 3;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2836:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(375));

    i++;
    if (i==N) return;

//RomLine2837:

    // op = op + inC*C, where inC=-8
    op0 = fr.mul(-8, C0);
    op1 = fr.mul(-8, C1);
    op2 = fr.mul(-8, C2);
    op3 = fr.mul(-8, C3);
    op4 = fr.mul(-8, C4);
    op5 = fr.mul(-8, C5);
    op6 = fr.mul(-8, C6);
    op7 = fr.mul(-8, C7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine2838:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2839:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine2840:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2841:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2842:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine2843:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2844:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2845:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2846:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine2847:

    // op0 = op0 + CONST
    op0 = 4;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2848:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(375));

    i++;
    if (i==N) return;

//RomLine2849:

    // op = op + inC*C, where inC=-8
    op0 = fr.mul(-8, C0);
    op1 = fr.mul(-8, C1);
    op2 = fr.mul(-8, C2);
    op3 = fr.mul(-8, C3);
    op4 = fr.mul(-8, C4);
    op5 = fr.mul(-8, C5);
    op6 = fr.mul(-8, C6);
    op7 = fr.mul(-8, C7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine2850:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2851:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine2852:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2853:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2854:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine2855:

    // op0 = op0 + CONST
    op0 = 2856;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2856:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2857:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2858:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2859:

    // op0 = op0 + CONST
    op0 = 2860;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2860:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2861:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2862:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2863:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2864:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(375));

    i++;
    if (i==N) return;

//RomLine2865:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2866:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2867:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2868:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2869:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2870:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2871:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inE*E, where inE=-1
    op0 = fr.add(op0, fr.neg(E0));
    op1 = fr.add(op1, fr.neg(E1));
    op2 = fr.add(op2, fr.neg(E2));
    op3 = fr.add(op3, fr.neg(E3));
    op4 = fr.add(op4, fr.neg(E4));
    op5 = fr.add(op5, fr.neg(E5));
    op6 = fr.add(op6, fr.neg(E6));
    op7 = fr.add(op7, fr.neg(E7));

    i++;
    if (i==N) return;

//RomLine2872:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2873:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2874:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine2875:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2876:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2877:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2878:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2879:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine2880:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2881:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine2882:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2883:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2884:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2885:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2886:

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = PC;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2887:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2888:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine2889:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2890:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2891:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2892:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2893:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2894:

    // op = op + inSR*SR, where inSR=1
    op0 = SR0;
    op1 = SR1;
    op2 = SR2;
    op3 = SR3;
    op4 = SR4;
    op5 = SR5;
    op6 = SR6;
    op7 = SR7;

    i++;
    if (i==N) return;

//RomLine2895:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine2896:

    // op0 = op0 + CONST
    op0 = 2897;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2897:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2898:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2899:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2900:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2901:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2902:

    // op0 = op0 + CONST
    op0 = 2903;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2903:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2904:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2905:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2906:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2907:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2908:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2909:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2910:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2911:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2912:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2913:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2914:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2915:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2916:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2917:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2918:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2919:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine2920:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2921:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2922:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2923:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2924:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2925:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2926:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine2927:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2928:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2929:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2930:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2931:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2932:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2933:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2934:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2935:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine2936:

    // op0 = op0 + CONST
    op0 = 2937;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2937:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2938:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2939:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2940:

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = PC;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2941:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2942:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine2943:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2944:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2945:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2946:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2947:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine2948:

    // op = op + inSR*SR, where inSR=1
    op0 = SR0;
    op1 = SR1;
    op2 = SR2;
    op3 = SR3;
    op4 = SR4;
    op5 = SR5;
    op6 = SR6;
    op7 = SR7;

    i++;
    if (i==N) return;

//RomLine2949:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2950:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2951:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2952:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2953:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2954:

    // op = op + inD*D, where inD=-2500
    op0 = fr.mul(-2500, D0);
    op1 = fr.mul(-2500, D1);
    op2 = fr.mul(-2500, D2);
    op3 = fr.mul(-2500, D3);
    op4 = fr.mul(-2500, D4);
    op5 = fr.mul(-2500, D5);
    op6 = fr.mul(-2500, D6);
    op7 = fr.mul(-2500, D7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(100));

    i++;
    if (i==N) return;

//RomLine2955:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2956:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2957:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2958:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2959:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(9000));

    i++;
    if (i==N) return;

//RomLine2960:

    // op0 = op0 + CONST
    op0 = 2961;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2961:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2962:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine2963:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2964:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2965:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2966:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2967:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2968:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2969:

    // op0 = op0 + CONST
    op0 = 2970;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2970:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine2971:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2972:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2973:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2974:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2975:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2976:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2977:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2978:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2979:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2980:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine2981:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2982:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2983:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2984:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2985:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine2986:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2987:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine2988:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2989:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2990:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine2991:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine2992:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine2993:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2994:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine2995:

    // op0 = op0 + CONST
    op0 = 2996;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine2996:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine2997:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine2998:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine2999:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3000:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3001:

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = PC;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3002:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3003:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine3004:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3005:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3006:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3007:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine3008:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine3009:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine3010:

    // op = op + inSR*SR, where inSR=1
    op0 = SR0;
    op1 = SR1;
    op2 = SR2;
    op3 = SR3;
    op4 = SR4;
    op5 = SR5;
    op6 = SR6;
    op7 = SR7;

    i++;
    if (i==N) return;

//RomLine3011:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3012:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3013:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3014:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3015:

    // op = op + inD*D, where inD=-2500
    op0 = fr.mul(-2500, D0);
    op1 = fr.mul(-2500, D1);
    op2 = fr.mul(-2500, D2);
    op3 = fr.mul(-2500, D3);
    op4 = fr.mul(-2500, D4);
    op5 = fr.mul(-2500, D5);
    op6 = fr.mul(-2500, D6);
    op7 = fr.mul(-2500, D7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(100));

    i++;
    if (i==N) return;

//RomLine3016:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3017:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3018:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine3019:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(9000));

    i++;
    if (i==N) return;

//RomLine3020:

    // op0 = op0 + CONST
    op0 = 3021;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3021:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3022:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine3023:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3024:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3025:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3026:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3027:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3028:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3029:

    // op0 = op0 + CONST
    op0 = 3030;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3030:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3031:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine3032:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3033:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3034:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine3035:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3036:

    // op = op + inD*D, where inD=-1
    op0 = fr.neg(D0);
    op1 = fr.neg(D1);
    op2 = fr.neg(D2);
    op3 = fr.neg(D3);
    op4 = fr.neg(D4);
    op5 = fr.neg(D5);
    op6 = fr.neg(D6);
    op7 = fr.neg(D7);

    i++;
    if (i==N) return;

//RomLine3037:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3038:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine3039:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3040:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3041:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3042:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine3043:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3044:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine3045:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3046:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3047:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine3048:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    i++;
    if (i==N) return;

//RomLine3049:

    // op0 = op0 + CONST
    op0 = 3050;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3050:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3051:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3052:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3053:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3054:

    // op0 = op0 + CONST
    op0 = 3055;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3055:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3056:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3057:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(32));

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3058:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3059:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3060:

    // op0 = op0 + CONST
    op0 = 3061;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3061:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3062:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3063:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3064:

    // op0 = op0 + CONST
    op0 = 3065;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3065:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3066:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3067:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3068:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3069:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3070:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3071:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3072:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inE*E, where inE=-1
    op0 = fr.add(op0, fr.neg(E0));
    op1 = fr.add(op1, fr.neg(E1));
    op2 = fr.add(op2, fr.neg(E2));
    op3 = fr.add(op3, fr.neg(E3));
    op4 = fr.add(op4, fr.neg(E4));
    op5 = fr.add(op5, fr.neg(E5));
    op6 = fr.add(op6, fr.neg(E6));
    op7 = fr.add(op7, fr.neg(E7));

    i++;
    if (i==N) return;

//RomLine3073:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3074:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3075:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3076:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op = op + inE*E, where inE=-1
    op0 = fr.add(op0, fr.neg(E0));
    op1 = fr.add(op1, fr.neg(E1));
    op2 = fr.add(op2, fr.neg(E2));
    op3 = fr.add(op3, fr.neg(E3));
    op4 = fr.add(op4, fr.neg(E4));
    op5 = fr.add(op5, fr.neg(E5));
    op6 = fr.add(op6, fr.neg(E6));
    op7 = fr.add(op7, fr.neg(E7));

    i++;
    if (i==N) return;

//RomLine3077:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3078:

    // op = op + inC*C, where inC=-200
    op0 = fr.mul(-200, C0);
    op1 = fr.mul(-200, C1);
    op2 = fr.mul(-200, C2);
    op3 = fr.mul(-200, C3);
    op4 = fr.mul(-200, C4);
    op5 = fr.mul(-200, C5);
    op6 = fr.mul(-200, C6);
    op7 = fr.mul(-200, C7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine3079:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3080:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    i++;
    if (i==N) return;

//RomLine3081:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3082:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3083:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3084:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3085:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3086:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3087:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine3088:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3089:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3090:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3091:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3092:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine3093:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3094:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3095:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3096:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine3097:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3098:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3099:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3100:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3101:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3102:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3103:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3104:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3105:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3106:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3107:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3108:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine3109:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3110:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3111:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3112:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3113:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3114:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine3115:

    // op0 = op0 + CONST
    op0 = 3116;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3116:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3117:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3118:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3119:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine3120:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3121:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3122:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine3123:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3124:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3125:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3126:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3127:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine3128:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3129:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3130:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3131:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3132:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3133:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3134:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3135:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3136:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine3137:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3138:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3139:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine3140:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3141:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine3142:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3143:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3144:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine3145:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3146:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3147:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3148:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine3149:

    // op0 = op0 + CONST
    op0 = 3150;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3150:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3151:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3152:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3153:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3154:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3155:

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = PC;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3156:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3157:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine3158:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3159:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3160:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3161:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine3162:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine3163:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine3164:

    // op = op + inSR*SR, where inSR=1
    op0 = SR0;
    op1 = SR1;
    op2 = SR2;
    op3 = SR3;
    op4 = SR4;
    op5 = SR5;
    op6 = SR6;
    op7 = SR7;

    i++;
    if (i==N) return;

//RomLine3165:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3166:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine3167:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3168:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3169:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3170:

    // op = op + inD*D, where inD=-2500
    op0 = fr.mul(-2500, D0);
    op1 = fr.mul(-2500, D1);
    op2 = fr.mul(-2500, D2);
    op3 = fr.mul(-2500, D3);
    op4 = fr.mul(-2500, D4);
    op5 = fr.mul(-2500, D5);
    op6 = fr.mul(-2500, D6);
    op7 = fr.mul(-2500, D7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(100));

    i++;
    if (i==N) return;

//RomLine3171:

    // op0 = op0 + CONST
    op0 = 3172;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3172:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3173:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine3174:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3175:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3176:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3177:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3178:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3179:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3180:

    // op0 = op0 + CONST
    op0 = 3181;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3181:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3182:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3183:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine3184:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine3185:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3186:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3187:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3188:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine3189:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3190:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine3191:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3192:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3193:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3194:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3195:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3196:

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = PC;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3197:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3198:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine3199:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3200:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3201:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine3202:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3203:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3204:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine3205:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine3206:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine3207:

    // op0 = op0 + CONST
    op0 = 3208;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3208:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3209:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3210:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3211:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3212:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine3213:

    // op0 = op0 + CONST
    op0 = 3214;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3214:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3215:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3216:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3217:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3218:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3219:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3220:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine3221:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3222:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3223:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3224:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3225:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3226:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3227:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3228:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3229:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine3230:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3231:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3232:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine3233:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3234:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine3235:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3236:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3237:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine3238:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3239:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3240:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3241:

    // op = op + inB*B, where inB=-1
    op0 = fr.neg(B0);
    op1 = fr.neg(B1);
    op2 = fr.neg(B2);
    op3 = fr.neg(B3);
    op4 = fr.neg(B4);
    op5 = fr.neg(B5);
    op6 = fr.neg(B6);
    op7 = fr.neg(B7);

    // op = op + inE*E, where inE=1
    op0 = fr.add(op0, E0);
    op1 = fr.add(op1, E1);
    op2 = fr.add(op2, E2);
    op3 = fr.add(op3, E3);
    op4 = fr.add(op4, E4);
    op5 = fr.add(op5, E5);
    op6 = fr.add(op6, E6);
    op7 = fr.add(op7, E7);

    i++;
    if (i==N) return;

//RomLine3242:

    // op0 = op0 + CONST
    op0 = 3243;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3243:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3244:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3245:

    // op = op + inSR*SR, where inSR=1
    op0 = SR0;
    op1 = SR1;
    op2 = SR2;
    op3 = SR3;
    op4 = SR4;
    op5 = SR5;
    op6 = SR6;
    op7 = SR7;

    i++;
    if (i==N) return;

//RomLine3246:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3247:

    // op0 = op0 + inPC*PC, where inPC=1
    op0 = PC;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3248:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3249:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, 1);

    i++;
    if (i==N) return;

//RomLine3250:

    // op0 = op0 + inCTX*CTX, where inCTX=1
    op0 = CTX;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3251:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3252:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3253:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3254:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3255:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine3256:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine3257:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3258:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3259:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3260:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3261:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3262:

    // op = op + inD*D, where inD=-2500
    op0 = fr.mul(-2500, D0);
    op1 = fr.mul(-2500, D1);
    op2 = fr.mul(-2500, D2);
    op3 = fr.mul(-2500, D3);
    op4 = fr.mul(-2500, D4);
    op5 = fr.mul(-2500, D5);
    op6 = fr.mul(-2500, D6);
    op7 = fr.mul(-2500, D7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(100));

    i++;
    if (i==N) return;

//RomLine3263:

    // op0 = op0 + CONST
    op0 = 3264;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3264:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3265:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine3266:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3267:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3268:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3269:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3270:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3271:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3272:

    // op0 = op0 + CONST
    op0 = 3273;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3273:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3274:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine3275:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine3276:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3277:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3278:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3279:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine3280:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    i++;
    if (i==N) return;

//RomLine3281:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3282:

    // op = op + inC*C, where inC=1
    op0 = C0;
    op1 = C1;
    op2 = C2;
    op3 = C3;
    op4 = C4;
    op5 = C5;
    op6 = C6;
    op7 = C7;

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine3283:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3284:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3285:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3286:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3287:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3288:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3289:

    // op = op + inA*A, where inA=-1
    op0 = fr.neg(A0);
    op1 = fr.neg(A1);
    op2 = fr.neg(A2);
    op3 = fr.neg(A3);
    op4 = fr.neg(A4);
    op5 = fr.neg(A5);
    op6 = fr.neg(A6);
    op7 = fr.neg(A7);

    i++;
    if (i==N) return;

//RomLine3290:

    // op0 = op0 + CONST
    op0 = 24000;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3291:

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = GAS;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(5000));

    i++;
    if (i==N) return;

//RomLine3292:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3293:

    // op0 = op0 + CONST
    op0 = 2;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3294:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3295:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3296:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine3297:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3298:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3299:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine3300:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3301:

    // op0 = op0 + inSP*SP, where inSP=1
    op0 = SP;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine3302:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3303:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3304:

    // op0 = op0 + CONST
    op0 = 3305;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3305:

    // op = op + inE*E, where inE=-25000
    op0 = fr.mul(-25000, E0);
    op1 = fr.mul(-25000, E1);
    op2 = fr.mul(-25000, E2);
    op3 = fr.mul(-25000, E3);
    op4 = fr.mul(-25000, E4);
    op5 = fr.mul(-25000, E5);
    op6 = fr.mul(-25000, E6);
    op7 = fr.mul(-25000, E7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine3306:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3307:

    // op = op + inE*E, where inE=-2600
    op0 = fr.mul(-2600, E0);
    op1 = fr.mul(-2600, E1);
    op2 = fr.mul(-2600, E2);
    op3 = fr.mul(-2600, E3);
    op4 = fr.mul(-2600, E4);
    op5 = fr.mul(-2600, E5);
    op6 = fr.mul(-2600, E6);
    op7 = fr.mul(-2600, E7);

    // op0 = op0 + inGAS*GAS, where inGAS=1
    op0 = fr.add(op0, GAS);

    i++;
    if (i==N) return;

//RomLine3308:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3309:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // E' = op
    E0 = op0;
    E1 = op1;
    E2 = op2;
    E3 = op3;
    E4 = op4;
    E5 = op5;
    E6 = op6;
    E7 = op7;

    i++;
    if (i==N) return;

//RomLine3310:

    // op = op + inD*D, where inD=1
    op0 = D0;
    op1 = D1;
    op2 = D2;
    op3 = D3;
    op4 = D4;
    op5 = D5;
    op6 = D6;
    op7 = D7;

    i++;
    if (i==N) return;

//RomLine3311:

    // op = op + inE*E, where inE=1
    op0 = E0;
    op1 = E1;
    op2 = E2;
    op3 = E3;
    op4 = E4;
    op5 = E5;
    op6 = E6;
    op7 = E7;

    i++;
    if (i==N) return;

//RomLine3312:

    // op0 = op0 + CONST
    op0 = 3313;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3313:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // D' = op
    D0 = op0;
    D1 = op1;
    D2 = op2;
    D3 = op3;
    D4 = op4;
    D5 = op5;
    D6 = op6;
    D7 = op7;

    i++;
    if (i==N) return;

//RomLine3314:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3315:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    // C' = op
    C0 = op0;
    C1 = op1;
    C2 = op2;
    C3 = op3;
    C4 = op4;
    C5 = op5;
    C6 = op6;
    C7 = op7;

    i++;
    if (i==N) return;

//RomLine3316:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // SR' = op
    SR0 = op0;
    SR1 = op1;
    SR2 = op2;
    SR3 = op3;
    SR4 = op4;
    SR5 = op5;
    SR6 = op6;
    SR7 = op7;

    i++;
    if (i==N) return;

//RomLine3317:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3318:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3319:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // B' = op
    B0 = op0;
    B1 = op1;
    B2 = op2;
    B3 = op3;
    B4 = op4;
    B5 = op5;
    B6 = op6;
    B7 = op7;

    i++;
    if (i==N) return;

//RomLine3320:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    // op0 = op0 + CONST
    op0 = fr.add(op0, fr.neg(1));

    i++;
    if (i==N) return;

//RomLine3321:

    // op = op + inB*B, where inB=1
    op0 = B0;
    op1 = B1;
    op2 = B2;
    op3 = B3;
    op4 = B4;
    op5 = B5;
    op6 = B6;
    op7 = B7;

    i++;
    if (i==N) return;

//RomLine3322:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3323:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3324:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3325:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3326:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3327:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    // A' = op
    A0 = op0;
    A1 = op1;
    A2 = op2;
    A3 = op3;
    A4 = op4;
    A5 = op5;
    A6 = op6;
    A7 = op7;

    i++;
    if (i==N) return;

//RomLine3328:

    // op = op + inA*A, where inA=1
    op0 = A0;
    op1 = A1;
    op2 = A2;
    op3 = A3;
    op4 = A4;
    op5 = A5;
    op6 = A6;
    op7 = A7;

    i++;
    if (i==N) return;

//RomLine3329:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3330:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

//RomLine3331:

    // op0 = op0 + CONST
    op0 = 1;
    op1 = fr.zero();
    op2 = fr.zero();
    op3 = fr.zero();
    op4 = fr.zero();
    op5 = fr.zero();
    op6 = fr.zero();
    op7 = fr.zero();

    i++;
    if (i==N) return;

//RomLine3332:

    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero

    i++;
    if (i==N) return;

}
