#ifndef ZKEVM_API_H
#define ZKEVM_API_H
    #include <stdint.h>

    int zkevm_main(char *configFile, void* pAddress, void** pSMRequests, void* pSMRequestsOut, void *pStarkInfo = nullptr);
    int zkevm_delete_sm_requests(void **pSMRequests);
    int zkevm_arith(void * inputs, int ninputs, void * pAddress);
    int zkevm_arith_req(void* pSMRequests,  void * pAddress);
    int zkevm_binary_req(void* pSMRequests,  void * pAddress);
    int zkevm_memory(void * inputs_, int ninputs, void * pAddress);
    int zkevm_memory_req(void* pSMRequests,  void * pAddress);
    int zkevm_mem_align(void * inputs_, int ninputs, void* pAddress);
    int zkevm_mem_align_req(void* pSMRequests,  void * pAddress);
    
    int zkevm_padding_kk(void * inputs_, int ninputs, void * pAddress, void* pSMRequests, void* pSMRequestsOut); 
    int zkevm_padding_kk_req(void* pSMRequests,  void * pAddress); 
    int zkevm_padding_kk_bit(void * inputs_, int ninputs, void * pAddress, void* pSMRequests, void* pSMRequestsOut);
    int zkevm_padding_kk_bit_req(void* pSMRequests,  void * pAddress);
    int zkevm_bits2field_kk(void * inputs_, int ninputs, void * pAddress, void* pSMRequests, void* pSMRequestsOut);
    int zkevm_bits2field_kk_req(void* pSMRequests,  void * pAddress);
    int zkevm_keccak_f(void * inputs_, int ninputs, void * pAddress);
    int zkevm_keccak_f_req(void* pSMRequests,  void * pAddress);

    int zkevm_padding_sha256(void * inputs_, int ninputs, void * pAddress, void* pSMRequests, void* pSMRequestsOut);
    int zkevm_padding_sha256_req(void* pSMRequests,  void * pAddress);
    int zkevm_padding_sha256_bit(void * inputs_, int ninputs, void * pAddress, void* pSMRequests, void* pSMRequestsOut);
    int zkevm_padding_sha256_bit_req(void* pSMRequests,  void * pAddress);
    int zkevm_bits2field_sha256(void * inputs_, int ninputs, void * pAddress, void* pSMRequests, void* pSMRequestsOut);
    int zkevm_bits2field_sha256_req(void* pSMRequests,  void * pAddress);
    int zkevm_sha256_f(void * inputs_, int ninputs, void * pAddress);
    int zkevm_sha256_f_req(void* pSMRequests,  void * pAddress);

    int zkevm_storage_req(void* pSMRequests,  void * pAddress);
    int zkevm_padding_pg_req(void* pSMRequests,  void * pAddress);
    int zkevm_climb_key_req(void* pSMRequests,  void * pAddress);
    int zkevm_poseidon_g_req(void* pSMRequests,  void * pAddress);
#endif