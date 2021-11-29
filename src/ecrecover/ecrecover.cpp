#include <string>
#include <sstream>
#include <array>
#include <iomanip>
#include "keccak-tiny.h"
#include "secp256k1.h"
#include "secp256k1_recovery.h"
#include "ecrecover.hpp"

using namespace std;

std::string bytes_to_hex_string(const uint8_t *str, const uint64_t s)
{
  std::ostringstream ret;

  for (size_t i = 0; i < s; ++i)
    ret << std::hex << std::setfill('0') << std::setw(2) << std::nouppercase << (int) str[i];

  return ret.str();
}

std::string hex_to_string(const std::string& input)
{
  static const char* const lut = "0123456789abcdef";
  size_t len = input.length();
  if (len & 1) throw std::invalid_argument("odd length");

  std::string output;
  output.reserve(len / 2);
  for (size_t i = 0; i < len; i += 2)
    {
      char a = input[i];
      const char* p = std::lower_bound(lut, lut + 16, a);
      if (*p != a) throw std::invalid_argument("not a hex digit");

      char b = input[i + 1];
      const char* q = std::lower_bound(lut, lut + 16, b);
      if (*q != b) throw std::invalid_argument("not a hex digit");

      output.push_back(((p - lut) << 4) | (q - lut));
    }
  return output;
}

std::string ecrecover(std::string sig, std::string msg) // hex-encoded sig, plain text msg
{
  std::string _sig = hex_to_string(sig.substr(2)); // strip 0x
  
  if(_sig.size() != 65)
    return ("0x00000000000000000000000000000000");
  
  int v = _sig[64];
  _sig = _sig.substr(0,64);

  if(v>3)
    v-=27;

  auto* ctx = secp256k1_context_create( SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY );

  secp256k1_ecdsa_recoverable_signature rawSig;
  if(!secp256k1_ecdsa_recoverable_signature_parse_compact(ctx, &rawSig, (unsigned char*)_sig.data(), v))
      return ("0x00000000000000000000000000000000");

  std::array<uint8_t,32> hash;
  keccak_256(hash.data(), hash.size(), (unsigned char*)msg.data(), msg.length());                                       // hash message
  msg = string("\x19")+"Ethereum Signed Message:\n32"+string(hash.begin(),hash.end());                                  // wrap message hash
  keccak_256(hash.data(), hash.size(), (unsigned char*)msg.data(), msg.length());                                       // hash wrapped message hash

  secp256k1_pubkey rawPubkey;
  if(!secp256k1_ecdsa_recover(ctx, &rawPubkey, &rawSig, hash.data()))                                                   // 32 bit hash
      return ("0x00000000000000000000000000000000");

  std::array<uint8_t,65> pubkey;
  size_t biglen = 65;

  secp256k1_ec_pubkey_serialize(ctx, pubkey.data(), &biglen, &rawPubkey, SECP256K1_EC_UNCOMPRESSED);

  std::string out = std::string(pubkey.begin(),pubkey.end()).substr(1);

  keccak_256(hash.data(), hash.size(), (const unsigned char*)out.data(), out.length());

  return("0x"+bytes_to_hex_string(hash.data(),hash.size()).substr(24));
}

/*int main(){
 std::string a = ecrecover("0x508875071183e0839fc7992b309c3a9d557b7274782d307f33706fddbea193fb3d799f700f43296a8432594ed7de022eab1d43b608949ca098da7824739f57041c","hola"); 
 printf("%s\n",a.c_str());
        return 0;
}*/