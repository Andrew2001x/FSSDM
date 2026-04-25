#include <shark/protocols/common.hpp>
#include <shark/protocols/mul.hpp>
# English note: localized comment removed.
# English note: localized comment removed.
# English note: localized comment removed.
# English note: localized comment removed.
# English note: localized comment removed.

namespace shark {
namespace protocols {
namespace invertsqrt {

// English note: localized comment removed.
// English note: localized comment removed.
// English note: localized comment removed.
void sep_gen(const span<u64>& x_tilde, int f, span<u64>& u_tilde, span<u8>& z) {
    // English note: localized comment removed.
    // English note: localized comment removed.
}

void sep_eval(const span<u64>& x_tilde, int f, span<u64>& u_tilde, span<u8>& z) {
    // English note: localized comment removed.
}

void sep_call(const span<u64>& x_tilde, int f, span<u64>& u_tilde, span<u8>& z) {
    if (party == DEALER) sep_gen(x_tilde, f, u_tilde, z);
    else sep_eval(x_tilde, f, u_tilde, z);
}

// English note: localized comment removed.
// English note: localized comment removed.
u64 sqrtcomp_gen(int f, const span<u8>& z) {
    // English note: localized comment removed.
    int k = z.size();
    int k_prime = k / 2;
    int f_prime = f / 2;
    u64 c0 = 1ULL << ((f + 1) / 2); // 2^{(f+1)/2} 
    u64 c1 = 1ULL << (f / 2);
    span<u8> a(k_prime + 1);
    for (int i = 0; i <= k_prime; ++i) {
        a[i] = or_call(z[2*i], z[2*i + 1]);
    }
    // English note: localized comment removed.
    // English note: localized comment removed.
    int e_prime = 0; // TODO
    u64 b = lsb_mux(z); // LSB(e + f)
    u64 result = mux(c0, c1, b) * (1ULL << (2 * f_prime - (e_prime + 1)));
    return result;
}

u64 sqrtcomp_eval(int f, const span<u8>& z) {
    // English note: localized comment removed.
    // English note: localized comment removed.
    return 0;
}

u64 sqrtcomp_call(int f, const span<u8>& z) {
    if (party == DEALER) return sqrtcomp_gen(f, z);
    else return sqrtcomp_eval(f, z);
}

// English note: localized comment removed.
// English note: localized comment removed.
void gen(const span<u64>& x_tilde, span<u64>& y_tilde, int f) {
    span<u64> u_tilde(x_tilde.size());
    span<u8> z(/*k*/); // k
    sep_call(x_tilde, f, u_tilde, z);
    // English note: localized comment removed.
    // English note: localized comment removed.
    u64 const1 = /*3.14736 * (1ULL << f)*/;
    u64 const2 = /*4.63887 * (1ULL << f)*/;
    u64 const3 = /*5.77789 * (1ULL << f)*/;
    span<u64> temp1 = mul_call(u_tilde, const2);
    span<u64> temp2 = sub_call(temp1, const3);
    span<u64> temp3 = truncate_call(temp2, f);
    span<u64> temp4 = mul_call(u_tilde, temp3);
    span<u64> temp5 = truncate_call(temp4, f);
    span<u64> c_tilde = add_call(const1, temp5);
    // Step 3
    u64 pow2 = sqrtcomp_call(f, z);
    // Step 4
    span<u64> temp6 = mul_call(c_tilde, pow2);
    y_tilde = truncate_call(temp6, f);
}

void eval(const span<u64>& x_tilde, span<u64>& y_tilde, int f) {
    // English note: localized comment removed.
}

void call(const span<u64>& x_tilde, span<u64>& y_tilde, int f) {
    if (party == DEALER) gen(x_tilde, y_tilde, f);
    else eval(x_tilde, y_tilde, f);
}

} // namespace invertsqrt
} // namespace protocols
} // namespace shark

