#include "../include/gfloat.h"

// Constructor (used by Hopper_simulator)
Gfloat::Gfloat(bool sign, int16_t exponent, int32_t significand) 
    : sign_(sign), exponent_(exponent), significand_(significand) {}

// Copy constructor (used implicitly)
Gfloat::Gfloat(const Gfloat& other) : sign_(other.sign_), exponent_(other.exponent_), significand_(other.significand_) {}

// Casting operator (used for static_cast<float>)
Gfloat::operator float() const {
    // If significand is 0, return 0
    if (significand_ == 0) {
        return 0.0f;
    }
    
    uint32_t float_bits = 0;
    
    // Set sign bit
    if (sign_) {
        float_bits |= 0x80000000;
    }
    
    // Calculate exponent bits: exponent + 127
    int exponent_bits = exponent_ + 127;
    
    // If bit 23 of significand is 0, subtract 1 from exponent_bits
    if ((significand_ & 0x800000) == 0) {
        exponent_bits -= 1;
    }
    
    // Set exponent bits (bits 23-30)
    float_bits |= (exponent_bits & 0xFF) << 23;
    
    // Set mantissa bits: first 23 bits of significand (less important bits)
    float_bits |= significand_ & 0x7FFFFF;
    
    // Reinterpret bits as float
    return *reinterpret_cast<const float*>(&float_bits);
}

// Getter methods (used by group_sum functions)
int32_t Gfloat::get_significand() const {
    return significand_;
}

bool Gfloat::get_sign() const {
    return sign_;
}

int16_t Gfloat::get_exponent() const {
    return exponent_;
}
