#ifndef GFLOAT_H
#define GFLOAT_H

#include <iostream>
#include <cstdint>

/**
 * @brief Gfloat class for GPU floating-point simulation
 * 
 * This class represents a floating-point number format that can be used
 * for GPU simulation purposes. It provides basic arithmetic operations
 * and conversion capabilities.
 */
class Gfloat {
private:
    int32_t significand_;  // Mantissa/significand
    bool sign_;            // Sign bit (true = negative, false = positive)
    int16_t exponent_;     // Exponent

public:
    // Constructors (used by Hopper_simulator)
    Gfloat(bool sign, int16_t exponent, int32_t significand);
    Gfloat() : sign_(false), exponent_(-133), significand_(0) {}  // Default constructor (Hopper zero exponent)
    
    // Copy constructor (used implicitly)
    Gfloat(const Gfloat& other);
    
    // Casting operator (used for static_cast<float>)
    operator float() const;  // Cast to float (fp32)
    
    // Destructor
    ~Gfloat() = default;
    
    // Getter methods (used by group_sum functions)
    int32_t get_significand() const;
    bool get_sign() const;
    int16_t get_exponent() const;
};

#endif // GFLOAT_H 