#include <string>
#include "units.h"
#include <stdexcept>
#include <vector>
#include <set>

// This definition file defines the member functions of the
// class "UValue" declared in declaration "units.h", as well
// as the non-member function "convert_to".

// Constructor sets value and unit of number
UValue::UValue(double value, const std::string &units) {
    this->value = value;
    this->units = units;
}

// gets value of the number
double UValue::get_value() const {
    return this->value;
}

// gets units of the number
std::string UValue::get_units() const {
    return this->units;
}

// UnitConverter member functions
/* Constructor initializes a UnitConverter object.
 * Conversions must be added separately. */
UnitConverter::UnitConverter() {
}

/* Adds a conversion rule to a UnitConverter object.
 * The reverse conversion is also added.
 * 
 * Throws invalid_argument exception if the conversion 
 * already exists in the UnitConverter. */
void UnitConverter::add_conversion(const std::string &from_units,
        double multiplier, const std::string &to_units) {
    // verifies that conversion isn't already in UnitConverter
    // by iterating through all elements
    for (UnitConverter::Conversion c : this->conversions) {
        // check if conversion from input to output units exists
        if (c.from_units.compare(from_units) == 0 && 
                c.to_units.compare(to_units) == 0) {
            std::string err_msg = "Already have a conversion from " 
                             + from_units +  " to " + to_units;
            throw std::invalid_argument(err_msg);
        } 
    }
    // conversion is not in UnitConverter, so we add it
    Conversion c_new = {from_units, multiplier, to_units};
    this->conversions.push_back(c_new);
    // also add opposite conversion
    Conversion c_rev = {to_units, 1.0/multiplier, from_units};
    this->conversions.push_back(c_rev);
    return;
}

/* convert_to() converts an input unit-ed value to the desired
 * output units. It can recursively loop through conversions to
 * identify conversions requiring multiple steps.
 *
 * Throws an invalid_argument exception if the desired conversion
 * is not available in the UnitConverter objects library. */
UValue UnitConverter::convert_to(UValue input, 
        const std::string &to_units, std::set<std::string> seen) const {
    // records that we've seen the units of the unit-ed number
    seen.insert(input.get_units());
    // loops through existing, unseen conversions from the units of the input
    for (UnitConverter::Conversion c : this->conversions) {
        if (c.from_units.compare(input.get_units()) == 0
                && seen.count(c.to_units) == 0) {
            // found an unseen conversion from the input units
            if (c.to_units.compare(to_units) == 0) {
                // successful conversion to to_units! returns result
                UValue output {c.multiplier * input.get_value(), to_units};
                return output;
            // we have a conversion but not for to_units--keep searching
            } else {
                // perform conversion
                UValue new_value {c.multiplier * input.get_value(), c.to_units};
                // try recursive computation of conversion
                try {
                    UValue result = UnitConverter::convert_to(new_value, to_units, seen);
                    return result;
                // ignore failed recursion and move onto next conversion
                } catch (std::invalid_argument) {}
            }
        }
    }
    // throws exception if no conversion found
    std::string err_msg = "Don't know how to convert from "
                      + input.get_units() + " to "
                          + to_units;
    throw std::invalid_argument(err_msg);   
}

/* 2-argument convert_to option that 
 * works as previously with already-included empty 
 * set for the seen elements. */
UValue UnitConverter::convert_to(UValue v, const std::string &to_units) const {
    return UnitConverter::convert_to(v, to_units, std::set<std::string>{});
}
