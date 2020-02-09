#include <string>
#include <vector>
#include <set>


/* The UValue class manages numbers with units by 
 * storing both value (double) and unit (string).
 * Ultimately, this structure will allow for easy
 * conversion between different units. */
class UValue {
    // Data-members
    // Value of the number
    double value;
    // Units of the number
    std::string units;

public:
    // Constructor sets value and units
    UValue(double value, const std::string &units);

    // Member functions
    double get_value() const;       // Accessors
    std::string get_units() const; 
};

/* The UnitConverter class will keep track of all
 * conversions we know how to perform.
 */
class UnitConverter {
    // Structure to store conversion info
    struct Conversion {
        // Units of input value
        std::string from_units;
        // amount to multiply value to get output
        // units
        double multiplier;
        // units of output value
        std::string to_units;
        // set of units seen during recursive search for conversion
        std::set<std::string> seen;
    };
    
    // Member variables
    // vector of Conversion structs to hold all
    // known conversions using dynamic allocation
    std::vector<Conversion> conversions;
public:
    // Member functions
    // Constructor
    UnitConverter();
    // adds a new conversion rule to the UnitConverter (mutator)
    void add_conversion(const std::string &from_units,
            double multiplier, const std::string &to_units);
    // converts directly from input value to the output units (accessor)
    UValue convert_to(UValue input, const std::string &to_units) const;
    // recursively converts from the input value to the output units (accessor)
    UValue convert_to(UValue input, const std::string &to_units, 
            std::set<std::string> seen) const;
};
