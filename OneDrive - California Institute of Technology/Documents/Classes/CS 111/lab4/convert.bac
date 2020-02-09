#include <string>
#include <set>
#include <iostream>
#include <fstream>
#include "units.h"
#include <stdexcept>
#include <cstdio>

using namespace std;

/* This script allows the user to type in a
 * number with units in the format
 * <integer value> <characters of units>,
 * request a new unit for the value, and
 * receive the converted number if the capability
 * has been implemented in the program. Otherwise,
 * a failure message is issued. */

// Function declarations
UnitConverter init_converter(const string &filename); // initiates conversions

// Function definitions
/* init_converter() reads a file of unit
 * conversions and adds them to a UnitConverter 
 * object.
 *
 * Throws an invalid_argument exception if the file
 * cannot be read.
 * If a conversion is repeated in the file, the 
 * add_conversion function from the UnitConverter class
 * will throw an invalid_argument exception. */
UnitConverter init_converter(const string &filename) {
    // declare UnitConverter object to store conversions
    UnitConverter converter;
    // file stream for reading conversion file
    ifstream ifs{filename};
    // makes sure the file was opened successfully
    if (!ifs) {
        throw invalid_argument("Couldn't open file");
    } else {
        cout << "Loading units from " << filename << endl;
    }
    int c = 0; // counter for assigning values to conversion struct
    // declare arguments for conversion
    string from_units;
    double multiplier;
    string to_units;
    // Read file until EOF, parsing into arguments for 
    // add_conversion
    while (ifs) {
        // index for assigning input from filestream
        int ind = c % 3;
        // read in the from_units
        if (ind == 0) {
            // if values loaded, add conversion
            if (c > 0) {
                converter.add_conversion(from_units, multiplier, to_units);
            }
            ifs >> from_units;
        // reads in the multiplier
        } else if (ind == 1) {
            ifs >> multiplier;
        // reads in the to units
        } else {
            ifs >> to_units;
        }
        c++;
    }
    return converter;
}

int main() {
    // declares variables
    double value;
    string units, to_units;
    string filename = "rules.txt";

    // initialize UnitConvter to convert values
    UnitConverter u;
    try {
        u = init_converter(filename);
    // if file cannot be read, reports error and returns 1, indicating failure
    } catch (invalid_argument e) {
        cout << e.what() << endl;
        return 1;
    }

    // asks for the inputs; expects <value> <units>
    cout << "Enter value with units: ";
    cin >> value >> units;
    // creates input UValue object
    UValue input{value, units};

    // asks for units to convert to
    cout << "Convert to units: ";
    cin >> to_units;
    // converts units if conversion exists; otherwise throws
    // invalid_argument exception
    UValue output{value, units}; // initialize output with input params
    try {
        // attempts conversion using UnitConverter
        output = u.convert_to(input, to_units);
        // if no exception thrown, prints out successfully converted number
        cout << "Converted to: " <<  output.get_value() 
             << " " << output.get_units() << endl;
    // if invalid_argument exception thrown, print out error message
    } catch (invalid_argument e) {
        cout << "Couldn't convert to " + to_units + "!" << endl;
        cout << e.what() << endl;
        // indicate failure
        return 1;
    }
   
    return 0;
}
