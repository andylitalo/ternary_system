#include "testbase.h"
#include "units.h"
#include <stdexcept>

#include <cstdlib>
#include <iostream>


using namespace std;


/*===========================================================================
 * TEST FUNCTIONS
 *
 * These are called by the main() function at the end of this file.
 */


/*!
 * Perform some basic tests using the UnitConverter
 */
void test_converter(TestContext &ctx) {
    ctx.DESC("UnitConverter operations");

    UnitConverter u;

    u.add_conversion("km", 1000, "m");  // 1km = 1000m

    UValue v = u.convert_to(UValue{5, "km"}, "m");
    ctx.CHECK(v.get_value() == 5000);
    ctx.CHECK(v.get_units() == "m");

    v = u.convert_to(UValue{10000, "m"}, "km");
    ctx.CHECK(v.get_value() == 10);
    ctx.CHECK(v.get_units() == "km");

    ctx.result();

    ctx.DESC("UnitConverter allows multiple conversions for a type");

    u.add_conversion("m", 39.4, "in");  // 1m = 39.4in

    v = u.convert_to(UValue{5, "m"}, "in");
    ctx.CHECK(v.get_value() == 5 * 39.4);
    ctx.CHECK(v.get_units() == "in");

    v = u.convert_to(UValue{394, "in"}, "m");
    ctx.CHECK(v.get_value() == 394 / 39.4);
    ctx.CHECK(v.get_units() == "m");

    ctx.result();

    ctx.DESC("UnitConverter doesn't allow a conversion more than once");

    try {
        u.add_conversion("m", 39.4, "in");  // Already recorded
        ctx.CHECK(false);  // If no exception, it's a failure
    }
    catch (invalid_argument) {
        ctx.CHECK(true);  // Success!
    }
    catch (...) {
        ctx.CHECK(false);  // Failure:  wrong exception
    }

    try {
        u.add_conversion("in", 1.0 / 39.4, "m");  // Already recorded
        ctx.CHECK(false);  // If no exception, it's a failure
    }
    catch (invalid_argument) {
        ctx.CHECK(true);  // Success!
    }
    catch (...) {
        ctx.CHECK(false);  // Failure:  wrong exception
    }

    ctx.result();

    ctx.DESC("UnitConverter throws on unrecognized conversion");

    try {
        v = u.convert_to(UValue{15, "furlong"}, "km");
        ctx.CHECK(false);  // If no exception, it's a failure
    }
    catch (invalid_argument) {
        ctx.CHECK(true);  // Success!
    }
    catch (...) {
        ctx.CHECK(false);  // Failure:  wrong exception
    }

    ctx.result();
}


void test_multistep_conversions(TestContext &ctx) {

    UnitConverter u;
    UValue v{0, "invalid"};

    u.add_conversion("A", 2, "B");  // 1 A = 2 B
    u.add_conversion("B", 3, "C");  // 1 B = 3 C
    u.add_conversion("A", 5, "D");  // 1 A = 5 D

    u.add_conversion("E", 10, "F"); // 1 E = 10 F

    ctx.DESC("Multi-step conversion operations that are possible");

    try {
        // Should be able to convert from A to C, through B
        v = u.convert_to({1, "A"}, "C");
        ctx.CHECK(v.get_value() == 6);
        ctx.CHECK(v.get_units() == "C");
    }
    catch (...) {
        ctx.CHECK(false); // Caught an unexpected exception.
    }

    try {
        // Should be able to convert from C to A, through B
        v = u.convert_to({18, "C"}, "A");
        ctx.CHECK(v.get_value() == 3);
        ctx.CHECK(v.get_units() == "A");
    }
    catch (...) {
        ctx.CHECK(false); // Caught an unexpected exception.
    }

    ctx.result();

    ctx.DESC("Multi-step conversions that are impossible should throw");

    try {
        // Should not be able to convert from B to F.
        v = u.convert_to({20, "B"}, "F");
        ctx.CHECK(false);  // This operation should throw!
    }
    catch (invalid_argument) {
        ctx.CHECK(true);   // Expected exception.
    }
    catch (...) {
        ctx.CHECK(false);  // Exception was expected, but wrong type was thrown
    }

    ctx.result();
}


/*! This program is a simple test-suite for the units code. */
int main() {
  
    cout << "Lab 3:  Testing the UnitConverter\n\n";

    TestContext ctx(cout);

    test_converter(ctx);
    test_multistep_conversions(ctx);

    // Return 0 if everything passed, nonzero if something failed.
    return !ctx.ok();
}
