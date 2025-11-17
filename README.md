# Overview of the repository

All the code is written in `src`, with `analytic_integrals.rs`, `green_integrals`, `quadratic_surface_approximation.rs` having the main integral evaluation functions for near singular K, G, and singular K respectively. 
These files also have the algorithm for finding the bounds of integration. Code for the specific integrals $I_0, I_x, I_y, \ldots$ are in `src/integrals.rs`.  Some examples for calculating the integral over the triangles are in `src/lib/test_functions.rs`, which is also the code used to generate the data in the paper. 


# Using the Rust code in other languages

## Steps to Create the Rust Function Callable from MATLAB

### 1. Set Up the Rust Project

All functions that are to be called must be written in `lib.rs` in the following format: 
```
#[no_mangle] // Prevent name mangling
pub extern "C" fn function_name(a: f64, b: f64) -> f64 {
  a + b
}
```
The `#[no_mangle]` attribute allows the function name to be called from C. The `extern "C"` specifies taht this function uses the C calling convention, making it compatible with C/C++.

## 2. Update Cargo.toml
Append the `Cargo.toml` file with
```
[lib]
crate-type = ["cdylib"]
```
This makes Rust build a dynamic library, which will be used by the C wrapper.

## 3. Build the shared library
Run
```
cargo build --release
```
and copy the `.dll` file (or `.so`, `.dylib` if not using windows) into a path that can be found by C/C++.

## 4. Create the C Wrapper for MEX

Create a new file `mex_wrapper.c`:
```
#include "mex.h"

// Declare the Rust function signature
extern double function_name(double a, double b);

// The MEX function interface
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("rust_mex:add_two_numbers", "Two input arguments required.");
    }

    // Extract inputs from MATLAB
    double a = mxGetScalar(prhs[0]);
    double b = mxGetScalar(prhs[1]);

    // Call the Rust function
    double result = function_name(a, b);

    // Return the result to MATLAB
    plhs[0] = mxCreateDoubleScalar(result);
}
```
If it doesn't work, maybe you need to also include
```
// Load the DLL
    HMODULE hLib = LoadLibrary("my_rust_lib.dll");
    if (hLib == NULL) {
        mexErrMsgIdAndTxt("MATLAB:sum_elements:loadLibrary", "Failed to load the DLL.");
    }
```
This allows C to find the `.dll` file.

## 5. Compile the MEX Function
Open MATLAB and run the following command, specifying the paths to the Rust shared library and the C wrapper:
```
mex -I/path/to/rust/include -L/path/to/rust/target/release -lrust_mex mex_wrapper.c
```
Replace /path/to/rust/ with the actual path to your Rust project directory.
-I specifies the include path for any headers.
-L specifies the path to the compiled Rust shared library.
-l tells the compiler to link with the rust_mex library.

## Conclusion
All functions written in the `lib.rs` can be called by MATLAB.
