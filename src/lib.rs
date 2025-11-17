use ndarray::{Array1, Array2};

use crate::{analytic_integrals::evaluate_near_singular_integral_analytically, green_integrals::evaluate_green_analytically, quadratic_surface_approximation::qsa_integrate_singular};

type A1 = Array1<f64>;
type A2 = Array2<f64>;

extern crate ndarray;
extern crate ndarray_parallel;
extern crate clap;

mod test_functions;
mod utils;
pub mod integral_functions;
pub mod integrals;
mod structures;
mod side_map_table;
mod transforms;

mod quadratic_surface_approximation;
mod analytic_integrals;
mod green_integrals;

// qsa method to integrate the function K(x, y)l(y) over a triangle
// K(x, y) = 1/4pi (x-y) * n(x) / |x-y|^3 
// l(y) is the barycentric interpolation polynomials of a 3 point quadrature on a triangle
// current integral types:
// 0: l(y) = 1, 1: l(y) = y_1, 2: l(y) = y_2, where y = (y_1, y_2) in R^2. 
#[no_mangle]
pub extern "C" fn qsa_integration_method(triangle_raw: *const f64, normal_x_raw: *const f64, second_fundamental_form_raw: *const f64, permutation_list_raw: *const usize, integral_type: usize) -> f64 {

    unsafe {
        // create triangle
        let triangle_vec = Vec::from_raw_parts(triangle_raw as *mut f64, 9, 9);
        let triangle = A2::from_shape_vec((3, 3), triangle_vec).unwrap();

        // create normal vector
        let normal_x_vec = Vec::from_raw_parts(normal_x_raw as *mut f64, 3, 3);
        let normal_x = A1::from_vec(normal_x_vec);
        
        // create second fundamental form
        let second_fundamental_form_vec = Vec::from_raw_parts(second_fundamental_form_raw as *mut f64, 9, 9);
        let second_fundamental_form = A2::from_shape_vec((3, 3), second_fundamental_form_vec).unwrap();
        
        // create permutation vector
        let triangle_permutation_vector = Vec::from_raw_parts(permutation_list_raw as *mut usize, 3, 3);
        
        // integrates the function
        let val = qsa_integrate_singular(&triangle, &normal_x, &second_fundamental_form, triangle_permutation_vector, integral_type);

        val
    }
}

// Analytic method to integrate near-singular integral K(x, y)l(y) over a triangle
// K(x, y) = 1/4pi (x-y) * n(x) / |x-y|^3 
// l(y) is the barycentric interpolation polynomials of a 3 point quadrature on a triangle
// current integral types:
// 0: l(y) = 1, 1: l(y) = y_1, 2: l(y) = y_2, where y = (y_1, y_2) in R^2. 
#[no_mangle]
pub extern "C" fn analytic_integration_method(triangle_raw: *const f64, normal_x_raw: *const f64, x_raw: *const f64, _permutation_list_raw: *const usize, integral_type: usize) -> f64 {

    unsafe {
        // create normal vector
        let normal_x_vec = Vec::from_raw_parts(normal_x_raw as *mut f64, 3, 3);
        let normal_x = A1::from_vec(normal_x_vec);
        
        // create x
        let x_vec = Vec::from_raw_parts(x_raw as *mut f64, 3, 3);
        let x = A1::from_vec(x_vec);
        
        // create triangle
        let triangle_vec = Vec::from_raw_parts(triangle_raw as *mut f64, 9, 9);
        let triangle = A2::from_shape_vec((3, 3), triangle_vec).unwrap();

        let (_new_x, _new_normal, _new_triangle, analytic_val) = evaluate_near_singular_integral_analytically(&x, &normal_x, &triangle, integral_type);

        analytic_val
    }
}

// Analytic method to integrate integral G(x, y)l(y) over a triangle
// G(x, y) = -1/4pi 1/ |x-y| 
// l(y) is the barycentric interpolation polynomials of a 3 point quadrature on a triangle
// current integral types:
// 0: l(y) = 1, 1: l(y) = y_1, 2: l(y) = y_2, where y = (y_1, y_2) in R^2. 
#[no_mangle]
pub extern "C" fn analytic_integration_method_green(triangle_raw: *const f64, normal_x_raw: *const f64, x_raw: *const f64, _permutation_list_raw: *const usize, integral_type: usize) -> f64 {

    unsafe {
        // create normal vector
        let normal_x_vec = Vec::from_raw_parts(normal_x_raw as *mut f64, 3, 3);
        let normal_x = A1::from_vec(normal_x_vec);
        
        // create x
        let x_vec = Vec::from_raw_parts(x_raw as *mut f64, 3, 3);
        let x = A1::from_vec(x_vec);
        
        // create triangle
        let triangle_vec = Vec::from_raw_parts(triangle_raw as *mut f64, 9, 9);
        let triangle = A2::from_shape_vec((3, 3), triangle_vec).unwrap();

        let (_new_x, _new_normal, _new_triangle, analytic_val) = evaluate_green_analytically(&x, &normal_x, &triangle, integral_type);

        analytic_val
    }
}
