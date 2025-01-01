use integral_functions::geometric_method_on_singular_integral;
use ndarray::{Array1, Array2};

type A1 = Array1<f64>;
type A2 = Array2<f64>;

extern crate ndarray;
extern crate ndarray_parallel;
extern crate clap;


mod test_functions;
mod utils;
mod integral_functions;
mod integrals;
mod structures;
mod side_map_table;
mod transforms;

#[no_mangle]
pub extern "C" fn geometric_integration_method(triangle_raw: *const f64, normal_x_raw: *const f64, second_fundamental_form_raw: *const f64, permutation_list_raw: *const usize, integral_type: usize) -> f64 {

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

        let geo_gamma = geometric_method_on_singular_integral(&triangle, &normal_x, &second_fundamental_form, triangle_permutation_vector, 0, integral_type);
        geo_gamma
    }
}



pub fn run_tests() {
    test_functions::tester();
    test_functions::analytic_tester();
}