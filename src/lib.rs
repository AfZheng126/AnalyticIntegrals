use integral_functions::{evaluate_integral_analytically, geometric_method_on_singular_integral, integrate_kernal_with_duffy};
use ndarray::{concatenate, s, Array1, Array2, Axis};
use ndarray_stats::DeviationExt;
use structures::node::Node;
use utils::{change_triangle_info_for_duffy_method, cross_product, get_gauss_legendre_nodes, meshgrid};

type A1 = Array1<f64>;
type A2 = Array2<f64>;

extern crate ndarray;
extern crate ndarray_parallel;
extern crate clap;

mod test_functions;
mod utils;
pub mod integral_functions;
mod integrals;
mod structures;
mod side_map_table;
mod transforms;

// geometric method to integrate singular integral
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

// Duffy method to integrate singular integral
#[no_mangle]
pub extern "C" fn duffy_integration_method(triangle_raw: *const f64, normal_x_raw: *const f64, x_raw: *const f64, _permutation_list_raw: *const usize, integral_type: usize) -> f64 {

    unsafe {
        // create normal vector
        let normal_x_vec = Vec::from_raw_parts(normal_x_raw as *mut f64, 3, 3);
        let normal_x = A1::from_vec(normal_x_vec);

        // create x
        let x_vec = Vec::from_raw_parts(x_raw as *mut f64, 3, 3);
        let x = A1::from_vec(x_vec);
        let node_for_x = Node::new_from_array(999, &x, &normal_x);

        // turn random triangle into a Triangle
        let triangle_vec = Vec::from_raw_parts(triangle_raw as *mut f64, 9, 9);
        let triangle = A2::from_shape_vec((3, 3), triangle_vec).unwrap();
        
        let (mapping_matrix, distance_matrix, triangle_normal, jacobian) = turn_matrix_into_correct_information(&triangle, x);

        // generate Gauss Legendre Nodes
        const NUMBER_OF_GAUSS_LEGENDRE_NODES:usize = 7;
        let (nodes, weights) = get_gauss_legendre_nodes(NUMBER_OF_GAUSS_LEGENDRE_NODES);
        let (gauss_legendre_x_nodes, gauss_legendre_y_nodes) = meshgrid(nodes.clone(), nodes);
        let weights = weights.t().dot(&weights);

        let (w, w2, w3) = integrate_kernal_with_duffy(&triangle_normal, &node_for_x, 
            &gauss_legendre_x_nodes, &gauss_legendre_y_nodes, &weights, NUMBER_OF_GAUSS_LEGENDRE_NODES, 
            &mapping_matrix, &distance_matrix, 0.34, 0.44);
        if integral_type == 0 {
            jacobian * w
        } else if integral_type == 1 {
            jacobian * w2
        } else if integral_type == 2 {
            jacobian * w3
        } else {
            panic!("integral type is not implemented")
        }
    }
}

// Analytic method to integrate near-singular integral or when the kernal is the greens function
#[no_mangle]
pub extern "C" fn analytic_integration_method(triangle_raw: *const f64, normal_x_raw: *const f64, x_raw: *const f64, _permutation_list_raw: *const usize, kernal_type: usize, integral_type: usize) -> f64 {

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

        let (_new_x, _new_normal, _new_triangle, analytic_val) = evaluate_integral_analytically(&x, &normal_x, &triangle, 0, kernal_type, integral_type, &A2::eye(2));

        analytic_val
    }
}


fn turn_matrix_into_correct_information(triangle: &A2, x: A1) -> (A2, A2, A1, f64) {
    let v1 = triangle.slice(s![0, ..]).to_owned();
    let v2 = triangle.slice(s![1, ..]).to_owned();
    let v3 = triangle.slice(s![2, ..]).to_owned();
    let d1 = (&x - &v1).into_shape((1, 3)).unwrap();
    let d2 = (&x - &v2).into_shape((1, 3)).unwrap();
    let d3 = (&x - &v3).into_shape((1, 3)).unwrap();

    let distance_matrix = concatenate(Axis(0), &[d1.view(), d2.view(), d3.view()]).unwrap();
    let cross = cross_product(&(v2 - &v1), &(v3 - &v1));
    let cross_norm = cross.l2_dist(&A1::zeros(3)).unwrap();
    let triangle_normal = cross.clone() / cross_norm;

    let jacobian = cross.l2_dist(&A1::zeros(3)).unwrap();
    let mapping_matrix = triangle.clone();
    let (_triangle_permutation_vector, mut mapping_matrix) = change_triangle_info_for_duffy_method(mapping_matrix, &distance_matrix);
    
    let inverse = A2::from_shape_vec((3, 3), vec![-1., -1., 1., 0., 1., 0., 1., 0., 0.]).unwrap();
    mapping_matrix = mapping_matrix.t().dot(&inverse);

    (mapping_matrix, distance_matrix, triangle_normal, jacobian)
}

pub fn run_tests() {
    test_functions::tester();
    test_functions::analytic_tester();
}