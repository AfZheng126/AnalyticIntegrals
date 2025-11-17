
use std::f64::consts::PI;

use libm::atan2;
use ndarray::s;
use ndarray_stats::DeviationExt;

use crate::{A1, A2, integrals::{qsa_xx, qsa_xxx, qsa_xxy, qsa_xyy, qsa_yy, qsa_yyy}, utils::{get_theta_2, rotate_surface_to_be_tangent}};


// Quadratic surface approximation method to integrate K(x, y) on a triangle where x is the first vertex of the triangle
pub fn qsa_integrate_singular(triangle: &A2, normal_x: &A1, second_fundamental_form: &A2, triangle_permutation_vector: Vec<usize>, integral_type: usize) -> f64 {
    // rotate to make normal_x be (0, 0, 1), V1 = (0, 0, 0), and V2 = (x, 0, 0)
    let (new_triangle, permutation_vector, _first_rotation_matrix) = rotate_surface_to_be_tangent(triangle, normal_x);

    let mut new_triangle_permutation_vector = vec![0, 1, 2];
    
    for k in 0..3 {
        new_triangle_permutation_vector[k] = triangle_permutation_vector[permutation_vector[k]];
    }
    
    // integrate analytically
    let val = integral_helper_1(&new_triangle, &second_fundamental_form, integral_type);

    - val / 2.  // the -1/2 is due to the second fundamental form expansion 
}

// Geometric Method for algorithm
fn integral_helper_1(triangle: &A2, second_fundamental_form: &A2, integral_type: usize) -> f64 {

    let mut final_value = Vec::new();

    // calculate values needed for integral
    let theta_end = atan2(triangle[[2, 1]], triangle[[2, 0]]);
    let theta_2 = get_theta_2(triangle);
    if theta_2 == 0.0 {
        panic!("theta 2 should not be 0 as the triangle is not degenerate")
    }
    let vertex_2_norm = triangle.slice(s![1,..]).l2_dist(&A1::zeros(3)).unwrap();

    // evaluate integrals
    let mut weights = integral_helper_2(theta_2, theta_end, vertex_2_norm, second_fundamental_form, integral_type);
    final_value.append(&mut weights);

    // sum up all the values
    let mut val = final_value.iter().sum();

    // divide by the 4 pi
    val = val / (4.0 * PI);

    val
}

// integrate in radius and then in theta for the singular integrals using the second fundamental form
fn integral_helper_2(theta_2: f64, theta_end: f64, vertex_2_norm: f64, second_fundamental_form: &A2, method: usize) -> Vec<f64> {

    let mut integral_values = Vec::new();

    let sin_theta_2 = libm::sin(theta_2);
    let vertex_2_norm_squared = vertex_2_norm.powi(2);
    let sin_theta_2_squared = sin_theta_2.powi(2);
    
    let i1_val = qsa_xx(theta_2, theta_end);
    let i2_val = qsa_yy(theta_2, theta_end);
    let i3_val = qsa_xxy(theta_2, theta_end);
    let i4_val = qsa_xyy(theta_2, theta_end);
    let i5_val = qsa_xxx(theta_2, theta_end);
    let i6_val = qsa_yyy(theta_2, theta_end);
    
    if method == 0 {
        // integrate only kernal
        let i1 = second_fundamental_form[[0, 0]] * vertex_2_norm * sin_theta_2 * i1_val;
        let i2 = second_fundamental_form[[1, 1]] * vertex_2_norm * sin_theta_2 * i2_val;
        integral_values.append(&mut vec![i1, i2]);
    } else if method == 1 {
        // integrate kernal * x
        let i4 = 0.5 * second_fundamental_form[[1, 1]] * vertex_2_norm_squared * sin_theta_2_squared * i4_val;
        let i5 = 0.5 * second_fundamental_form[[0, 0]] * vertex_2_norm_squared * sin_theta_2_squared * i5_val;
        integral_values.append(&mut vec![i4, i5]);
    } else if method == 2 {
        // integrate kernal * y
        let i3 = 0.5 * second_fundamental_form[[0, 0]] * vertex_2_norm_squared * sin_theta_2_squared * i3_val;
        let i6 = 0.5 * second_fundamental_form[[1, 1]] * vertex_2_norm_squared * sin_theta_2_squared * i6_val;
        integral_values.append(&mut vec![i3, i6]);
    } else {
        panic!("method not available")
    }
    integral_values
}