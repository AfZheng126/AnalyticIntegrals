use std::f64::{consts::PI};

use libm::{sqrt};
use ndarray::{Zip};
use ndarray_stats::DeviationExt;
use crate::transforms::{affine_transform, duffy_transform};
use crate::utils::{cross_product, get_intepolation_constants};

use crate::{A1, A2};

// evaluation of the different analytic integrals

pub fn centroid_method_integrate_integral(triangle: &A2, permutation_vector: Vec<usize>) -> A2 {
    // get interpolation constants
    let interpolation_constants = get_intepolation_constants(&triangle);
    
    // initialize gamma
    let mut gamma_weights = vec![0.0, 0.0, 0.0];

    // get centroid
    let centroid_x = triangle[[0, 0]] + triangle[[1, 0]] + triangle[[2, 0]];
    let centroid_y = triangle[[0, 1]] + triangle[[1, 1]] + triangle[[2, 1]];
    let centroid_z = triangle[[0, 2]] + triangle[[1, 2]] + triangle[[2, 2]];
    let centroid_norm = sqrt(centroid_x.powi(2) + centroid_y.powi(2) + centroid_z.powi(2));


    // get area of triangle
    let a = A1::from_vec(vec![triangle[[1, 0]] - triangle[[0, 0]], triangle[[1, 1]] - triangle[[0, 1]], triangle[[1, 2]] - triangle[[0, 2]]]);
    let b = A1::from_vec(vec![triangle[[2, 0]] - triangle[[0, 0]], triangle[[2, 1]] - triangle[[0, 1]], triangle[[2, 2]] - triangle[[0, 2]]]);

    let c = cross_product(&a, &b);
    let norm = c.l2_dist(&A1::zeros(3)).unwrap();
    
    let area = norm / 2.0;

    // evaluate functions at centroid
    
    // gamma 1
    let mut weights = vec![0.0, 0.0, 0.0];
    weights[0] += interpolation_constants[[0, 0]] * (-centroid_z) / (4.0 * PI * centroid_norm.powi(3)) * area;
    weights[0] += interpolation_constants[[0, 1]] * (-centroid_z * centroid_x) / (4.0 * PI * centroid_norm.powi(3)) * area;
    weights[0] += interpolation_constants[[0, 2]] * (-centroid_z * centroid_y) / (4.0 * PI * centroid_norm.powi(3)) * area;
    
    // gamma 2
    weights[1] += interpolation_constants[[1, 0]] * (-centroid_z) / (4.0 * PI * centroid_norm.powi(3)) * area;
    weights[1] += interpolation_constants[[1, 1]] * (-centroid_z * centroid_x) / (4.0 * PI * centroid_norm.powi(3)) * area;
    weights[1] += interpolation_constants[[1, 2]] * (-centroid_z * centroid_y) / (4.0 * PI * centroid_norm.powi(3)) * area;

    // gamma 3
    weights[2] += interpolation_constants[[2, 0]] * (-centroid_z) / (4.0 * PI * centroid_norm.powi(3)) * area;
    weights[2] += interpolation_constants[[2, 1]] * (-centroid_z * centroid_x) / (4.0 * PI * centroid_norm.powi(3)) * area;
    weights[2] += interpolation_constants[[2, 2]] * (-centroid_z * centroid_y) / (4.0 * PI * centroid_norm.powi(3)) * area;

    for k in 0..3 {
        gamma_weights[k] = weights[permutation_vector[k]];
    }

    A2::from_shape_vec((1, 3), gamma_weights).unwrap()
}

// Duffy Method
pub fn integrate_green_with_duffy(x: &A1, gauss_legendre_x_nodes:&A2, gauss_legendre_y_nodes:&A2, weights: &A2, number_of_nodes: usize, mapping_matrix: &A2) -> (f64, f64, f64) {

    // calculate Kvals
    let mut k_vals = A2::zeros((number_of_nodes, number_of_nodes));
    Zip::from(&mut k_vals).and(gauss_legendre_x_nodes).and(gauss_legendre_y_nodes).for_each(|k, s1, s2| {
        *k = compute_green_with_duffy(*s1, *s2, x, &mapping_matrix);
    });

    let mut l1_vals = A2::zeros((number_of_nodes, number_of_nodes));
    Zip::from(&mut l1_vals).and(gauss_legendre_x_nodes).for_each(|k, s1| {
        *k = 1.0 - s1;          // 1 - s - t
    });

    let mut l2_vals = A2::zeros((number_of_nodes, number_of_nodes));
    Zip::from(&mut l2_vals).and(gauss_legendre_x_nodes).and(gauss_legendre_y_nodes).for_each(|k, s1, s2| {
        let simplex_point = duffy_transform(A1::from_vec(vec![*s1, *s2, 0.0]));
        let original_points = affine_transform(simplex_point, mapping_matrix);
        *k = original_points[0]; // x
    });

    let mut l3_vals = A2::zeros((number_of_nodes, number_of_nodes));
    Zip::from(&mut l3_vals).and(gauss_legendre_x_nodes).and(gauss_legendre_y_nodes).for_each(|k, s1, s2| {
        let simplex_point = duffy_transform(A1::from_vec(vec![*s1, *s2, 0.0]));
        let original_points = affine_transform(simplex_point, mapping_matrix);
        *k = original_points[1]; // y
    });
    
    // integrate to get weights
    // let w1 = weights * &k_vals * l1_vals;
    let w2 = weights * &k_vals * l2_vals;
    let w3 = weights * &k_vals * l3_vals;
    // let sum_of_weights = vec![w1.sum(), w2.sum(), w3.sum()];
    // A2::from_shape_vec((1, 3), sum_of_weights).unwrap()

    let w = weights * &k_vals;
    // println!("W: {:?}, W1 + W2 + W3: {:?}", w.sum(), w1.sum() + w2.sum() + w3.sum());
    (w.sum(), w2.sum(), w3.sum())

}

// calculate kernal function using Duffy transform when the kernal is the normal derivative of the Green's function in 3D
fn compute_green_with_duffy(s1: f64, s2: f64, x: &A1, mapping_matrix: &A2) -> f64 {
    // println!("duffy");
    // first map points back to original triangle
    let simplex_point = duffy_transform(A1::from_vec(vec![s1,s2, 0.0]));

    let original_points = affine_transform(simplex_point, mapping_matrix);

    // look at how far the node is from the gauss legendre point
    let d = x - &original_points;

    // evaluate Kernal
    let result = - s1 / (4.0*PI*((&d.mapv(|d| d.powi(2))).sum()).sqrt());
    result
}