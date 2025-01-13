use std::f64::consts::PI;
use libm::{atan2, exp, sin};
use ndarray::{arr1, s, Axis, Zip};
use ndarray_stats::DeviationExt;

use crate::{integrals::{integrate_j1, integrate_j2, integrate_j3}, side_map_table::SIDE_TABLE, transforms::{affine_transform, duffy_transform}, A1, A2};
use crate::structures::{analytic_triangle::AnalyticTriangle, node::Node, projection_point::ProjectionPoint};
use crate::utils::{compensated_summation, special_sum, get_theta_2, is_positively_oriented, rotate_surface_to_be_tangent, rotate_triangle_on_xy_plane};
use super::integrals::{geometric_1, geometric_2, geometric_3, geometric_4, geometric_5, geometric_6, integrate_i0, integrate_ix, integrate_iy, integrate_ixx, integrate_iyy, integrate_ixy};

// evaluate the integral analytically
// for greens function, kernal type is 1
// for normal derivative of greens function, kernal type is 2
// integral types are what the polynomial is {0: 1, 1: x, 2: y}
// second_fundamental_form is currently not used so when creating inputs for the function just use &A2::eye(2)
pub(crate) fn evaluate_integral_analytically(x: &A1, normal: &A1, triangle: &A2, method: usize, kernal_type: usize, integral_type: usize, _second_fundamental_form: &A2) -> (A1, A1, A2, f64) {
    // first perform transformations and create analytic triangles
    // this transformation doesn't change the order of the vertices
    let (triangle, x, normal) = rotate_triangle_on_xy_plane(triangle, x, normal);
    let c = x[2];

    // make first row the vertex with the smallest norm and that the triangle is positively oriented
    let (triangle, _triangle_permutation_vector) = is_positively_oriented(triangle.clone());

    // save this triangle
    let new_triangle = triangle.clone();
    let new_x = x.clone();
    let new_normal = normal.clone();

    // create analytic triangle
    let analytic_triangle = AnalyticTriangle::new(triangle.clone());
    
    // ordered in the original triangle, not the permuted triangle
    let mut values_to_sum = Vec::new();
    
    // first check if the origin is in the triangle
    if analytic_triangle.contains_origin() {
        // println!("integrating starting at origin");
        let mut weights;
        if kernal_type == 0 {
            if integral_type == 0 {
                weights = integrate_only_kernal(Vec::new(), Vec::new(), 0.0, analytic_triangle.get_critical_radius(0), c, &normal);
            } else if integral_type == 1 {
                weights = integrate_kernal_x(Vec::new(), Vec::new(), 0.0, analytic_triangle.get_critical_radius(0), c, &normal);
            } else if integral_type == 2 {
                weights = integrate_kernal_y(Vec::new(), Vec::new(), 0.0, analytic_triangle.get_critical_radius(0), c, &normal);
            } else {  
                panic!("integral type not supported")
            }
        } else if kernal_type == 1 {
            weights = integrate_green_analytically(Vec::new(), Vec::new(), 0.0, analytic_triangle.get_critical_radius(0), c, integral_type);
        } else {
            panic!("kernal type not supported")
        }
        values_to_sum.append(&mut weights);
    }

    // interate through the different regions in which theta is continuous
    let critical_points_order = analytic_triangle.get_critical_points_order();    

    let n = analytic_triangle.get_number_of_critical_points();
    // initial start is always [inactive, inactive, inactive]
    let mut activity = vec![-1, -1, -1];

    for i in 0..n-1 { // don't loop the last one as that one just makes all sides inactive

        /* Example: activity is currently of the form
        [active, inactive, active]
        if we hit v2, so d1+ changes to d2( + or - [one of them])
             hence, we get [inactive, active, active]
        if we hit d2, then integral splits to become
              [active, split, active]

        if we hit a vertex, inactive and active switch, but split always becomes active (what if two vertex have same radius?)
        if we hit projection, inactive becomes split (it will always be initially inactive)
        */

        // check what is the current type
        let current_point = critical_points_order[i];

        match current_point {
            0 | 1 | 2 => {
                if activity[SIDE_TABLE[current_point][0]] == 0 {
                    activity[SIDE_TABLE[current_point][0]] = 1;
                } else {
                    activity[SIDE_TABLE[current_point][0]] = - activity[SIDE_TABLE[current_point][0]];
                }
                if activity[SIDE_TABLE[current_point][1]] == 0 {
                    activity[SIDE_TABLE[current_point][1]] = 1;
                } else {
                    activity[SIDE_TABLE[current_point][1]] = - activity[SIDE_TABLE[current_point][1]];
                }
            },
            3 | 4 | 5 => {
                if activity[SIDE_TABLE[current_point][0]] == -1 {
                    activity[SIDE_TABLE[current_point][0]] = 0;
                } else {
                    println!("triangle: {:?}", &triangle);
                    println!("vertex 1: {:?}\nvertex2: {:?}\nvertex 3: {:?}", analytic_triangle.get_vertex(0), analytic_triangle.get_vertex(1), analytic_triangle.get_vertex(2));
                    println!("projection points 1: {:?} \nprojection points 2: {:?}\nprojection points 3: {:?}", analytic_triangle.get_orthogonal_projection(0), analytic_triangle.get_orthogonal_projection(1), analytic_triangle.get_orthogonal_projection(2));
                    println!("current point: {:?}", current_point);
                    println!("activity: {:?}", &activity);
                    println!("critical point order: {:?}", &critical_points_order);
                    panic!("activity went wrong somewhere")
                }
            },
            _ => panic!("permutation vector has a problem")
        }

        // now decide the arccos and which signs to use
        let mut angle_bounds = Vec::new();
        let mut angle_signs = Vec::new();

        let mut last_sign = 0;
        let mut last_side = 0;
        let mut add_last = 0;
        let mut right_left_order = true;

        for side_number in 0..3 {
            let vertex = analytic_triangle.get_vertex(side_number);
            let projection = analytic_triangle.get_orthogonal_projection(side_number);
            if activity[side_number] == 1 {
                // add only one side of projection point, which is the right side and NOT the left side
                if vertex.get_right() == 0 {
                    // find which one to add
                    if right_left_order {
                        angle_signs.push(projection.get_right());
                    } else {
                        angle_signs.push(projection.get_left());
                    }
                } else {
                    angle_signs.push(vertex.get_right());
                }
                add_last = -1; // first edge has passed, no longer need to add the left side of a split edge last
                angle_bounds.push(projection);
                right_left_order = !right_left_order; // switch order for next edge
            } else if activity[side_number] == 0 {
                if add_last == 0 {
                    // add left side at the end if this is the first edge that is not inactive
                    // println!("adding left last");
                    last_sign = projection.get_left();
                    add_last = 1;
                    last_side = side_number;

                    angle_signs.push(projection.get_right());
                    angle_bounds.push(projection);
                    right_left_order = !right_left_order; // switch order for next edge
                } else {
                    // add both left and right of projection point
                    if right_left_order {
                        angle_signs.push(projection.get_right());
                        angle_bounds.push(projection);
                        angle_signs.push(projection.get_left());
                        angle_bounds.push(projection);
                    } else {
                        angle_signs.push(projection.get_left());
                        angle_bounds.push(projection);
                        angle_signs.push(projection.get_right());
                        angle_bounds.push(projection);
                    }
                    // added two points, so don't change order
                }
            } else {
                // do nothing as edge is inactive
            }
        }
        if add_last == 1 {
            angle_signs.push(last_sign);
            angle_bounds.push(analytic_triangle.get_orthogonal_projection(last_side));
            right_left_order = !right_left_order;
        }
        if right_left_order == false {
            panic!("something went wrong")
        }

        let mut weights;
        if integral_type == 0 {
            weights = integrate_only_kernal(angle_signs, angle_bounds, analytic_triangle.get_critical_radius(i), analytic_triangle.get_critical_radius(i+1), c, &normal);
        } else if integral_type == 1 {
            weights = integrate_kernal_x(angle_signs, angle_bounds, analytic_triangle.get_critical_radius(i), analytic_triangle.get_critical_radius(i+1), c, &normal);
        } else if integral_type == 2 {
            weights = integrate_kernal_y(angle_signs, angle_bounds, analytic_triangle.get_critical_radius(i), analytic_triangle.get_critical_radius(i+1), c, &normal);
        } else {
            panic!("integral type not supported")
        }
        values_to_sum.append(&mut weights);
    }

    let mut final_val;
    if method == 0 {
        final_val = values_to_sum.iter().sum();
    } else if method == 1 {
        final_val = special_sum(&values_to_sum);
    } else {
        final_val = compensated_summation(&values_to_sum);
    }

    // divide by the 4 pi
    final_val = final_val / (4.0 * PI);
    
    (new_x, new_normal, new_triangle, final_val)
}

pub(crate) fn integrate_only_kernal(angle_signs: Vec<i8>, angle_bounds: Vec<&ProjectionPoint>, radius_start: f64, radius_end: f64, c:f64, normal: &A1) -> Vec<f64> {
    let n = angle_signs.len();

    // check that n is even
    if (n % 2) != 0 {
        panic!("number of boundaries is not even.")
    }

    let mut integral_vals = Vec::new();

    if n == 0 {
        // println!("integrating around origin ------------------------");
        // integrate from 0 to 2 pi
        let phi_start = PI/2.;
        let d_norm_start = 0.0;
        let sign_start = -1.0;
        // this makes phi - arccos(0) = 0

        let phi_end = 3.*PI / 2.;
        let d_norm_end = 0.0;
        let sign_end = 1.0;
        // this makes phi + arccos(0) = 2pi

        let i1_val = integrate_i0(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
        let i2_val = integrate_ix(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
        let i3_val = integrate_iy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);

        let i1 = c * normal[2] * i1_val;
        let i2 = - normal[0] * i2_val;
        let i3 = - normal[1] * i3_val;
        integral_vals = vec![i1, i2, i3];
    } else {
        let n2 = n / 2;
        for k in 0..n2 {
            // println!("new region---------------------------------");
            let d_start = angle_bounds.get(2*k).unwrap().to_owned();
            let phi_start = d_start.get_angle();
            let d_norm_start = d_start.get_norm();
            let sign_start = angle_signs.get(2*k).unwrap().to_owned() as f64; // 1.0 or -1.0

            let d_end = angle_bounds.get(2*k + 1).unwrap().to_owned();
            let phi_end = d_end.get_angle();
            let d_norm_end = d_end.get_norm();
            let sign_end = angle_signs.get(2*k + 1).unwrap().to_owned() as f64; // 1.0 or -1.0

            let i1_val = integrate_i0(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            let i2_val = integrate_ix(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            let i3_val = integrate_iy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);

            // println!("d norm start: {:?}, phi start: {:?}, sign start: {:?}\nd norm end: {:?}, phi end: {:?}, sign end: {:?}\nradius start: {:?}, radius end: {:?}\nc: {:?}\ni1: {:?}\ti2: {:?}\ti3: {:?}\ni4: {:?}\ti5: {:?}\ti6: {:?}", 
            //         d_norm_start, phi_start, sign_start, d_norm_end, phi_end, sign_end, radius_start, radius_end, &c, i1_val, i2_val, i3_val, i4_val, i5_val, i6_val);

            let i1 = c * normal[2] * i1_val;
            let i2 = - normal[0] * i2_val;
            let i3 = - normal[1] * i3_val;
            integral_vals.append(&mut vec![i1, i2, i3]);
        }
    }
    integral_vals
}

pub(crate) fn integrate_kernal_x(angle_signs: Vec<i8>, angle_bounds: Vec<&ProjectionPoint>, radius_start: f64, radius_end: f64, c:f64, normal: &A1) -> Vec<f64> {
    let n = angle_signs.len();

    // check that n is even
    if (n % 2) != 0 {
        panic!("number of boundaries is not even.")
    }

    let mut integral_vals = Vec::new();

    if n == 0 {
        // println!("integrating around origin ------------------------");
        // integrate from 0 to 2 pi
        let phi_start = PI/2.;
        let d_norm_start = 0.0;
        let sign_start = -1.0;
        // this makes phi - arccos(0) = 0

        let phi_end = 3.*PI / 2.;
        let d_norm_end = 0.0;
        let sign_end = 1.0;
        // this makes phi + arccos(0) = 2pi

        let i2_val = integrate_ix(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
        let i4_val = integrate_ixx(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
        let i6_val = integrate_ixy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);

        // println!("d norm start: {:?}, phi start: {:?}, sign start: {:?}\nd norm end: {:?}, phi end: {:?}, sign end: {:?}\nradius start: {:?}, radius end: {:?}\nc: {:?}\ni1: {:?}\ti2: {:?}\ti3: {:?}\ni4: {:?}\ti5: {:?}\ti6: {:?}", 
        //         d_norm_start, phi_start, sign_start, d_norm_end, phi_end, sign_end, radius_start, radius_end, &c, i1_val, i2_val, i3_val, i4_val, i5_val, i6_val);
        
        let i2 = c* normal[2] * i2_val;
        let i4 = - normal[0] * i4_val;
        let i6 = - normal[1] * i6_val; 
        integral_vals.append(&mut vec![i2, i4, i6]);
    } else {
        let n2 = n / 2;
        for k in 0..n2 {
            // println!("new region---------------------------------");
            let d_start = angle_bounds.get(2*k).unwrap().to_owned();
            let phi_start = d_start.get_angle();
            let d_norm_start = d_start.get_norm();
            let sign_start = angle_signs.get(2*k).unwrap().to_owned() as f64; // 1.0 or -1.0

            let d_end = angle_bounds.get(2*k + 1).unwrap().to_owned();
            let phi_end = d_end.get_angle();
            let d_norm_end = d_end.get_norm();
            let sign_end = angle_signs.get(2*k + 1).unwrap().to_owned() as f64; // 1.0 or -1.0

            let i2_val = integrate_ix(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            let i4_val = integrate_ixx(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            let i6_val = integrate_ixy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);

            // println!("d norm start: {:?}, phi start: {:?}, sign start: {:?}\nd norm end: {:?}, phi end: {:?}, sign end: {:?}\nradius start: {:?}, radius end: {:?}\nc: {:?}\ni1: {:?}\ti2: {:?}\ti3: {:?}\ni4: {:?}\ti5: {:?}\ti6: {:?}", 
            //         d_norm_start, phi_start, sign_start, d_norm_end, phi_end, sign_end, radius_start, radius_end, &c, i1_val, i2_val, i3_val, i4_val, i5_val, i6_val);

            let i2 = c* normal[2] * i2_val;
            let i4 = - normal[0] * i4_val;
            let i6 = - normal[1] * i6_val; 
            integral_vals.append(&mut vec![i2, i4, i6]);
        }
    }
    integral_vals
}

pub(crate) fn integrate_kernal_y(angle_signs: Vec<i8>, angle_bounds: Vec<&ProjectionPoint>, radius_start: f64, radius_end: f64, c:f64, normal: &A1) -> Vec<f64> {
    let n = angle_signs.len();

    // check that n is even
    if (n % 2) != 0 {
        panic!("number of boundaries is not even.")
    }

    let mut integral_vals = Vec::new();

    if n == 0 {
        // println!("integrating around origin ------------------------");
        // integrate from 0 to 2 pi
        let phi_start = PI/2.;
        let d_norm_start = 0.0;
        let sign_start = -1.0;
        // this makes phi - arccos(0) = 0

        let phi_end = 3.*PI / 2.;
        let d_norm_end = 0.0;
        let sign_end = 1.0;
        // this makes phi + arccos(0) = 2pi

        let i3_val = integrate_iy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
        let i5_val = integrate_iyy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
        let i6_val = integrate_ixy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);

        // println!("d norm start: {:?}, phi start: {:?}, sign start: {:?}\nd norm end: {:?}, phi end: {:?}, sign end: {:?}\nradius start: {:?}, radius end: {:?}\nc: {:?}\ni1: {:?}\ti2: {:?}\ti3: {:?}\ni4: {:?}\ti5: {:?}\ti6: {:?}", 
        //         d_norm_start, phi_start, sign_start, d_norm_end, phi_end, sign_end, radius_start, radius_end, &c, i1_val, i2_val, i3_val, i4_val, i5_val, i6_val);
        
        let i3 = c * normal[2] * i3_val;
        let i5 = - normal[1] * i5_val;
        let i6 = - normal[0] * i6_val;
        integral_vals.append(&mut vec![i3, i5, i6]);
    } else {
        let n2 = n / 2;
        for k in 0..n2 {
            // println!("new region---------------------------------");
            let d_start = angle_bounds.get(2*k).unwrap().to_owned();
            let phi_start = d_start.get_angle();
            let d_norm_start = d_start.get_norm();
            let sign_start = angle_signs.get(2*k).unwrap().to_owned() as f64; // 1.0 or -1.0

            let d_end = angle_bounds.get(2*k + 1).unwrap().to_owned();
            let phi_end = d_end.get_angle();
            let d_norm_end = d_end.get_norm();
            let sign_end = angle_signs.get(2*k + 1).unwrap().to_owned() as f64; // 1.0 or -1.0

            let i3_val = integrate_iy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            let i5_val = integrate_iyy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            let i6_val = integrate_ixy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);

            // println!("d norm start: {:?}, phi start: {:?}, sign start: {:?}\nd norm end: {:?}, phi end: {:?}, sign end: {:?}\nradius start: {:?}, radius end: {:?}\nc: {:?}\ni3: {:?}\ti5: {:?}\ni6: {:?}", 
            //         d_norm_start, phi_start, sign_start, d_norm_end, phi_end, sign_end, radius_start, radius_end, &c, i3_val, i5_val, i6_val);

            let i3 = c * normal[2] * i3_val;
            let i5 = - normal[1] * i5_val;
            let i6 = - normal[0] * i6_val; 
            integral_vals.append(&mut vec![i3, i5, i6]);
        }
    }
    integral_vals
}

pub(crate) fn integrate_kernal_with_duffy(triangle_normal: &A1, node: &Node, gauss_legendre_x_nodes:&A2, gauss_legendre_y_nodes:&A2, weights: &A2, number_of_nodes: usize, mapping_matrix: &A2, distance_matrix: &A2, mut a1: f64, mut a2: f64) -> (f64, f64, f64) {
    // define how far the node is from the singularity node on the triangle
    let zero = A1::zeros(3);
    let d1 = distance_matrix.index_axis(Axis(0), 0).l2_dist(&zero).expect("could not compute l2 distance when calculating kernal");
    let d2 = distance_matrix.index_axis(Axis(0), 1).l2_dist(&zero).expect("could not compute l2 distance when calculating kernal");
    let d3 = distance_matrix.index_axis(Axis(0), 2).l2_dist(&zero).expect("could not compute l2 distance when calculating kernal");
    let d = vec![d1, d2, d3];
    let max_omega = d.into_iter().max_by(|a, b| a.total_cmp(b)).ok_or_else(|| 1.0).expect("max omega is 0");
    // find the best a1 and a2 (kind of best)
    if a1 != 0.0 {
        // based on max omega, change other parameters
        if max_omega > 0.7 {
            // if triangle is large, then 
            a1 = 0.409 - 0.37*max_omega;
            // a1 = 0.40 - 0.37 * max_omega;
            // a1 = 0.36;
        } else if max_omega > 0.1 {
            // not sure if linear relation is the best
            // a1 = 0.40 - 0.37 * max_omega;
            a1 = 0.409 - 0.37 * max_omega;
            // a1 = 0.36;
        } else if max_omega > 0.05 {
            a1 = 0.418 - 0.55 * max_omega;
            // a1 = 1.0;
        } else if max_omega > 0.01 {
            a1 = 0.4187 - 0.62 * max_omega;
        } else {
            // if max_omega is very small (<0.01) just set the integral to 0
            a1 = 1.0;
        }
        if a1 < 0.0 {
            a1 = 0.1;
        }

        a2 = a1 + 0.1;
        if a2 > 1.0 {
            a2 = 1.0;
        }
    }

    // calculate Kvals
    let mut k_vals = A2::zeros((number_of_nodes, number_of_nodes));
    Zip::from(&mut k_vals).and(gauss_legendre_x_nodes).and(gauss_legendre_y_nodes).for_each(|k, s1, s2| {
        *k = compute_kernal_with_duffy(*s1, *s2, &node, &triangle_normal, &mapping_matrix, max_omega, a1, a2);
    });

    // let mut l1_vals = A2::zeros((number_of_nodes, number_of_nodes));
    // Zip::from(&mut l1_vals).and(gauss_legendre_x_nodes).for_each(|k, s1| {
    //     *k = 1.0 - s1;          // 1 - s - t
    // });

    let mut l2_vals = A2::zeros((number_of_nodes, number_of_nodes));
    Zip::from(&mut l2_vals).and(gauss_legendre_x_nodes).and(gauss_legendre_y_nodes).for_each(|k, s1, s2| {
        let simplex_point = duffy_transform(arr1(&[*s1, *s2, 0.0]));
        let original_points = affine_transform(simplex_point, mapping_matrix);
        *k = original_points[0]; // x
    });

    let mut l3_vals = A2::zeros((number_of_nodes, number_of_nodes));
    Zip::from(&mut l3_vals).and(gauss_legendre_x_nodes).and(gauss_legendre_y_nodes).for_each(|k, s1, s2| {
        let simplex_point = duffy_transform(arr1(&[*s1, *s2, 0.0]));
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
fn compute_kernal_with_duffy(s1: f64, s2: f64, node: &Node, triangle_normal_vector: &A1, mapping_matrix: &A2, max_omega: f64, a1: f64, a2: f64) -> f64 {
    // println!("duffy");
    // first map points back to original triangle
    let simplex_point = duffy_transform(arr1(&[s1,s2, 0.0]));

    let original_points = affine_transform(simplex_point, mapping_matrix);

    // look at how far the node is from the gauss legendre point
    let d = node.get_coordinate() - &original_points;
    let alpha:f64;
    let omega = d.l2_dist(&A1::zeros(3)).unwrap() / max_omega;

    let a3;
    if a1 == a2 {
        a3 = a1;
    } else {
        a3 = 1.0/(a2 - a1);
    } 

    if omega < a1 {
        alpha = 0.0;
    } else if (omega >= a1) && (omega < a2) {
        alpha = 1. - exp(1.0 + 1.0/((a3*(omega - a1)).powi(2) - 1.0));
    } else {
        alpha = 1.0;
    }
    let true_normal_vector = node.get_normal_vector();
    let normal_vector = alpha * true_normal_vector + (1.0 - alpha) * triangle_normal_vector;

    // evaluate Kernal
    let result = (&d.dot(&normal_vector) * s1) / (4.0*PI*((&d.mapv(|d| d.powi(2))).sum()).powf(1.5));
    result
}

// Geometric Method
pub(crate) fn geometric_method_on_singular_integral(triangle: &A2, normal_x: &A1, second_fundamental_form: &A2, triangle_permutation_vector: Vec<usize>, summation_method: usize, integral_type: usize) -> f64 {
    // rotate to make normal_x be (0, 0, 1), V1 = (0, 0, 0), and V2 = (x, 0, 0)
    let (new_triangle, permutation_vector) = rotate_surface_to_be_tangent(triangle, normal_x);
    let mut new_triangle_permutation_vector = vec![0, 1, 2];
    for k in 0..3 {
        new_triangle_permutation_vector[k] = triangle_permutation_vector[permutation_vector[k]];
    }
    // println!("new triangle: {:?}, {:?}", &new_triangle, &permutation_vector);
    
    // integrate analytically
    let mut final_value = Vec::new();

    // calculate values needed for integral
    let theta_end = atan2(new_triangle[[2, 1]], new_triangle[[2, 0]]);
    let theta_2 = get_theta_2(&new_triangle);
    if theta_2 == 0.0 {
        panic!("theta 2 should not be 0 as the triangle is not degenerate")
    }
    let vertex_2_norm = new_triangle.slice(s![1,..]).l2_dist(&A1::zeros(3)).unwrap();

    // evaluate integrals
    let mut weights = geometric_integral(theta_2, theta_end, vertex_2_norm, &second_fundamental_form, integral_type);
    final_value.append(&mut weights);

    // sum up all the values
    let mut val;
    if summation_method == 0 {
        val = final_value.iter().sum();
    } else if summation_method == 1 {
        val = special_sum(&final_value);

    } else {
        val = compensated_summation(&final_value);
    }
    // sum all the values

    // divide by the 4 pi
    val = val / (4.0 * PI);

    - val / 2.0
}

// integrate in radius and then in theta for the singular integrals using the second fundamental form
pub(crate) fn geometric_integral(theta_2: f64, theta_end: f64, vertex_2_norm: f64, second_fundamental_form: &A2, method: usize) -> Vec<f64> {

    let mut integral_values = Vec::new();

    let sin_theta_2 = sin(theta_2);
    let vertex_2_norm_squared = vertex_2_norm.powi(2);
    let sin_theta_2_squared = sin_theta_2.powi(2);
    
    let i1_val = geometric_1(theta_2, theta_end);
    let i2_val = geometric_2(theta_2, theta_end);
    let i3_val = geometric_3(theta_2, theta_end);
    let i4_val = geometric_4(theta_2, theta_end);
    let i5_val = geometric_5(theta_2, theta_end);
    let i6_val = geometric_6(theta_2, theta_end);
    
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


// Analytic equations for Green function of Laplace in 3D
pub(crate) fn integrate_green_analytically(angle_signs: Vec<i8>, angle_bounds: Vec<&ProjectionPoint>, radius_start: f64, radius_end: f64, c:f64, integral_type: usize) -> Vec<f64> {
    let n = angle_signs.len();

    // check that n is even
    if (n % 2) != 0 {
        panic!("number of boundaries is not even.")
    }

    let mut integral_vals = Vec::new();
    if n == 0 {
        // integrate from 0 to 2 pi
        let phi_start = PI/2.;
        let d_norm_start = 0.0;
        let sign_start = -1.0;
        // this makes phi - arccos(0) = 0

        let phi_end = 3.*PI / 2.;
        let d_norm_end = 0.0;
        let sign_end = 1.0;
        // this makes phi + arccos(0) = 2pi

        let j1_val = integrate_j1(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
        let j2_val = integrate_j2(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
        let j3_val = integrate_j3(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);

        match integral_type {
            0 => {
                integral_vals.push(j1_val);
            }, 1 => {
                integral_vals.push(j2_val);
            }, 2 => {
                integral_vals.push(j3_val);
            }, _ => {
                panic!("integral type not supported for green currently")
            }
        }
    } else {
        let n2 = n / 2;
        for k in 0..n2 {
            let d_start = angle_bounds.get(2*k).unwrap().to_owned();
            let phi_start = d_start.get_angle();
            let d_norm_start = d_start.get_norm();
            let sign_start = angle_signs.get(2*k).unwrap().to_owned() as f64; // 1.0 or -1.0

            let d_end = angle_bounds.get(2*k + 1).unwrap().to_owned();
            let phi_end = d_end.get_angle();
            let d_norm_end = d_end.get_norm();
            let sign_end = angle_signs.get(2*k + 1).unwrap().to_owned() as f64; // 1.0 or -1.0

            let j1_val = integrate_j1(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            let j2_val = integrate_j2(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            let j3_val = integrate_j3(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);

            match integral_type {
                0 => {
                    integral_vals.push(j1_val);
                }, 1 => {
                    integral_vals.push(j2_val);
                }, 2 => {
                    integral_vals.push(j3_val);
                }, _ => {
                    panic!("integral type not supported for green currently")
                }
            }
        }
    }
    integral_vals
}