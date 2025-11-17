use std::f64::consts::PI;

use crate::{A1, A2, integrals::{integrate_i0, integrate_ix, integrate_ixx, integrate_ixy, integrate_iy, integrate_iyy}, 
    side_map_table::SIDE_TABLE, 
    structures::{AnalyticTriangle, ProjectionPoint}, utils::{is_positively_oriented, rotate_triangle_onto_xy_plane}};

pub fn evaluate_near_singular_integral_analytically(x: &A1, normal: &A1, triangle: &A2, integral_type: usize) -> (A1, A1, A2, f64) {
    // first perform transformations and create analytic triangles
    // this transformation doesn't change the order of the vertices
    let (triangle, x, normal, _rotation_matrix, _preimage_of_origin) = rotate_triangle_onto_xy_plane(triangle, x, normal);
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
        let mut weights;
        if integral_type == 0 {
            weights = integrate_only_kernal(Vec::new(), Vec::new(), 0.0, analytic_triangle.get_critical_radius(0), c, &normal);
        } else if integral_type == 1 {
            weights = integrate_kernal_x(Vec::new(), Vec::new(), 0.0, analytic_triangle.get_critical_radius(0), c, &normal);
        } else if integral_type == 2 {
            weights = integrate_kernal_y(Vec::new(), Vec::new(), 0.0, analytic_triangle.get_critical_radius(0), c, &normal);
        } else {  
            panic!("integral type not supported")
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

    let mut final_val = values_to_sum.iter().sum();

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
        
        let i2 = c* normal[2] * i2_val;
        let i4 = - normal[0] * i4_val;
        let i6 = - normal[1] * i6_val; 
        integral_vals.append(&mut vec![i2, i4, i6]);
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

            let i2_val = integrate_ix(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            let i4_val = integrate_ixx(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            let i6_val = integrate_ixy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);

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

        let i3 = c * normal[2] * i3_val;
        let i5 = - normal[1] * i5_val;
        let i6 = - normal[0] * i6_val;
        integral_vals.append(&mut vec![i3, i5, i6]);
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

            let i3_val = integrate_iy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            let i5_val = integrate_iyy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            let i6_val = integrate_ixy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);

            let i3 = c * normal[2] * i3_val;
            let i5 = - normal[1] * i5_val;
            let i6 = - normal[0] * i6_val; 
            integral_vals.append(&mut vec![i3, i5, i6]);
        }
    }
    integral_vals
}