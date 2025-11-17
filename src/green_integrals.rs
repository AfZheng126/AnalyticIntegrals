use std::f64::consts::PI;

use crate::{A1, A2, integrals::{integrate_j0, integrate_jx, integrate_jy}, side_map_table::SIDE_TABLE, structures::{AnalyticTriangle, ProjectionPoint}, utils::{is_positively_oriented, rotate_triangle_onto_xy_plane}};

// Integrate of (Green * interpolation polynomial)
pub fn evaluate_green_analytically(x: &A1, normal: &A1, triangle: &A2, integral_type: usize) -> (A1, A1, A2, f64) {    
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
        // println!("integrating starting at origin");
        let mut weights;
        if integral_type == 0 {
            weights = integrate_only_green(Vec::new(), Vec::new(), 0.0, analytic_triangle.get_critical_radius(0), c);
        } else if integral_type == 1 {
            weights = integrate_green_x(Vec::new(), Vec::new(), 0.0, analytic_triangle.get_critical_radius(0), c);
        } else if integral_type == 2 {
            weights = integrate_green_y(Vec::new(), Vec::new(), 0.0, analytic_triangle.get_critical_radius(0), c);
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
            weights = integrate_only_green(angle_signs, angle_bounds, analytic_triangle.get_critical_radius(i), analytic_triangle.get_critical_radius(i+1), c);
        } else if integral_type == 1 {
            weights = integrate_green_x(angle_signs, angle_bounds, analytic_triangle.get_critical_radius(i), analytic_triangle.get_critical_radius(i+1), c);
        } else if integral_type == 2 {
            weights = integrate_green_y(angle_signs, angle_bounds, analytic_triangle.get_critical_radius(i), analytic_triangle.get_critical_radius(i+1), c);
        } else {
            panic!("integral type not supported")
        }
        values_to_sum.append(&mut weights);
    }

    let mut final_val = values_to_sum.iter().sum();

    // divide by the - 4 pi
    final_val =  -1.0 * final_val / (4.0 * PI);
    
    (new_x, new_normal, new_triangle, final_val)
}

fn integrate_only_green(angle_signs: Vec<i8>, angle_bounds: Vec<&ProjectionPoint>, radius_start: f64, radius_end: f64, c:f64) -> Vec<f64> {
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

        let j0_val = integrate_j0(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);

        integral_vals = vec![j0_val];
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

            let j0_val = integrate_j0(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            
            integral_vals.push(j0_val);
        }

    }
    integral_vals
}

fn integrate_green_x(angle_signs: Vec<i8>, angle_bounds: Vec<&ProjectionPoint>, radius_start: f64, radius_end: f64, c:f64) -> Vec<f64> {
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

        let jx_val = integrate_jx(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);

        integral_vals = vec![jx_val];
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

            let jx_val = integrate_jx(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            
            integral_vals.push(jx_val);
        }

    }
    integral_vals
}

fn integrate_green_y(angle_signs: Vec<i8>, angle_bounds: Vec<&ProjectionPoint>, radius_start: f64, radius_end: f64, c:f64) -> Vec<f64> {
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

        let jy_val = integrate_jy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);

        integral_vals = vec![jy_val];
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

            let jy_val = integrate_jy(phi_end, phi_start, c, radius_end, radius_start, d_norm_end, d_norm_start, sign_end, sign_start);
            
            integral_vals.push(jy_val);
        }

    }
    integral_vals
}

