use std::f64::consts::PI;
use std::fs;

use libm::{acos, atan, atan2, cos, sin};
use ndarray::{concatenate, s, Axis};
use ndarray_stats::DeviationExt;

use crate::{A1, A2};

pub(crate) fn cross_product(a:&A1, b: &A1) -> A1 {
    let v = vec![a[1]*b[2] - b[1]*a[2], b[0]*a[2] - a[0]*b[2], a[0]*b[1] - b[0]*a[1]];
    A1::from_vec(v)
}

// meshgrid function for vectors
pub(crate) fn meshgrid(x: A1, y: A1) -> (A2, A2) {
    let nx = x.dim();
    let ny = y.dim();
    let x = x.into_shape((nx, 1)).unwrap();
    let y = y.into_shape((1, ny)).unwrap();
    
    let mut x_nodes = concatenate(Axis(0), &[x.view()]).unwrap();
    let mut y_nodes = concatenate(Axis(0), &[y.view()]).unwrap();

    for _ in 0..ny-1 {
        x_nodes = concatenate(Axis(1), &[x_nodes.view(), x.view()]).unwrap();
    }
    for _ in 0..nx-1 {
        y_nodes = concatenate(Axis(0), &[y_nodes.view(), y.view()]).unwrap();
    }

    (x_nodes, y_nodes)
}

pub(crate) fn change_triangle_info_for_duffy_method(mut mapping_matrix: A2, distance_matrix: &A2) -> (Vec<usize>, A2) {
    let zero = A1::zeros(3);
    let d1 = distance_matrix.index_axis(Axis(0), 0).l2_dist(&zero).expect("could not compute l2 distance when calculating kernal");
    let d2 = distance_matrix.index_axis(Axis(0), 1).l2_dist(&zero).expect("could not compute l2 distance when calculating kernal");
    let d3 = distance_matrix.index_axis(Axis(0), 2).l2_dist(&zero).expect("could not compute l2 distance when calculating kernal");
    let d = vec![d1, d2, d3];
    let mut min_row = 0;
    let mut min_d = d1;
    for i in 0..3 {
        if d[i] < min_d {
            min_d = d[i];
            min_row = i;
        }
    }
    let triangle_permutation_vector;

    if min_row == 1 {
        triangle_permutation_vector = vec![1, 2, 0];
    } else if min_row == 2 {
        triangle_permutation_vector = vec![2, 0, 1];
    } else {
        triangle_permutation_vector = vec![0, 1, 2];
    } 

    // create new triangle
    let old_triangle = mapping_matrix.clone();
    for i in 0..3 {
        mapping_matrix.slice_mut(s![i, ..]).assign(&old_triangle.slice(s![triangle_permutation_vector[i], ..]));
    }

    // // get inverse of permutation
    // if min_row == 1 {
    //     triangle_permutation_vector = vec![2, 0, 1];
    // } else if min_row == 2 {
    //     triangle_permutation_vector = vec![1, 2, 0];
    // } else {
    //     triangle_permutation_vector = vec![0, 1, 2];
    // } 

    (triangle_permutation_vector, mapping_matrix)
}


// get Gauss Legendre Node Data
pub(crate) fn get_gauss_legendre_nodes(number_of_nodes: usize) -> (A1, A2) {

    let contents;
    if number_of_nodes == 7 {
        contents = fs::read_to_string("gauss_legendre_data/gauss_legendre_nodes_7.txt").expect("Gauss Nodes for N = 7 is missing.");
    } else if number_of_nodes == 8 {
        contents = fs::read_to_string("gauss_legendre_data/gauss_legendre_nodes_8.txt").expect("Gauss Nodes for N = 8 is missing.");
    } else if number_of_nodes == 10 {
        contents = fs::read_to_string("gauss_legendre_data/gauss_legendre_nodes_10.txt").expect("Gauss Nodes for N = 10 is missing.");
    } else if number_of_nodes == 16 {
        contents = fs::read_to_string("gauss_legendre_data/gauss_legendre_nodes_16.txt").expect("Gauss Nodes for N = 16 is missing.");
    } else if number_of_nodes == 50 {
        contents = fs::read_to_string("gauss_legendre_data/gauss_legendre_nodes_50.txt").expect("Gauss Nodes for N = 50 is missing.");
    } else if number_of_nodes == 200 {
        contents = fs::read_to_string("gauss_legendre_data/gauss_legendre_nodes_200.txt").expect("Gauss Nodes for N = 200 is missing.");
    } else {
        panic!("unable to get those gauss legendre nodes.");
    }
    
    let txt = contents.split("\n");
    let mut x = Vec::new();
    let mut w = Vec::new();
    for line in txt {
        let vals:Vec<&str> = line.split_whitespace().collect();
        let x_val:f64 = vals[0].parse().expect("Not a valid number");
        let w_val:f64 = vals[1].parse().expect("Not a valid number");
        x.push(x_val);
        w.push(w_val);
    }
    (A1::from_vec(x), A2::from_shape_vec((1, number_of_nodes), w).unwrap())
}


pub(crate) fn project_onto_line(point: &A1, vertex1: &A1, vertex2: &A1) -> A1 {
    // calculate direction vector of the line
    let mut direction = vertex2 - vertex1;
    
    // normalize direction vector
    direction = &direction / direction.l2_dist(&A1::zeros(3)).unwrap();

    // calcualte the vector from vertex1 to the given point
    let point_vector = point - vertex1;

    // calculate the projection of point vector onto the line direction
    let projection_length = point_vector.dot(&direction);
    
    // calculate the projected point
    let mut projection_point = vertex1 + projection_length * direction;

    // check if projection point has smaller norm than the vertices, otherwise just set projection point to equal to vertex since they are so close together
    let zero = A1::zeros(3);
    let norm = projection_point.l2_dist(&zero).unwrap();
    let norm1 = vertex1.l2_dist(&zero).unwrap();
    let norm2 = vertex2.l2_dist(&zero).unwrap();
    if norm >= norm1 {
        projection_point = vertex1.clone();
    } else if norm >= norm2 {
        projection_point = vertex2.clone();
    } else {
        // println!("norm: {:?}, norm1: {:?}, norm2: {:?}", norm, norm1, norm2);
    }

    projection_point
}

pub(crate) fn get_angle(point: &A1) -> f64 {
    let theta = atan(point[1] / point[0]);
    if theta < 0. {
        // if x-coordinate is negative, then shift by pi/2 to the other quadrant, otherwise shift by 2 pi to make it positive
        if point[0] < 0.0 {
            theta + PI
        } else {
            theta + 2. * PI
        }
    } else {
        if point[0] < 0.0 {
            theta + PI
        } else {
            theta
        }
    }
}

pub(crate) fn is_point_on_segment(point: &A1, vertex1: &A1, vertex2: &A1) -> bool {
    // check if point is within the bounding box of the segment
    let min_x = vertex1[0].min(vertex2[0]);
    let max_x = vertex1[0].max(vertex2[0]);
    let min_y = vertex1[1].min(vertex2[1]);
    let max_y = vertex1[1].max(vertex2[1]);

    if point[0] < min_x || point[0] > max_x || point[1] < min_y || point[1] > max_y {
        return false;
    }
    // check if the point is one of the vertices
    if point.l2_dist(&vertex1).unwrap() < 1e-10 || point.l2_dist(&vertex2).unwrap() < 1e-10 {
        return false;
    }

    // check if the point satisfies the equation of the line
    let dx = vertex2[0] - vertex1[0];
    if dx.abs() > 1e-14 {
        let dy = vertex2[1] - vertex1[1];
        let m = dy / dx;
        let b = vertex2[1] - m * vertex2[0];

        let y_expected = m * point[0] + b;
        //println!("y expected: {:?}, y true: {:?}, {:?}", &y_expected, point[1], (y_expected - point[1]).abs());
        if (y_expected - point[1]).abs() < 1e-10 {
            return true;
        } else {
            return false;
        }
    } else {
        return true;
    }
    
}

pub(crate) fn sort_by_norm(critical_points: &A2, permutation_vector: Vec<usize>) -> (A2, Vec<usize>) {
    let mut norms = Vec::new();
    for i in 0..critical_points.shape()[0] {
        let row = critical_points.slice(s![i, ..]).to_owned();
        norms.push(row.l2_dist(&A1::zeros(critical_points.shape()[1])).expect("unable to compute l2 norm"));
    }

    let prev_norms = norms.clone();
    let mut temp_p: Vec<usize> = (0..critical_points.shape()[0]).collect();
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    temp_p.sort_by(|a, b| prev_norms[*a].partial_cmp(&prev_norms[*b]).unwrap());

    let mut result = A2::zeros((critical_points.shape()[0], critical_points.shape()[1]));
    let mut new_permutation_vector = Vec::new();
    temp_p.into_iter().enumerate().for_each(|(row, val)| {
        result.slice_mut(s![row, ..]).assign(&critical_points.slice(s![val,..]));
        new_permutation_vector.push(permutation_vector[val]);
    });

    (result, new_permutation_vector)
}

pub(crate) fn is_positively_oriented(mut triangle: A2) -> (A2, Vec<usize>) {
    let mut permutation_vector = vec![0, 1, 2];
    let mut v1 = triangle.slice(s![0,..]).to_owned();
    let mut v2 = triangle.slice(s![1,..]).to_owned();
    let mut v3 = triangle.slice(s![2,..]).to_owned();
    let zero = A1::zeros(3);

    // first make trianlge have the first vertex be the smallest
    let mut new_triangle = triangle.clone();
    if v1.l2_dist(&zero).unwrap() > v2.l2_dist(&zero).unwrap() {
        new_triangle.slice_mut(s![0,..]).assign(&triangle.slice(s![1,..]));
        new_triangle.slice_mut(s![1,..]).assign(&triangle.slice(s![0,..]));
        
        triangle = new_triangle.clone();
        let temp = v2.clone();
        v2 = v1;
        v1 = temp;
        permutation_vector.swap(1, 0);
    }

    if v1.l2_dist(&zero).unwrap() > v3.l2_dist(&zero).unwrap() {
        new_triangle.slice_mut(s![0,..]).assign(&triangle.slice(s![2,..]));
        new_triangle.slice_mut(s![2,..]).assign(&triangle.slice(s![0,..]));

        triangle = new_triangle.clone();
        let temp = v3.clone();
        v3 = v1;
        v1 = temp;
        permutation_vector.swap(2, 0);
    }

    let vector1 = v2 - &v1;
    let vector2 = v3 - &v1;

    let determinant = vector1[0 as usize] * vector2[1 as usize] - vector1[1 as usize] * vector2[0 as usize];

    if determinant < 0. {
        new_triangle.slice_mut(s![1,..]).assign(&triangle.slice(s![2,..]));
        new_triangle.slice_mut(s![2,..]).assign(&triangle.slice(s![1,..]));
        permutation_vector.swap(1, 2);
        
        (new_triangle, permutation_vector)
    } else {
        (new_triangle, permutation_vector)
    }

}

// triangle: each row is a point
// rotates so that x is on the z-axis and the triangle is on the xy-plane
#[allow(dead_code)]
pub(crate) fn rotate_triangle_onto_xy_plane(triangle: &A2, x: &A1, normal_x: &A1) -> (A2, A1, A1, A2, A1) {
    // println!("triangle: {:?}\nx: {:?}\nnormal: {:?}", triangle, x, normal_x);
    // first get the normal vector of the triangle
    let v1 = A1::from_vec(vec![triangle[[0, 0]], triangle[[0, 1]], triangle[[0, 2]]]);
    let v2 = A1::from_vec(vec![triangle[[1, 0]], triangle[[1, 1]], triangle[[1, 2]]]);
    let v3 = A1::from_vec(vec![triangle[[2, 0]], triangle[[2, 1]], triangle[[2, 2]]]);
    let zero = A1::zeros(3);
    
    let vector1 = v2 - &v1;
    let vector2 = v3 - &v1;

    let mut normal = cross_product(&vector1, &vector2);
    let n = normal.l2_dist(&zero).unwrap();
    normal = normal / n;
    // println!("triangle normal: {:?}, n: {:?}", normal, n);

    // find the rotation matrix
    let xy_normal = A1::from_vec(vec![0., 0., 1.]);
    let v = cross_product(&normal, &xy_normal);
    let s = v.l2_dist(&zero).expect("unable to compute l2 norm");
    // println!("s: {:?}", s);
    let rotation_matrix;
    if s < 1e-15 {
        // triangle is already parallel to XY plane
        rotation_matrix = A2::eye(3);
    } else {
        let c = normal.dot(&xy_normal);
        rotation_matrix = A2::eye(3) + skew_symmetric(&v) + skew_symmetric(&v).dot(&skew_symmetric(&v)) * (1.0 - c)/s.powi(2);
    }
    // println!("rotation matrix: {:?}", rotation_matrix);
    let mut new_x = rotation_matrix.dot(x);
    let new_normal_x = rotation_matrix.dot(normal_x); // normal is only changed by rotation, not translation
    let mut new_triangle = rotation_matrix.dot(&triangle.t()).t().to_owned(); // each row is a point, so first transpose the triangle, then transpose the result to be consistent
    // println!("\nnew x: {:?}\nnew normal: {:?}\nnew triangle: {:?}", new_x, new_normal_x, new_triangle);

    let mut preimage_of_origin = A1::zeros(3);  // rotations doesn't shift the origin currently

    // now translate triangle to XY-axis
    let z_val = new_triangle[[0, 2]];
    new_triangle.slice_mut(s![..,2]).assign(&A1::zeros(3));
    new_x[2] = new_x[2] - z_val;
    preimage_of_origin[2] = preimage_of_origin[2] + z_val;
    // println!("\nnew x: {:?}\nnew normal: {:?}\nnew triangle: {:?}", new_x, new_normal_x, new_triangle);
    
    // now translate x to z-axis
    let shift_matrix = A2::from_shape_vec((3, 3), vec![new_x[0], new_x[1], 0.0, new_x[0], new_x[1], 0.0, new_x[0], new_x[1], 0.0]).unwrap();
    new_triangle = new_triangle - shift_matrix;
    preimage_of_origin[0] = preimage_of_origin[0] + new_x[0];
    preimage_of_origin[1] = preimage_of_origin[1] + new_x[1];

    new_x[0] = 0.0;
    new_x[1] = 0.0;

    (new_triangle, new_x, new_normal_x, rotation_matrix, preimage_of_origin)
}


// assuming that the target is V1
// first rotates so that nx = [0, 0, 1]
// then translates so that x=V1 is the origin
// then projects triangle to XY-plane
// then rotates so that V1V2 edge is on X-axis
// returns the rotated triangle and the changed permutations of the nodes to make the triangle positively oriented
// the permutation is only in the nodes, so the coordinate variables were not switched, only rotated
// also returns the two rotational matrices as we need to invert them to get the correct second fundamental form.
pub(crate) fn rotate_surface_to_be_tangent(triangle: &A2, normal_x: &A1) -> (A2, Vec<usize>, A2) {
    // translate so that V1 is the origin (which also makes it the smallest vertex in norm)
    let translation_x = triangle[[0, 0]];
    let translation_y = triangle[[0, 1]];
    let translation_z = triangle[[0, 2]];
    let shift_matrix = A2::from_shape_vec((3, 3), vec![translation_x, translation_y, translation_z, translation_x, translation_y, translation_z, translation_x, translation_y, translation_z]).unwrap();
    let mut new_triangle = triangle - shift_matrix;

    // find the rotation matrix that makes n(x) equal to [0, 0, 1]
    let zero = A1::zeros(3);
    let final_normal = A1::from_vec(vec![0., 0., 1.]);
    let mut v = cross_product(&normal_x, &final_normal);
    let angle = acos(normal_x.dot(&final_normal));

    if v.l2_dist(&zero).unwrap() != 0.0 {
        let norm_of_v = v.l2_dist(&zero).unwrap();
        v = v / norm_of_v;
    }
    let skew_v = skew_symmetric(&v);
    let v_as_2d_array = v.into_shape((1, 3)).unwrap();
    let rotation_matrix = cos(angle) * A2::eye(3) + (1.0 - cos(angle)) * (v_as_2d_array.t().dot(&v_as_2d_array)) + sin(angle) * skew_v;

    let temp = new_triangle.t().to_owned();
    new_triangle = rotation_matrix.dot(&temp).t().to_owned();
    
    // now project points onto the XY-plane
    new_triangle[[0, 2]] = 0.0;
    new_triangle[[1, 2]] = 0.0;
    new_triangle[[2, 2]] = 0.0;
    
    // check if the projected triangle is positively oriented
    let permutation_vector;
    (new_triangle, permutation_vector) = is_positively_oriented(new_triangle);

    // now rotate so V1V2 lies on the positive x-axis
    // calculate angle to rotate the triangle
    let theta = atan2(new_triangle[[1, 1]], new_triangle[[1, 0]]);
    let second_rotation_matrix = A2::from_shape_vec((3, 3), vec![cos(-theta), - sin(-theta), 0.0, sin(-theta), cos(-theta), 0.0, 0.0, 0.0, 1.0]).unwrap();
    new_triangle = (second_rotation_matrix.dot(&(new_triangle.t()))).t().to_owned();
    let smaller_rotation_matrix = A2::from_shape_vec((2, 2), vec![cos(-theta), - sin(-theta), sin(-theta), cos(-theta)]).unwrap();

    (new_triangle, permutation_vector, smaller_rotation_matrix)
}

#[allow(dead_code)]
fn skew_symmetric(v: &A1) -> A2 {
    A2::from_shape_vec((3, 3), vec![0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0]).expect("couldn't create matrix")
}

// get interior angle using cosine law
#[allow(dead_code)]
pub(crate) fn get_theta_2(triangle: &A2) -> f64 {
    let v1 = triangle.slice(s![0,..]).to_owned();
    let v2 = triangle.slice(s![1,..]).to_owned();
    let v3 = triangle.slice(s![2,..]).to_owned();
    
    let a = v2.l2_dist(&v3).unwrap(); // length of V2V3
    let b = v1.l2_dist(&v3).unwrap(); // length of V1V3
    let c = v1.l2_dist(&v2).unwrap(); // length of V1V2

    let cos_angle_2 = (a.powi(2) + c.powi(2) - b.powi(2)) / (2. * a * c);
    let theta2 = acos(cos_angle_2);
    theta2
}

// barycentric interpolation
pub(crate) fn get_intepolation_constants(triangle: &A2) -> A2 {
    let a = triangle[[1, 1]] - triangle[[2, 1]]; // P_2y - P_3y
    let b = triangle[[0, 0]] - triangle[[2, 0]]; // P_1x - P_3x
    let c = triangle[[2, 0]] - triangle[[1, 0]]; // P_3x - P_2x
    let d = triangle[[0, 1]] - triangle[[2, 1]]; // P_1y - P_3y

    let v = a*b + c*d;
    if !v.is_normal() {
        println!("triangle: {:?}\nv:{:?}", &triangle, &v);
        panic!("cannot get interpolation constants");
    }
    
    let w1_const = (-a*triangle[[2, 0]] - c*triangle[[2, 1]]) / v;
    let w1_x = a / v;
    let w1_y = c / v;

    let w2_const = (d * triangle[[0, 0]] - b * triangle[[0, 1]]) / v;
    let w2_x = -d / v;
    let w2_y = b / v;

    let w3_const = (-(triangle[[0, 1]] - triangle[[1, 1]])*triangle[[1, 0]] - (triangle[[1, 0]] - triangle[[0, 0]])*triangle[[1, 1]]) / v;
    let w3_x = (triangle[[0, 1]] - triangle[[1, 1]]) / v;
    let w3_y = (triangle[[1, 0]] - triangle[[0, 0]]) / v;

    A2::from_shape_vec((3, 3), vec![w1_const, w1_x, w1_y, w2_const, w2_x, w2_y, w3_const, w3_x, w3_y]).unwrap()
}

pub(crate) fn turn_matrix_into_correct_information(triangle: &A2, x: A1) -> (A2, A2, A1, f64) {
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