use libm::{acos, cos, sin};
use ndarray_stats::DeviationExt;
use crate::{A1, A2};

pub mod analytic_triangle;
pub mod node;
pub mod projection_point;
pub mod triangle;
pub mod vertex_point;

fn check_if_positive(v: &A1, d: &A1, angle: f64) -> i8 {
    // check sign of acos
    // println!("----------point: {:?}, v: {:?}", &d, &v);
    let origin = A1::zeros(3);
    let r = v.l2_dist(&origin).unwrap();
    let temp_angle = acos(d.l2_dist(&origin).unwrap() / v.l2_dist(&origin).unwrap());
    let positive_point_x = r * cos(angle + temp_angle);
    let positive_point_y = r * sin(angle + temp_angle);
    let negative_point_x = r * cos(angle - temp_angle);
    let negative_point_y = r * sin(angle - temp_angle);
    let positive_point = A1::from_vec(vec![positive_point_x, positive_point_y, 0.0]);
    let negative_point = A1::from_vec(vec![negative_point_x, negative_point_y, 0.0]);
    
    // println!("+: [{:?}, {:?}], -: [{:?}, {:?}]", positive_point_x, positive_point_y, negative_point_x, negative_point_y);
    if v.l2_dist(&positive_point).unwrap() < v.l2_dist(&negative_point).unwrap() {
        // println!("result = positive");
        1
    } else {
        -1
    }
}
