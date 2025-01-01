use libm::{acos, cos};
use ndarray_stats::DeviationExt;
use crate::{A1, A2};

pub mod analytic_triangle;
pub mod node;
pub mod projection_point;
pub mod triangle;
pub mod vertex_point;

fn check_if_positive(v: &A1, d: &A1, angle: f64) -> i8 {
    // check sign of acos
    let origin = A1::zeros(3);
    if (v.l2_dist(&origin).unwrap() * cos(acos(d.l2_dist(&origin).unwrap() / v.l2_dist(&origin).unwrap()) + angle) - v[0]).abs() < 1e-10 {
        1
    } else {
        -1
    }
}

