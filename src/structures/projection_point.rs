use ndarray_stats::DeviationExt;
use crate::A1;

#[derive(Clone, Debug)]
pub(crate) struct ProjectionPoint {
    // sign of left and right represents sign in front of arccos
    // 0 means that projection point is not on the triangle
    coordinate: A1,
    norm: f64,
    angle: f64,
    left: i8,
    right: i8,
}

impl ProjectionPoint {
    pub(super) fn new(coordinate: A1, angle: f64) -> ProjectionPoint {
        let norm = coordinate.l2_dist(&A1::zeros(3)).unwrap();
        // // round norm to 12 digits
        // norm = (norm * 1e12).round() / 1e12;

        ProjectionPoint { coordinate, norm, angle , left: 0, right: 0}
    }

    pub(super) fn change_left(&mut self, new_left: i8) {
        self.left = new_left;
    }

    pub(super) fn change_right(&mut self, new_right: i8) {
        self.right = new_right;
    }

    pub(crate) fn get_right(&self) -> i8 {
        self.right
    }

    pub(crate) fn get_left(&self) -> i8 {
        self.left
    }

    pub(crate) fn get_angle(&self) -> f64 {
        self.angle
    }

    pub(crate) fn get_norm(&self) -> f64 {
        self.norm
    }
}