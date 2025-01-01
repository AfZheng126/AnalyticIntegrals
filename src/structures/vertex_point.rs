use crate::A1;

#[derive(Clone, Debug)]
pub(crate) struct VertexPoint {
    // sign of the d on its left and right (oriented positively)
    // -1 means a negative sign in front of arccos
    // +1 means a positive sign in front of arccos
    // 0 means both, which happens when the orthogonal projection point lies on the triangle
    coordinate: A1,
    left: i8,
    right: i8,
}

impl VertexPoint {
    pub(super) fn new(coordinate: A1, left: i8, right: i8) -> VertexPoint {
        VertexPoint { coordinate, left, right }
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

    pub(super) fn get_coordinate(&self) -> A1 {
        self.coordinate.clone()
    }
}