use ndarray::arr1;
use super::{A1, A2};

// map square to unit simplex shifted up by 1 in the z-direction
pub(crate) fn duffy_transform(s:A1) -> A1 {
    let s1 = (1.0 - s[1])*s[0];
    let s2 = s[0]*s[1];
    arr1(&[s1, s2, 1.0])
}

// map points from unit simplex to original triangle
pub(crate) fn affine_transform(s:A1, mapping_matrix: &A2) -> A1 {
    let b = mapping_matrix.dot(&s);
    b
}