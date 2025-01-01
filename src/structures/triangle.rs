use std::{collections::HashMap, sync::Mutex};

use ndarray::{concatenate, s, Axis};
use super::{node::Node, A1, A2};

#[derive(Clone)]
pub(crate) struct Triangle {
    /*
    id: id of triangle (unique)
    nodes: list of id of 3 nodes in this triangle
    triangle_normal: normal vector of the triangle
    area: area of triangle
    on_electrode_boundary: if the triangle intersects the boundary of an electrode
     */
    id: usize,
    nodes: Vec<usize>,
    triangle_normal: A1,
    area: f64,
    on_electrode: u8, // 0 if not on it, 1 if on the electrode, 2 if intersects the electrode boundary
}

impl Triangle {
    pub(crate) fn new(id: usize, nodes: Vec<usize>, triangle_normal: A1, area: f64) -> Triangle {
        Triangle { id, nodes, triangle_normal, area , on_electrode: 0}
    }

    pub(crate) fn get_area(&self) -> f64 {
        self.area
    }

    pub(crate) fn get_id(&self) -> usize {
        self.id
    }

    pub(crate) fn get_normal_vector(&self) -> A1 {
        self.triangle_normal.clone()
    }

    pub(crate) fn get_nodes(&self) -> Vec<usize> {
        self.nodes.to_vec()
    }

    // get the coordinates of the nodes of the triangles. Each row is a different node.
    pub(crate) fn unmutable_get_coordinates(&self, all_nodes: &HashMap<usize, Node>) -> A2 {
        let u1 = all_nodes.get(&self.nodes[0]).unwrap().get_coordinate();
        let u1 = u1.into_shape((1, 3)).unwrap();
        
        let u2 = all_nodes.get(&self.nodes[1]).unwrap().get_coordinate();
        let u2 = u2.into_shape((1, 3)).unwrap();

        let u3 = all_nodes.get(&self.nodes[2]).unwrap().get_coordinate();
        let u3 = u3.into_shape((1, 3)).unwrap();
        
        concatenate(Axis(0), &[u1.view(), u2.view(), u3.view()]).unwrap()
    }

    // get the distances between a node and the nodes of the triangle. 
    // each row is the difference between x and a node of the triangle.
    pub(crate) fn get_dist(&self, node: &A1, list_of_nodes: &HashMap<usize, Mutex<Node>>) -> A2 {        
        let u1 = list_of_nodes.get(&self.nodes[0]).unwrap().lock().unwrap();
        let d1 = (node - u1.get_coordinate()).into_shape((1, 3)).unwrap();
        drop(u1);

        let u2 = list_of_nodes.get(&self.nodes[1]).unwrap().lock().unwrap();
        let d2 = (node - u2.get_coordinate()).into_shape((1, 3)).unwrap();
        drop(u2);

        let u3 = list_of_nodes.get(&self.nodes[2]).unwrap().lock().unwrap();
        let d3 = (node - u3.get_coordinate()).into_shape((1, 3)).unwrap();
        drop(u3);
        
        concatenate(Axis(0), &[d1.view(), d2.view(), d3.view()]).unwrap()
    }

    // each row is the difference between v and a vertex of the triangle
    pub(crate) fn unmutable_get_dist(&self, v: &A1, list_of_nodes: &HashMap<usize, Node>) -> A2 {

        let u1 = list_of_nodes.get(&self.nodes[0]).unwrap().get_coordinate();
        let d1 = (v - u1.clone()).into_shape((1, 3)).unwrap();
        
        let u2 = list_of_nodes.get(&self.nodes[1]).unwrap().get_coordinate();
        let d2 = (v - u2).into_shape((1, 3)).unwrap();

        let u3 = list_of_nodes.get(&self.nodes[2]).unwrap().get_coordinate();
        let d3 = (v - u3).into_shape((1, 3)).unwrap();
        
        let dist = concatenate(Axis(0), &[d1.view(), d2.view(), d3.view()]).unwrap();
        // println!("v: {:?}, u1: {:?}\ndist: {:?}", v, u1, dist);
        // let sum = dist.sum_axis(Axis(1));
        // println!("sum: {:?}", sum);
        dist
    }

    // check if the triangle has a node as a vertex
    pub(crate) fn contains_node(&self, node_id: usize) -> bool {
        self.nodes.contains(&node_id)
    }

    // code for CEM model

    pub(crate) fn set_if_on_electrode(&mut self, on_electrode: u8) {
        self.on_electrode = on_electrode;
    }

    pub(crate) fn on_electode(&self) -> u8 {
        self.on_electrode
    }

    // get the centroid and the normal vector of the centroid (calculated as the average of the normal vectors at the vertices)
    pub(crate) fn get_centroid(&self, list_of_nodes: &HashMap<usize, Node>) -> (A1, A1) {
        let n1 = list_of_nodes.get(&self.nodes[0]).unwrap();
        let n2 = list_of_nodes.get(&self.nodes[1]).unwrap();
        let n3 = list_of_nodes.get(&self.nodes[2]).unwrap();

        let v1 = n1.get_coordinate();
        let v2 = n2.get_coordinate();
        let v3 = n3.get_coordinate();

        let normal1 = n1.get_normal_vector();
        let normal2 = n2.get_normal_vector();
        let normal3 = n3.get_normal_vector();
        
        ( (v1 + v2 + v3) / 3. , (normal1 + normal2 + normal3) / 3.)
    }

    // get the midpoints of the edges and project them onto the interface
    pub(crate) fn get_points_for_refinement_array_version(&self, array_of_nodes: &A2) -> A2 {
        // get vertices of triangle
        let vertices = self.get_nodes();
        let vertex1 = array_of_nodes.slice(s![vertices[0] - 1,..]).to_owned();
        let vertex2 = array_of_nodes.slice(s![vertices[1] - 1,..]).to_owned();
        let vertex3 = array_of_nodes.slice(s![vertices[2] - 1,..]).to_owned();

        // find their midpoints
        let m12 = (&vertex1 + &vertex2) / 2.;
        let m23 = (&vertex2 + &vertex3) / 2.;
        let m31 = (&vertex3 + &vertex1) / 2.;

        let mut mid_point_array = A2::zeros((3, 3));
        mid_point_array.slice_mut(s![0,..]).assign(&m12);
        mid_point_array.slice_mut(s![1,..]).assign(&m23);
        mid_point_array.slice_mut(s![2,..]).assign(&m31);

        mid_point_array
    }
    
}