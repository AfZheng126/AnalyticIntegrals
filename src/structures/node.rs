use std::{collections::HashMap, sync::Mutex};

use ndarray::Axis;
use ndarray_stats::QuantileExt;

use super::{triangle::Triangle, A1};

#[derive(Clone, Debug)]
pub struct Node {
    id: usize, 
    coordinate: A1,
    normal_vector: A1,
    nearby_triangles: Vec<usize>,
    on_electode: u8,    // 0 if not on it, 1, 2, ... if it is on electrode i
    neighbors: Vec<usize>,
}

impl Node {
    //intitialization functions
    pub(crate) fn new(id: usize, coordinate: &Vec<f64>, normal_vector: &Vec<f64>) -> Node {
        Node {id, coordinate: A1::from_vec(coordinate.to_vec()), normal_vector: A1::from_vec(normal_vector.to_vec()), nearby_triangles: Vec::new(), on_electode: 0, neighbors: Vec::new()}
    }
    
    pub(crate) fn new_from_array(id: usize, coordinate: &A1, normal_vector: &A1) -> Node {
        Node {id, coordinate: coordinate.to_owned(), normal_vector: normal_vector.to_owned(), nearby_triangles: Vec::new(), on_electode: 0, neighbors: Vec::new()}
    }

    pub(crate) fn get_id(&self) -> usize {
        self.id
    }

    pub(crate) fn get_neighbors(&self) -> Vec<usize> {
        self.neighbors.clone()
    }

    pub(crate) fn add_nearby_triangle(&mut self, triangle: &Triangle, param: f64, list_of_nodes: &HashMap<usize, Mutex<Node>>) -> bool {
        if self.nearby_triangles.contains(&triangle.get_id()) {     // if the triangle is already included in nearby triangles, do nothing
            return true;
        } else if triangle.get_nodes().contains(&self.get_id()) {   // if the triangle contains the node, add it to nearby triangles and also add its vertices as neighbors
            self.nearby_triangles.push(triangle.get_id());
            for node_id in triangle.get_nodes() {
                if (node_id != self.get_id()) && (self.get_neighbors().contains(&node_id) == false) {
                    self.neighbors.push(node_id);
                }
            }
            return true;
        }else {
            let d = triangle.get_dist(&self.get_coordinate(), list_of_nodes);

            let binding = (&d * &d).sum_axis(Axis(0));
            let val = binding.min().unwrap();
            if val.powf(1.5) < param {
                self.nearby_triangles.push(triangle.get_id());
                return true;
            } else {
                return false;
            }
        }
    }

    pub(crate) fn add_nearby_triangle_unmutable(&mut self, triangle: &Triangle, param: f64, list_of_nodes: &HashMap<usize, Node>) -> bool {
        if self.nearby_triangles.contains(&triangle.get_id()) {
            return true;
        } else if triangle.get_nodes().contains(&self.get_id()) {
            self.nearby_triangles.push(triangle.get_id());
            for node_id in triangle.get_nodes() {
                if (node_id != self.get_id()) && (self.get_neighbors().contains(&node_id) == false) {
                    self.neighbors.push(node_id);
                }
            }
            return true;
        }else {
            let d = triangle.unmutable_get_dist(&self.get_coordinate(), list_of_nodes);

            let binding = (&d * &d).sum_axis(Axis(1));
            let val = binding.min().unwrap();
            if val.powf(1.5) < param {
                self.nearby_triangles.push(triangle.get_id());
                return true;
            } else {
                return false;
            }
        }
    }

    pub(crate) fn get_coordinate(&self) -> A1 {
        self.coordinate.clone()
    }

    pub(crate) fn get_normal_vector(&self) -> A1 {
        self.normal_vector.clone()
    }

    pub(crate) fn close_to(&self, triangle: &Triangle) -> bool {
        self.nearby_triangles.contains(&triangle.get_id())
    }

    fn get_near_by_triangles(&self) -> Vec<usize> {
        self.nearby_triangles.to_owned()
    }

    pub(crate) fn add_near_by_triangles(&mut self, triangle: &Triangle, list_of_nodes: &HashMap<usize, Node>) {
        let n1 = list_of_nodes.get(&triangle.get_nodes()[0]).unwrap();
        let n2 = list_of_nodes.get(&triangle.get_nodes()[1]).unwrap();
        let n3 = list_of_nodes.get(&triangle.get_nodes()[2]).unwrap();

        for id in n1.get_near_by_triangles() {
            if self.nearby_triangles.contains(&id) == false {
                self.nearby_triangles.push(id);
            }
        }
        for id in n2.get_near_by_triangles() {
            if self.nearby_triangles.contains(&id) == false {
                self.nearby_triangles.push(id);
            }
        }
        for id in n3.get_near_by_triangles() {
            if self.nearby_triangles.contains(&id) == false {
                self.nearby_triangles.push(id);
            }
        }
    }

    pub(crate) fn set_on_electrode(&mut self, on_electrode: u8) {
        self.on_electode = on_electrode;
    }

    pub(crate) fn on_electode(&self) -> u8 {
        self.on_electode
    }
}