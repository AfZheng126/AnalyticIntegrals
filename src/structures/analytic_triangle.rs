use std::f64::consts::PI;

use libm::cos;
use ndarray::{concatenate, s, Axis};
use ndarray_stats::DeviationExt;

use crate::{utils::{cross_product, get_angle, is_point_on_segment, project_onto_line, sort_by_norm}, A1, A2};

use super::{check_if_positive, projection_point::ProjectionPoint, vertex_point::VertexPoint};


#[derive(Clone, Debug)]
pub(crate) struct AnalyticTriangle {
    /*
    vertices: 2d array of the coordiantes of the vertices
    vertex_points: vector of VertexPoint stuctures
    projections: orthogonal projections of origin 
    critical points: 2d array of the critical radius points
    critical_points_order: lists the critical points in terms of their norm from smallest to greatest
     */
    vertices: A2,
    vertex_points: Vec<VertexPoint>,
    orthogonal_projections: Vec<ProjectionPoint>,
    critical_points: A2,
    critical_points_order: Vec<usize>
}

impl AnalyticTriangle {

    pub(crate) fn new(triangle: A2) -> AnalyticTriangle {
        let origin = A1::zeros(3);
        
        // triangle vertices
        let v1 = triangle.slice(s![0, ..]).to_owned();
        let v2 = triangle.slice(s![1, ..]).to_owned();
        let v3 = triangle.slice(s![2, ..]).to_owned();

        // orthogonal projections
        let mut d1 = project_onto_line(&origin, &v1, &v2);
        let d2 = project_onto_line(&origin, &v2, &v3);
        let mut d3 = project_onto_line(&origin, &v3, &v1);

        // angles of orthogonal projections (phi)
        let mut angle1;
        if d1.l2_dist(&origin).unwrap() < 1e-15 {
            // if d1 is the origin, then angles are based on the angle of v2
            // - pi/2 so that later phi + acos(0) = angle of V2
            angle1 = get_angle(&v2) - PI / 2.;
            if angle1 < 0.0 {
                angle1 = angle1 + 2. * PI;
            }
            // also change d1 and d3 to origin
            d1 = A1::zeros(3);
            d3 = A1::zeros(3);
        } else {
            angle1 = get_angle(&d1);
        }

        let angle2 = get_angle(&d2);

        let mut angle3;
        if d3.l2_dist(&origin).unwrap() < 1e-14 {
            // if d3 is the origin, then angles are based on the angle of v3
            angle3 = get_angle(&v3) - PI / 2.;
            if angle3 < 0.0 {
                angle3 = angle3 + 2. * PI;
            }
        } else {
            angle3 = get_angle(&d3);
        }

        let mut projection_point_1 = ProjectionPoint::new(d1.clone(), angle1);
        let mut projection_point_2 = ProjectionPoint::new(d2.clone(), angle2);
        let mut projection_point_3 = ProjectionPoint::new(d3.clone(), angle3);
        
        // cast into 2d array
        let v1_temp = v1.clone().into_shape((1, 3)).unwrap();
        let v2_temp = v2.clone().into_shape((1, 3)).unwrap();
        let v3_temp = v3.clone().into_shape((1, 3)).unwrap();

        let mut critical_points = concatenate(Axis(0), &[v1_temp.view(), v2_temp.view(), v3_temp.view()]).expect("unable to concatenate");
        
        let mut vertex_1 = VertexPoint::new(v1.clone(), 0, 0);
        let mut vertex_2 = VertexPoint::new(v2.clone(), 0, 0);
        let mut vertex_3 = VertexPoint::new(v3.clone(), 0, 0);
        let mut p = vec![0, 1, 2];

        if v1.l2_dist(&origin).unwrap() < 1e-14 {
            // if v1 is the origin, then don't include projections d1 and d3, which are also the origin
            // if d2 is the same point as v2 or v3, don't include it

            if is_point_on_segment(&d2, &v2, &v3) {
                let is_positive = check_if_positive(&v2, &d2, angle2);
                projection_point_2.change_left(is_positive);
                projection_point_2.change_right(-is_positive);

                critical_points = concatenate(Axis(0), &[critical_points.view(), d2.into_shape((1, 3)).unwrap().view()]).unwrap();
                p.push(4);
            } else {
                let is_positive = check_if_positive(&v2, &d2, angle2);
                vertex_2.change_right(is_positive);
                vertex_3.change_left(is_positive);
            }

            // still check for signs on sides v1->v2 and v3->v1
            if (v2.l2_dist(&origin).unwrap() * cos(PI/2. + angle1) - v2[0]).abs() < 1e-12 {
                vertex_1.change_right(1);
                vertex_2.change_left(1);
            } else {
                vertex_1.change_right(-1);
                vertex_2.change_left(-1);
            }

            if (v3.l2_dist(&origin).unwrap() * cos(PI/2. + angle3) - v3[0]).abs() < 1e-12 {
                vertex_3.change_right(1);
                vertex_1.change_left(1);
            } else {
                vertex_3.change_right(-1);
                vertex_1.change_left(-1);
            }
        } else {
            if is_point_on_segment(&d1, &v1, &v2) {
                let is_positive = check_if_positive(&v1, &d1, angle1);
                projection_point_1.change_left(is_positive);
                projection_point_1.change_right(-is_positive);

                critical_points = concatenate(Axis(0), &[critical_points.view(), d1.into_shape((1, 3)).unwrap().view()]).unwrap();
                p.push(3);
            } else {
                // find if arccos should be negative or positive
                // compare to point furthest away from d1 (in this case it is v2 as v1 is always closer to the origin)
                let is_positive = check_if_positive(&v2, &d1, angle1);
                vertex_1.change_right(is_positive);
                vertex_2.change_left(is_positive);
            }
    
            if is_point_on_segment(&d2, &v2, &v3) {
                let is_positive = check_if_positive(&v2, &d2, angle2);
                projection_point_2.change_left(is_positive);
                projection_point_2.change_right(-is_positive);

                critical_points = concatenate(Axis(0), &[critical_points.view(), d2.into_shape((1, 3)).unwrap().view()]).unwrap();
                p.push(4);
            } else {
                // compare to point furthest away from d2 (in case d2 == v2 or d2 == v3)
                let is_positive;
                if d2.l2_dist(&v2).unwrap() < d2.l2_dist(&v3).unwrap() {
                    is_positive = check_if_positive(&v3, &d2, angle2);
                } else {
                    is_positive = check_if_positive(&v2, &d2, angle2);
                }
                vertex_2.change_right(is_positive);
                vertex_3.change_left(is_positive);
            }

            if is_point_on_segment(&d3, &v3, &v1) {
                let is_positive = check_if_positive(&v3, &d3, angle3);
                projection_point_3.change_left(is_positive);
                projection_point_3.change_right(-is_positive);

                critical_points = concatenate(Axis(0), &[critical_points.view(), d3.into_shape((1, 3)).unwrap().view()]).unwrap();
                p.push(5);
            } else {
                // compare to point furthest away from d3 (in this case it is v3 as v1 is always closer to the origin)
                let is_positive = check_if_positive(&v3, &d3, angle3);
                vertex_3.change_right(is_positive);
                vertex_1.change_left(is_positive);
            }
        }
        
        let orthogonal_projections = vec![projection_point_1, projection_point_2, projection_point_3];
        let vertex_sides = vec![vertex_1, vertex_2, vertex_3];
        // sort critical points
        let (critical_points, critical_points_order) = sort_by_norm(&critical_points, p);
        // in very special cases, the norm of a projection point can equal to the norm of the vertex in finite precision even though the distance between them is > 1e-10
        // let (critical_points, critical_points_order) = check_critical_points_order(critical_points, critical_points_order);
        
        AnalyticTriangle { vertices: triangle, vertex_points: vertex_sides, orthogonal_projections, critical_points, critical_points_order }
    }

    pub(crate) fn contains_origin(&self) -> bool {
        // if v1 is the origin, return false since we are integrating the radius from 0 to 0
        let vertex_1 = self.vertex_points[0].get_coordinate();
        if vertex_1.l2_dist(&A1::zeros(3)).unwrap() < 1e-16 {
            return false;
        }

        // calculate the area of the original triangle
        let original_area = self.get_area();
        
        // calculate the areas of triangles formed by each vertex and the origin
        let v1 = self.get_vertices().slice(s![0, ..]).to_owned();
        let v2 = self.get_vertices().slice(s![1, ..]).to_owned();
        let v3 = self.get_vertices().slice(s![2, ..]).to_owned();
        let zeros = A1::zeros(3);
        
        let a1 = &v1 - &zeros;
        let a2 = &v2 - &zeros;
        let a3 = &v3 - &zeros;

        let c1 = cross_product(&a1, &a2);
        let val1 = c1.l2_dist(&zeros).unwrap() / 2.;

        let c2 = cross_product(&a1, &a3);
        let val2 = c2.l2_dist(&zeros).unwrap() / 2.;
        
        let c3 = cross_product(&a2, &a3);
        let val3 = c3.l2_dist(&zeros).unwrap() / 2.;

        if (val1 + val2 + val3 - original_area).abs() < 10e-15 {
            return true;
        } else {
            return false;
        }
        
    }

    fn get_area(&self) -> f64 {
        let v1 = self.get_vertices().slice(s![0, ..]).to_owned();
        let v2 = self.get_vertices().slice(s![1, ..]).to_owned();
        let v3 = self.get_vertices().slice(s![2, ..]).to_owned();

        let a = v2 - &v1;
        let b = v3 - &v1;
        let c = cross_product(&a, &b);
        let val = c.l2_dist(&A1::zeros(3)).unwrap();
        val / 2.
    }

    fn get_vertices(&self) -> &A2 {
        &self.vertices
    }

    pub(crate) fn get_critical_radius(&self, i: usize) -> f64 {
        let row = self.critical_points.slice(s![i,..]);
        let r = row.l2_dist(&A1::zeros(3)).unwrap();
        r
    }
    
    pub(crate) fn get_number_of_critical_points(&self) -> usize {
        self.critical_points.shape()[0]
    }

    pub(crate) fn get_critical_points_order(&self) -> Vec<usize> {
        self.critical_points_order.to_owned()
    }

    pub(crate) fn get_vertex(&self, i: usize) -> &VertexPoint {
        let vertex = self.vertex_points.get(i).unwrap();
        vertex
    }

    pub(crate) fn get_orthogonal_projection(&self, i: usize) -> &ProjectionPoint {
        let point = self.orthogonal_projections.get(i).unwrap();
        point
    }
}