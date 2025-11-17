use std::{fs::File, io::Write, process, time::Instant};


use libm::sqrt;
use ndarray_stats::DeviationExt;
use rand::Rng;

use crate::{A1, A2, analytic_integrals::evaluate_near_singular_integral_analytically, green_integrals::evaluate_green_analytically, integral_functions::integrate_green_with_duffy, 
    utils::{get_gauss_legendre_nodes, meshgrid, turn_matrix_into_correct_information}};

#[allow(dead_code)]
pub(crate) fn analytic_tester() {
    let folder_name= "test_data/analytic_test/".to_string();

    let mut file1 = File::create(folder_name.to_string() + "x_constant.txt").expect("unable to create file");
    let mut file2 = File::create(folder_name.to_string() + "random_triangle.txt").expect("unable to create file");
    let mut normal_file = File::create(folder_name.to_string() + "random_normal.txt").expect("unable to create file");

    let mut analytic_file = File::create(folder_name.to_string() + "analytic_values.txt").expect("unable to create file");
    let mut time_file = File::create(folder_name.to_string() + "time_values.txt").expect("unable to create file");

    let mut random_constant = "".to_string();
    let mut random_normal = "".to_string();
    let mut triangles = "".to_string();
    let mut analytic_results = "".to_string();
    let mut time_results = "".to_string();
    let bound = 0.01;

    // define the values for the normal of x
    let mut normal = A1::from_vec(vec![1.0, 1.0, 1.0]);
    normal = &normal / normal.l2_dist(&A1::zeros(3)).unwrap();

    let num_of_tests = 2000;
    for _ in 0..num_of_tests {
        let mut rng = rand::rng();
        
        // define the fixed x
        // let x = A1::from_vec(vec![rng.gen_range(-0.25..0.25), rng.gen_range(-0.25..0.25), rng.gen_range(-0.25..0.25)]);
        let x = A1::from_vec(vec![0.0, 0.0, rng.random_range(-0.25..0.25)]);
        // let x = A1::from_vec(vec![0.0, 0.0, 0.1]);
        // let shape_operator = - A2::eye(3);  // shaper operator for the unit sphere
        // let node_for_x = Node::new_from_array(999, &x, &normal, Some(shape_operator));

        // create random triangles 
        let mut triangle = A2::from_shape_vec((3, 3), vec![
            rng.random_range(-bound..bound), rng.random_range(-bound..bound), 0.0, 
            rng.random_range(-bound..bound), rng.random_range(-bound..bound), 0.0, 
            rng.random_range(-bound..bound), rng.random_range(-bound..bound), 0.0]).unwrap();

        // let mut triangle = A2::from_shape_vec((3, 3), vec![
        //     0.0, 0.0, 0.0, 
        //     0.1, 0.1, 0.0, 
        //     -0.1, 0.1, 0.0]).unwrap();

        // map points onto sphere
        triangle[[0, 2]] = get_z_value(triangle[[0, 0]], triangle[[0, 1]]);
        triangle[[1, 2]] = get_z_value(triangle[[1, 0]], triangle[[1, 1]]);
        triangle[[2, 2]] = get_z_value(triangle[[2, 0]], triangle[[2, 1]]);
        // println!("triangle: {:?}", triangle);
        
        // turn random triangle into a Triangle
        // let v1 = triangle.slice(s![0, ..]).to_owned();
        // let v2 = triangle.slice(s![1, ..]).to_owned();
        // let v3 = triangle.slice(s![2, ..]).to_owned();
        // let d1 = (&x - &v1).into_shape((1, 3)).unwrap();
        // let d2 = (&x - &v2).into_shape((1, 3)).unwrap();
        // let d3 = (&x - &v3).into_shape((1, 3)).unwrap();

        let time = Instant::now();
        let (new_x, new_normal, new_triangle, analytic_val) = evaluate_near_singular_integral_analytically(&x, &normal, &triangle, 0);
        let analytic_time = time.elapsed();
        
        // println!("\nold version: ----------------------------------\n");
        // let analytic_old = evaluate_integral_analytically(&x, &normal, &triangle, 0, 0, &A2::eye(2));
        // println!("w: \t\t{:?}\nw2:\t\t{:?}\nw3:\t\t{:?}\nnew analytic: \t{:?}\nnew analytic: \t{:?}\nnew analytic: \t{:?}\nold sum: \t{:?}\nold: {:?}", duffy_gamma, jacobian*w2, jacobian*w3, analytic_val, avx, avy, analytic_old.sum(), analytic_old);
        
        // save x and normal
        for j in 0..3 {
            random_constant.push_str(&new_x[j].to_string());
            random_constant.push_str("\t");
            random_normal.push_str(&new_normal[j].to_string());
            random_normal.push_str("\t");
            
        }
        random_constant.push_str("\n");
        random_normal.push_str("\n");

        // save triangle
        for i in 0..3 {
            for j in 0..3 {
                triangles.push_str(&new_triangle[[i,j]].to_string());
                triangles.push_str("\t");
            }
        }
        triangles.push_str("\n");
        
        analytic_results.push_str(&analytic_val.to_string());
        analytic_results.push_str("\n");

        time_results.push_str("\t");
        time_results.push_str(&analytic_time.as_micros().to_string());
        time_results.push_str("\n");
        
    }
    // println!("time: {:?}", time.elapsed());
    file1.write_all(random_constant.as_bytes()).expect("unable to write result");
    file2.write_all(triangles.as_bytes()).expect("unable to write result");
    normal_file.write_all(random_normal.as_bytes()).expect("unable to write normal data");
    // file3.write_all(results1.as_bytes()).expect("unable to write result");
    analytic_file.write_all(analytic_results.as_bytes()).expect("unable to write analytic results");
    time_file.write_all(time_results.as_bytes()).expect("unable to write time results");
    
    println!("data written");
    process::exit(0x0100);
}

#[allow(dead_code)]
pub(crate) fn analytic_green_tester() {

    let folder_name = "test_data/analytic_green_test_singular/".to_string();
    let num_of_tests = 1000;
    let bound = 0.001;

    let mut file1 = File::create(folder_name.to_string() + "x_constant.txt").expect("unable to create file");
    let mut file2 = File::create(folder_name.to_string() + "random_triangle.txt").expect("unable to create file");

    let mut analytic_file = File::create(folder_name.to_string() + "analytic_values.txt").expect("unable to create file");
    let mut analytic_file2 = File::create(folder_name.to_string() + "analytic_values_x.txt").expect("unable to create file");

    let mut duffy_file = File::create(folder_name.to_string() + "duffy_values.txt").expect("unable to create file");
    let mut time_file = File::create(folder_name.to_string() + "time_values.txt").expect("unable to create file");

    let mut random_constant = "".to_string();
    let mut random_normal = "".to_string();
    let mut triangles = "".to_string();
    let mut analytic_results = "".to_string();
    let mut analytic_results2 = "".to_string();

    let mut duffy_results = "".to_string();

    let mut time_results = "".to_string();

    // define the values for the normal of x
    let mut normal = A1::from_vec(vec![1.0, 1.0, 1.0]);
    normal = &normal / normal.l2_dist(&A1::zeros(3)).unwrap();

    const NUMBER_OF_GAUSS_LEGENDRE_NODES:usize = 16;
    let (nodes, weights) = get_gauss_legendre_nodes(NUMBER_OF_GAUSS_LEGENDRE_NODES);
    let (gauss_legendre_x_nodes, gauss_legendre_y_nodes) = meshgrid(nodes.clone(), nodes);
    let weights = weights.t().dot(&weights);

    for _ in 0..num_of_tests {
        let mut rng = rand::rng();
        
        // define the fixed x
        // let x = A1::from_vec(vec![rng.gen_range(-0.25..0.25), rng.gen_range(-0.25..0.25), rng.gen_range(-0.25..0.25)]);
        // let x = A1::from_vec(vec![0.0, 0.0, rng.random_range(-bound..bound)]);
        let x = A1::from_vec(vec![0.0, 0.0, 0.0]);
        
        // let x = A1::from_vec(vec![0.0, 0.0, 0.1]);
        // let shape_operator = - A2::eye(3);  // shaper operator for the unit sphere
        // let node_for_x = Node::new_from_array(999, &x, &normal, Some(shape_operator));

        // create random triangles 
        // let mut triangle = A2::from_shape_vec((3, 3), vec![
        //     rng.random_range(-bound..bound), rng.random_range(-bound..bound), 0.0, 
        //     rng.random_range(-bound..bound), rng.random_range(-bound..bound), 0.0, 
        //     rng.random_range(-bound..bound), rng.random_range(-bound..bound), 0.0]).unwrap();

        // fix first edge to be origin, which is the same as x
        let mut triangle = A2::from_shape_vec((3, 3), vec![
            0.0, 0.0, 0.0, 
            rng.random_range(-bound..bound), rng.random_range(-bound..bound), 0.0, 
            rng.random_range(-bound..bound), rng.random_range(-bound..bound), 0.0]).unwrap();

        // map points onto sphere
        triangle[[0, 2]] = get_z_value(triangle[[0, 0]], triangle[[0, 1]]);
        triangle[[1, 2]] = get_z_value(triangle[[1, 0]], triangle[[1, 1]]);
        triangle[[2, 2]] = get_z_value(triangle[[2, 0]], triangle[[2, 1]]);
        // println!("triangle: {:?}", triangle);
        

        // analytic method
        let time = Instant::now();
        let (new_x, _, new_triangle, analytic_val) = evaluate_green_analytically(&x, &normal, &triangle, 0);
        let analytic_time = time.elapsed();

        // Duffy method
        let time = Instant::now();
        let (mapping_matrix, _, _, jacobian) = turn_matrix_into_correct_information(&triangle, x.clone());
        let (mut w, _w1, _w2) = integrate_green_with_duffy(&x, &gauss_legendre_x_nodes, &gauss_legendre_y_nodes, &weights, NUMBER_OF_GAUSS_LEGENDRE_NODES, &mapping_matrix);
        let duffy_time = time.elapsed();

        w = jacobian * w;



        let (_, _, _, analytic_val2) = evaluate_green_analytically(&x, &normal, &triangle, 1);

        // save x and normal
        for j in 0..3 {
            random_constant.push_str(&new_x[j].to_string());
            random_constant.push_str("\t");
            
        }
        random_constant.push_str("\n");
        random_normal.push_str("\n");

        // save triangle
        for i in 0..3 {
            for j in 0..3 {
                triangles.push_str(&new_triangle[[i,j]].to_string());
                triangles.push_str("\t");
            }
        }
        triangles.push_str("\n");
        
        analytic_results.push_str(&analytic_val.to_string());
        analytic_results.push_str("\n");

        analytic_results2.push_str(&analytic_val2.to_string());
        analytic_results2.push_str("\n");

        duffy_results.push_str(&w.to_string());
        duffy_results.push_str("\n");

        time_results.push_str("\t");
        time_results.push_str(&analytic_time.as_micros().to_string());
        time_results.push_str("\t");
        time_results.push_str(&duffy_time.as_micros().to_string());
        time_results.push_str("\n");
        
    }
    // println!("time: {:?}", time.elapsed());
    file1.write_all(random_constant.as_bytes()).expect("unable to write result");
    file2.write_all(triangles.as_bytes()).expect("unable to write result");
    // file3.write_all(results1.as_bytes()).expect("unable to write result");
    analytic_file.write_all(analytic_results.as_bytes()).expect("unable to write analytic results");
    analytic_file2.write_all(analytic_results2.as_bytes()).expect("unable to write analytic results");
    time_file.write_all(time_results.as_bytes()).expect("unable to write time results");
    duffy_file.write_all(duffy_results.as_bytes()).expect("unable to write duffy results");
    
    println!("data written");
    process::exit(0x0100);
}

fn get_z_value(x: f64, y: f64) -> f64 {
    let z = 1.0 - sqrt(1.0 - x.powi(2) - y.powi(2));
    return z;
}