use std::{collections::HashMap, fs::{self, File}, io::Write, process, sync::Mutex, time::Instant};

use libm::sqrt;
use ndarray::{concatenate, s, Axis};
use ndarray_stats::DeviationExt;
use rand::Rng;

use crate::{integral_functions::{evaluate_integral_analytically, geometric_method_on_singular_integral, integrate_kernal_with_duffy}, structures::{node::Node, triangle::Triangle}, utils::{change_triangle_info_for_duffy_method, cross_product, get_gauss_legendre_nodes, meshgrid}, A1, A2};

pub(crate) fn tester() {
    //create_mesh();

    let mut file1 = File::create("x_constant.txt").expect("unable to create file");
    let mut file2 = File::create("random_triangle.txt").expect("unable to create file");
    // let mut file3 = File::create("green_values.txt").expect("unable to create file");
    let mut duffy_file = File::create("duffy_values.txt").expect("unable to create file");
    let mut geo_file = File::create("geo_values.txt").expect("unable to create file");
    let mut time_file = File::create("time_values.txt").expect("unable to create file");

    let mut random_constant = "".to_string();
    let mut triangles = "".to_string();
    let mut duffy_results = "".to_string();
    let mut geo_results = "".to_string();
    let mut time_results = "".to_string();
    let bound = 0.05;

    // define the values for the normal of x
    let mut normal = A1::from_vec(vec![0.0, 0.0, 1.0]);
    normal = &normal / normal.l2_dist(&A1::zeros(3)).unwrap();

    let num_of_tests = 2000;
    for _ in 0..num_of_tests {
        let mut rng = rand::thread_rng();
        let c = rng.gen_range(-bound..bound);
        random_constant.push_str(&c.to_string());
        random_constant.push_str("\n");
        
        // define the fixed x
        let x = A1::from_vec(vec![0.0, 0.0, 0.0]);
        let node_for_x = Node::new_from_array(999, &x, &normal);
    
        // create random triangles
        let mut triangle = A2::from_shape_vec((3, 3), vec![
            0.0, 0.0, 0.0, 
            rng.gen_range(-bound..bound), rng.gen_range(-bound..bound), 0.0, 
            rng.gen_range(-bound..bound), rng.gen_range(-bound..bound), 0.0]).unwrap();

        // let r_val = (10.0 as f64).powi(-r);
        // let mut triangle = A2::from_shape_vec((3, 3), vec![
        //     0.0, 0.0, 0.0, 
        //     r_val, 0.0, 0.0, 
        //     0.02, r_val, 0.0]).unwrap();

        // map points onto the sphere
        triangle[[0, 2]] = get_z_value(triangle[[0, 0]], triangle[[0, 1]]);
        triangle[[1, 2]] = get_z_value(triangle[[1, 0]], triangle[[1, 1]]);
        triangle[[2, 2]] = get_z_value(triangle[[2, 0]], triangle[[2, 1]]);
        // println!("triangle: {:?}", &triangle);
        
        // turn random triangle into a Triangle
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
        let (triangle_permutation_vector, mut mapping_matrix) = change_triangle_info_for_duffy_method(mapping_matrix, &distance_matrix);
        
        // save triangle
        for i in 0..3 {
            for j in 0..3 {
                triangles.push_str(&mapping_matrix[[i,j]].to_string());
                triangles.push_str("\t");
            }
        }
        triangles.push_str("\n");
        
        let inverse = A2::from_shape_vec((3, 3), vec![-1., -1., 1., 0., 1., 0., 1., 0., 0.]).unwrap();
        mapping_matrix = mapping_matrix.t().dot(&inverse);

        // generate Gauss Legendre Nodes
        const NUMBER_OF_GAUSS_LEGENDRE_NODES:usize = 7;
        let (nodes, weights) = get_gauss_legendre_nodes(NUMBER_OF_GAUSS_LEGENDRE_NODES);
        let (gauss_legendre_x_nodes, gauss_legendre_y_nodes) = meshgrid(nodes.clone(), nodes);
        let weights = weights.t().dot(&weights);

        // get second fundamental form
        let second_fundamental_form = A2::from_shape_vec((2, 2), vec![-1.0, 0.0, 0.0, -1.0]).unwrap();

        // evaluate the weights for gamma
        let time = Instant::now();
        
        let (w, w2, w3) = integrate_kernal_with_duffy(&triangle_normal, &node_for_x, 
            &gauss_legendre_x_nodes, &gauss_legendre_y_nodes, &weights, NUMBER_OF_GAUSS_LEGENDRE_NODES, 
            &mapping_matrix, &distance_matrix, 0.34, 0.44);
        let duffy_gamma = jacobian * w;
        let duffy_time = time.elapsed();

        let time = Instant::now();
        let geo_gamma = geometric_method_on_singular_integral(&triangle, &normal, &second_fundamental_form, triangle_permutation_vector, 0, 0);
        let geo_time = time.elapsed();
        
        duffy_results.push_str(&duffy_gamma.to_string());
        duffy_results.push_str("\n");
        geo_results.push_str(&geo_gamma.to_string());
        geo_results.push_str("\n");

        time_results.push_str(&duffy_time.as_micros().to_string());
        time_results.push_str("\t");
        time_results.push_str(&geo_time.as_micros().to_string());
        time_results.push_str("\n");
        
    }
    // println!("time: {:?}", time.elapsed());
    file1.write_all(random_constant.as_bytes()).expect("unable to write result");
    file2.write_all(triangles.as_bytes()).expect("unable to write result");
    // file3.write_all(results1.as_bytes()).expect("unable to write result");
    duffy_file.write_all(duffy_results.as_bytes()).expect("unable to write duffy results");
    geo_file.write_all(geo_results.as_bytes()).expect("unable to write duffy results");
    time_file.write_all(time_results.as_bytes()).expect("unable to write duffy results");
    
    println!("data written");
    process::exit(0x0100);
}

pub(crate) fn analytic_tester() {
    let mut file1 = File::create("x_constant.txt").expect("unable to create file");
    let mut file2 = File::create("random_triangle.txt").expect("unable to create file");
    let mut normal_file = File::create("random_normal.txt").expect("unable to create file");

    let mut duffy_file = File::create("duffy_values.txt").expect("unable to create file");
    let mut analytic_file = File::create("analytic_values.txt").expect("unable to create file");
    let mut time_file = File::create("time_values.txt").expect("unable to create file");

    let mut random_constant = "".to_string();
    let mut random_normal = "".to_string();
    let mut triangles = "".to_string();
    let mut duffy_results = "".to_string();
    let mut analytic_results = "".to_string();
    let mut time_results = "".to_string();
    let bound = 0.5;

    // define the values for the normal of x
    let mut normal = A1::from_vec(vec![1.0, 1.0, 1.0]);
    normal = &normal / normal.l2_dist(&A1::zeros(3)).unwrap();

    let num_of_tests = 2000;
    for _ in 0..num_of_tests {
        let mut rng = rand::thread_rng();
        
        // define the fixed x
        // let x = A1::from_vec(vec![rng.gen_range(-0.25..0.25), rng.gen_range(-0.25..0.25), rng.gen_range(-0.25..0.25)]);
        let x = A1::from_vec(vec![0.0, 0.0, rng.gen_range(-0.25..0.25)]);
        // let x = A1::from_vec(vec![0.0, 0.0, 0.1]);
        let node_for_x = Node::new_from_array(999, &x, &normal);

        // create random triangles 
        let mut triangle = A2::from_shape_vec((3, 3), vec![
            rng.gen_range(-bound..bound), rng.gen_range(-bound..bound), 0.0, 
            rng.gen_range(-bound..bound), rng.gen_range(-bound..bound), 0.0, 
            rng.gen_range(-bound..bound), rng.gen_range(-bound..bound), 0.0]).unwrap();

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
        // println!("triangle normal: {:?}", triangle_normal);

        let jacobian = cross.l2_dist(&A1::zeros(3)).unwrap();
        
        let mapping_matrix = triangle.clone();
        let (_triangle_permutation_vector, mut mapping_matrix) = change_triangle_info_for_duffy_method(mapping_matrix, &distance_matrix);
        
        let inverse = A2::from_shape_vec((3, 3), vec![-1., -1., 1., 0., 1., 0., 1., 0., 0.]).unwrap();
        mapping_matrix = mapping_matrix.t().dot(&inverse);

        // generate Gauss Legendre Nodes
        const NUMBER_OF_GAUSS_LEGENDRE_NODES:usize = 50;
        let (nodes, weights) = get_gauss_legendre_nodes(NUMBER_OF_GAUSS_LEGENDRE_NODES);
        let (gauss_legendre_x_nodes, gauss_legendre_y_nodes) = meshgrid(nodes.clone(), nodes);
        let weights = weights.t().dot(&weights);

        // evaluate the weights for gamma
        let time = Instant::now();
        let (w, w2, w3) = integrate_kernal_with_duffy(&triangle_normal, &node_for_x, 
            &gauss_legendre_x_nodes, &gauss_legendre_y_nodes, &weights, NUMBER_OF_GAUSS_LEGENDRE_NODES, 
            &mapping_matrix, &distance_matrix, 0.0, 0.0);
        let duffy_gamma = jacobian * w3;
        let duffy_time = time.elapsed();

        let time = Instant::now();
        let (new_x, new_normal, new_triangle, analytic_val) = evaluate_integral_analytically(&x, &normal, &triangle, 0, 0, 2, &A2::eye(2));
        let analytic_time = time.elapsed();
        // let (new_x, new_normal, new_triangle, avx) = evaluate_integral_analytically_for_paper(&x, &normal, &triangle, 0, 1, &A2::eye(2));
        // let (new_x, new_normal, new_triangle, avy) = evaluate_integral_analytically_for_paper(&x, &normal, &triangle, 0, 2, &A2::eye(2));

        
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
        
        duffy_results.push_str(&duffy_gamma.to_string());
        analytic_results.push_str(&analytic_val.to_string());
        duffy_results.push_str("\n");
        analytic_results.push_str("\n");

        time_results.push_str(&duffy_time.as_micros().to_string());
        time_results.push_str("\t");
        time_results.push_str(&analytic_time.as_micros().to_string());
        time_results.push_str("\n");
        
    }
    // println!("time: {:?}", time.elapsed());
    file1.write_all(random_constant.as_bytes()).expect("unable to write result");
    file2.write_all(triangles.as_bytes()).expect("unable to write result");
    normal_file.write_all(random_normal.as_bytes()).expect("unable to write normal data");
    // file3.write_all(results1.as_bytes()).expect("unable to write result");
    duffy_file.write_all(duffy_results.as_bytes()).expect("unable to write duffy results");
    analytic_file.write_all(analytic_results.as_bytes()).expect("unable to write analytic results");
    time_file.write_all(time_results.as_bytes()).expect("unable to write time results");
    
    println!("data written");
    process::exit(0x0100);
}

fn get_z_value(x: f64, y: f64) -> f64 {
    let z = sqrt(1.0 - x.powi(2) - y.powi(2)) - 1.0;
    // println!("x^2 + y^2: {:?}, z: {:?}", x.powi(2) + y.powi(2), z);
    z
}

pub(crate) fn divergence_test() {
    const DISTANCE_PARAMETER:f64 = 0.0005;

    // integrate the kernal on a sphere and check if it equals to 0, 1, or 1/2

    // read the data from the txt files
    let node_file_path = "test_data/nodes.txt";
    let node_normal_file_path = "test_data/node_normals.txt";
    let triangle_file_path = "test_data/triangles.txt";

    // read node data:
    let node_locations = fs::read_to_string(node_file_path).expect("node location file is missing");
    let node_data: Vec<Vec<f64>> = node_locations.split("\n").map(|v| {
        let loc:Vec<f64> = v.split_whitespace().map(|u| u.parse().unwrap()).collect();
        loc
    }).collect();
    let node_normals = fs::read_to_string(node_normal_file_path).expect("node normals file is missing");
    let node_normals_data: Vec<Vec<f64>> = node_normals.split("\n").map(|v| {
        let loc:Vec<f64> = v.split_whitespace().map(|u| u.parse().unwrap()).collect();
        loc
    }).collect();
    let number_of_nodes = node_data.len();

    // create nodes
    let mut list_of_nodes = HashMap::new();
    for node_number in 1..number_of_nodes+1 {
        let current_node = Node::new(node_number, &node_data[node_number - 1], &node_normals_data[node_number - 1]);
        list_of_nodes.insert(node_number, Mutex::new(current_node));
    }

    // read triangle data
    let node_of_triangles = fs::read_to_string(triangle_file_path).expect("triangle file is missing");
    let triangle_data: Vec<Vec<usize>> = node_of_triangles.split("\n").map(|v| {
        let loc:Vec<usize> = v.split_whitespace().map(|u| u.parse().unwrap()).collect();
        loc
    }).collect();      
    
    let number_of_triangles = triangle_data.len();

    // create all triangles
    let mut list_of_triangles = HashMap::new();
    (1..number_of_triangles + 1).into_iter().for_each(|triangle_id|{
        let triangle_nodes = triangle_data.get(triangle_id - 1).expect("could not find triangle id");
        let area = area(&triangle_nodes, &list_of_nodes);
        let triangle_normal_vector = get_triangle_normal(triangle_nodes, &node_normals_data, &node_data);

        // create instance of triangle
        let current_triangle = Triangle::new(triangle_id, triangle_nodes.to_vec(), triangle_normal_vector, area);
        
        // see which nodes this triangle is close to
        for node_number in 1..number_of_nodes + 1 {
            let mut v = list_of_nodes.get(&node_number).unwrap().lock().unwrap();
            v.add_nearby_triangle(&current_triangle, DISTANCE_PARAMETER, &list_of_nodes);
        }

        list_of_triangles.insert(triangle_id, current_triangle);
    });

    println!("finished reading data");

    let x1 = A1::from_vec(vec![0.0, 0.0, 0.1]);
    let mut normal = A1::from_vec(vec![0.0, 0.0, 1.0]);
    normal = &normal / normal.l2_dist(&A1::zeros(3)).unwrap();

    let x2 = A1::from_vec(vec![0.0, 0.0, 1.0]);
    let x3 = A1::from_vec(vec![0.0, 0.0, 2.0]);


    // let node_for_x = Node::new_from_array(9999, &x, &normal);

    println!("finished creating nodes x1: {:?},\n x2: {:?},\n x3: {:?},\n now integrating over the sphere", x1, x2, x3);

    // iterate over all triangles and integrate only the kernal

    let mut final_val_1 = 0.0;
    let mut final_val_2 = 0.0;
    let mut final_val_3 = 0.0;
    for (trianlge_id, triangle_struct) in list_of_triangles {
        let triangle_nodes = triangle_struct.get_nodes();
        let mut triangle = A2::eye(3);
        let v1 = list_of_nodes.get(&triangle_nodes[0]).unwrap().lock().unwrap().get_coordinate();
        let v2 = list_of_nodes.get(&triangle_nodes[1]).unwrap().lock().unwrap().get_coordinate();
        let v3 = list_of_nodes.get(&triangle_nodes[2]).unwrap().lock().unwrap().get_coordinate();

        triangle.slice_mut(s![0,..]).assign(&v1);
        triangle.slice_mut(s![1,..]).assign(&v2);
        triangle.slice_mut(s![2,..]).assign(&v3);
        
        // write code to determine when to switch to geometric or duffy method for x2

        let (new_x, new_normal, new_triangle, analytic_val1) = evaluate_integral_analytically(&x1, &normal, &triangle, 0, 0, 0, &A2::eye(2));
        let (new_x, new_normal, new_triangle, analytic_val2) = evaluate_integral_analytically(&x2, &normal, &triangle, 0, 0, 0, &A2::eye(2));
        let (new_x, new_normal, new_triangle, analytic_val3) = evaluate_integral_analytically(&x3, &normal, &triangle, 0, 0, 0, &A2::eye(2));

        final_val_1 += analytic_val1;
        final_val_2 += analytic_val2;
        final_val_3 += analytic_val3;
    }
    println!("final values: {:?}, {:?}, {:?}", final_val_1, final_val_2, final_val_3);

}

// functions used when reading mesh from file ------------------------------------------------
fn area(nodes: &Vec<usize>, list_of_nodes: &HashMap<usize, Mutex<Node>>) -> f64 {
    // do not make nodes mutable as they should not be changed here
    let v0 = list_of_nodes.get(&nodes[0]).unwrap().lock().unwrap();
    let v1 = list_of_nodes.get(&nodes[1]).unwrap().lock().unwrap();
    let v2 = list_of_nodes.get(&nodes[2]).unwrap().lock().unwrap();

    let a = v1.get_coordinate() - v0.get_coordinate();
    let b = v2.get_coordinate() - v0.get_coordinate();
    let c = cross_product(&a, &b);
    let val = c.l2_dist(&A1::zeros(3)).unwrap();
    val/2.
}

fn get_triangle_normal(triangle_nodes: &Vec<usize>, node_normals_data: &Vec<Vec<f64>>, node_data: &Vec<Vec<f64>>) -> A1 {
    // find triangle normal (use average of node normals to determine direction)
    let normal1 = A1::from_vec(node_normals_data[triangle_nodes[0] - 1].to_vec());
    let normal2 = A1::from_vec(node_normals_data[triangle_nodes[1] - 1].to_vec());
    let normal3 = A1::from_vec(node_normals_data[triangle_nodes[2] - 1].to_vec());
    let average_normal = (normal1 + normal2 + normal3) / 3.;
    
    let node1 = A1::from_vec(node_data[triangle_nodes[0] - 1].to_vec());
    let node2 = A1::from_vec(node_data[triangle_nodes[1] - 1].to_vec());
    let node3 = A1::from_vec(node_data[triangle_nodes[2] - 1].to_vec());

    let d1 = node2 - &node1;
    let d2 = node3 - &node1;

    let cross = cross_product(&d1, &d2);
    let norm = cross.l2_dist(&A1::zeros(3)).expect("could not compute l2 normal for triangle normal");
    
    let direction = cross.dot(&average_normal);
    if direction < 0.0 {
        return -1.0/norm * cross;
    } else {
        return 1.0/norm * cross;
    }
}