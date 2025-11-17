
use std::f64::consts::PI;

use libm::{acos, asin, atan, atanh, cos, log, sin, sqrt, tan};

pub fn integrate_i0(phi_e: f64, phi_s: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64{
    let mut angle_difference = phi_e - phi_s;
    let branch_check = check_for_branch_cut(phi_e, phi_s, (radius_start + radius_end) / 2.0 , d_norm_e, d_norm_s, sign_e, sign_s);
    // println!("branch check: {:?}", &branch_check);
    if branch_check < 0.0 {
        angle_difference = angle_difference + 2.0 * PI;
    } else if branch_check > 2.0 * PI {
        angle_difference = angle_difference - 2.0 * PI;
    }
    //println!("angle difference: {:?}", &angle_difference);
    //println!("--- test: {:?}, {:?}", integral_1(c, radius_start, radius_end), integral_2(c, d_norm_e, radius_start, radius_end));
    let value = angle_difference * integral_1(c, radius_start, radius_end) 
    + sign_e * integral_2(c, d_norm_e, radius_start, radius_end) 
    - sign_s * integral_2(c, d_norm_s, radius_start, radius_end);
    value
}

pub fn integrate_ix(phi_e: f64, phi_s: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64 {
    //println!("--- test: {:?}", integral_3(c, d_norm_e, radius_start, radius_end));
    let value = (d_norm_e * sin(phi_e) - d_norm_s * sin(phi_s)) * integral_1(c, radius_start, radius_end)
    + sign_e * cos(phi_e) * integral_3(c, d_norm_e, radius_start, radius_end)
    - sign_s * cos(phi_s) * integral_3(c, d_norm_s, radius_start, radius_end);
    value
}

pub fn integrate_iy(phi_e: f64, phi_s: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64 {
    let value = (d_norm_s * cos(phi_s) - d_norm_e * cos(phi_e)) * integral_1(c, radius_start, radius_end)
    + sign_e * sin(phi_e) * integral_3(c, d_norm_e, radius_start, radius_end)
    - sign_s * sin(phi_s) * integral_3(c, d_norm_s, radius_start, radius_end);
    value
}

pub fn integrate_ixx(phi_e: f64, phi_s: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64 {
    let mut angle_difference = phi_e - phi_s;
    let branch_check = check_for_branch_cut(phi_e, phi_s, (radius_start + radius_end) / 2.0 , d_norm_e, d_norm_s, sign_e, sign_s);
    if branch_check < 0.0 {
        angle_difference = angle_difference + 2.0 * PI;
    } else if branch_check > 2.0 * PI {
        angle_difference = angle_difference - 2.0 * PI;
    }

    let value = (angle_difference - sin(2. * phi_e)/2. + sin(2. * phi_s)/2. ) * integral_4(c, radius_start, radius_end)
    + sign_e * integral_5(c, d_norm_e, radius_start, radius_end)
    - sign_s * integral_5(c, d_norm_s, radius_start, radius_end)
    + (-d_norm_s.powi(2) * sin(2. * phi_s) + d_norm_e.powi(2) * sin(2. * phi_e)) * integral_1(c, radius_start, radius_end)
    + sign_e*d_norm_e*cos(2. * phi_e) * integral_3(c, d_norm_e, radius_start, radius_end)
    - sign_s*d_norm_s*cos(2. * phi_s) * integral_3(c, d_norm_s, radius_start, radius_end);

    value / 2.
}

pub fn integrate_iyy(phi_e: f64, phi_s: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64 {
    let mut angle_difference = phi_e - phi_s;
    let branch_check = check_for_branch_cut(phi_e, phi_s, (radius_start + radius_end) / 2.0 , d_norm_e, d_norm_s, sign_e, sign_s);
    if branch_check < 0.0 {
        angle_difference = angle_difference + 2.0 * PI;
    } else if branch_check > 2.0 * PI {
        angle_difference = angle_difference - 2.0 * PI;
    }

    // println!("--- angle difference: {:?}, i4: {:?}, i5: {:?}", &angle_difference, integral_4(c, radius_start, radius_end), integral_5(c, d_norm_e, radius_start, radius_end));
    let value = (angle_difference + sin(2. * phi_e)/2. - sin(2. * phi_s)/2. ) * integral_4(c, radius_start, radius_end)
    + sign_e * integral_5(c, d_norm_e, radius_start, radius_end)
    - sign_s * integral_5(c, d_norm_s, radius_start, radius_end)
    + (d_norm_s.powi(2) * sin(2. * phi_s) - d_norm_e.powi(2) * sin(2. * phi_e)) * integral_1(c, radius_start, radius_end)
    - sign_e*d_norm_e*cos(2. * phi_e) * integral_3(c, d_norm_e, radius_start, radius_end)
    + sign_s*d_norm_s*cos(2. * phi_s) * integral_3(c, d_norm_s, radius_start, radius_end);

    value / 2.
}

pub fn integrate_ixy(phi_e: f64, phi_s: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64 {
    let value = (sin(phi_s).powi(2) - sin(phi_e).powi(2)) * integral_4(c, radius_start, radius_end)
    + (d_norm_s.powi(2)*cos(2.*phi_s) - d_norm_e.powi(2)*cos(2.*phi_e)) * integral_1(c, radius_start, radius_end)
    + sign_e*d_norm_e*sin(2. * phi_e) * integral_3(c, d_norm_e, radius_start, radius_end)
    - sign_s*d_norm_s*sin(2. * phi_s) * integral_3(c, d_norm_s, radius_start, radius_end);

    value / 2.
}

pub fn integrate_ixxx(phi_e: f64, phi_s: f64, c: f64, radius_e: f64, radius_s: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64 {
    let value = (sin(phi_e) - sin(phi_e)*cos(phi_e).powi(2))*d_norm_e * integral_6(c, d_norm_e, radius_s, radius_e)
        - (sin(phi_s) - sin(phi_s)*cos(phi_s).powi(2))*d_norm_s * integral_6(c, d_norm_s, radius_s, radius_e)
        + sign_e * cos(phi_e) * integral_7(c, d_norm_e, radius_s, radius_e)
        - sign_s * cos(phi_s) * integral_7(c, d_norm_s, radius_s, radius_e)
        + (sin(phi_e)*cos(phi_e).powi(2) - sin(phi_e).powi(3) / 3.0)*d_norm_e.powi(3) * integral_1(c, radius_s, radius_e)
        - (sin(phi_s)*cos(phi_s).powi(2) - sin(phi_s).powi(3) / 3.0)*d_norm_s.powi(3) * integral_1(c, radius_s, radius_e)
        - sign_e * sin(phi_e).powi(2) * cos(phi_e) * d_norm_e.powi(2) * integral_3(c, d_norm_e, radius_s, radius_e)
        + sign_s * sin(phi_s).powi(2) * cos(phi_s) * d_norm_s.powi(2) * integral_3(c, d_norm_s, radius_s, radius_e)
        - sign_e * cos(phi_e).powi(3) * integral_8(c, d_norm_e, radius_s, radius_e) / 3.0
        + sign_s * cos(phi_s).powi(3) * integral_8(c, d_norm_s, radius_s, radius_e) / 3.0;
    value
}

pub fn integrate_iyyy(phi_e: f64, phi_s: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64 {
    let value = (cos(phi_e)*sin(phi_e).powi(2) - cos(phi_e))*d_norm_e * integral_6(c, d_norm_e, radius_start, radius_end)
        - (cos(phi_s)*sin(phi_s).powi(2) - cos(phi_s))*d_norm_s * integral_6(c, d_norm_s, radius_start, radius_end)
        - sign_e * sin(phi_e) * integral_7(c, d_norm_e, radius_start, radius_end)
        + sign_s * sin(phi_s) * integral_7(c, d_norm_s, radius_start, radius_end)
        + (cos(phi_e).powi(3)/3.0 - cos(phi_e)*sin(phi_e).powi(2))*d_norm_e.powi(3) * integral_1(c, radius_start, radius_end)
        - (cos(phi_s).powi(3)/3.0 - cos(phi_s)*sin(phi_s).powi(2))*d_norm_s.powi(3) * integral_1(c, radius_start, radius_end)
        + sign_e * cos(phi_e).powi(2) * sin(phi_e) * d_norm_e.powi(2) * integral_3(c, d_norm_e, radius_start, radius_end)
        - sign_s * cos(phi_s).powi(2) * sin(phi_s) * d_norm_s.powi(2) * integral_3(c, d_norm_s, radius_start, radius_end)
        + sign_e * sin(phi_e).powi(3) * integral_8(c, d_norm_e, radius_start, radius_end) / 3.0
        - sign_s * sin(phi_s).powi(3) * integral_8(c, d_norm_s, radius_start, radius_end) / 3.0;
    value
}

pub fn integrate_ixxy(phi_e: f64, phi_s: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64 {
    let value = -cos(phi_e)*sin(phi_e).powi(2)*d_norm_e * integral_6(c, d_norm_e, radius_start, radius_end)
        + cos(phi_s)*sin(phi_s).powi(2)*d_norm_s * integral_6(c, d_norm_s, radius_start, radius_end)
        - (cos(phi_e).powi(3)/3.0 - cos(phi_e)*sin(phi_e).powi(2))*d_norm_e.powi(3) * integral_1(c, radius_start, radius_end)
        + (cos(phi_s).powi(3)/3.0 - cos(phi_s)*sin(phi_s).powi(2))*d_norm_s.powi(3) * integral_1(c, radius_start, radius_end)
        - sign_e * cos(phi_e).powi(2) * sin(phi_e) * d_norm_e.powi(2) * integral_3(c, d_norm_e, radius_start, radius_end)
        + sign_s * cos(phi_s).powi(2) * sin(phi_s) * d_norm_s.powi(2) * integral_3(c, d_norm_s, radius_start, radius_end)
        - sign_e * sin(phi_e).powi(3) * integral_8(c, d_norm_e, radius_start, radius_end) / 3.0
        + sign_s * sin(phi_s).powi(3) * integral_8(c, d_norm_s, radius_start, radius_end) / 3.0;
    value
}

pub fn integrate_ixyy(phi_e: f64, phi_s: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64 {
    let value = sin(phi_e)*cos(phi_e).powi(2)*d_norm_e * integral_6(c, d_norm_e, radius_start, radius_end)
        - sin(phi_s)*cos(phi_s).powi(2)*d_norm_s * integral_6(c, d_norm_s, radius_start, radius_end)
        - (sin(phi_e)*cos(phi_e).powi(2) - sin(phi_e).powi(3) / 3.0)*d_norm_e.powi(3) * integral_1(c, radius_start, radius_end)
        + (sin(phi_s)*cos(phi_s).powi(2) - sin(phi_s).powi(3) / 3.0)*d_norm_s.powi(3) * integral_1(c, radius_start, radius_end)
        + sign_e * sin(phi_e).powi(2) * cos(phi_e) * d_norm_e.powi(2) * integral_3(c, d_norm_e, radius_start, radius_end)
        - sign_s * sin(phi_s).powi(2) * cos(phi_s) * d_norm_s.powi(2) * integral_3(c, d_norm_s, radius_start, radius_end)
        + sign_e * cos(phi_e).powi(3) * integral_8(c, d_norm_e, radius_start, radius_end) / 3.0
        - sign_s * cos(phi_s).powi(3) * integral_8(c, d_norm_s, radius_start, radius_end) / 3.0;
    value
}


fn integral_1(c: f64, radius_start: f64, radius_end: f64) -> f64 {
    if c == 0.0 {
        let val_end = - 1.0 / radius_end;
        let val_start = - 1.0 / radius_start;
        val_end - val_start
    } else {
        let val_end = - 1.0 / (radius_end.powi(2) + c.powi(2)).sqrt();
        let val_start = - 1.0 / (radius_start.powi(2) + c.powi(2)).sqrt();
        val_end - val_start
    }
}

fn integral_2(c: f64, d_norm: f64, radius_start: f64, radius_end: f64) -> f64 {
    if d_norm == 0.0 {
        return PI * integral_1(c, radius_start, radius_end) / 2.;
    } else {
        if c == 0.0 {
            let val_end = - acos(d_norm / radius_end) / radius_end + sqrt(radius_end.powi(2) - d_norm.powi(2)) / (radius_end * d_norm);
            let val_start = - acos(d_norm / radius_start) / radius_start + sqrt(radius_start.powi(2) - d_norm.powi(2)) / (radius_start * d_norm);

            return val_end - val_start;
        } else {
            let val_end = -acos(d_norm / radius_end) / (radius_end.powi(2) + c.powi(2)).sqrt() + atan(c * ((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))).sqrt() / d_norm ) / c;
            let val_start = -acos(d_norm / radius_start) / (radius_start.powi(2) + c.powi(2)).sqrt() + atan(c * ((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))).sqrt() / d_norm ) / c;

            return val_end - val_start;
        }
    }
}

fn integral_3(c: f64, d_norm: f64, radius_start: f64, radius_end: f64) -> f64 {
    let val_end;
    let val_start;
    if c == 0.0 {
        if d_norm != 0.0 {
            val_end = -sqrt(radius_end.powi(2) - d_norm.powi(2)) / radius_end + log((sqrt(radius_end.powi(2) - d_norm.powi(2)) + radius_end) / d_norm);
            val_start = -sqrt(radius_start.powi(2) - d_norm.powi(2)) / radius_start + log((sqrt(radius_start.powi(2) - d_norm.powi(2)) + radius_start) / d_norm);
        } else {
            return log(radius_end) - log(radius_start);
        }
    } else {
        val_end = log(2.0 * (radius_end.powi(2) + c.powi(2)).sqrt() * (radius_end.powi(2) - d_norm.powi(2)).sqrt() + c.powi(2) - d_norm.powi(2) + 2.0 * radius_end.powi(2)) / 2. 
            - ((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))).sqrt();

        val_start = log(2.0 * (radius_start.powi(2) + c.powi(2)).sqrt() * (radius_start.powi(2) - d_norm.powi(2)).sqrt() + c.powi(2) - d_norm.powi(2) + 2.0 * radius_start.powi(2)) / 2.
            - ((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))).sqrt();
    }
    
    if !val_end.is_finite() || !val_start.is_finite() {
        println!("c = {:?}, d = {:?}, r = [{:?}, {:?}]", c, d_norm, radius_start, radius_end);
        panic!("error with i3");
    }

    return val_end - val_start;
}

fn integral_4(c: f64, radius_start: f64, radius_end: f64) -> f64 {
    if c == 0.0 {
        return radius_end - radius_start;
    } else {
        let val_end = (radius_end.powi(2) + 2.0 * c.powi(2)) / (radius_end.powi(2) + c.powi(2)).sqrt();
        let val_start = (radius_start.powi(2) + 2.0 * c.powi(2)) / (radius_start.powi(2) + c.powi(2)).sqrt();
        return val_end - val_start;
    }
}

fn integral_5(c: f64, d_norm: f64, radius_start: f64, radius_end: f64) -> f64 {
    let val_end;
    let val_start;
    if d_norm == 0.0 {
        return PI * integral_4(c, radius_start, radius_end) / 2.;
    } else if d_norm < 1e-7 {
        // if d is very small
        if c == 0.0 {
            val_end = radius_end * acos(d_norm / radius_end) - d_norm * log((sqrt(radius_end.powi(2) - d_norm.powi(2)) + radius_end) / d_norm);
            val_start = radius_start * acos(d_norm / radius_start) - d_norm * log((sqrt(radius_start.powi(2) - d_norm.powi(2)) + radius_start) / d_norm);
        } else if c.abs() < 1e-8 {
            let temp_val_end = (radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2));
            let temp_val_start = (radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2));

            // use taylor expansion to approximate atanh
            val_end = -2.0 * c * atan(c * ((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))).sqrt() / d_norm ) 
            + (radius_end.powi(2) + 2.0*c.powi(2)) * acos(d_norm / radius_end) / (radius_end.powi(2) + c.powi(2)).sqrt()
            - d_norm * (temp_val_end.powf(0.5) + temp_val_end.powf(1.5) + temp_val_end.powf(2.5)); 
            
            val_start = -2.0 * c * atan(c * ((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))).sqrt() / d_norm ) 
            + (radius_start.powi(2) + 2.0*c.powi(2)) * acos(d_norm / radius_start) / (radius_start.powi(2) + c.powi(2)).sqrt()
            - d_norm * (temp_val_start.powf(0.5) + temp_val_start.powf(1.5) + temp_val_start.powf(2.5));

        } else {
            val_end = -2.0 * c * atan(c * ((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))).sqrt() / d_norm ) 
            + (radius_end.powi(2) + 2.0*c.powi(2)) * acos(d_norm / radius_end) / (radius_end.powi(2) + c.powi(2)).sqrt() 
            - d_norm * atanh(((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))).sqrt());
            val_start = -2.0 * c * atan(c * ((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))).sqrt() / d_norm ) 
            + (radius_start.powi(2) + 2.0*c.powi(2)) * acos(d_norm / radius_start) / (radius_start.powi(2) + c.powi(2)).sqrt() 
            - d_norm * atanh(((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))).sqrt());
        }
    } else {
        if c == 0.0 {
            val_end = radius_end * acos(d_norm / radius_end);
            val_start = radius_start * acos(d_norm / radius_start);
        } else {
            val_end = -2.0 * c * atan(c * ((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))).sqrt() / d_norm ) 
            + (radius_end.powi(2) + 2.0*c.powi(2)) * acos(d_norm / radius_end) / (radius_end.powi(2) + c.powi(2)).sqrt() 
            - d_norm * atanh(((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))).sqrt());
            val_start = -2.0 * c * atan(c * ((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))).sqrt() / d_norm ) 
            + (radius_start.powi(2) + 2.0*c.powi(2)) * acos(d_norm / radius_start) / (radius_start.powi(2) + c.powi(2)).sqrt() 
            - d_norm * atanh(((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))).sqrt());
        }
    }
    
    if !val_end.is_finite() || !val_start.is_finite() {
        println!("c = {:?}, d = {:?}, r = [{:?}, {:?}]", c, d_norm, radius_start, radius_end);
        panic!("error with i5");
    }

    return val_end - val_start;
}

fn integral_6(c: f64, _d_norm: f64, radius_start: f64, radius_end: f64) -> f64 {
    if c == 0.0 {
        return radius_end - radius_start;
    } else {
        let val_end = (radius_end.powi(2) + 2. * c.powi(2)) / sqrt(radius_end.powi(2) + c.powi(2));
        let val_start = (radius_start.powi(2) + 2. * c.powi(2)) / sqrt(radius_start.powi(2) + c.powi(2));
        return val_end - val_start;
    }
}

fn integral_7(c: f64, d_norm: f64, radius_s: f64, radius_e: f64) -> f64 {
    let val_s;
    let val_e;
    if c == 0.0 {
        val_e = radius_e * sqrt(radius_e.powi(2) - d_norm.powi(2)) / 2.0 - d_norm.powi(2) * log(sqrt(radius_e.powi(2) - d_norm.powi(2)) + radius_e) / 2.0;
        val_s = radius_s * sqrt(radius_s.powi(2) - d_norm.powi(2)) / 2.0 - d_norm.powi(2) * log(sqrt(radius_s.powi(2) - d_norm.powi(2)) + radius_s) / 2.0;
    } else {
        let r_e_square = radius_e.powi(2);
        
        let c_square = c.powi(2);
        let d_square = d_norm.powi(2);
        
        val_e = (r_e_square + 3.0 * c_square) * sqrt(r_e_square - d_square) / (2.0 * sqrt(r_e_square + c_square))
            - (3.0 * c_square + d_square) * log(c_square - d_square + 2.0 * r_e_square + 2.0 * sqrt((r_e_square + c_square) * (r_e_square - d_square))) / 4.0;
        
        let r_s_square = radius_s.powi(2);
        val_s = (r_s_square + 3.0 * c_square) * sqrt(r_s_square - d_square) / (2.0 * sqrt(r_s_square + c_square))
            - (3.0 * c_square + d_square) * log(c_square - d_square + 2.0 * r_s_square + 2.0 * sqrt((r_s_square + c_square) * (r_s_square - d_square))) / 4.0;
        
    }
    return val_e - val_s;
}

fn integral_8(c: f64, d_norm: f64, radius_s: f64, radius_e: f64) -> f64 {
    let val_s;
    let val_e;

    let d_square = d_norm.powi(2);
    if c == 0.0 {
        val_e = (d_square / radius_e + radius_e / 2.0) * sqrt(radius_e.powi(2) - d_square) - 3.0 * d_square * log(sqrt(radius_e.powi(2) - d_square) + radius_e)/ 2.0;
        val_s = (d_square / radius_s + radius_s / 2.0) * sqrt(radius_s.powi(2) - d_square) - 3.0 * d_square * log(sqrt(radius_s.powi(2) - d_square) + radius_s)/ 2.0;
    } else {
        let c_square = c.powi(2);
        val_e = sqrt(radius_e.powi(2) - d_square) * (3.0 * c_square + 2.0 * d_square + radius_e.powi(2)) / (2.0 * sqrt(radius_e.powi(2) + c_square))
            - (3.0/4.0) * (c_square + d_square) * log(2.0 * sqrt((radius_e.powi(2) + c_square) * (radius_e.powi(2) - d_square)) + c_square - d_square + 2.0 * radius_e.powi(2));
        
        val_s = sqrt(radius_s.powi(2) - d_square) * (3.0 * c_square + 2.0 * d_square + radius_s.powi(2)) / (2.0 * sqrt(radius_s.powi(2) + c_square))
            - (3.0/4.0) * (c_square + d_square) * log(2.0 * sqrt((radius_s.powi(2) + c_square) * (radius_s.powi(2) - d_square)) + c_square - d_square + 2.0 * radius_s.powi(2));       
    }
    return val_e - val_s;
}


// integrals for Greens function
pub fn integrate_j0(phi_e: f64, phi_s: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64{
    let mut angle_difference = phi_e - phi_s;
    let branch_check = check_for_branch_cut(phi_e, phi_s, (radius_start + radius_end) / 2.0 , d_norm_e, d_norm_s, sign_e, sign_s);
    if branch_check < 0.0 {
        angle_difference = angle_difference + 2.0 * PI;
    } else if branch_check > 2.0 * PI {
        angle_difference = angle_difference - 2.0 * PI;
    }

    let value = angle_difference * integral_j1(c, radius_start, radius_end) 
    + sign_e * integral_j2(c, d_norm_e, radius_start, radius_end) 
    - sign_s * integral_j2(c, d_norm_s , radius_start, radius_end);
    value
}

pub fn integrate_jx(phi_e: f64, phi_s: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64{

    let value = (d_norm_e*sin(phi_e) - d_norm_s*sin(phi_s)) * integral_j1(c, radius_start, radius_end) 
    + sign_e * cos(phi_e)* integral_j3(c, d_norm_e, radius_start, radius_end) 
    - sign_s * cos(phi_s) * integral_j3(c, d_norm_s , radius_start, radius_end);
    value
}

pub fn integrate_jy(phi_e: f64, phi_s: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64{

    let value = - (d_norm_e*cos(phi_e) - d_norm_s*cos(phi_s)) * integral_j1(c, radius_start, radius_end) 
    + sign_e * sin(phi_e)* integral_j3(c, d_norm_e, radius_start, radius_end) 
    - sign_s * sin(phi_s) * integral_j3(c, d_norm_s , radius_start, radius_end);
    value
}

fn integral_j1(c: f64, radius_start: f64, radius_end: f64) -> f64 {
    let val;
    if c == 0.0 {
        val = radius_end - radius_start;
    } else {
        let val_end = sqrt(radius_end.powi(2) + c.powi(2));
        let val_start = sqrt(radius_start.powi(2) + c.powi(2));
        val = val_end - val_start;
    }
    if !val.is_finite() {
        println!("j1 cannot be integrated, rs: {:?}, re: {:?}, c: {:?}", radius_start, radius_end, c);
    }
    val
}

fn integral_j2(c: f64, d_norm: f64, radius_start: f64, radius_end: f64) -> f64 {
    let val_end;
    let val_start;

    if c.abs() < 1e-15 {
        if d_norm == 0.0  {
            return PI * integral_j1(c, radius_start, radius_end) / 2.;
        } else if d_norm.abs() < 1e-8 {
            let temp_val_end = -(sqrt(radius_end.powi(2) - d_norm.powi(2)) - radius_end) / (sqrt(radius_end.powi(2) - d_norm.powi(2)) + radius_end);
            let temp_val_start = -(sqrt(radius_start.powi(2) - d_norm.powi(2)) - radius_start) / (sqrt(radius_start.powi(2) - d_norm.powi(2)) + radius_start);

            // aproximate log using its taylor expansion at 1
            val_end = radius_end * acos(d_norm / radius_end)
                + d_norm/2. * ((temp_val_end - 1.) - (temp_val_end - 1.).powi(2) / 2. + (temp_val_end - 1.).powi(3) / 3. - (temp_val_end - 1.).powi(4) / 4.);
            val_start = radius_start * acos(d_norm / radius_start)
                + d_norm/2. * ((temp_val_start - 1.) - (temp_val_start - 1.).powi(2) / 2. + (temp_val_start - 1.).powi(3) / 3. - (temp_val_start - 1.).powi(4) / 4.);
        } else {
            val_end = radius_end * acos(d_norm / radius_end) 
                + d_norm/2. * log(-(sqrt(radius_end.powi(2) - d_norm.powi(2)) - radius_end) / (sqrt(radius_end.powi(2) - d_norm.powi(2)) + radius_end));
            val_start = radius_start * acos(d_norm / radius_start) 
                + d_norm/2. * log(-(sqrt(radius_start.powi(2) - d_norm.powi(2)) - radius_start) / (sqrt(radius_start.powi(2) - d_norm.powi(2)) + radius_start));
        }
    } else {
        if d_norm == 0.0 {
            return PI * integral_j1(c, radius_start, radius_end) / 2.;
        } else {    // TODO: add the taylor expansion for atanh
            val_end = sqrt(radius_end.powi(2) + c.powi(2)) * acos(d_norm / radius_end) 
            - c * asin(c * sqrt((radius_end.powi(2) - d_norm.powi(2)) / radius_end.powi(2)) / sqrt(c.powi(2) + d_norm.powi(2))) 
            - taylor_expansion_for_atanh(radius_end, c, d_norm);
            if !val_end.is_finite() {
                println!("error in computing j2 end, {:?}, {:?}, {:?}", sqrt(radius_end.powi(2) + c.powi(2)) * acos(d_norm / radius_end), 
                    c * asin(c * sqrt((radius_end.powi(2) - d_norm.powi(2)) / radius_end.powi(2)) / sqrt(c.powi(2) + d_norm.powi(2))), 
                    d_norm * atanh(sqrt((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2)))));
                    println!("c: {:?}, d: {:?}, radius: [{:?}, {:?}]", c, d_norm, radius_start, radius_end);
                    panic!("something is not finite")
            }
            val_start = sqrt(radius_start.powi(2) + c.powi(2)) * acos(d_norm / radius_start) 
            - c * asin(c * sqrt((radius_start.powi(2) - d_norm.powi(2)) / radius_start.powi(2)) / sqrt(c.powi(2) + d_norm.powi(2))) 
            - taylor_expansion_for_atanh(radius_start, c, d_norm);

            if !val_start.is_finite() {
                println!("error in computing j2 start, {:?}, {:?}, {:?}", sqrt(radius_start.powi(2) + c.powi(2)) * acos(d_norm / radius_start), 
                c * asin(c * sqrt((radius_start.powi(2) - d_norm.powi(2)) / radius_start.powi(2)) / sqrt(c.powi(2) + d_norm.powi(2))), 
                d_norm * atanh(sqrt((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2)))));
                println!("c: {:?}, d: {:?}, radius: [{:?}, {:?}]", c, d_norm, radius_start, radius_end);
                println!("sqrt = {:?}, acos = {:?}", sqrt(radius_start.powi(2) + c.powi(2)), acos(d_norm / radius_start));
                panic!("something is not finite")
            }
        }
    }

    if !val_end.is_finite() || !val_start.is_finite() {
        panic!("error with j2");
    }

    return val_end - val_start;
}

fn taylor_expansion_for_atanh(r: f64, c: f64, d: f64) -> f64 {
    let val = sqrt((r.powi(2) - d.powi(2)) / (r.powi(2) + c.powi(2)));
    if val == 1.0 {
        return 0.0;
    } else {
        return d * atanh(val);
    }
}

fn integral_j3(c: f64,  d_norm: f64, radius_start: f64, radius_end: f64) -> f64 {
    let val_end;
    let val_start;
    if c.abs() <= 1e-8 {
        if d_norm == 0.0 {
            val_end = radius_end.powi(2);
            val_start = radius_start.powi(2);
        } else if d_norm.abs() <= 1e-8 {
            let temp_val_end = radius_end - sqrt(radius_end.powi(2) - d_norm.powi(2));
            let temp_val_start = radius_start - sqrt(radius_start.powi(2) - d_norm.powi(2));

            // use taylor expansion to approximate log
            val_end = radius_end * sqrt(radius_end.powi(2) - d_norm.powi(2)) 
                + d_norm.powi(2) * ((temp_val_end - 1.) - (temp_val_end - 1.).powi(2) / 2. + (temp_val_end - 1.).powi(3) / 3. - (temp_val_end - 1.).powi(4) / 4.);
            val_start = radius_start * sqrt(radius_start.powi(2) - d_norm.powi(2)) 
                + d_norm.powi(2) * ((temp_val_start - 1.) - (temp_val_start - 1.).powi(2) / 2. + (temp_val_start - 1.).powi(3) / 3. - (temp_val_start - 1.).powi(4) / 4.);
        } else {
            val_end = radius_end * sqrt(radius_end.powi(2) - d_norm.powi(2)) + d_norm.powi(2) * log((-radius_end + sqrt(radius_end.powi(2) - d_norm.powi(2))).abs());
            val_start = radius_start * sqrt(radius_start.powi(2) - d_norm.powi(2)) + d_norm.powi(2) * log((-radius_start + sqrt(radius_start.powi(2) - d_norm.powi(2))).abs());
        }
    } else if c.abs() < 1e-8 {
        
        // use taylor expansion to approximate log
        let temp_val_end = sqrt(radius_end.powi(2) + c.powi(2)) -  sqrt(radius_end.powi(2) - d_norm.powi(2));
        let temp_val_start = sqrt(radius_start.powi(2) + c.powi(2)) -  sqrt(radius_start.powi(2) - d_norm.powi(2));

        let log_temp_val_end = (temp_val_end - 1.) - (temp_val_end - 1.).powi(2) / 2. + (temp_val_end - 1.).powi(3) / 3. - (temp_val_end - 1.).powi(4) / 4.;
        let log_temp_val_start = (temp_val_start - 1.) - (temp_val_start - 1.).powi(2) / 2. + (temp_val_start - 1.).powi(3) / 3. - (temp_val_start - 1.).powi(4) / 4.;

        val_end = sqrt(radius_end.powi(2) + c.powi(2)) * sqrt(radius_end.powi(2) - d_norm.powi(2)) + (c.powi(2) + d_norm.powi(2)) * log_temp_val_end;
        val_start = sqrt(radius_start.powi(2) + c.powi(2)) * sqrt(radius_start.powi(2) - d_norm.powi(2)) + (c.powi(2) + d_norm.powi(2)) * log_temp_val_start;
    } else {
        val_end = sqrt(radius_end.powi(2) + c.powi(2)) * sqrt(radius_end.powi(2) - d_norm.powi(2)) + (c.powi(2) + d_norm.powi(2)) * log(sqrt(radius_end.powi(2) + c.powi(2)) -  sqrt(radius_end.powi(2) - d_norm.powi(2)));
        val_start = sqrt(radius_start.powi(2) + c.powi(2)) * sqrt(radius_start.powi(2) - d_norm.powi(2)) + (c.powi(2) + d_norm.powi(2)) * log(sqrt(radius_start.powi(2) + c.powi(2)) -  sqrt(radius_start.powi(2) - d_norm.powi(2)));
    }

    if !val_end.is_finite() || !val_start.is_finite() {
        panic!("error with j3");
    }
    return (val_end - val_start) / 2.0 ;
}

fn check_for_branch_cut(phi_e: f64, phi_s: f64, radius: f64, d_norm_e: f64, d_norm_s: f64, sign_e: f64, sign_s: f64) -> f64 {
    let start_angle;
    if d_norm_s == 0.0 {
        start_angle = phi_s + sign_s * PI / 2.;
    } else {
        start_angle = phi_s + sign_s * acos(d_norm_s / radius);
    }
    
    let end_angle;
    if d_norm_e == 0.0 {
        end_angle = phi_e + sign_e * PI / 2.;
    } else {
        end_angle = phi_e + sign_e * acos(d_norm_e / radius);
    }
    //println!("end angle: {:?}, start angle: {:?}", &end_angle, &start_angle);
    return end_angle - start_angle
}


// ------------------ integrals for geometric integral -----------------------
// pub fn geometric_0(theta_2: f64, theta_end: f64) -> f64 {
//     let val_end = sin(theta_end - theta_2) - sin(2.0 * theta_2) * log(tan((theta_end + theta_2) / 2.)) / 2.;
//     let val_start = sin(- theta_2) - sin(2.0 * theta_2) * log(tan(theta_2 / 2.)) / 2.;
//     val_end - val_start
// }

pub fn qsa_xx(theta_2: f64, theta_end: f64) -> f64 {
    let val_end = cos(theta_end - theta_2) + cos(theta_2).powi(2) * log(tan((theta_end + theta_2) / 2.));
    let val_start = cos(- theta_2) + cos(theta_2).powi(2) * log(tan((theta_2) / 2.));
    val_end - val_start
}

pub fn qsa_yy(theta_2: f64, theta_end: f64) -> f64 {
    let val_end = -cos(theta_end - theta_2) + sin(theta_2).powi(2) * log(tan((theta_end + theta_2) / 2.));
    let val_start = -cos(- theta_2) + sin(theta_2).powi(2) * log(tan((theta_2) / 2.));
    val_end - val_start
}

pub fn qsa_xxy(theta_2: f64, theta_end: f64) -> f64 {
    let val_end = -2.*(cos(theta_2) + 3.*cos(3.*theta_2)) * atanh(cos(theta_2) - sin(theta_2)*tan(theta_end/2.))
        + (2.*sin(2.*theta_end - theta_2) + sin(theta_2) + 3.*sin(3.*theta_2)) / sin(theta_end + theta_2);
    let val_start = -2.*(cos(theta_2) + 3.*cos(3.*theta_2)) * atanh(cos(theta_2))
        + (2.*sin( - theta_2) + sin(theta_2) + 3.*sin(3.*theta_2)) / sin(theta_2);
    (val_end - val_start) / 4.
}

pub fn qsa_xyy(theta_2: f64, theta_end: f64) -> f64 {
    let val_end = -2.*(sin(theta_2) - 3.*sin(3.*theta_2)) * atanh(cos(theta_2) - sin(theta_2)*tan(theta_end/2.))
        - (2.*cos(2.*theta_end - theta_2) + cos(theta_2) - 3.*cos(3.*theta_2)) / sin(theta_end + theta_2);
    let val_start = -2.*(sin(theta_2) - 3.*sin(3.*theta_2)) * atanh(cos(theta_2))
        - (2.*cos( - theta_2) + cos(theta_2) - 3.*cos(3.*theta_2)) / sin(theta_2);
    (val_end - val_start) / 4.
}

pub fn qsa_xxx(theta_2: f64, theta_end: f64) -> f64 {
    let val_end = -cos(theta_end - 2.*theta_2) - 6.*cos(theta_2)*sin(theta_2).powi(2)*atanh(cos(theta_2) - sin(theta_2)*tan(theta_end/2.))
        + sin(theta_2).powi(3) / sin(theta_end + theta_2);
    let val_start = -cos( - 2.*theta_2) - 6.*cos(theta_2)*sin(theta_2).powi(2)*atanh(cos(theta_2))
        + sin(theta_2).powi(3) / sin(theta_2);
    val_end - val_start
}

pub fn qsa_yyy(theta_2: f64, theta_end: f64) -> f64 {
    let val_end = -sin(theta_end - 2.*theta_2) - 6.*cos(theta_2).powi(2)*sin(theta_2)*atanh(cos(theta_2) - sin(theta_2)*tan(theta_end/2.))
        - cos(theta_2).powi(3) / sin(theta_end + theta_2);
    let val_start = -sin( - 2.*theta_2) - 6.*cos(theta_2).powi(2)*sin(theta_2)*atanh(cos(theta_2))
        - cos(theta_2).powi(3) / sin(theta_2);
    val_end - val_start
}
