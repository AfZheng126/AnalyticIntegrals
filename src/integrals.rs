
use std::f64::consts::PI;

use libm::{acos, asin, atan, atanh, cos, log, sin, sqrt, tan};

pub(crate) fn integrate_i0(phi_end: f64, phi_start: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_end: f64, d_norm_start: f64, sign_end: f64, sign_start: f64) -> f64{
    let mut angle_difference = phi_end - phi_start;
    let branch_check = check_for_branch_cut(phi_end, phi_start, (radius_start + radius_end) / 2.0 , d_norm_end, d_norm_start, sign_end, sign_start);
    if branch_check < 0.0 {
        angle_difference = angle_difference + 2.0 * PI;
    } else if branch_check > 2.0 * PI {
        angle_difference = angle_difference - 2.0 * PI;
    }
    let value = angle_difference * integral_1(c, radius_start, radius_end) 
    + sign_end * integral_2(c, d_norm_end, radius_start, radius_end) 
    - sign_start * integral_2(c, d_norm_start, radius_start, radius_end);
    value
}

pub(crate) fn integrate_ix(phi_end: f64, phi_start: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_end: f64, d_norm_start: f64, sign_end: f64, sign_start: f64) -> f64 {
    //println!("--- test: {:?}", integral_3(c, d_norm_end, radius_start, radius_end));
    let value = (d_norm_end * sin(phi_end) - d_norm_start * sin(phi_start)) * integral_1(c, radius_start, radius_end)
    + sign_end * cos(phi_end) * integral_3(c, d_norm_end, radius_start, radius_end)
    - sign_start * cos(phi_start) * integral_3(c, d_norm_start, radius_start, radius_end);
    value
}

pub(crate) fn integrate_iy(phi_end: f64, phi_start: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_end: f64, d_norm_start: f64, sign_end: f64, sign_start: f64) -> f64 {
    let value = (d_norm_start * cos(phi_start) - d_norm_end * cos(phi_end)) * integral_1(c, radius_start, radius_end)
    + sign_end * sin(phi_end) * integral_3(c, d_norm_end, radius_start, radius_end)
    - sign_start * sin(phi_start) * integral_3(c, d_norm_start, radius_start, radius_end);
    value
}

pub(crate) fn integrate_ixx(phi_end: f64, phi_start: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_end: f64, d_norm_start: f64, sign_end: f64, sign_start: f64) -> f64 {
    let mut angle_difference = phi_end - phi_start;
    let branch_check = check_for_branch_cut(phi_end, phi_start, (radius_start + radius_end) / 2.0 , d_norm_end, d_norm_start, sign_end, sign_start);
    if branch_check < 0.0 {
        angle_difference = angle_difference + 2.0 * PI;
    } else if branch_check > 2.0 * PI {
        angle_difference = angle_difference - 2.0 * PI;
    }

    let value = (angle_difference - sin(2. * phi_end)/2. + sin(2. * phi_start)/2. ) * integral_4(c, radius_start, radius_end)
    + sign_end * integral_5(c, d_norm_end, radius_start, radius_end)
    - sign_start * integral_5(c, d_norm_start, radius_start, radius_end)
    + (-d_norm_start.powi(2) * sin(2. * phi_start) + d_norm_end.powi(2) * sin(2. * phi_end)) * integral_1(c, radius_start, radius_end)
    + sign_end*d_norm_end*cos(2. * phi_end) * integral_3(c, d_norm_end, radius_start, radius_end)
    - sign_start*d_norm_start*cos(2. * phi_start) * integral_3(c, d_norm_start, radius_start, radius_end);

    value / 2.
}

pub(crate) fn integrate_iyy(phi_end: f64, phi_start: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_end: f64, d_norm_start: f64, sign_end: f64, sign_start: f64) -> f64 {
    let mut angle_difference = phi_end - phi_start;
    let branch_check = check_for_branch_cut(phi_end, phi_start, (radius_start + radius_end) / 2.0 , d_norm_end, d_norm_start, sign_end, sign_start);
    if branch_check < 0.0 {
        angle_difference = angle_difference + 2.0 * PI;
    } else if branch_check > 2.0 * PI {
        angle_difference = angle_difference - 2.0 * PI;
    }

    // println!("--- angle difference: {:?}, i4: {:?}, i5: {:?}", &angle_difference, integral_4(c, radius_start, radius_end), integral_5(c, d_norm_end, radius_start, radius_end));
    let value = (angle_difference + sin(2. * phi_end)/2. - sin(2. * phi_start)/2. ) * integral_4(c, radius_start, radius_end)
    + sign_end * integral_5(c, d_norm_end, radius_start, radius_end)
    - sign_start * integral_5(c, d_norm_start, radius_start, radius_end)
    + (d_norm_start.powi(2) * sin(2. * phi_start) - d_norm_end.powi(2) * sin(2. * phi_end)) * integral_1(c, radius_start, radius_end)
    - sign_end*d_norm_end*cos(2. * phi_end) * integral_3(c, d_norm_end, radius_start, radius_end)
    + sign_start*d_norm_start*cos(2. * phi_start) * integral_3(c, d_norm_start, radius_start, radius_end);

    value / 2.
}

pub(crate) fn integrate_ixy(phi_end: f64, phi_start: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_end: f64, d_norm_start: f64, sign_end: f64, sign_start: f64) -> f64 {
    let value = (sin(phi_start).powi(2) - sin(phi_end).powi(2)) * integral_4(c, radius_start, radius_end)
    + (d_norm_start.powi(2)*cos(2.*phi_start) - d_norm_end.powi(2)*cos(2.*phi_end)) * integral_1(c, radius_start, radius_end)
    + sign_end*d_norm_end*sin(2. * phi_end) * integral_3(c, d_norm_end, radius_start, radius_end)
    - sign_start*d_norm_start*sin(2. * phi_start) * integral_3(c, d_norm_start, radius_start, radius_end);

    value / 2.
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
    if c == 0.0 {
        if d_norm == 0.0 {
            let val_end = -sqrt(radius_end.powi(2) - d_norm.powi(2)) / radius_end + log((sqrt(radius_end.powi(2) - d_norm.powi(2)) + radius_end) / d_norm);
            let val_start = -sqrt(radius_start.powi(2) - d_norm.powi(2)) / radius_start + log((sqrt(radius_start.powi(2) - d_norm.powi(2)) + radius_start) / d_norm);
            return val_end - val_start;
        } else {
            return log(radius_end) - log(radius_start);
        }
    } else {
        let val_end = log(2.0 * (radius_end.powi(2) + c.powi(2)).sqrt() * (radius_end.powi(2) - d_norm.powi(2)).sqrt() + c.powi(2) - d_norm.powi(2) + 2.0 * radius_end.powi(2)) / 2. 
            - ((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))).sqrt();

        let val_start = log(2.0 * (radius_start.powi(2) + c.powi(2)).sqrt() * (radius_start.powi(2) - d_norm.powi(2)).sqrt() + c.powi(2) - d_norm.powi(2) + 2.0 * radius_start.powi(2)) / 2.
            - ((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))).sqrt();
        return val_end - val_start;
    }
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

fn integral_5(c: f64, d_norm: f64, radius_start: f64, radius_end: f64)  -> f64 {
    if d_norm == 0.0 {
        return PI * integral_4(c, radius_start, radius_end) / 2.;
    } else if d_norm < 1e-7 {
        // if d is very small
        if c == 0.0 {
            let val_end = radius_end * acos(d_norm / radius_end) - d_norm * log((sqrt(radius_end.powi(2) - d_norm.powi(2)) + radius_end) / d_norm);
            let val_start = radius_start * acos(d_norm / radius_start) - d_norm * log((sqrt(radius_start.powi(2) - d_norm.powi(2)) + radius_start) / d_norm);
            return  val_end - val_start;
        } else if c.abs() < 1e-8 {
            let temp_val_end = (radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2));
            let temp_val_start = (radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2));

            // use taylor expansion to approximate atanh
            let val_end = -2.0 * c * atan(c * ((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))).sqrt() / d_norm ) 
            + (radius_end.powi(2) + 2.0*c.powi(2)) * acos(d_norm / radius_end) / (radius_end.powi(2) + c.powi(2)).sqrt()
            - d_norm * (temp_val_end.powf(0.5) + temp_val_end.powf(1.5) + temp_val_end.powf(2.5)); 
            
            let val_start = -2.0 * c * atan(c * ((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))).sqrt() / d_norm ) 
            + (radius_start.powi(2) + 2.0*c.powi(2)) * acos(d_norm / radius_start) / (radius_start.powi(2) + c.powi(2)).sqrt()
            - d_norm * (temp_val_start.powf(0.5) + temp_val_start.powf(1.5) + temp_val_start.powf(2.5));

            return val_end - val_start;
        } else {
            let val_end = -2.0 * c * atan(c * ((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))).sqrt() / d_norm ) 
            + (radius_end.powi(2) + 2.0*c.powi(2)) * acos(d_norm / radius_end) / (radius_end.powi(2) + c.powi(2)).sqrt() 
            - d_norm * atanh(((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))).sqrt());
            let val_start = -2.0 * c * atan(c * ((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))).sqrt() / d_norm ) 
            + (radius_start.powi(2) + 2.0*c.powi(2)) * acos(d_norm / radius_start) / (radius_start.powi(2) + c.powi(2)).sqrt() 
            - d_norm * atanh(((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))).sqrt());
            return val_end - val_start;
        }
    } else {
        if c == 0.0 {
            let val_end = radius_end * acos(d_norm / radius_end);
            let val_start = radius_start * acos(d_norm / radius_start);
            return  val_end - val_start;
        } else {
            let val_end = -2.0 * c * atan(c * ((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))).sqrt() / d_norm ) 
            + (radius_end.powi(2) + 2.0*c.powi(2)) * acos(d_norm / radius_end) / (radius_end.powi(2) + c.powi(2)).sqrt() 
            - d_norm * atanh(((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))).sqrt());
            let val_start = -2.0 * c * atan(c * ((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))).sqrt() / d_norm ) 
            + (radius_start.powi(2) + 2.0*c.powi(2)) * acos(d_norm / radius_start) / (radius_start.powi(2) + c.powi(2)).sqrt() 
            - d_norm * atanh(((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))).sqrt());
            return val_end - val_start;
        }
    }
}

// integrals for Greens function
pub(crate) fn integrate_j1(phi_end: f64, phi_start: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_end: f64, d_norm_start: f64, sign_end: f64, sign_start: f64) -> f64{
    let mut angle_difference = phi_end - phi_start;
    let branch_check = check_for_branch_cut(phi_end, phi_start, (radius_start + radius_end) / 2.0 , d_norm_end, d_norm_start, sign_end, sign_start);
    if branch_check < 0.0 {
        angle_difference = angle_difference + 2.0 * PI;
    } else if branch_check > 2.0 * PI {
        angle_difference = angle_difference - 2.0 * PI;
    }

    let value = angle_difference * integral_j1(c, radius_start, radius_end) 
    + sign_end * integral_j2(c, d_norm_end, radius_start, radius_end) 
    - sign_start * integral_j2(c, d_norm_start , radius_start, radius_end);
    value
}

pub(crate) fn integrate_j2(phi_end: f64, phi_start: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_end: f64, d_norm_start: f64, sign_end: f64, sign_start: f64) -> f64{

    let value = (d_norm_end*sin(phi_end) - d_norm_start*sin(phi_start)) * integral_j1(c, radius_start, radius_end) 
    + sign_end * cos(phi_end)* integral_j3(c, d_norm_end, radius_start, radius_end) 
    - sign_start * cos(phi_start) * integral_j3(c, d_norm_start , radius_start, radius_end);
    value
}

pub(crate) fn integrate_j3(phi_end: f64, phi_start: f64, c: f64, radius_end: f64, radius_start: f64, d_norm_end: f64, d_norm_start: f64, sign_end: f64, sign_start: f64) -> f64{

    let value = - (d_norm_end*cos(phi_end) - d_norm_start*cos(phi_start)) * integral_j1(c, radius_start, radius_end) 
    + sign_end * sin(phi_end)* integral_j3(c, d_norm_end, radius_start, radius_end) 
    - sign_start * sin(phi_start) * integral_j3(c, d_norm_start , radius_start, radius_end);
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
    val
}

fn integral_j2(c: f64, d_norm: f64, radius_start: f64, radius_end: f64) -> f64 {
    if c.abs() < 1e-8 {
        if d_norm == 0.0  {
            return PI * integral_j1(c, radius_start, radius_end) / 2.;
        } else if d_norm.abs() < 1e-8 {
            let temp_val_end = -(sqrt(radius_end.powi(2) - d_norm.powi(2)) - radius_end) / (sqrt(radius_end.powi(2) - d_norm.powi(2)) + radius_end);
            let temp_val_start = -(sqrt(radius_start.powi(2) - d_norm.powi(2)) - radius_start) / (sqrt(radius_start.powi(2) - d_norm.powi(2)) + radius_start);

            // aproximate log using its taylor expansion at 1
            let val_end = radius_end * acos(d_norm / radius_end)
                + d_norm/2. * ((temp_val_end - 1.) - (temp_val_end - 1.).powi(2) / 2. + (temp_val_end - 1.).powi(3) / 3. - (temp_val_end - 1.).powi(4) / 4.);
            let val_start = radius_start * acos(d_norm / radius_start)
                + d_norm/2. * ((temp_val_start - 1.) - (temp_val_start - 1.).powi(2) / 2. + (temp_val_start - 1.).powi(3) / 3. - (temp_val_start - 1.).powi(4) / 4.);
            return  val_end - val_start;
        } else {
            let val_end = radius_end * acos(d_norm / radius_end) 
                + d_norm/2. * log(-(sqrt(radius_end.powi(2) - d_norm.powi(2)) - radius_end) / (sqrt(radius_end.powi(2) - d_norm.powi(2)) + radius_end));
            let val_start = radius_start * acos(d_norm / radius_start) 
                + d_norm/2. * log(-(sqrt(radius_start.powi(2) - d_norm.powi(2)) - radius_start) / (sqrt(radius_start.powi(2) - d_norm.powi(2)) + radius_start));
            return  val_end - val_start;
        }
    } else {
        if d_norm == 0.0 {
            return PI * integral_j1(c, radius_start, radius_end) / 2.;
        } else {
            let val_end = sqrt(radius_end.powi(2) + c.powi(2)) * acos(d_norm / radius_end) 
            - c * asin(c * sqrt((radius_end.powi(2) - d_norm.powi(2)) / radius_end.powi(2)) / sqrt(c.powi(2) + d_norm.powi(2))) 
            - d_norm * atanh(sqrt((radius_end.powi(2) - d_norm.powi(2)) / (radius_end.powi(2) + c.powi(2))));
            let val_start = sqrt(radius_start.powi(2) + c.powi(2)) * acos(d_norm / radius_start) 
            - c * asin(c * sqrt((radius_start.powi(2) - d_norm.powi(2)) / radius_start.powi(2)) / sqrt(c.powi(2) + d_norm.powi(2))) 
            - d_norm * atanh(sqrt((radius_start.powi(2) - d_norm.powi(2)) / (radius_start.powi(2) + c.powi(2))));
            return val_end - val_start;
        }
    }
}

fn integral_j3(c: f64,  d_norm: f64, radius_start: f64, radius_end: f64) -> f64 {
    if c.abs() <= 1e-8 {
        if d_norm == 0.0 {
            (radius_end.powi(2) - radius_start.powi(2)) / 2.
        } else if d_norm.abs() <= 1e-8 {
            let temp_val_end = radius_end - sqrt(radius_end.powi(2) - d_norm.powi(2));
            let temp_val_start = radius_start - sqrt(radius_start.powi(2) - d_norm.powi(2));

            // use taylor expansion to approximate log
            let val_end = radius_end * sqrt(radius_end.powi(2) - d_norm.powi(2)) 
                + d_norm.powi(2) * ((temp_val_end - 1.) - (temp_val_end - 1.).powi(2) / 2. + (temp_val_end - 1.).powi(3) / 3. - (temp_val_end - 1.).powi(4) / 4.);
            let val_start = radius_start * sqrt(radius_start.powi(2) - d_norm.powi(2)) 
                + d_norm.powi(2) * ((temp_val_start - 1.) - (temp_val_start - 1.).powi(2) / 2. + (temp_val_start - 1.).powi(3) / 3. - (temp_val_start - 1.).powi(4) / 4.);
            (val_end - val_start) / 2.
        } else {
            let val_end = radius_end * sqrt(radius_end.powi(2) - d_norm.powi(2)) + d_norm.powi(2) * log((-radius_end + sqrt(radius_end.powi(2) - d_norm.powi(2))).abs());
            let val_start = radius_start * sqrt(radius_start.powi(2) - d_norm.powi(2)) + d_norm.powi(2) * log((-radius_start + sqrt(radius_start.powi(2) - d_norm.powi(2))).abs());
            (val_end - val_start) / 2.
        }
    } else {
        let val_end = sqrt(radius_end.powi(2) + c.powi(2)) * sqrt(radius_end.powi(2) - d_norm.powi(2)) + (c.powi(2) + d_norm.powi(2)) * log(sqrt(radius_end.powi(2) + c.powi(2)) -  sqrt(radius_end.powi(2) - d_norm.powi(2)));
        let val_start = sqrt(radius_start.powi(2) + c.powi(2)) * sqrt(radius_start.powi(2) - d_norm.powi(2)) + (c.powi(2) + d_norm.powi(2)) * log(sqrt(radius_start.powi(2) + c.powi(2)) -  sqrt(radius_start.powi(2) - d_norm.powi(2)));
        (val_end - val_start) / 2.
    }
}

fn check_for_branch_cut(phi_end: f64, phi_start: f64, radius: f64, d_norm_end: f64, d_norm_start: f64, sign_end: f64, sign_start: f64) -> f64 {
    let start_angle;
    if d_norm_start == 0.0 {
        start_angle = phi_start + sign_start * PI / 2.;
    } else {
        start_angle = phi_start + sign_start * acos(d_norm_start / radius);
    }
    
    let end_angle;
    if d_norm_end == 0.0 {
        end_angle = phi_end + sign_end * PI / 2.;
    } else {
        end_angle = phi_end + sign_end * acos(d_norm_end / radius);
    }
    return end_angle - start_angle
}


// ------------------ integrals for geometric integral -----------------------
pub(crate) fn geometric_1(theta_2: f64, theta_end: f64) -> f64 {
    let val_end = cos(theta_end - theta_2) + cos(theta_2).powi(2) * log(tan((theta_end + theta_2) / 2.));
    let val_start = cos(- theta_2) + cos(theta_2).powi(2) * log(tan((theta_2) / 2.));
    val_end - val_start
}

pub(crate) fn geometric_2(theta_2: f64, theta_end: f64) -> f64 {
    let val_end = -cos(theta_end - theta_2) + sin(theta_2).powi(2) * log(tan((theta_end + theta_2) / 2.));
    let val_start = -cos(- theta_2) + sin(theta_2).powi(2) * log(tan((theta_2) / 2.));
    val_end - val_start
}

pub(crate) fn geometric_3(theta_2: f64, theta_end: f64) -> f64 {
    let val_end = -2.*(cos(theta_2) + 3.*cos(3.*theta_2)) * atanh(cos(theta_2) - sin(theta_2)*tan(theta_end/2.))
        + (2.*sin(2.*theta_end - theta_2) + sin(theta_2) + 3.*sin(3.*theta_2)) / sin(theta_end + theta_2);
    let val_start = -2.*(cos(theta_2) + 3.*cos(3.*theta_2)) * atanh(cos(theta_2))
        + (2.*sin( - theta_2) + sin(theta_2) + 3.*sin(3.*theta_2)) / sin(theta_2);
    (val_end - val_start) / 4.
}

pub(crate) fn geometric_4(theta_2: f64, theta_end: f64) -> f64 {
    let val_end = -2.*(sin(theta_2) - 3.*sin(3.*theta_2)) * atanh(cos(theta_2) - sin(theta_2)*tan(theta_end/2.))
        - (2.*cos(2.*theta_end - theta_2) + cos(theta_2) - 3.*cos(3.*theta_2)) / sin(theta_end + theta_2);
    let val_start = -2.*(sin(theta_2) - 3.*sin(3.*theta_2)) * atanh(cos(theta_2))
        - (2.*cos( - theta_2) + cos(theta_2) - 3.*cos(3.*theta_2)) / sin(theta_2);
    (val_end - val_start) / 4.
}

pub(crate) fn geometric_5(theta_2: f64, theta_end: f64) -> f64 {
    let val_end = -cos(theta_end - 2.*theta_2) - 6.*cos(theta_2)*sin(theta_2).powi(2)*atanh(cos(theta_2) - sin(theta_2)*tan(theta_end/2.))
        + sin(theta_2).powi(3) / sin(theta_end + theta_2);
    let val_start = -cos( - 2.*theta_2) - 6.*cos(theta_2)*sin(theta_2).powi(2)*atanh(cos(theta_2))
        + sin(theta_2).powi(3) / sin(theta_2);
    val_end - val_start
}

pub(crate) fn geometric_6(theta_2: f64, theta_end: f64) -> f64 {
    let val_end = -sin(theta_end - 2.*theta_2) - 6.*cos(theta_2).powi(2)*sin(theta_2)*atanh(cos(theta_2) - sin(theta_2)*tan(theta_end/2.))
        - cos(theta_2).powi(3) / sin(theta_end + theta_2);
    let val_start = -sin( - 2.*theta_2) - 6.*cos(theta_2).powi(2)*sin(theta_2)*atanh(cos(theta_2))
        - cos(theta_2).powi(3) / sin(theta_2);
    val_end - val_start
}
