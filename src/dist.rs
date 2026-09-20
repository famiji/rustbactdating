//! R-compatible probability distributions.
//!
//! Implementations follow R's conventions so that likelihoods computed here
//! can be compared directly against the R reference implementation.

/// log Gamma(x) via Lanczos approximation (matches lgamma to ~1e-14).
pub fn lgamma(x: f64) -> f64 {
    // Lanczos coefficients, g=7, n=9
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection formula
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// R's dgamma(x, shape, scale, log). Returns 0 (or -inf) for x < 0.
pub fn dgamma(x: f64, shape: f64, scale: f64, log: bool) -> f64 {
    if x < 0.0 || shape <= 0.0 || scale <= 0.0 {
        return if log { f64::NEG_INFINITY } else { 0.0 };
    }
    // density = x^(shape-1) * exp(-x/scale) / (scale^shape * Gamma(shape))
    let v = (shape - 1.0) * x.ln() - x / scale - shape * scale.ln() - lgamma(shape);
    if log {
        v
    } else {
        v.exp()
    }
}

/// Regularized lower incomplete gamma P(a,x) via series / continued fraction.
/// (Numerical Recipes gammp/gammq, accurate to ~1e-14.)
fn gammp(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // series
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        sum * (-x + a * x.ln() - lgamma(a)).exp()
    } else {
        // continued fraction for Q(a,x)
        const FPMIN: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        let q = (-x + a * x.ln() - lgamma(a)).exp() * h;
        1.0 - q
    }
}

/// R's pgamma(q, shape, scale, lower.tail, log.p).
pub fn pgamma(q: f64, shape: f64, scale: f64, lower_tail: bool, log_p: bool) -> f64 {
    if q <= 0.0 {
        // P(X <= 0) = 0
        return if lower_tail {
            if log_p { f64::NEG_INFINITY } else { 0.0 }
        } else {
            if log_p { 0.0 } else { 1.0 }
        };
    }
    let p = gammp(shape, q / scale);
    let val = if lower_tail { p } else { 1.0 - p };
    if log_p {
        val.ln()
    } else {
        val
    }
}

/// R's dnbinom(x, size, prob, log).
pub fn dnbinom(x: f64, size: f64, prob: f64, log: bool) -> f64 {
    if x < 0.0 {
        return if log { f64::NEG_INFINITY } else { 0.0 };
    }
    if prob <= 0.0 || prob > 1.0 {
        return f64::NAN;
    }
    // R convention: size == 0 degenerates to a point mass at x = 0.
    // dnbinom(0, 0, p, log=TRUE) == 0 and dnbinom(x>0, 0, p, log=TRUE) == -Inf.
    if size == 0.0 {
        return if x == 0.0 {
            if log { 0.0 } else { 1.0 }
        } else if log {
            f64::NEG_INFINITY
        } else {
            0.0
        };
    }
    // log C(x+size-1, x) + size*log(prob) + x*log(1-prob)
    let v = lgamma(x + size) - lgamma(size) - lgamma(x + 1.0)
        + size * prob.ln()
        + x * (1.0 - prob).ln();
    if log {
        v
    } else {
        v.exp()
    }
}

/// R's dpois(x, lambda, log).
pub fn dpois(x: f64, lambda: f64, log: bool) -> f64 {
    if lambda < 0.0 {
        return f64::NAN;
    }
    if x < 0.0 {
        return if log { f64::NEG_INFINITY } else { 0.0 };
    }
    let v = -lambda + x * lambda.ln() - lgamma(x + 1.0);
    if log {
        v
    } else {
        v.exp()
    }
}

/// R's dnorm(x, mean, sd, log).
pub fn dnorm(x: f64, mean: f64, sd: f64, log: bool) -> f64 {
    let z = (x - mean) / sd;
    let v = -0.5 * z * z - sd.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
    if log {
        v
    } else {
        v.exp()
    }
}

// ---------------- Random sampling ----------------

use rand::Rng;

/// rgamma(1, shape, scale) via Marsaglia-Tsang.
pub fn rgamma<R: Rng>(rng: &mut R, shape: f64, scale: f64) -> f64 {
    if shape < 1.0 {
        // boost: X = Y * U^(1/shape) with Y ~ Gamma(shape+1)
        let u: f64 = rng.gen();
        return rgamma(rng, shape + 1.0, scale) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = rnorm(rng, 0.0, 1.0);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u: f64 = rng.gen();
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v * scale;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v * scale;
        }
    }
}

/// rnorm(1, mean, sd) via Box-Muller.
pub fn rnorm<R: Rng>(rng: &mut R, mean: f64, sd: f64) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-300);
    let u2: f64 = rng.gen();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mean + sd * z
}

/// rpois(1, lambda) via Knuth (small lambda) / rejection (large).
pub fn rpois<R: Rng>(rng: &mut R, lambda: f64) -> f64 {
    if lambda < 30.0 {
        let l = (-lambda).exp();
        let mut k = 0.0;
        let mut p = 1.0;
        loop {
            k += 1.0;
            p *= rng.gen::<f64>();
            if p <= l {
                return k - 1.0;
            }
        }
    }
    // large lambda: normal approximation with continuity correction
    let v = rnorm(rng, 0.0, 1.0);
    (lambda + v * lambda.sqrt() + 0.5).floor().max(0.0)
}

/// rnbinom(1, size, prob): Poisson-Gamma mixture.
pub fn rnbinom<R: Rng>(rng: &mut R, size: f64, prob: f64) -> f64 {
    // If size is not integer, use gamma mixture: lambda ~ Gamma(size, (1-prob)/prob), X ~ Poisson(lambda)
    let lambda = rgamma(rng, size, (1.0 - prob) / prob);
    rpois(rng, lambda)
}

/// runif(1) in [0,1).
pub fn runif<R: Rng>(rng: &mut R) -> f64 {
    rng.gen()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lgamma() {
        assert!((lgamma(1.0) - 0.0).abs() < 1e-10);
        assert!((lgamma(5.0) - 24f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_dgamma_matches_r() {
        // R: dgamma(2, shape=3, scale=1, log=TRUE) = -1.306853  =>  exp = 0.2706706
        let v = dgamma(2.0, 3.0, 1.0, false);
        assert!((v - 0.270_670_566_6).abs() < 1e-8, "got {}", v);
    }

    #[test]
    fn test_pgamma_matches_r() {
        // R: pgamma(2, shape=3, scale=1, lower.tail=TRUE, log.p=TRUE) = -1.129102
        let v = pgamma(2.0, 3.0, 1.0, true, true);
        assert!((v - (-1.129_101_6)).abs() < 1e-6, "got {}", v);
    }

    #[test]
    fn test_dnbinom_matches_r() {
        // R: dnbinom(3, size=5, prob=0.4, log=TRUE) = -2.558582
        let v = dnbinom(3.0, 5.0, 0.4, true);
        assert!((v - (-2.558_582_5)).abs() < 1e-6, "got {}", v);
    }
    #[test]
    fn test_dnbinom_size_zero_matches_r() {
        // R: dnbinom(0, 0, 0.085, log=TRUE) == 0 ; dnbinom(1, 0, 0.085, log=TRUE) == -Inf
        assert_eq!(dnbinom(0.0, 0.0, 0.085, true), 0.0);
        assert_eq!(dnbinom(1.0, 0.0, 0.085, true), f64::NEG_INFINITY);
    }

}