module chart

pub struct LinearScale {
pub:
	domain_min f64
	domain_max f64
	range_min  f64
	range_max  f64
}

pub fn (s LinearScale) map(v f64) f64 {
	if s.domain_max == s.domain_min {
		return (s.range_min + s.range_max) / 2.0
	}
	t := (v - s.domain_min) / (s.domain_max - s.domain_min)
	return s.range_min + t * (s.range_max - s.range_min)
}
