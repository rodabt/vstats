module symbol

import math

@[params]
struct Opt {
	n		int = 1
}

type Func = fn (f64) f64 | fn (f64, f64) f64

fn parse_expr(s string) []string {
	return []string{}
}

// WIP
// Usage: to_f(expr) as fn (f64) f64
// Will be useful to have explicit derivatives when needed...
fn to_f(expr string) Func {
	// tokens := parse_expr(expr)...
	match true {
		expr.match_glob('x^*') { 
			dump(expr)
			args := expr.split('^')
			n := args[1].f64()
			return fn [n](x f64) f64 { return math.pow(x,n) } 
		}
		expr.match_glob('*[*]x') { 
			dump(expr)
			args := expr.split('*')
			a := args[0].f64()
			return fn [a](x f64) f64 { return a*x } 
		}
		else { return fn (x f64) f64 { return x } }
	}
}

fn main() {
	f := to_f('10*x') as fn (f64) f64
	println(f(5))
}