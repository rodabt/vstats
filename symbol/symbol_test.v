module symbol

fn test__to_f_default() {
	// Unknown expression should return identity function
	f := to_f('unknown') as fn (f64) f64
	
	assert f(5.0) == 5.0
	assert f(10.0) == 10.0
}

fn test__parse_expr() {
	// Currently returns empty slice as it's WIP
	result := parse_expr('x^2')
	assert result.len == 0
}

fn test__parse_expr_empty() {
	// Empty expression
	result := parse_expr('')
	assert result.len == 0
}

fn test__symbol_module_exists() {
	// Basic test that the module compiles and functions are accessible
	f := to_f('x') as fn (f64) f64
	// Identity function behavior
	assert f(0.0) == 0.0
	assert f(1.0) == 1.0
}
