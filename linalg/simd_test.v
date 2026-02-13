module linalg

// ============================================================================
// SIMD Vector Operations Tests
// ============================================================================

fn test_dot_simd_basic() {
	v := [f64(1.0), 2.0, 3.0, 4.0]
	w := [f64(5.0), 6.0, 7.0, 8.0]
	
	result := dot_simd(v, w)
	expected := f64(1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0)
	
	assert result == expected, 'dot_simd should compute correct dot product'
}

fn test_dot_simd_large() {
	// Test with large vector that triggers SIMD path
	mut v := []f64{len: 100, init: f64(index)}
	mut w := []f64{len: 100, init: f64(index + 1)}
	
	result := dot_simd(v, w)
	expected := dot_scalar(v, w)
	
	// Allow small floating point tolerance
	diff := result - expected
	assert diff < 1e-10 && diff > -1e-10, 'dot_simd large should match scalar'
}

fn test_dot_simd_with_config() {
	v := [f64(1.0), 2.0, 3.0, 4.0]
	w := [f64(5.0), 6.0, 7.0, 8.0]
	
	// Test with SIMD disabled (should use scalar)
	cfg := SIMDConfig{ use_simd: false, min_size: 64 }
	result := dot_simd_with_config(v, w, cfg)
	expected := dot_scalar(v, w)
	
	assert result == expected, 'dot_simd_with_config should use scalar when disabled'
}

fn test_add_simd() {
	v := [f64(1.0), 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
	w := [f64(1.0), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
	
	result := add_simd(v, w)
	
	assert result.len == 8, 'add_simd should preserve length'
	assert result[0] == 2.0, 'add_simd should add elements correctly'
	assert result[7] == 9.0, 'add_simd should add last element correctly'
}

fn test_subtract_simd() {
	v := [f64(10.0), 20.0, 30.0, 40.0]
	w := [f64(1.0), 2.0, 3.0, 4.0]
	
	result := subtract_simd(v, w)
	
	assert result.len == 4, 'subtract_simd should preserve length'
	assert result[0] == 9.0, 'subtract_simd should subtract elements correctly'
	assert result[3] == 36.0, 'subtract_simd should subtract last element correctly'
}

fn test_scalar_multiply_simd() {
	v := [f64(1.0), 2.0, 3.0, 4.0]
	c := f64(2.0)
	
	result := scalar_multiply_simd(c, v)
	
	assert result.len == 4, 'scalar_multiply_simd should preserve length'
	assert result[0] == 2.0, 'scalar_multiply_simd should multiply correctly'
	assert result[3] == 8.0, 'scalar_multiply_simd should multiply last element correctly'
}

fn test_sum_simd() {
	v := [f64(1.0), 2.0, 3.0, 4.0, 5.0]
	
	result := sum_simd(v)
	expected := f64(15.0)
	
	assert result == expected, 'sum_simd should compute correct sum'
}

fn test_relu_simd() {
	v := [f64(-1.0), 0.0, 1.0, -2.0, 3.0, -4.0, 5.0]
	
	result := relu_simd(v)
	
	assert result.len == 7, 'relu_simd should preserve length'
	assert result[0] == 0.0, 'relu_simd should zero negative values'
	assert result[2] == 1.0, 'relu_simd should keep positive values'
	assert result[4] == 3.0, 'relu_simd should keep larger positive values'
}

fn test_sigmoid_simd() {
	v := [f64(0.0), 1.0, -1.0, 2.0, -2.0]
	
	result := sigmoid_simd(v)
	
	assert result.len == 5, 'sigmoid_simd should preserve length'
	// sigmoid(0) = 0.5
	assert result[0] > 0.49 && result[0] < 0.51, 'sigmoid(0) should be ~0.5'
	// sigmoid(1) > 0.5
	assert result[1] > 0.5, 'sigmoid(1) should be > 0.5'
	// sigmoid(-1) < 0.5
	assert result[2] < 0.5, 'sigmoid(-1) should be < 0.5'
}

fn test_tanh_simd() {
	v := [f64(0.0), 1.0, -1.0]
	
	result := tanh_simd(v)
	
	assert result.len == 3, 'tanh_simd should preserve length'
	// tanh(0) = 0
	assert result[0] == 0.0, 'tanh(0) should be 0'
	// tanh(1) > 0
	assert result[1] > 0.5, 'tanh(1) should be > 0.5'
	// tanh(-1) < 0
	assert result[2] < -0.5, 'tanh(-1) should be < -0.5'
}

fn test_elementwise_multiply_simd() {
	v := [f64(1.0), 2.0, 3.0, 4.0]
	w := [f64(2.0), 3.0, 4.0, 5.0]
	
	result := elementwise_multiply_simd(v, w)
	
	assert result.len == 4, 'elementwise_multiply_simd should preserve length'
	assert result[0] == 2.0, 'elementwise_multiply_simd should multiply correctly'
	assert result[1] == 6.0, 'elementwise_multiply_simd should multiply correctly'
	assert result[2] == 12.0, 'elementwise_multiply_simd should multiply correctly'
	assert result[3] == 20.0, 'elementwise_multiply_simd should multiply correctly'
}

fn test_matvec_multiply_simd() {
	m := [
		[f64(1.0), 2.0, 3.0],
		[f64(4.0), 5.0, 6.0],
	]
	v := [f64(1.0), 1.0, 1.0]
	
	result := matvec_multiply_simd(m, v)
	
	assert result.len == 2, 'matvec_multiply_simd should return correct length'
	assert result[0] == 6.0, 'matvec_multiply_simd row 0 should be sum of row'
	assert result[1] == 15.0, 'matvec_multiply_simd row 1 should be sum of row'
}

fn test_simd_fallback() {
	// Test with small vectors that should use scalar fallback
	v := [f64(1.0), 2.0]
	w := [f64(3.0), 4.0]
	
	result := dot_simd(v, w)
	expected := f64(11.0)
	
	assert result == expected, 'dot_simd should handle small vectors'
}

fn test_simd_config() {
	// Test configuration functions
	cfg := SIMDConfig{ use_simd: true, min_size: 64 }
	assert should_use_simd(100, cfg), 'should_use_simd should return true for large vectors'
	
	cfg2 := SIMDConfig{ use_simd: false, min_size: 64 }
	assert !should_use_simd(100, cfg2), 'should_use_simd should return false when disabled'
}

fn test_batch_sum_simd() {
	vectors := [
		[f64(1.0), 2.0, 3.0, 4.0],
		[f64(1.0), 2.0, 3.0, 4.0],
		[f64(1.0), 2.0, 3.0, 4.0],
	]
	
	result := batch_sum_simd(vectors)
	
	assert result.len == 4, 'batch_sum_simd should preserve vector length'
	assert result[0] == 3.0, 'batch_sum_simd should sum first elements'
	assert result[3] == 12.0, 'batch_sum_simd should sum last elements'
}

fn test_default_simd_config() {
	cfg := default_simd_config()
	
	assert cfg.use_simd == true, 'default config should enable SIMD'
	assert cfg.min_size == 64, 'default min_size should be 64'
	assert cfg.unroll_factor == 4, 'default unroll_factor should be 4'
}
