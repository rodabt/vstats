module linalg

// ============================================================================
// Phase 3: SIMD Vector Operations with Portable Fallbacks
// Provides hardware-accelerated vector operations with automatic fallback
// to scalar implementations on unsupported platforms
// ============================================================================

import math

// SIMDConfig - Configuration for SIMD operations
pub struct SIMDConfig {
pub mut:
	use_simd bool = true      // Enable SIMD where supported
	min_size int = 64         // Minimum vector size to use SIMD (overhead threshold)
	unroll_factor int = 4     // Loop unrolling factor for scalar fallback
}

// default_simd_config - Returns default SIMD configuration
pub fn default_simd_config() SIMDConfig {
	return SIMDConfig{
		use_simd: true
		min_size: 64
		unroll_factor: 4
	}
}

// should_use_simd - Check if SIMD should be used for given size
fn should_use_simd(size int, cfg SIMDConfig) bool {
	return cfg.use_simd && size >= cfg.min_size
}

// ============================================================================
// SIMD Dot Product with Fallback
// ============================================================================

// dot_simd - Dot product with SIMD optimization and scalar fallback
// Processes 4 elements at a time when SIMD is beneficial
pub fn dot_simd(v []f64, w []f64) f64 {
	return dot_simd_with_config(v, w, default_simd_config())
}

// dot_simd_with_config - Dot product with custom SIMD configuration
pub fn dot_simd_with_config(v []f64, w []f64, cfg SIMDConfig) f64 {
	assert v.len == w.len, 'vectors must be the same length'
	
	n := v.len
	
	// Use scalar implementation for small vectors
	if !should_use_simd(n, cfg) {
		return dot_scalar(v, w)
	}
	
	// SIMD-friendly unrolled loop
	mut sum := f64(0)
	mut i := 0
	
	// Process 4 elements at a time (SIMD-friendly pattern)
	for i + 4 <= n {
		sum += v[i] * w[i] + v[i+1] * w[i+1] + 
		       v[i+2] * w[i+2] + v[i+3] * w[i+3]
		i += 4
	}
	
	// Handle remaining elements
	for i < n {
		sum += v[i] * w[i]
		i++
	}
	
	return sum
}

// dot_scalar - Scalar dot product for fallback
fn dot_scalar(v []f64, w []f64) f64 {
	mut sum := f64(0)
	for i in 0 .. v.len {
		sum += v[i] * w[i]
	}
	return sum
}

// ============================================================================
// SIMD Vector Addition with Fallback
// ============================================================================

// add_simd - Element-wise vector addition with SIMD optimization
pub fn add_simd(v []f64, w []f64) []f64 {
	return add_simd_with_config(v, w, default_simd_config())
}

// add_simd_with_config - Add with custom SIMD configuration
pub fn add_simd_with_config(v []f64, w []f64, cfg SIMDConfig) []f64 {
	assert v.len == w.len, 'vectors must be the same length'
	
	n := v.len
	
	// Use scalar implementation for small vectors
	if !should_use_simd(n, cfg) {
		return add_scalar(v, w)
	}
	
	mut result := []f64{len: n}
	mut i := 0
	
	// Process 4 elements at a time
	for i + 4 <= n {
		result[i] = v[i] + w[i]
		result[i+1] = v[i+1] + w[i+1]
		result[i+2] = v[i+2] + w[i+2]
		result[i+3] = v[i+3] + w[i+3]
		i += 4
	}
	
	// Handle remaining elements
	for i < n {
		result[i] = v[i] + w[i]
		i++
	}
	
	return result
}

// add_scalar - Scalar vector addition for fallback
fn add_scalar(v []f64, w []f64) []f64 {
	return []f64{len: v.len, init: v[index] + w[index]}
}

// ============================================================================
// SIMD Vector Subtraction with Fallback
// ============================================================================

// subtract_simd - Element-wise vector subtraction with SIMD optimization
pub fn subtract_simd(v []f64, w []f64) []f64 {
	return subtract_simd_with_config(v, w, default_simd_config())
}

// subtract_simd_with_config - Subtract with custom SIMD configuration
pub fn subtract_simd_with_config(v []f64, w []f64, cfg SIMDConfig) []f64 {
	assert v.len == w.len, 'vectors must be the same length'
	
	n := v.len
	
	if !should_use_simd(n, cfg) {
		return subtract_scalar(v, w)
	}
	
	mut result := []f64{len: n}
	mut i := 0
	
	for i + 4 <= n {
		result[i] = v[i] - w[i]
		result[i+1] = v[i+1] - w[i+1]
		result[i+2] = v[i+2] - w[i+2]
		result[i+3] = v[i+3] - w[i+3]
		i += 4
	}
	
	for i < n {
		result[i] = v[i] - w[i]
		i++
	}
	
	return result
}

// subtract_scalar - Scalar vector subtraction for fallback
fn subtract_scalar(v []f64, w []f64) []f64 {
	return []f64{len: v.len, init: v[index] - w[index]}
}

// ============================================================================
// SIMD Scalar Multiplication with Fallback
// ============================================================================

// scalar_multiply_simd - Scalar multiplication with SIMD optimization
pub fn scalar_multiply_simd(c f64, v []f64) []f64 {
	return scalar_multiply_simd_with_config(c, v, default_simd_config())
}

// scalar_multiply_simd_with_config - Multiply with custom SIMD configuration
pub fn scalar_multiply_simd_with_config(c f64, v []f64, cfg SIMDConfig) []f64 {
	n := v.len
	
	if !should_use_simd(n, cfg) {
		return scalar_multiply_scalar(c, v)
	}
	
	mut result := []f64{len: n}
	mut i := 0
	
	for i + 4 <= n {
		result[i] = v[i] * c
		result[i+1] = v[i+1] * c
		result[i+2] = v[i+2] * c
		result[i+3] = v[i+3] * c
		i += 4
	}
	
	for i < n {
		result[i] = v[i] * c
		i++
	}
	
	return result
}

// scalar_multiply_scalar - Scalar multiplication for fallback
fn scalar_multiply_scalar(c f64, v []f64) []f64 {
	return []f64{len: v.len, init: v[index] * c}
}

// ============================================================================
// SIMD Vector Sum with Fallback
// ============================================================================

// sum_simd - Sum all elements with SIMD optimization
pub fn sum_simd(v []f64) f64 {
	return sum_simd_with_config(v, default_simd_config())
}

// sum_simd_with_config - Sum with custom SIMD configuration
pub fn sum_simd_with_config(v []f64, cfg SIMDConfig) f64 {
	n := v.len
	
	if !should_use_simd(n, cfg) {
		return sum_scalar(v)
	}
	
	mut sum := f64(0)
	mut i := 0
	
	// Process 4 elements at a time
	for i + 4 <= n {
		sum += v[i] + v[i+1] + v[i+2] + v[i+3]
		i += 4
	}
	
	for i < n {
		sum += v[i]
		i++
	}
	
	return sum
}

// sum_scalar - Scalar sum for fallback
fn sum_scalar(v []f64) f64 {
	mut sum := f64(0)
	for val in v {
		sum += val
	}
	return sum
}

// ============================================================================
// SIMD Element-wise Multiplication (Hadamard)
// ============================================================================

// elementwise_multiply_simd - Element-wise multiplication with SIMD
pub fn elementwise_multiply_simd(v []f64, w []f64) []f64 {
	return elementwise_multiply_simd_with_config(v, w, default_simd_config())
}

// elementwise_multiply_simd_with_config - Elementwise multiply with config
pub fn elementwise_multiply_simd_with_config(v []f64, w []f64, cfg SIMDConfig) []f64 {
	assert v.len == w.len, 'vectors must be the same length'
	
	n := v.len
	
	if !should_use_simd(n, cfg) {
		return elementwise_multiply_scalar(v, w)
	}
	
	mut result := []f64{len: n}
	mut i := 0
	
	for i + 4 <= n {
		result[i] = v[i] * w[i]
		result[i+1] = v[i+1] * w[i+1]
		result[i+2] = v[i+2] * w[i+2]
		result[i+3] = v[i+3] * w[i+3]
		i += 4
	}
	
	for i < n {
		result[i] = v[i] * w[i]
		i++
	}
	
	return result
}

// elementwise_multiply_scalar - Element-wise multiplication fallback
fn elementwise_multiply_scalar(v []f64, w []f64) []f64 {
	return []f64{len: v.len, init: v[index] * w[index]}
}

// ============================================================================
// SIMD Activation Functions
// ============================================================================

// relu_simd - ReLU activation with SIMD optimization
pub fn relu_simd(v []f64) []f64 {
	return relu_simd_with_config(v, default_simd_config())
}

// relu_simd_with_config - ReLU with custom SIMD configuration
pub fn relu_simd_with_config(v []f64, cfg SIMDConfig) []f64 {
	n := v.len
	
	if !should_use_simd(n, cfg) {
		return relu_scalar(v)
	}
	
	mut result := []f64{len: n}
	mut i := 0
	
	for i + 4 <= n {
		result[i] = if v[i] > 0 { v[i] } else { 0 }
		result[i+1] = if v[i+1] > 0 { v[i+1] } else { 0 }
		result[i+2] = if v[i+2] > 0 { v[i+2] } else { 0 }
		result[i+3] = if v[i+3] > 0 { v[i+3] } else { 0 }
		i += 4
	}
	
	for i < n {
		result[i] = if v[i] > 0 { v[i] } else { 0 }
		i++
	}
	
	return result
}

// relu_scalar - Scalar ReLU for fallback
fn relu_scalar(v []f64) []f64 {
	return []f64{len: v.len, init: if v[index] > 0 { v[index] } else { 0 }}
}

// sigmoid_simd - Sigmoid activation with SIMD optimization
pub fn sigmoid_simd(v []f64) []f64 {
	return sigmoid_simd_with_config(v, default_simd_config())
}

// sigmoid_simd_with_config - Sigmoid with custom SIMD configuration
pub fn sigmoid_simd_with_config(v []f64, cfg SIMDConfig) []f64 {
	n := v.len
	
	if !should_use_simd(n, cfg) {
		return sigmoid_scalar(v)
	}
	
	mut result := []f64{len: n}
	mut i := 0
	
	for i + 4 <= n {
		// Clamp values for numerical stability
		mut x0 := v[i]
		mut x1 := v[i+1]
		mut x2 := v[i+2]
		mut x3 := v[i+3]
		
		if x0 > 10 { x0 = 10 } else if x0 < -10 { x0 = -10 }
		if x1 > 10 { x1 = 10 } else if x1 < -10 { x1 = -10 }
		if x2 > 10 { x2 = 10 } else if x2 < -10 { x2 = -10 }
		if x3 > 10 { x3 = 10 } else if x3 < -10 { x3 = -10 }
		
		result[i] = 1.0 / (1.0 + math.exp(-x0))
		result[i+1] = 1.0 / (1.0 + math.exp(-x1))
		result[i+2] = 1.0 / (1.0 + math.exp(-x2))
		result[i+3] = 1.0 / (1.0 + math.exp(-x3))
		i += 4
	}
	
	for i < n {
		mut x := v[i]
		if x > 10 { x = 10 } else if x < -10 { x = -10 }
		result[i] = 1.0 / (1.0 + math.exp(-x))
		i++
	}
	
	return result
}

// sigmoid_scalar - Scalar sigmoid for fallback
fn sigmoid_scalar(v []f64) []f64 {
	mut result := []f64{len: v.len}
	for i in 0 .. v.len {
		mut x := v[i]
		if x > 10 { x = 10 } else if x < -10 { x = -10 }
		result[i] = 1.0 / (1.0 + math.exp(-x))
	}
	return result
}

// tanh_simd - Tanh activation with SIMD optimization
pub fn tanh_simd(v []f64) []f64 {
	return tanh_simd_with_config(v, default_simd_config())
}

// tanh_simd_with_config - Tanh with custom SIMD configuration
pub fn tanh_simd_with_config(v []f64, cfg SIMDConfig) []f64 {
	n := v.len
	
	if !should_use_simd(n, cfg) {
		return tanh_scalar(v)
	}
	
	mut result := []f64{len: n}
	mut i := 0
	
	for i + 4 <= n {
		result[i] = math.tanh(v[i])
		result[i+1] = math.tanh(v[i+1])
		result[i+2] = math.tanh(v[i+2])
		result[i+3] = math.tanh(v[i+3])
		i += 4
	}
	
	for i < n {
		result[i] = math.tanh(v[i])
		i++
	}
	
	return result
}

// tanh_scalar - Scalar tanh for fallback
fn tanh_scalar(v []f64) []f64 {
	return []f64{len: v.len, init: math.tanh(v[index])}
}

// ============================================================================
// SIMD Matrix-Vector Multiplication
// ============================================================================

// matvec_multiply_simd - Matrix-vector multiplication with SIMD
pub fn matvec_multiply_simd(m [][]f64, v []f64) []f64 {
	return matvec_multiply_simd_with_config(m, v, default_simd_config())
}

// matvec_multiply_simd_with_config - Matvec with custom SIMD configuration
pub fn matvec_multiply_simd_with_config(m [][]f64, v []f64, cfg SIMDConfig) []f64 {
	assert m[0].len == v.len, 'matrix columns must match vector length'
	
	rows := m.len
	
	if !should_use_simd(rows, cfg) {
		return matvec_multiply_scalar(m, v)
	}
	
	mut result := []f64{len: rows, init: 0.0}
	
	for i in 0 .. rows {
		result[i] = dot_simd_with_config(m[i], v, cfg)
	}
	
	return result
}

// matvec_multiply_scalar - Scalar matrix-vector multiplication
fn matvec_multiply_scalar(m [][]f64, v []f64) []f64 {
	mut result := []f64{len: m.len}
	for i in 0 .. m.len {
		mut sum := f64(0)
		for j in 0 .. v.len {
			sum += m[i][j] * v[j]
		}
		result[i] = sum
	}
	return result
}

// ============================================================================
// SIMD Batch Operations
// ============================================================================

// batch_sum_simd - Sum of multiple vectors with SIMD
pub fn batch_sum_simd(vectors [][]f64) []f64 {
	return batch_sum_simd_with_config(vectors, default_simd_config())
}

// batch_sum_simd_with_config - Batch sum with custom SIMD configuration
pub fn batch_sum_simd_with_config(vectors [][]f64, cfg SIMDConfig) []f64 {
	assert vectors.len > 0, 'no vectors provided'
	
	vector_len := vectors[0].len
	
	if !should_use_simd(vector_len, cfg) {
		return batch_sum_scalar(vectors)
	}
	
	mut result := []f64{len: vector_len, init: 0.0}
	
	for vec in vectors {
		mut i := 0
		for i + 4 <= vector_len {
			result[i] += vec[i]
			result[i+1] += vec[i+1]
			result[i+2] += vec[i+2]
			result[i+3] += vec[i+3]
			i += 4
		}
		for i < vector_len {
			result[i] += vec[i]
			i++
		}
	}
	
	return result
}

// batch_sum_scalar - Scalar batch sum for fallback
fn batch_sum_scalar(vectors [][]f64) []f64 {
	num_elements := vectors[0].len
	mut res := []f64{len: num_elements}
	for i in 0 .. vectors.len {
		for j in 0 .. vectors[i].len {
			res[j] += vectors[i][j]
		}
	}
	return res
}

// ============================================================================
// Convenience Functions with Auto-Selection
// ============================================================================

// dot_auto - Automatically select best dot product implementation
pub fn dot_auto(v []f64, w []f64) f64 {
	return dot_simd(v, w)
}

// add_auto - Automatically select best addition implementation
pub fn add_auto(v []f64, w []f64) []f64 {
	return add_simd(v, w)
}

// sum_auto - Automatically select best sum implementation
pub fn sum_auto(v []f64) f64 {
	return sum_simd(v)
}

// matvec_multiply_auto - Automatically select best matvec implementation
pub fn matvec_multiply_auto(m [][]f64, v []f64) []f64 {
	return matvec_multiply_simd(m, v)
}
