import linalg
import math

fn test__shape() {
	n_rows, n_cols := linalg.shape([[1, 2, 3], [4, 5, 6]])
	assert n_rows == 2
	assert n_cols == 3
}

fn test__identity() {
	assert linalg.identity_matrix[int](3) == [[1,0,0], [0, 1, 0], [0, 0, 1]]
}

fn test__reshape() {
	// 6-element vector → 2×3 matrix
	m := linalg.reshape([1, 2, 3, 4, 5, 6], 2, 3)
	assert m == [[1, 2, 3], [4, 5, 6]]
	// 9-element vector → 3×3 matrix
	m2 := linalg.reshape([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3)
	assert m2[0] == [1, 2, 3]
	assert m2[1] == [4, 5, 6]
	assert m2[2] == [7, 8, 9]
}

fn test__matmul() {
	// 2×2 identity times itself = identity
	id := linalg.identity_matrix[f64](2)
	result := linalg.matmul(id, id)
	assert math.abs(result[0][0] - 1.0) < 1e-10
	assert math.abs(result[0][1] - 0.0) < 1e-10
	assert math.abs(result[1][0] - 0.0) < 1e-10
	assert math.abs(result[1][1] - 1.0) < 1e-10

	// Known 2×2 product: [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
	a := [[1.0, 2.0], [3.0, 4.0]]
	b := [[5.0, 6.0], [7.0, 8.0]]
	c := linalg.matmul(a, b)
	assert math.abs(c[0][0] - 19.0) < 1e-10
	assert math.abs(c[0][1] - 22.0) < 1e-10
	assert math.abs(c[1][0] - 43.0) < 1e-10
	assert math.abs(c[1][1] - 50.0) < 1e-10
}

fn test__matvec_mul() {
	// [[1,2],[3,4]] × [1,1] = [3,7]
	m := [[1.0, 2.0], [3.0, 4.0]]
	v := [1.0, 1.0]
	result := linalg.matvec_mul(m, v)
	assert math.abs(result[0] - 3.0) < 1e-10
	assert math.abs(result[1] - 7.0) < 1e-10

	// Identity × [5,6] = [5,6]
	id := linalg.identity_matrix[f64](2)
	v2 := [5.0, 6.0]
	r2 := linalg.matvec_mul(id, v2)
	assert math.abs(r2[0] - 5.0) < 1e-10
	assert math.abs(r2[1] - 6.0) < 1e-10
}

fn test__transpose() {
	// 2×3 → 3×2
	m := [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
	t := linalg.transpose(m)
	assert t.len == 3
	assert t[0].len == 2
	assert math.abs(t[0][0] - 1.0) < 1e-10
	assert math.abs(t[0][1] - 4.0) < 1e-10
	assert math.abs(t[1][0] - 2.0) < 1e-10
	assert math.abs(t[2][1] - 6.0) < 1e-10
}

fn test__gaussian_elimination() {
	// Solve: 2x + y = 5, x + 3y = 10  → x=1, y=3
	a := [[2.0, 1.0], [1.0, 3.0]]
	b := [5.0, 10.0]
	sol := linalg.gaussian_elimination(a, b)
	assert math.abs(sol[0] - 1.0) < 1e-9
	assert math.abs(sol[1] - 3.0) < 1e-9

	// Solve: x + 2y + 3z = 14, 2x + y + z = 8, x + y + 2z = 10 → x=2, y=0, z=4
	a2 := [[1.0, 2.0, 3.0], [2.0, 1.0, 1.0], [1.0, 1.0, 2.0]]
	b2 := [14.0, 8.0, 10.0]
	sol2 := linalg.gaussian_elimination(a2, b2)
	assert math.abs(sol2[0] - 2.0) < 1e-9
	assert math.abs(sol2[1] - 0.0) < 1e-9
	assert math.abs(sol2[2] - 4.0) < 1e-9
}
