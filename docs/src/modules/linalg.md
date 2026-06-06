# linalg

`import vstats.linalg`

Vectors, matrices, and SIMD-accelerated operations. Foundational layer — all other
modules build on this.

> **vs Python:** replaces `numpy` array operations. Functions are generic in,
> same-type out: `add[T](v []T, w []T) []T`.

## Vectors

```v
add[T](v []T, w []T) []T
subtract[T](v []T, w []T) []T
vector_sum[T](vector_list [][]T) []T
scalar_multiply[T](c f64, v []T) []T
vector_mean[T](vector_list [][]T) []T
dot[T](v []T, w []T) T
magnitude[T](v []T) T
distance[T](v []T, w []T) T
squared_distance[T](v []T, w []T) T
sum_of_squares[T](v []T) T
```

## Matrices

```v
shape[T](a [][]T) (int, int)
get_row[T](a [][]T, i int) []T
get_column[T](a [][]T, j int) []T
flatten[T](m [][]T) []T
reshape[T](v []T, rows int, columns int) [][]T
make_matrix[T](num_rows int, num_cols int, op fn(int,int) T) [][]T
identity_matrix[T](n int) [][]T
matmul[T](a [][]T, b [][]T) [][]T
transpose[T](m [][]T) [][]T
matvec_mul[T](m [][]T, v []T) []T
gaussian_elimination(a [][]f64, b []f64) []f64
arange[T](end int) []T
```

## 3D Rotations

```v
rotation_x[T](angle T) [][]T
rotation_y[T](angle T) [][]T
rotation_z[T](angle T) [][]T
rotation[T](alpha T, beta T, gamma T) [][]T
```

## SIMD (simd.v)

Performance variants for `[]f64` with auto-dispatch (picks SIMD vs scalar at runtime):

```v
dot_simd(v []f64, w []f64) f64
add_simd(v []f64, w []f64) []f64
sum_simd(v []f64) f64
relu_simd(v []f64) []f64
sigmoid_simd(v []f64) []f64
// Use *_auto variants to let runtime choose:
dot_auto(v []f64, w []f64) f64
```
