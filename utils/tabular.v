module utils

// rows_to_vector extracts a single column from row-oriented data into a flat []f64,
// preserving row order. Any DataFrame library's row-map output (e.g. vframes'
// DataFrame.values()!, converted to []map[string]f64) can feed this directly.
pub fn rows_to_vector(rows []map[string]f64, column string) ![]f64 {
	if rows.len == 0 {
		return error('rows_to_vector: rows must not be empty')
	}
	mut result := []f64{cap: rows.len}
	for i, row in rows {
		if column !in row {
			return error('rows_to_vector: column "${column}" missing in row ${i}')
		}
		result << row[column]
	}
	return result
}

// rows_to_matrix extracts multiple columns from row-oriented data into a row-major
// [][]f64 -- result[i][j] is row i, column columns[j] -- matching the shape ml
// functions like linear_regression(x [][]T, y []T) already consume.
pub fn rows_to_matrix(rows []map[string]f64, columns []string) ![][]f64 {
	if rows.len == 0 {
		return error('rows_to_matrix: rows must not be empty')
	}
	if columns.len == 0 {
		return error('rows_to_matrix: columns must not be empty')
	}
	mut result := [][]f64{cap: rows.len}
	for i, row in rows {
		mut r := []f64{cap: columns.len}
		for col in columns {
			if col !in row {
				return error('rows_to_matrix: column "${col}" missing in row ${i}')
			}
			r << row[col]
		}
		result << r
	}
	return result
}
