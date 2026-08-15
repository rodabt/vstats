import vstats.utils

fn test__rows_to_vector_extracts_column() {
	rows := [
		{'x': 1.0, 'y': 10.0},
		{'x': 2.0, 'y': 20.0},
		{'x': 3.0, 'y': 30.0},
	]
	result := utils.rows_to_vector(rows, 'y') or { panic(err.msg()) }
	assert result == [10.0, 20.0, 30.0]
}

fn test__rows_to_vector_errors_on_empty_rows() {
	rows := []map[string]f64{}
	utils.rows_to_vector(rows, 'y') or {
		assert err.msg().contains('empty')
		return
	}
	assert false, 'expected an error for empty rows'
}

fn test__rows_to_vector_errors_on_missing_column() {
	rows := [
		{'x': 1.0, 'y': 10.0},
		{'x': 2.0},
	]
	utils.rows_to_vector(rows, 'y') or {
		assert err.msg().contains('y')
		return
	}
	assert false, 'expected an error for a missing column'
}

fn test__rows_to_matrix_builds_row_major() {
	rows := [
		{'x1': 1.0, 'x2': 2.0},
		{'x1': 3.0, 'x2': 4.0},
	]
	result := utils.rows_to_matrix(rows, ['x1', 'x2']) or { panic(err.msg()) }
	assert result == [[1.0, 2.0], [3.0, 4.0]]
}

fn test__rows_to_matrix_errors_on_empty_rows() {
	rows := []map[string]f64{}
	utils.rows_to_matrix(rows, ['x1']) or {
		assert err.msg().contains('empty')
		return
	}
	assert false, 'expected an error for empty rows'
}

fn test__rows_to_matrix_errors_on_missing_column() {
	rows := [
		{'x1': 1.0, 'x2': 2.0},
		{'x1': 3.0},
	]
	utils.rows_to_matrix(rows, ['x1', 'x2']) or {
		assert err.msg().contains('x2')
		return
	}
	assert false, 'expected an error for a missing column'
}

fn test__rows_to_dataset_truncates_target_to_int() {
	rows := [
		{'x': 1.0, 'label': 2.9},
		{'x': 2.0, 'label': 0.4},
	]
	ds := utils.rows_to_dataset(rows, ['x'], 'label', 'synthetic') or { panic(err.msg()) }
	assert ds.target == [2, 0]
	assert ds.features == [[1.0], [2.0]]
	assert ds.feature_names == ['x']
	assert ds.target_name == 'label'
	assert ds.name == 'synthetic'
}

fn test__rows_to_dataset_propagates_missing_column_error() {
	rows := [
		{'x': 1.0},
	]
	utils.rows_to_dataset(rows, ['x'], 'label', 'synthetic') or {
		assert err.msg().contains('label')
		return
	}
	assert false, 'expected an error for a missing target column'
}

fn test__rows_to_regression_dataset_keeps_target_as_f64() {
	rows := [
		{'x': 1.0, 'y': 2.9},
		{'x': 2.0, 'y': 0.4},
	]
	ds := utils.rows_to_regression_dataset(rows, ['x'], 'y', 'synthetic') or { panic(err.msg()) }
	assert ds.target == [2.9, 0.4]
	assert ds.features == [[1.0], [2.0]]
	assert ds.feature_names == ['x']
	assert ds.target_name == 'y'
	assert ds.name == 'synthetic'
}
