import stats

fn test__mean() {
	assert stats.mean([f64(1), 2 , 3]) == f64(2.0)
}

fn test__median() {
	assert stats.median([f64(1), 10, 2, 9, 5]) == f64(5)
	assert stats.median([f64(1), 9, 2, 10]) == f64(5.5)
}

fn test__mode() {
	assert stats.mode([f64(1),2,2,2,3,3,3,5,6,7,8,8,9]) == [f64(2), 3]
}

fn test__range() {
	assert stats.range([f64(1),2,3,4,5,7]) == f64(6.0)
}