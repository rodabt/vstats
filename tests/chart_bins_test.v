import chart

fn test__histogram_bins_explicit_count() {
	b := chart.histogram_bins([1.0, 2.0, 3.0, 4.0], 2)
	assert b.edges == [1.0, 2.5, 4.0]
	assert b.counts == [2, 2]
}

fn test__histogram_bins_auto_count_is_positive() {
	b := chart.histogram_bins([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 0)
	assert b.counts.len > 0
	mut total := 0
	for c in b.counts {
		total += c
	}
	assert total == 8 // every value lands in a bin
}
