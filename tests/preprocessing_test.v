module main

import utils

fn test__winsorize_clips_upper_tail() {
	// upper=0.8 on n=5: hi_idx = int(0.8*4) = 3, sorted=[1,2,3,4,100], hi_val=4.0
	x := [1.0, 2.0, 3.0, 4.0, 100.0]
	r := utils.winsorize(x, 0.0, 0.8)
	assert r[4] == 4.0
	assert r[0] == 1.0
	assert r.len == 5
}

fn test__winsorize_preserves_input_order() {
	// unsorted input: 5.0 should be clipped, position 0 gets 4.0, position 1 stays 1.0
	x := [5.0, 1.0, 3.0, 2.0, 4.0]
	r := utils.winsorize(x, 0.0, 0.8)
	assert r[0] == 4.0
	assert r[1] == 1.0
}

fn test__winsorize_no_op_when_bounds_are_full() {
	x := [1.0, 2.0, 3.0, 4.0, 5.0]
	r := utils.winsorize(x, 0.0, 1.0)
	assert r == x
}

fn test__trim_removes_upper_tail() {
	// upper=0.8 on n=5: hi_val=4.0, value 100.0 is removed
	x := [1.0, 2.0, 3.0, 4.0, 100.0]
	r := utils.trim(x, 0.0, 0.8)
	assert r.len == 4
	assert 100.0 !in r
}

fn test__trim_preserves_order_of_kept_values() {
	// lower=0.25 on n=5: lo_idx=int(0.25*4)=1, sorted=[1,2,3,4,5], lo_val=2.0
	// 1.0 is removed; kept values [5,3,2,4] preserve original order
	x := [5.0, 1.0, 3.0, 2.0, 4.0]
	r := utils.trim(x, 0.25, 1.0)
	assert r.len == 4
	assert r[0] == 5.0
	assert 1.0 !in r
}
