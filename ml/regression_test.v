module ml

import math

fn test__mse() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.1, 2.0, 2.9]
	
	error := mse(y_true, y_pred)
	// MSE = (0.1^2 + 0^2 + 0.1^2) / 3 = 0.02 / 3 ≈ 0.00667
	assert error > 0.006 && error < 0.007
}

fn test__mse_perfect() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.0, 2.0, 3.0]
	
	error := mse(y_true, y_pred)
	assert error == 0.0, "MSE should be 0 for perfect predictions"
}

fn test__rmse() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.0, 2.0, 3.0]
	
	error := rmse(y_true, y_pred)
	assert error == 0.0, "RMSE should be 0 for perfect predictions"
}

fn test__mae() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.1, 2.1, 3.1]
	
	error := mae(y_true, y_pred)
	// MAE = (0.1 + 0.1 + 0.1) / 3 = 0.1
	assert math.abs(error - 0.1) < 0.001
}

fn test__mae_perfect() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.0, 2.0, 3.0]
	
	error := mae(y_true, y_pred)
	assert error == 0.0, "MAE should be 0 for perfect predictions"
}

fn test__r_squared() {
	y_true := [1.0, 2.0, 3.0, 4.0, 5.0]
	y_pred := [1.0, 2.0, 3.0, 4.0, 5.0]
	
	r2 := r_squared(y_true, y_pred)
	assert r2 == 1.0, "R² should be 1 for perfect predictions"
}

fn test__r_squared_poor() {
	y_true := [1.0, 2.0, 3.0, 4.0, 5.0]
	y_pred := [3.0, 3.0, 3.0, 3.0, 3.0]  // constant prediction
	
	r2 := r_squared(y_true, y_pred)
	// R² should be poor (< 0.5)
	assert r2 < 0.5
}

fn test__sigmoid() {
	// sigmoid(0) = 0.5
	assert math.abs(sigmoid(0.0) - 0.5) < 0.001
	// sigmoid(x) is between 0 and 1
	assert sigmoid(10.0) > 0.99
	assert sigmoid(-10.0) < 0.01
}
