import ml
import math

fn test__linear_regression() {
	// Perfect linear data: y = 2*x + 1
	x := [[1.0], [2.0], [3.0], [4.0], [5.0]]
	y := [3.0, 5.0, 7.0, 9.0, 11.0]
	model := ml.linear_regression(x, y)
	assert math.abs(model.intercept - 1.0) < 1e-6
	assert math.abs(model.coefficients[0] - 2.0) < 1e-6
}

fn test__linear_predict() {
	// Fit y = 3*x + 0 and verify predictions match
	x := [[1.0], [2.0], [3.0]]
	y := [3.0, 6.0, 9.0]
	model := ml.linear_regression(x, y)
	preds := ml.linear_predict(model, x)
	for i in 0 .. preds.len {
		assert math.abs(preds[i] - y[i]) < 1e-5
	}
	// R² should be 1.0 for perfectly collinear data
	r2 := ml.r_squared(y, preds)
	assert math.abs(r2 - 1.0) < 1e-6
}

fn test__logistic_regression() {
	// Two clearly separated clusters: class 0 around x=1, class 1 around x=10
	mut x := [][]f64{}
	mut y := []f64{}
	for i in 0 .. 20 {
		x << [f64(i + 1)]
		y << if i < 10 { f64(0) } else { f64(1) }
	}
	model := ml.logistic_regression(x, y, 1000, 0.1)
	preds := ml.logistic_predict(model, x, 0.5)
	mut correct := 0
	for i in 0 .. preds.len {
		if preds[i] == y[i] {
			correct++
		}
	}
	accuracy := f64(correct) / f64(preds.len)
	assert accuracy > 0.9
}

fn test__mse() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.1, 2.0, 2.9]
	
	error := ml.mse(y_true, y_pred)
	// MSE = (0.1^2 + 0^2 + 0.1^2) / 3 = 0.02 / 3 ≈ 0.00667
	assert error > 0.006 && error < 0.007
}

fn test__mse_perfect() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.0, 2.0, 3.0]
	
	error := ml.mse(y_true, y_pred)
	assert error == 0.0, "MSE should be 0 for perfect predictions"
}

fn test__rmse() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.0, 2.0, 3.0]
	
	error := ml.rmse(y_true, y_pred)
	assert error == 0.0, "RMSE should be 0 for perfect predictions"
}

fn test__mae() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.1, 2.1, 3.1]
	
	error := ml.mae(y_true, y_pred)
	// MAE = (0.1 + 0.1 + 0.1) / 3 = 0.1
	assert math.abs(error - 0.1) < 0.001
}

fn test__mae_perfect() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.0, 2.0, 3.0]
	
	error := ml.mae(y_true, y_pred)
	assert error == 0.0, "MAE should be 0 for perfect predictions"
}

fn test__r_squared() {
	y_true := [1.0, 2.0, 3.0, 4.0, 5.0]
	y_pred := [1.0, 2.0, 3.0, 4.0, 5.0]
	
	r2 := ml.r_squared(y_true, y_pred)
	assert r2 == 1.0, "R² should be 1 for perfect predictions"
}

fn test__r_squared_poor() {
	y_true := [1.0, 2.0, 3.0, 4.0, 5.0]
	y_pred := [3.0, 3.0, 3.0, 3.0, 3.0]  // constant prediction
	
	r2 := ml.r_squared(y_true, y_pred)
	// R² should be poor (< 0.5)
	assert r2 < 0.5
}

fn test__sigmoid() {
	// sigmoid(0) = 0.5
	assert math.abs(ml.sigmoid(0.0) - 0.5) < 0.001
	// sigmoid(x) is between 0 and 1
	assert ml.sigmoid(10.0) > 0.99
	assert ml.sigmoid(-10.0) < 0.01
}
