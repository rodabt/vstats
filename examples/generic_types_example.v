import stats
import nn
import optim

fn main() {
	println("=" .repeat(80))
	println("VStats: Generic Type Support Example")
	println("=" .repeat(80))
	println("")

	// Demonstrate that many statistical functions now accept both int and f64
	
	println("1. DESCRIPTIVE STATISTICS WITH GENERIC TYPES")
	println("-" .repeat(80))
	
	// With int array
	int_data := [1, 2, 3, 4, 5]
	int_mean := stats.mean(int_data)
	int_variance := stats.variance(int_data)
	int_std := stats.standard_deviation(int_data)
	
	println("Integer array: ${int_data}")
	println("Mean: ${int_mean:.4f}")
	println("Variance: ${int_variance:.4f}")
	println("Std Dev: ${int_std:.4f}")
	println("")
	
	// With f64 array
	f64_data := [1.5, 2.5, 3.5, 4.5, 5.5]
	f64_mean := stats.mean(f64_data)
	f64_variance := stats.variance(f64_data)
	f64_std := stats.standard_deviation(f64_data)
	
	println("Float array: ${f64_data}")
	println("Mean: ${f64_mean:.4f}")
	println("Variance: ${f64_variance:.4f}")
	println("Std Dev: ${f64_std:.4f}")
	println("")

	println("2. CORRELATION WITH MIXED TYPES")
	println("-" .repeat(80))
	
	x_int := [1, 2, 3, 4, 5]
	y_int := [2, 4, 5, 4, 6]
	
	x_float := [1.0, 2.0, 3.0, 4.0, 5.0]
	y_float := [2.0, 4.0, 5.0, 4.0, 6.0]
	
	corr_int := stats.correlation(x_int, y_int)
	corr_float := stats.correlation(x_float, y_float)
	
	println("Correlation (int data): ${corr_int:.4f}")
	println("Correlation (f64 data): ${corr_float:.4f}")
	println("")

	println("3. NEURAL NETWORK LOSS FUNCTIONS WITH GENERIC TYPES")
	println("-" .repeat(80))
	
	// MSE loss with int
	y_true_int := [1, 2, 3, 4, 5]
	y_pred_int := [1, 2, 2, 5, 5]
	mse_int := nn.mse_loss(y_true_int, y_pred_int)
	
	println("MSE Loss (int data): ${mse_int:.4f}")
	
	// MAE loss with f64
	y_true_f64 := [1.0, 2.0, 3.0, 4.0, 5.0]
	y_pred_f64 := [1.1, 2.1, 2.9, 4.2, 4.8]
	mae_f64 := nn.mae_loss(y_true_f64, y_pred_f64)
	
	println("MAE Loss (f64 data): ${mae_f64:.4f}")
	println("")

	println("4. OPTIMIZATION WITH GENERIC TYPES")
	println("-" .repeat(80))
	
	// Gradient descent with int
	v_int := [1, 2, 3]
	grad_int := [0, 1, 0]
	result_int := optim.gradient_step(v_int, grad_int, 1)
	
	println("Gradient step (int): ${result_int}")
	
	// Gradient descent with f64
	v_f64 := [1.0, 2.0, 3.0]
	grad_f64 := [0.0, 0.1, 0.0]
	result_f64 := optim.gradient_step(v_f64, grad_f64, 0.1)
	
	println("Gradient step (f64): ${result_f64}")
	println("")

	println("5. KEY DESIGN PRINCIPLE")
	println("-" .repeat(80))
	println("Generic Input [T] → F64 Output")
	println("- Functions accept both int and f64 arrays")
	println("- Always return f64 for numerical precision")
	println("- Supports type flexibility while maintaining mathematical accuracy")
}
