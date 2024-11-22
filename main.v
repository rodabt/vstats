import prob
import optim
import rand
import linalg
import utils

fn main() {
    // Example usage
    println(prob.inverse_normal_cdf(0.90, 0, 1))
    println(optim.difference_quotient(fn (x f64) f64 { return x*x }, 10,0.01))

    mut v := utils.range(5).map(rand.f64_in_range(-10,10) or {0})
    mut epoch := 1

    for {
        grad := optim.sum_of_squares_gradient(v)
        v = optim.gradient_step(v, grad, -0.01)
        println('epoch: ${epoch}, v: ${v}')
        if linalg.distance(v, [f64(0), 0, 0, 0, 0]) < 0.001 {
            println('Coverged at: ${epoch}')
            break
        }
        epoch = epoch + 1
    }
    println(linalg.distance(v, [f64(0), 0, 0, 0, 0])) 
}
