# nn

`import vstats.nn`

Neural network layers, forward/backward pass, and training loop.

> **vs Python:** replaces a minimal PyTorch or Keras sequential model.
> No autograd — gradients are computed manually layer by layer.

## Layers

```v
dense_layer(input_size int, output_size int, activation string) DenseLayer
// activation: "relu", "sigmoid", "tanh", "softmax", "linear"

layer.forward(input []f64) []f64   // returns reference to output_buffer — copy before reuse!
layer.backward(grad []f64) []f64
```

> **Warning:** `forward` returns a reference to the layer's `output_buffer`.
> If you call `forward` again on the same layer before consuming the result,
> the previous output will be overwritten. Copy with `result[0]` or `arrays.clone`.

## Network

```v
create_network(layers []DenseLayer) Network
network.forward(input []f64) []f64
network.train(x [][]f64, y [][]f64, epochs int, lr f64)
```

## Loss Functions

```v
mse_loss(y_true []f64, y_pred []f64) f64
cross_entropy_loss(y_true []f64, y_pred []f64) f64
binary_cross_entropy_loss(y_true []f64, y_pred []f64) f64
```
