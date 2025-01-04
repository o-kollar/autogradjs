# AutoGradJS

This repository contains a minimalistic deep learning framework written in JavaScript. The framework includes classes for building and training neural networks, implementing common machine learning layers, activation functions, optimizers, and loss functions. The core functionality allows users to define and train models using stochastic gradient descent (SGD), Adam, RMSprop, Adagrad, and other optimization algorithms.

## Features

- **Layers:**
  - Linear (Fully Connected Layer)
  - ReLU (Rectified Linear Unit)
  - Tanh (Hyperbolic Tangent)
  - Softmax
  - Dropout
  - Layer Normalization

- **Activation Functions:**
  - ReLU
  - Tanh
  - Softmax

- **Loss Functions:**
  - Mean Squared Error (MSE)
  - Huber Loss
  - Cross-Entropy Loss
  - Accuracy (for classification tasks)

- **Optimizers:**
  - Stochastic Gradient Descent (SGD)
  - Adam
  - RMSprop
  - Adagrad

- **Backpropagation and Autograd:**
  - Backpropagation is implemented for computing gradients and performing weight updates.

- **Model:**
  - The `Model` class allows you to build and train neural networks by stacking different layers.

- **Serialization:**
  - You can save and load models using JSON serialization, which includes model architecture and weights.

## Getting Started

### Prerequisites

- Node.js (v12 or later)


### Usage

1. **Create a Simple Model:**

You can define a simple model by creating layers and adding them to the model:

```javascript
const model = new Model(
  new Linear(2, 3),  // Input size: 2, Output size: 3
  new ReLU(),
  new Linear(3, 1)   // Input size: 3, Output size: 1
);
```

2. **Forward Pass:**

To perform a forward pass (inference) with an input tensor:

```javascript
const input = new Value([[1, 2]]);
const output = model.forward(input);
console.log(output.data);
```

3. **Training:**

To train a model, you will typically use an optimizer, a loss function, and perform backward propagation:

```javascript
const optimizer = new SGD(model.parameters(), 0.01);

for (let epoch = 0; epoch < 1000; epoch++) {
  const input = new Value([[1, 2]]);
  const target = new Value([[1]]);  // Target output
  
  // Forward pass
  const output = model.forward(input);
  
  // Compute loss
  const loss = output.mse(target);
  
  // Backward pass
  loss.backward();
  
  // Update parameters
  optimizer.step();
  
  // Zero gradients for next iteration
  optimizer.zero_grad();
  
  if (epoch % 100 === 0) {
    console.log(`Epoch ${epoch}: Loss = ${loss.data[0][0]}`);
  }
}
```

4. **Save and Load Models:**

You can save the model's parameters and architecture to a JSON file:

```javascript
model.save('model.json');
```

To load a model:

```javascript
const modelData = fs.readFileSync('model.json', 'utf8');
const loadedModel = Model.load(modelData);
```

## Model Layers

The framework supports the following layers:

- **Linear**: A fully connected layer with weights and bias.
- **ReLU**: Rectified Linear Unit activation function.
- **Tanh**: Hyperbolic tangent activation function.
- **Softmax**: Used for multi-class classification tasks.
- **Dropout**: Regularization layer to reduce overfitting by randomly setting some units to zero during training.
- **LayerNorm**: Normalizes inputs to a layer for better training stability.

## Optimizers

The following optimizers are implemented:

- **SGD**: Basic stochastic gradient descent with optional momentum and weight decay.
- **Adam**: Adaptive moment estimation (a combination of momentum and RMSprop).
- **RMSprop**: An adaptive learning rate optimizer.
- **Adagrad**: Adaptive gradient optimizer.

## Loss Functions

- **Mean Squared Error (MSE)**: Suitable for regression tasks.
- **Huber Loss**: A loss function that combines MSE and MAE for robustness.
- **Cross-Entropy Loss**: Used for classification tasks.
- **Accuracy**: Metric to compute accuracy of a model's predictions.

## Example

Here is a simple example of training a neural network using this framework:

```javascript
const model = new Model(
  new Linear(2, 3),
  new ReLU(),
  new Linear(3, 1)
);

const optimizer = new Adam(model.parameters(), 0.001);

for (let epoch = 0; epoch < 1000; epoch++) {
  const input = new Value([[1, 2]]);
  const target = new Value([[1]]);  // Target output
  
  const output = model.forward(input);
  const loss = output.mse(target);
  
  loss.backward();
  optimizer.step();
  optimizer.zero_grad();
  
  if (epoch % 100 === 0) {
    console.log(`Epoch ${epoch}: Loss = ${loss.data[0][0]}`);
  }
}
```

## Contributing

If you'd like to contribute to this project, feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.