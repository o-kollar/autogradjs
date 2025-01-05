import {

    Value,

    Layer,

    LayerNorm,

    Linear,

    ReLU,

    Tanh,

    Dropout,

    Softmax,

    Optimizer,

    Adagrad,

    Adam,
    AdamW,

    RMSprop,

    SGD,

    Model

} from './index.js';


const batch_size = 4;

const input_size = 3;

const hidden_size = 4;

const num_classes = 3;  // 3 different classes


// Define model with Softmax layer

const model = new Model([

    new Linear(input_size, hidden_size),

    new LayerNorm(hidden_size),

    new Linear(hidden_size, hidden_size),

    new ReLU(),

    new LayerNorm(hidden_size),

    new Linear(hidden_size, num_classes),

    new ReLU(),

    new Softmax()    // Add dedicated Softmax layer

]);


// Create batched input data

const X = new Value([

    [2.0, 3.0, -1.0],

    [3.0, -1.0, 0.5],

    [0.5, 1.0, 1.0],

    [1.0, 1.0, -1.0]

]);


// Create one-hot encoded labels

const y = Value.oneHot([0, 1, 2, 0], num_classes);  // Example labels: class 0, 1, 2, 0


// Training loop

const lr = 0.01;

const n_epochs = 1600;


const optimizer = new AdamW(model.parameters());


const startTime = new Date().getTime(); // Record start time
let initialLoss = null;
let finalLoss = null;


for (let epoch = 0; epoch < n_epochs; epoch++) {

    // Forward pass

    model.zero_grad();

    const pred = model.forward(X);  // Softmax is now part of the model

    const loss = pred.crossEntropy(y);

    

    // Record initial loss

    if (epoch === 0) {

        initialLoss = loss.data[0][0];

    }

    

    // Calculate accuracy

    const accuracy = pred.accuracy(y);

    

    // Backward pass

    loss.backward();

    

    optimizer.step();

    

    if (epoch % 10 === 0) {

        console.log(`Epoch ${epoch}`);

        console.log(`Loss: ${loss.data[0][0]}`);

        console.log(`Accuracy: ${accuracy}`);

    }

    

    // Record final loss

    if (epoch === n_epochs - 1) {

        finalLoss = loss.data[0][0];

    }

}


const endTime = new Date().getTime(); // Record end time

const duration = (endTime - startTime) / 1000; // Convert to seconds


// Display training results

console.log('\nTraining completed.');

console.log(`Time taken for training: ${duration.toFixed(2)} seconds`);

console.log(`Initial loss: ${initialLoss.toFixed(6)}`);

console.log(`Final loss: ${finalLoss.toFixed(6)}`);

console.log(`Loss decrease: ${(initialLoss - finalLoss).toFixed(6)}`);


model.save('model.json')

console.log('Model saved');



const loadedModel = Model.load('model.json');


// After training, make predictions on new data

const test_X = new Value([

    [2.0, 3.0, -1.0],  // Test sample

]);


const test_pred = loadedModel.forward(test_X);  // Includes softmax

console.log('\nTest predictions (probabilities):');

console.log(test_pred.data);


// Get predicted class (argmax)

const predicted_class = test_pred.data[0].indexOf(Math.max(...test_pred.data[0]));

console.log('Predicted class:', predicted_class);
