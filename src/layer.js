import Value from "./value.js";
import { writeFileSync,readFileSync } from 'fs'; 


function randn_bm() {
    let u = 1 - Math.random(); 
    let v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// Base Layer class
export class Layer {
    constructor() {
        this._parameters = [];
    }

    parameters() {
        return this._parameters;
    }

    forward(x) {
        throw new Error("Not implemented");
    }
}

// Linear (Dense) layer
export class Linear extends Layer {
    constructor(in_features, out_features, bias=true) {
        super();
        this.in_features = in_features;
        this.out_features = out_features;
        
        // Initialize weights using He initialization
        this.weight = new Value(
            Array(in_features).fill().map(() => 
                Array(out_features).fill().map(() => 
                    randn_bm() * Math.sqrt(2.0 / in_features)
                )
            )
        );
        
        this.bias = bias ? new Value(
            [Array(out_features).fill().map(() => 0)]
        ) : null;

        this._parameters = bias ? [this.weight, this.bias] : [this.weight];
    }

    forward(x) {
        let out = x.matmul(this.weight);
        if (this.bias) {
            out = out.add(this.bias);
        }
        return out;
    }
}

// Activation functions
export class ReLU extends Layer {
    forward(x) {
        return x.relu();
    }
}

export class Tanh extends Layer {
    forward(x) {
        return x.tanh();
    }
}

export class Softmax extends Layer {
    forward(x) {
        return x.softmax();
    }
}


// Dropout layer
export class Dropout extends Layer {
    constructor(p=0.5) {
        super();
        this.p = p;
        this.training = true;
    }

    forward(x) {
        if (!this.training || this.p === 0) {
            return x;
        }

        const mask = x.data.map(row =>
            row.map(() => Math.random() > this.p ? 1 / (1 - this.p) : 0)
        );
        
        const result = new Value(
            x.data.map((row, i) =>
                row.map((val, j) => val * mask[i][j])
            ),
            [x],
            "dropout"
        );

        result._backward = () => {
            for (let i = 0; i < x.data.length; i++) {
                for (let j = 0; j < x.data[0].length; j++) {
                    x.grad[i][j] += mask[i][j] * result.grad[i][j];
                }
            }
        };

        return result;
    }
}

// LayerNorm layer
export class LayerNorm extends Layer {
    constructor(normalized_shape, eps=1e-5) {
        super();
        this.normalized_shape = normalized_shape;
        this.eps = eps;
        
        this.gamma = new Value(Array(1).fill().map(() =>
            Array(normalized_shape).fill().map(() => 1)
        ));
        this.beta = new Value(Array(1).fill().map(() =>
            Array(normalized_shape).fill().map(() => 0)
        ));
        
        this._parameters.push(this.gamma);
        this._parameters.push(this.beta);
    }

    forward(x) {
        const mean = x.data.map(row =>
            row.reduce((a, b) => a + b, 0) / row.length
        );
        
        const variance = x.data.map((row, i) =>
            row.reduce((a, b) => a + Math.pow(b - mean[i], 2), 0) / row.length
        );
        
        const normalized = x.data.map((row, i) =>
            row.map(val => 
                (val - mean[i]) / Math.sqrt(variance[i] + this.eps)
            )
        );
        
        const result = new Value(
            normalized.map((row, i) =>
                row.map((val, j) => 
                    val * this.gamma.data[0][j] + this.beta.data[0][j]
                )
            ),
            [x, this.gamma, this.beta],
            "layernorm"
        );

        const n = x.data[0].length;
        result._backward = () => {
            for (let i = 0; i < x.data.length; i++) {
                for (let j = 0; j < n; j++) {
                    const dx = result.grad[i][j] * this.gamma.data[0][j] / 
                        Math.sqrt(variance[i] + this.eps);
                    x.grad[i][j] += dx;
                    this.gamma.grad[0][j] += result.grad[i][j] * normalized[i][j];
                    this.beta.grad[0][j] += result.grad[i][j];
                }
            }
        };

        return result;
    }
}


// Sequential container
export class Model extends Layer{
    constructor(layers) {
        super();
        this.layers = Array.isArray(layers) ? layers : Array.from(arguments);
        this._parameters = this.layers.flatMap(layer => 
            layer.parameters ? layer.parameters() : []
        );
    }

    save(filename) {
        const modelData = {
            layers: this.layers.map(layer => {
                if (layer instanceof Linear) {
                    return {
                        type: "Linear",
                        in_features: layer.in_features,
                        out_features: layer.out_features,
                        bias: layer.bias !== null,
                        weight: layer.weight.data,
                        bias_data: layer.bias ? layer.bias.data : null
                    };
                } else if (layer instanceof ReLU) {
                    return { type: "ReLU" };
                } else if (layer instanceof Tanh) {
                    return { type: "Tanh" };
                } else if (layer instanceof Softmax) {
                    return { type: "Softmax" };
                } else if (layer instanceof Dropout) {
                    return { type: "Dropout", p: layer.p };
                } else if (layer instanceof LayerNorm) {
                    return {
                        type: "LayerNorm",
                        normalized_shape: layer.normalized_shape,
                        eps: layer.eps,
                        gamma: layer.gamma.data,
                        beta: layer.beta.data
                    };
                }
                return null
            }).filter(item => item !== null)
        };

        const jsonString = JSON.stringify(modelData);
        // In a browser environment, you'd typically download this:
     //  const blob = new Blob([jsonString], { type: 'application/json' });
     //   const link = document.createElement('a');
     //   link.href = URL.createObjectURL(blob);
     //   link.download = filename || 'model.json';
     //   link.click();
     //   URL.revokeObjectURL(link.href);

         // In a Node.js environment, you can use fs.writeFileSync
        writeFileSync(filename || 'model.json', jsonString);
    }

    static load(filename) {
        const jsonData = readFileSync(filename, 'utf-8');

        const modelData = JSON.parse(jsonData);
        const layers = modelData.layers.map(layerData => {
            switch (layerData.type) {
                case "Linear":
                    const linearLayer = new Linear(layerData.in_features, layerData.out_features, layerData.bias);
                    linearLayer.weight = new Value(layerData.weight);
                    if (layerData.bias) {
                        linearLayer.bias = new Value(layerData.bias_data);
                    }
                    return linearLayer;
                case "ReLU":
                    return new ReLU();
                case "Tanh":
                    return new Tanh();
                case "Softmax":
                    return new Softmax();
                case "Dropout":
                    return new Dropout(layerData.p);
                case "LayerNorm":
                    const layerNormLayer = new LayerNorm(layerData.normalized_shape, layerData.eps);
                    layerNormLayer.gamma = new Value(layerData.gamma);
                    layerNormLayer.beta = new Value(layerData.beta);
                    return layerNormLayer;
                default:
                    throw new Error(`Unknown layer type: ${layerData.type}`);
            }
        });
        return new Model(layers);
    }

    forward(x) {
        return this.layers.reduce((input, layer) => layer.forward(input), x);
    }

    zero_grad() {
        for (const param of this.parameters()) {
            param.grad = Array(param.grad.length).fill()
                .map(() => Array(param.grad[0].length).fill(0));
        }
    }
}