
export class Optimizer {
    constructor(parameters, lr=0.01) {
        this.parameters = parameters;
        this.lr = lr;
    }

    step() {
        throw new Error("Not implemented");
    }

    zero_grad() {
        for (const param of this.parameters) {
            param.grad = Array(param.grad.length).fill()
                .map(() => Array(param.grad[0].length).fill(0));
        }
    }
}

// SGD with momentum and weight decay
export class SGD extends Optimizer {
    constructor(parameters, lr=0.01, momentum=0, weight_decay=0) {
        super(parameters, lr);
        this.momentum = momentum;
        this.weight_decay = weight_decay;
        this.velocities = parameters.map(param => 
            Array(param.data.length).fill()
                .map(() => Array(param.data[0].length).fill(0))
        );
    }

    step() {
        this.parameters.forEach((param, i) => {
            for (let row = 0; row < param.data.length; row++) {
                for (let col = 0; col < param.data[0].length; col++) {
                    if (this.weight_decay !== 0) {
                        param.grad[row][col] += this.weight_decay * param.data[row][col];
                    }

                    this.velocities[i][row][col] = 
                        this.momentum * this.velocities[i][row][col] - 
                        this.lr * param.grad[row][col];
                    
                    param.data[row][col] += this.velocities[i][row][col];
                }
            }
        });
    }
}

// Adam optimizer
export class Adam extends Optimizer {
    constructor(parameters, lr=0.001, betas=[0.9, 0.999], eps=1e-8, weight_decay=0) {
        super(parameters, lr);
        this.betas = betas;
        this.eps = eps;
        this.weight_decay = weight_decay;
        this.t = 0;

        // Initialize momentum and velocity terms
        this.m = parameters.map(param => 
            Array(param.data.length).fill()
                .map(() => Array(param.data[0].length).fill(0))
        );
        this.v = parameters.map(param => 
            Array(param.data.length).fill()
                .map(() => Array(param.data[0].length).fill(0))
        );
    }

    step() {
        this.t += 1;
        const [beta1, beta2] = this.betas;

        this.parameters.forEach((param, i) => {
            for (let row = 0; row < param.data.length; row++) {
                for (let col = 0; col < param.data[0].length; col++) {
                    if (this.weight_decay !== 0) {
                        param.grad[row][col] += this.weight_decay * param.data[row][col];
                    }

                    // Update biased first moment estimate
                    this.m[i][row][col] = beta1 * this.m[i][row][col] + 
                        (1 - beta1) * param.grad[row][col];

                    // Update biased second raw moment estimate
                    this.v[i][row][col] = beta2 * this.v[i][row][col] + 
                        (1 - beta2) * param.grad[row][col] * param.grad[row][col];

                    // Compute bias-corrected first moment estimate
                    const m_hat = this.m[i][row][col] / (1 - Math.pow(beta1, this.t));

                    // Compute bias-corrected second raw moment estimate
                    const v_hat = this.v[i][row][col] / (1 - Math.pow(beta2, this.t));

                    // Update parameters
                    param.data[row][col] -= this.lr * m_hat / (Math.sqrt(v_hat) + this.eps);
                }
            }
        });
    }
}

// RMSprop optimizer
export class RMSprop extends Optimizer {
    constructor(parameters, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0) {
        super(parameters, lr);
        this.alpha = alpha;
        this.eps = eps;
        this.weight_decay = weight_decay;

        this.square_avg = parameters.map(param => 
            Array(param.data.length).fill()
                .map(() => Array(param.data[0].length).fill(0))
        );
    }

    step() {
        this.parameters.forEach((param, i) => {
            for (let row = 0; row < param.data.length; row++) {
                for (let col = 0; col < param.data[0].length; col++) {
                    if (this.weight_decay !== 0) {
                        param.grad[row][col] += this.weight_decay * param.data[row][col];
                    }

                    this.square_avg[i][row][col] = 
                        this.alpha * this.square_avg[i][row][col] + 
                        (1 - this.alpha) * param.grad[row][col] * param.grad[row][col];

                    param.data[row][col] -= this.lr * param.grad[row][col] / 
                        (Math.sqrt(this.square_avg[i][row][col]) + this.eps);
                }
            }
        });
    }
}

// AdaGrad optimizer
export class Adagrad extends Optimizer {
    constructor(parameters, lr=0.01, eps=1e-10, weight_decay=0) {
        super(parameters, lr);
        this.eps = eps;
        this.weight_decay = weight_decay;

        this.sum_squares = parameters.map(param => 
            Array(param.data.length).fill()
                .map(() => Array(param.data[0].length).fill(0))
        );
    }

    step() {
        this.parameters.forEach((param, i) => {
            for (let row = 0; row < param.data.length; row++) {
                for (let col = 0; col < param.data[0].length; col++) {
                    if (this.weight_decay !== 0) {
                        param.grad[row][col] += this.weight_decay * param.data[row][col];
                    }

                    this.sum_squares[i][row][col] += param.grad[row][col] * param.grad[row][col];

                    param.data[row][col] -= this.lr * param.grad[row][col] / 
                        (Math.sqrt(this.sum_squares[i][row][col]) + this.eps);
                }
            }
        });
    }
}