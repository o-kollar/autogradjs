
class Value {

    constructor(data, children=null, op="", label="") {
    
    this.data = Array.isArray(data) ? data : [[data]];
    
    this.grad = Array.isArray(data) ?
    
    Array(data.length).fill().map(() => Array(data[0].length).fill(0)) :
    
    [[0]];
    
    this.children = children != null ? children : [];
    
    this.op = op;
    
    this.label = label;
    
    this._backward = () => {};
    
    }
    
    
    matmul(rhs) {
    
    if (!(rhs instanceof Value)) {
    
    rhs = new Value([[rhs]]);
    
    }
    
    const m = this.data.length;
    
    const n = this.data[0].length;
    
    const p = rhs.data[0].length;
    
    let result_data = Array(m).fill().map(() => Array(p).fill(0));
    
    for (let i = 0; i < m; i++) {
    
    for (let j = 0; j < p; j++) {
    
    for (let k = 0; k < n; k++) {
    
    result_data[i][j] += this.data[i][k] * rhs.data[k][j];
    
    }
    
    }
    
    }
    
    const result = new Value(result_data, [this, rhs], "matmul");
    
    result._backward = () => {
    
    for (let i = 0; i < m; i++) {
    
    for (let j = 0; j < p; j++) {
    
    for (let k = 0; k < n; k++) {
    
    this.grad[i][k] += result.grad[i][j] * rhs.data[k][j];
    
    rhs.grad[k][j] += result.grad[i][j] * this.data[i][k];
    
    }
    
    }
    
    }
    
    };
    
    return result;
    
    }
    
    
    add(rhs) {
    
    if (!(rhs instanceof Value)) {
    
    rhs = new Value([[rhs]]);
    
    }
    
    const m = this.data.length;
    
    const n = this.data[0].length;
    
    const m2 = rhs.data.length;
    
    const n2 = rhs.data[0].length;
    
    const result_data = this.data.map((row, i) =>
    
    row.map((val, j) => {
    
    const rhsValue = rhs.data[i % m2][j % n2];
    
    return val + rhsValue;
    
    })
    
    );
    
    const result = new Value(result_data, [this, rhs], "+");
    
    result._backward = () => {
    
    for (let i = 0; i < m; i++) {
    
    for (let j = 0; j < n; j++) {
    
    this.grad[i][j] += result.grad[i][j];
    
    rhs.grad[i % m2][j % n2] += result.grad[i][j];
    
    }
    
    }
    
    };
    
    return result;
    
    }
    
    
    relu() {
    
    const result_data = this.data.map(row =>
    
    row.map(val => Math.max(0, val))
    
    );
    
    const result = new Value(result_data, [this], "relu");
    
    result._backward = () => {
    
    for (let i = 0; i < this.data.length; i++) {
    
    for (let j = 0; j < this.data[0].length; j++) {
    
    this.grad[i][j] += (this.data[i][j] > 0 ? 1 : 0) * result.grad[i][j];
    
    }
    
    }
    
    };
    
    return result;
    
    }
    
    
    tanh() {
    
    const result_data = this.data.map(row =>
    
    row.map(val => {
    
    const exp2x = Math.exp(2 * val);
    
    return (exp2x - 1) / (exp2x + 1);
    
    })
    
    );
    
    const result = new Value(result_data, [this], "tanh");
    
    result._backward = () => {
    
    for (let i = 0; i < this.data.length; i++) {
    
    for (let j = 0; j < this.data[0].length; j++) {
    
    const val = result_data[i][j];
    
    this.grad[i][j] += (1 - val * val) * result.grad[i][j];
    
    }
    
    }
    
    };
    
    return result;
    
    }
    
    
    mul(other) {
    
    other = other instanceof Value ? other : new Value(other);
    
    const out = new Value(this.data.map((row, i) => row.map((val, j) => val * other.data[i][j])), [this, other], '*');
    
    out._backward = () => {
    
    for (let i = 0; i < this.data.length; i++) {
    
    for (let j = 0; j < this.data[0].length; j++) {
    
    this.grad[i][j] += other.data[i][j] * out.grad[i][j];
    
    other.grad[i][j] += this.data[i][j] * out.grad[i][j];
    
    }
    
    }
    
    };
    
    return out;
    
    }
    
    
    exp() {
    
    const out = new Value(this.data.map(row => row.map(Math.exp)), [this], 'exp');
    
    out._backward = () => {
    
    for (let i = 0; i < this.data.length; i++) {
    
    for (let j = 0; j < this.data[0].length; j++) {
    
    this.grad[i][j] += out.data[i][j] * out.grad[i][j];
    
    }
    
    }
    
    };
    
    return out;
    
    }
    
    
    log() {
    
    const out = new Value(this.data.map(row => row.map(Math.log)), [this], 'log');
    
    out._backward = () => {
    
    for (let i = 0; i < this.data.length; i++) {
    
    for (let j = 0; j < this.data[0].length; j++) {
    
    this.grad[i][j] += (1 / this.data[i][j]) * out.grad[i][j];
    
    }
    
    }
    
    };
    
    return out;
    
    }
    
    
    neg() {
    
    return this.mul(-1);
    
    }
    
    
    sub(other) {
    
    return this.add(other.neg());
    
    }
    
    
    div(other) {
    
    return this.mul(other.pow(-1));
    
    }
    
    
    pow(k) {
    
    const out = new Value(this.data.map(row => row.map(val => Math.pow(val, k))), [this], `**${k}`);
    
    out._backward = () => {
    
    for (let i = 0; i < this.data.length; i++) {
    
    for (let j = 0; j < this.data[0].length; j++) {
    
    this.grad[i][j] += (k * Math.pow(this.data[i][j], k - 1)) * out.grad[i][j];
    
    }
    
    }
    
    };
    
    return out;
    
    }
    
    
    
    softmax() {
    
    const result_data = this.data.map(row => {
    
    const maxVal = Math.max(...row);
    
    const expVals = row.map(val => Math.exp(val - maxVal));
    
    const sumExp = expVals.reduce((a, b) => a + b, 0);
    
    return expVals.map(val => val / sumExp);
    
    });
    
    
    const result = new Value(result_data, [this], "softmax");
    
    
    result._backward = () => {
    
    for (let i = 0; i < this.data.length; i++) {
    
    for (let j = 0; j < this.data[0].length; j++) {
    
    const smVal = result_data[i][j];
    
    this.grad[i][j] += smVal * (1 - smVal) * result.grad[i][j];
    
    }
    
    }
    
    };
    
    
    return result;
    
    }
    
    
    mse(target) {
    
    if (!(target instanceof Value)) {
    
    target = new Value([[target]]);
    
    }
    
    const result_data = this.data.map((row, i) =>
    
    row.map((val, j) => {
    
    const diff = val - target.data[i][j];
    
    return diff * diff;
    
    })
    
    );
    
    const mean = result_data.flat().reduce((a, b) => a + b) / (this.data.length * this.data[0].length);
    
    const result = new Value([[mean]], [this, target], "mse");
    
    result._backward = () => {
    
    const scale = 2 / (this.data.length * this.data[0].length);
    
    for (let i = 0; i < this.data.length; i++) {
    
    for (let j = 0; j < this.data[0].length; j++) {
    
    this.grad[i][j] += scale * (this.data[i][j] - target.data[i][j]) * result.grad[0][0];
    
    }
    
    }
    
    };
    
    return result;
    
    }
    
    
    huberLoss(target, delta = 1.0) {
    
    if (!(target instanceof Value)) {
    
    target = new Value([[target]]);
    
    }
    
    let total_loss = 0;
    
    const batch_size = this.data.length;
    
    for (let i = 0; i < batch_size; i++) {
    
    for (let j = 0; j < this.data[0].length; j++) {
    
    const diff = Math.abs(this.data[i][j] - target.data[i][j]);
    
    total_loss += diff <= delta ?
    
    0.5 * diff * diff :
    
    delta * diff - 0.5 * delta * delta;
    
    }
    
    }
    
    const mean_loss = total_loss / batch_size;
    
    const result = new Value([[mean_loss]], [this, target], "huber");
    
    result._backward = () => {
    
    for (let i = 0; i < this.data.length; i++) {
    
    for (let j = 0; j < this.data[0].length; j++) {
    
    const diff = this.data[i][j] - target.data[i][j];
    
    this.grad[i][j] += (Math.abs(diff) <= delta ?
    
    diff : delta * Math.sign(diff)) / batch_size * result.grad[0][0];
    
    }
    
    }
    
    };
    
    return result;
    
    }
    
    
    // Modify the crossEntropy method in the Value class
    
    crossEntropy(target) {
    
    if (!(target instanceof Value)) {
    
    target = new Value([[target]]);
    
    }
    
    // Calculate cross entropy loss for each sample in batch
    
    let total_loss = 0;
    
    const batch_size = this.data.length;
    
    for (let i = 0; i < batch_size; i++) {
    
    for (let j = 0; j < this.data[0].length; j++) {
    
    // Add small epsilon to avoid log(0)
    
    const pred = Math.max(Math.min(this.data[i][j], 1 - 1e-7), 1e-7);
    
    total_loss += -target.data[i][j] * Math.log(pred);
    
    }
    
    }
    
    // Average loss over batch
    
    const mean_loss = total_loss / batch_size;
    
    const result = new Value([[mean_loss]], [this, target], "cross_entropy");
    
    result._backward = () => {
    
    for (let i = 0; i < this.data.length; i++) {
    
    for (let j = 0; j < this.data[0].length; j++) {
    
    // Gradient of cross entropy with respect to input
    
    const pred = Math.max(Math.min(this.data[i][j], 1 - 1e-7), 1e-7);
    
    this.grad[i][j] += (pred - target.data[i][j]) / batch_size * result.grad[0][0];
    
    }
    
    }
    
    };
    
    return result;
    
    }
    
    
    
    accuracy(target) {
    
    if (!(target instanceof Value)) {
    
    target = new Value([[target]]);
    
    }
    
    // Get predicted class (maximum probability)
    
    const predictions = this.data.map(row => {
    
    const maxIndex = row.indexOf(Math.max(...row));
    
    return maxIndex;
    
    });
    
    // Get target class
    
    const targetClasses = target.data.map(row => {
    
    const maxIndex = row.indexOf(Math.max(...row));
    
    return maxIndex;
    
    });
    
    // Calculate accuracy
    
    let correct = 0;
    
    for (let i = 0; i < predictions.length; i++) {
    
    if (predictions[i] === targetClasses[i]) {
    
    correct++;
    
    }
    
    }
    
    return correct / predictions.length;
    
    }
    
    
    // Helper method to convert labels to one-hot encoding
    
    static oneHot(labels, numClasses) {
    
    return new Value(
    
    labels.map(label => {
    
    const encoding = Array(numClasses).fill(0);
    
    encoding[label] = 1;
    
    return encoding;
    
    })
    
    );
    
    }
    
    
    backward() {
    
    const seen = new Set();
    
    const nodes = [];
    
    const sort = (root) => {
    
    if (!seen.has(root)) {
    
    seen.add(root);
    
    for (const child of root.children) {
    
    sort(child);
    
    }
    
    nodes.push(root);
    
    }
    
    };
    
    sort(this);
    
    nodes.reverse();
    
    this.grad[0][0] = 1;
    
    for (const node of nodes) {
    
    node._backward();
    
    }
    
    }
    
    }
export default Value;