class Value {
    constructor(data, children = null, op = "", label = "") {
      this.data = Array.isArray(data) ? data : [[data]];
      this.grad = Array.isArray(data) 
        ? Array(data.length).fill().map(() => Array(data[0].length).fill(0))
        : [[0]];
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
        row.map((val, j) => val + rhs.data[i % m2][j % n2])
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
  
    mul(other) {
      other = other instanceof Value ? other : new Value(other);
      const result = new Value(
        this.data.map((row, i) => row.map((val, j) => val * other.data[i][j])),
        [this, other],
        "*"
      );
  
      result._backward = () => {
        for (let i = 0; i < this.data.length; i++) {
          for (let j = 0; j < this.data[0].length; j++) {
            this.grad[i][j] += other.data[i][j] * result.grad[i][j];
            other.grad[i][j] += this.data[i][j] * result.grad[i][j];
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
  
    exp() {
      const result = new Value(
        this.data.map(row => row.map(Math.exp)),
        [this],
        "exp"
      );
      result._backward = () => {
        for (let i = 0; i < this.data.length; i++) {
          for (let j = 0; j < this.data[0].length; j++) {
            this.grad[i][j] += result.data[i][j] * result.grad[i][j];
          }
        }
      };
      return result;
    }
  
    log() {
      const result = new Value(
        this.data.map(row => row.map(Math.log)),
        [this],
        "log"
      );
      result._backward = () => {
        for (let i = 0; i < this.data.length; i++) {
          for (let j = 0; j < this.data[0].length; j++) {
            this.grad[i][j] += (1 / this.data[i][j]) * result.grad[i][j];
          }
        }
      };
      return result;
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
      const result = new Value(
        this.data.map(row => row.map(val => Math.pow(val, k))),
        [this],
        `**${k}`
      );
      result._backward = () => {
        for (let i = 0; i < this.data.length; i++) {
          for (let j = 0; j < this.data[0].length; j++) {
            this.grad[i][j] += k * Math.pow(this.data[i][j], k - 1) * result.grad[i][j];
          }
        }
      };
      return result;
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
  
      const mean = result_data.flat().reduce((a, b) => a + b) / 
                  (this.data.length * this.data[0].length);
  
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
  
    crossEntropy(target) {
      if (!(target instanceof Value)) {
        target = new Value([[target]]);
      }
  
      let total_loss = 0;
      const batch_size = this.data.length;
  
      for (let i = 0; i < batch_size; i++) {
        for (let j = 0; j < this.data[0].length; j++) {
          const pred = Math.max(Math.min(this.data[i][j], 1 - 1e-7), 1e-7);
          total_loss += -target.data[i][j] * Math.log(pred);
        }
      }
  
      const mean_loss = total_loss / batch_size;
      const result = new Value([[mean_loss]], [this, target], "cross_entropy");
  
      result._backward = () => {
        for (let i = 0; i < this.data.length; i++) {
          for (let j = 0; j < this.data[0].length; j++) {
            const pred = Math.max(Math.min(this.data[i][j], 1 - 1e-7), 1e-7);
            this.grad[i][j] += ((pred - target.data[i][j]) / batch_size) * result.grad[0][0];
          }
        }
      };
      return result;
    }
  
    accuracy(target) {
      if (!(target instanceof Value)) {
        target = new Value([[target]]);
      }
  
      const predictions = this.data.map(row => {
        const maxIndex = row.indexOf(Math.max(...row));
        return maxIndex;
      });
  
      const targetClasses = target.data.map(row => {
        const maxIndex = row.indexOf(Math.max(...row));
        return maxIndex;
      });
  
      let correct = 0;
      for (let i = 0; i < predictions.length; i++) {
        if (predictions[i] === targetClasses[i]) {
          correct++;
        }
      }
  
      return correct / predictions.length;
    }
  
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