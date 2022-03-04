const tf = require('@tensorflow/tfjs-node');

class Generator {

  constructor(modelPath) {
    this.modelPath = modelPath;
    this.ready = false;
    this.loading = false;
  }

  get isReady() {
    return this.ready;
  }

  get isLoading() {
    return this.loading;
  }

  async loadGraph() {
    this.loading = true;
    this.model = await tf.node.loadSavedModel(this.modelPath);
    this.ready = true;
    this.loading = false;
  }

  generateWaveform(params) {
    this.ready = false;
    const inputs = tf.tensor(params, [1, 8]);
    const out = this.model.predict(inputs).arraySync();
    const flat = tf.util.flatten(out);
    this.ready = true;
    return flat;
  }
}

module.exports = Generator;