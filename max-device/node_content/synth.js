const Generator = require('./src/generator');
const maxApi = require('max-api');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

async function main() {
  const modelPath = path.join(__dirname, 'model', 'generator_mod_v3');
  const generator = new Generator(modelPath);
  const maxSamples = 16384;

  maxApi.post(`Loading model from ${modelPath}`);
  await generator.loadGraph();
  maxApi.post('Model loaded!');

  maxApi.addHandler(maxApi.MESSAGE_TYPES.LIST, (a, b, c, d, e, f, g, h) => {
    if(generator.isReady) {
      const vector = [a, b, c, d, e, f, g, h];
      const samples = generator.generateWaveform(vector);

      maxApi.outlet(samples);
    }
  });
}

try {
  main();
} catch (e) {
  console.error(e);
  maxApi.post(e);
  process.exit(1);
}