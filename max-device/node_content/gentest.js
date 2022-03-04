const Generator = require('./src/generator');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

async function main() {
  const modelPath = path.join(__dirname, 'model', 'generator_mod_v2');
  const generator = new Generator(modelPath);

  console.log(`Loading model from ${modelPath}`);
  await generator.loadGraph();
  console.log('Model loaded!');

  if(generator.isReady) {
    const vector = [0, 0, 0, 0, 0, 0, 0, 0];
    const samples = generator.generateWaveform(vector);
    console.log(samples);
  }
}

main();