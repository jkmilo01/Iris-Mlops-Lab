import * as tf from '@tensorflow/tfjs';
import { loadIrisData, preprocessData } from './iris';

export interface TrainingMetrics {
  epoch: number;
  loss: number;
  acc: number;
}

export type ProgressCallback = (metrics: TrainingMetrics) => void;

export class MLEngine {
  private model: tf.LayersModel | null = null;

  async createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [4], units: 10, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    this.model = model;
    return model;
  }

  async train(epochs: number = 50, onProgress?: ProgressCallback) {
    const data = await loadIrisData();
    const { inputs, labels } = preprocessData(data);

    if (!this.model) await this.createModel();

    await this.model!.fit(inputs, labels, {
      epochs,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (onProgress && logs) {
            onProgress({
              epoch,
              loss: logs.loss,
              acc: logs.acc,
            });
          }
        },
      },
    });

    // Cleanup
    inputs.dispose();
    labels.dispose();

    return this.model;
  }

  async predict(input: number[]) {
    if (!this.model) throw new Error('Model not trained');
    
    return tf.tidy(() => {
      const inputTensor = tf.tensor2d([input], [1, 4]);
      const prediction = this.model!.predict(inputTensor) as tf.Tensor;
      const index = prediction.argMax(1).dataSync()[0];
      const probs = prediction.dataSync();
      return { index, probs };
    });
  }

  async saveModel() {
    if (!this.model) return;
    // In browser, we store in localstorage or indexedDB
    await this.model.save('localstorage://iris-model-v1');
  }

  async loadModel() {
    try {
      this.model = await tf.loadLayersModel('localstorage://iris-model-v1');
      return true;
    } catch (e) {
      return false;
    }
  }
}

export const engine = new MLEngine();
