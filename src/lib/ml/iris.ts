import * as tf from '@tensorflow/tfjs';

export interface IrisData {
  sepal_length: number;
  sepal_width: number;
  petal_length: number;
  petal_width: number;
  species: string;
}

export const IRIS_CLASSES = ['setosa', 'versicolor', 'virginica'];

/**
 * Loads the Iris dataset.
 * Since we are in a browser/node environment, we'll use a standard JSON-like Iris data.
 */
export async function loadIrisData(): Promise<IrisData[]> {
  // Standard Iris dataset
  // In a real MLOps app, we might fetch this from a DVC-tracked source or a URI.
  const response = await fetch('https://raw.githubusercontent.com/domoritz/iris/master/iris.json');
  const data = await response.json();
  return data.map((item: any) => ({
    sepal_length: item.sepalLength,
    sepal_width: item.sepalWidth,
    petal_length: item.petalLength,
    petal_width: item.petalWidth,
    species: item.species
  }));
}

export function preprocessData(data: IrisData[]) {
  return tf.tidy(() => {
    // Shuffling
    tf.util.shuffle(data);

    const inputs = data.map(d => [d.sepal_length, d.sepal_width, d.petal_length, d.petal_width]);
    const labels = data.map(d => IRIS_CLASSES.indexOf(d.species));

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 4]);
    const labelTensor = tf.oneHot(tf.tensor1d(labels, 'int32'), 3);

    return { inputs: inputTensor, labels: labelTensor };
  });
}
