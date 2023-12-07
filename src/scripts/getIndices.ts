/**
 * Returns an array with three subsets of random indices to create the train, test and validation datasets to train the model.
 * @param features The number of samples of the dataset to use.
 */
export function getIndices(
  features: number,
  options: { percent?: number } = {},
) {
  const { percent = 0.7 } = options;
  const randomIndex = getRandomIndicesArray(features);
  const limit = Math.floor(features * percent);
  const trainIndex: number[] = [];
  const testIndex: number[] = [];
  for (let i = 0; i < randomIndex.length; i++) {
    if (i < limit) {
      trainIndex.push(randomIndex[i]);
    } else {
      testIndex.push(randomIndex[i]);
    }
  }
  return [trainIndex, testIndex];
}

function getRandomIndicesArray(length: number): number[] {
  const numbers = Array.from({ length }, (_, i) => i);
  for (let i = numbers.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [numbers[i], numbers[j]] = [numbers[j], numbers[i]];
  }
  return numbers.slice(0, length);
}
