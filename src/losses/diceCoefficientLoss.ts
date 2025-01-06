import { Tensor, sum, scalar } from '@tensorflow/tfjs-node';

export function diceCoefficientLoss(yTrue: Tensor, yPred: Tensor): Tensor {
  const intersection = sum(yTrue.mul(yPred)).cast('float32');
  const union = sum(yTrue.add(yPred)).cast('float32');
  const dice = scalar(1.0).sub(
    intersection.mul(scalar(2.0)).div(union.add(intersection)),
  );
  return dice;
}
