import { writeFileSync } from 'fs';

import { Image, encode } from 'image-js';

import { randomColors } from './utilities/randomColors.js';

const size = 256;
const samples = 50;
const objects = 20;
const radius = 15;
for (let j = 0; j < samples; j++) {
  const colors = randomColors(objects);
  const x = new Image(size, size).fill(255);
  const y = x.clone();
  for (let i = 0; i < objects; i++) {
    const row = getNumber();
    const column = getNumber();
    x.drawCircle({ row, column }, radius, {
      color: colors[i],
      fill: colors[i],
      out: x,
    });
    y.drawCircle({ row, column }, radius, {
      color: [0, 0, 0],
      fill: [0, 0, 0],
      out: y,
    });
  }
  writeFileSync(
    new URL(`../data/train/x/${j}.png`, import.meta.url),
    encode(x),
  );
  writeFileSync(
    new URL(`../data/train/y/${j}.png`, import.meta.url),
    encode(y),
  );
}

function getNumber() {
  return Math.floor(Math.random() * 256);
}
