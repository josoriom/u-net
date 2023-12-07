export function randomColors(n: number) {
  const line: number[][] = [];
  for (let i = 0; i < n; i++) {
    const color: number[] = [];
    for (let j = 0; j < 3; j++) {
      color.push(Math.floor(Math.random() * 256));
    }
    line.push(color);
  }
  return line;
}
