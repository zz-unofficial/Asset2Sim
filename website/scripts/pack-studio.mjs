import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';

const scene = new URL('../public/studio-scene/', import.meta.url);
const output = new URL('packed/', scene);
mkdirSync(output, { recursive: true });
const read = (name) => readFileSync(new URL(name, scene));
const files = JSON.parse(read('files.json')).filter((name) => !name.startsWith('object_'));
const objects = JSON.parse(read('demo-assets.json'));
files.push(...Object.values(objects).flatMap((asset) => asset.collision));

// Pack a little-endian byte count and the raw data for each file in manifest order.
const chunks = files.flatMap((name) => {
  const data = read('assets/' + name);
  const size = Buffer.alloc(4);
  size.writeUInt32LE(data.length);
  return [size, data];
});
const wasm = readFileSync(new URL('../node_modules/@mujoco/mujoco/mujoco.wasm', import.meta.url));
// Use .bin names so local servers leave decompression to the browser code.
for (const [name, data] of [['meshes-gzip.bin', Buffer.concat(chunks)], ['mujoco-gzip.bin', wasm]]) {
  const compressed = gzipSync(data, { level: 6 });
  writeFileSync(new URL(name, output), compressed);
  console.log(`Studio ${name}: ${(data.length / 1048576).toFixed(1)} → ${(compressed.length / 1048576).toFixed(1)} MiB`);
}

// Remove files emitted by the previous pack naming scheme.
for (const name of ['meshes.bin.gz', 'mujoco.wasm.gz']) {
  rmSync(new URL(name, output), { force: true });
}
