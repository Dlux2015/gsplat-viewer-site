// Depth-sort worker: back-to-front ordering of splats for correct alpha
// blending. 16-bit counting sort over quantized view-space depth — O(n),
// ~20ms for 3M splats, and it runs HERE so the main thread never stalls.
//
// Protocol:
//   {type:"init", positions: Float32Array(n*3)}   — keeps a copy
//   {type:"sort", view: Float32Array(16)}         — column-major view matrix
// Replies: {type:"sorted", indices: Float32Array(n)} (transferred)

let positions = null;
let n = 0;

const BUCKETS = 65536; // 16-bit quantization
const counts = new Uint32Array(BUCKETS);

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    positions = msg.positions;
    n = positions.length / 3;
    return;
  }
  if (msg.type !== "sort" || !positions) return;

  const v = msg.view;
  // view-space z of point p = row 3 of view matrix · p (column-major indices)
  const vz0 = v[2], vz1 = v[6], vz2 = v[10], vz3 = v[14];

  // Pass 1: depth range (for stable quantization regardless of scene scale).
  let minD = Infinity, maxD = -Infinity;
  const depths = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = vz0 * positions[3 * i] + vz1 * positions[3 * i + 1] + vz2 * positions[3 * i + 2] + vz3;
    depths[i] = d;
    if (d < minD) minD = d;
    if (d > maxD) maxD = d;
  }
  const scale = (BUCKETS - 1) / Math.max(maxD - minD, 1e-9);

  // Pass 2: counting sort. Camera looks down -z in view space, so the most
  // distant splat has the MOST NEGATIVE z; ascending bucket order = far
  // first = back-to-front, which is exactly the blend order we need.
  counts.fill(0);
  const bucket = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    const b = ((depths[i] - minD) * scale) | 0;
    bucket[i] = b;
    counts[b]++;
  }
  let acc = 0;
  const starts = new Uint32Array(BUCKETS);
  for (let b = 0; b < BUCKETS; b++) {
    starts[b] = acc;
    acc += counts[b];
  }
  // Float32 indices because the instanced vertex attribute is float.
  const indices = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    indices[starts[bucket[i]]++] = i;
  }
  self.postMessage({ type: "sorted", indices }, [indices.buffer]);
};
