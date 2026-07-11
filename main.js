// Gaussian splat viewer: instanced quads + EWA covariance projection.
// Data path: .splat file (32 B/gaussian, antimatter15 layout) -> data
// textures; a Web Worker keeps splats depth-sorted back-to-front.
//
// Projection math follows the INRIA 3DGS rasterizer and antimatter15's
// splat viewer (both heavily field-tested) — see shader comments.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const TEX_WIDTH = 2048; // shader indexes texels as (i & 2047, i >> 11)

const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");
const message = document.getElementById("message");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x111111, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
camera.position.set(0, 0, 5);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; // orbit=drag, zoom=wheel, pan=right-drag (built in)

let mesh = null;
// Debug handle (harmless in production; invaluable when the screen is black).
window.__viewer = { renderer, scene, camera, controls, get mesh() { return mesh; } };
let splatCount = 0;
let sceneName = "—";
let worker = null;
let workerBusy = false;
let lastSortedView = new THREE.Matrix4().makeScale(0, 0, 0); // force first sort

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

const vertexShader = /* glsl */ `
  precision highp float;
  precision highp int;

  // Per-splat data lives in textures; each instance carries only its
  // (sorted) splat index. texPos: xyz | texCovA: cov(xx,xy,xz,yy) |
  // texCovB: cov(yz,zz) | texColor: rgba8.
  uniform sampler2D texPos;
  uniform sampler2D texCovA;
  uniform sampler2D texCovB;
  uniform sampler2D texColor;
  uniform vec2 uViewport; // px
  uniform vec2 uFocal;    // (fx, fy) px, derived from projectionMatrix

  in float splatIndex;

  out vec4 vColor;
  out vec2 vLocal; // quad-local coords; |vLocal|=2 is the quad edge

  void main() {
    int idx = int(splatIndex);
    ivec2 uv = ivec2(idx & 2047, idx >> 11);

    vec4 center = texelFetch(texPos, uv, 0);
    vec4 cam = modelViewMatrix * vec4(center.xyz, 1.0);
    vec4 clip = projectionMatrix * cam;

    // Cull splats behind the camera (or the quad smears across the screen).
    if (clip.w <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); vColor = vec4(0.0); vLocal = vec2(0.0); return; }

    // 3D covariance (symmetric, 6 unique values, precomputed CPU-side
    // from scale+rotation as R S S^T R^T).
    vec4 ca = texelFetch(texCovA, uv, 0);
    vec4 cb = texelFetch(texCovB, uv, 0);
    mat3 Vrk = mat3(
      ca.x, ca.y, ca.z,
      ca.y, ca.w, cb.x,
      ca.z, cb.x, cb.y
    );

    // EWA splatting: project the 3D covariance to 2D screen space.
    // J is the Jacobian of the perspective projection evaluated at the
    // splat center (linearizes the projection locally):
    //   J = [ fx/z   0    -fx*x/z^2 ]
    //       [ 0     fy/z  -fy*y/z^2 ]
    // W is the world->camera rotation. cov2D = J W Vrk W^T J^T.
    float z = cam.z;
    float z2 = z * z;
    mat3 J = mat3(
      uFocal.x / z, 0.0, 0.0,                              // column 0
      0.0, uFocal.y / z, 0.0,                              // column 1
      -uFocal.x * cam.x / z2, -uFocal.y * cam.y / z2, 0.0  // column 2
    );
    mat3 W = mat3(modelViewMatrix);
    mat3 T = J * W;
    mat3 cov2d = T * Vrk * transpose(T);

    // +0.3px low-pass filter (as in INRIA): guarantees every splat covers
    // at least a pixel, prevents aliasing shimmer on tiny splats.
    float a = cov2d[0][0] + 0.3;
    float b = cov2d[0][1];
    float d = cov2d[1][1] + 0.3;

    // Eigen-decomposition of the 2x2 covariance -> ellipse axes in px.
    float mid = 0.5 * (a + d);
    float rad = length(vec2(0.5 * (a - d), b));
    float lambda1 = mid + rad;
    float lambda2 = max(mid - rad, 0.1);
    vec2 dir = (b == 0.0 && a >= d) ? vec2(1.0, 0.0) : normalize(vec2(b, lambda1 - a));
    // sqrt(2*lambda)*[-2,2] quad => extends to ~2.8 sigma; clamp so a
    // degenerate splat can't produce a screen-sized quad.
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * dir;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(dir.y, -dir.x);

    vColor = texelFetch(texColor, uv, 0);
    vLocal = position.xy; // corners at (+-2, +-2)

    vec2 ndcCenter = clip.xy / clip.w;
    // px -> NDC is 2/viewport; corner already spans +-2 so 1/viewport here.
    gl_Position = vec4(
      ndcCenter
        + position.x * majorAxis / uViewport
        + position.y * minorAxis / uViewport,
      0.0, 1.0);
  }
`;

const fragmentShader = /* glsl */ `
  precision highp float;

  in vec4 vColor;
  in vec2 vLocal;

  out vec4 fragColor;

  void main() {
    // Gaussian falloff. vLocal is scaled so exp(-dot) == exp(-r^2/(2s^2)):
    // without this the splats render as hard-edged squares.
    float A = -dot(vLocal, vLocal);
    if (A < -4.0) discard;
    float alpha = exp(A) * vColor.a;
    if (alpha < 1.0 / 255.0) discard;
    // Premultiplied alpha (blend ONE, ONE_MINUS_SRC_ALPHA).
    fragColor = vec4(vColor.rgb * alpha, alpha);
  }
`;

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

function parseSplat(buffer) {
  const n = Math.floor(buffer.byteLength / 32);
  const f32 = new Float32Array(buffer);
  const u8 = new Uint8Array(buffer);
  const positions = new Float32Array(n * 3);
  const texH = Math.ceil(n / TEX_WIDTH);
  const posTex = new Float32Array(TEX_WIDTH * texH * 4);
  const covA = new Float32Array(TEX_WIDTH * texH * 4);
  const covB = new Float32Array(TEX_WIDTH * texH * 4);
  const colTex = new Uint8Array(TEX_WIDTH * texH * 4);

  for (let i = 0; i < n; i++) {
    const f = i * 8; // 8 floats per record
    const x = f32[f], y = f32[f + 1], z = f32[f + 2];
    const sx = f32[f + 3], sy = f32[f + 4], sz = f32[f + 5];
    positions[3 * i] = x; positions[3 * i + 1] = y; positions[3 * i + 2] = z;
    posTex[4 * i] = x; posTex[4 * i + 1] = y; posTex[4 * i + 2] = z;

    const bx = i * 32;
    colTex[4 * i] = u8[bx + 24];
    colTex[4 * i + 1] = u8[bx + 25];
    colTex[4 * i + 2] = u8[bx + 26];
    colTex[4 * i + 3] = u8[bx + 27];

    // quat stored as round(q*128)+128
    const qw = (u8[bx + 28] - 128) / 128;
    const qx = (u8[bx + 29] - 128) / 128;
    const qy = (u8[bx + 30] - 128) / 128;
    const qz = (u8[bx + 31] - 128) / 128;

    // Cov3D = R S (R S)^T with S = diag(scale). Rotation matrix from quat:
    const r00 = 1 - 2 * (qy * qy + qz * qz), r01 = 2 * (qx * qy - qw * qz), r02 = 2 * (qx * qz + qw * qy);
    const r10 = 2 * (qx * qy + qw * qz), r11 = 1 - 2 * (qx * qx + qz * qz), r12 = 2 * (qy * qz - qw * qx);
    const r20 = 2 * (qx * qz - qw * qy), r21 = 2 * (qy * qz + qw * qx), r22 = 1 - 2 * (qx * qx + qy * qy);
    // M = R*S columns scaled
    const m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
    const m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
    const m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;
    covA[4 * i] = m00 * m00 + m01 * m01 + m02 * m02;     // xx
    covA[4 * i + 1] = m00 * m10 + m01 * m11 + m02 * m12; // xy
    covA[4 * i + 2] = m00 * m20 + m01 * m21 + m02 * m22; // xz
    covA[4 * i + 3] = m10 * m10 + m11 * m11 + m12 * m12; // yy
    covB[4 * i] = m10 * m20 + m11 * m21 + m12 * m22;     // yz
    covB[4 * i + 1] = m20 * m20 + m21 * m21 + m22 * m22; // zz
  }
  return { n, texH, positions, posTex, covA, covB, colTex };
}

function makeDataTexture(data, texH, type) {
  const tex = new THREE.DataTexture(data, TEX_WIDTH, texH, THREE.RGBAFormat, type);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  return tex;
}

function loadBuffer(name, buffer) {
  if (buffer.byteLength < 32 || buffer.byteLength % 32 !== 0) {
    showMessage(`${name}: not a .splat file (size ${buffer.byteLength} not a multiple of 32)`);
    return;
  }
  const { n, texH, positions, posTex, covA, covB, colTex } = parseSplat(buffer);

  if (mesh) {
    scene.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
  }

  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array([-2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2, 0]), 3)
  );
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  const indices = new Float32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;
  geometry.setAttribute("splatIndex", new THREE.InstancedBufferAttribute(indices, 1));
  geometry.instanceCount = n;

  const material = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3, // texelFetch/ivec2 need GLSL ES 3.00
    vertexShader,
    fragmentShader,
    uniforms: {
      texPos: { value: makeDataTexture(posTex, texH, THREE.FloatType) },
      texCovA: { value: makeDataTexture(covA, texH, THREE.FloatType) },
      texCovB: { value: makeDataTexture(covB, texH, THREE.FloatType) },
      texColor: { value: makeDataTexture(colTex, texH, THREE.UnsignedByteType) },
      uViewport: { value: new THREE.Vector2() },
      uFocal: { value: new THREE.Vector2() },
    },
    // The (dir.y, -dir.x) minor-axis perpendicular makes the corner->screen
    // mapping orientation-reversing (det = -|major||minor|), so the quads
    // rasterize clockwise. Raw-WebGL splat viewers get away with it because
    // CULL_FACE is off by default there; three.js culls back faces unless:
    side: THREE.DoubleSide,
    transparent: true,
    depthTest: true,
    depthWrite: false, // painter's algorithm via the worker sort
    blending: THREE.CustomBlending,
    blendSrc: THREE.OneFactor, // shader outputs premultiplied alpha
    blendDst: THREE.OneMinusSrcAlphaFactor,
  });

  mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false; // quads are positioned in the shader
  scene.add(mesh);
  splatCount = n;
  sceneName = name;

  autoFrame(positions, n);
  restartWorker(positions);
  showMessage(null);
}

// Auto-frame from the splat bounding box — spawning the camera inside the
// cloud is the #1 cause of "black screen, no errors".
function autoFrame(positions, n) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      const v = positions[3 * i + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  const center = new THREE.Vector3(
    (min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2
  );
  const radius = Math.max(
    new THREE.Vector3(max[0] - min[0], max[1] - min[1], max[2] - min[2]).length() / 2,
    1e-3
  );
  controls.target.copy(center);
  // COLMAP worlds are Y-down (image convention), so +Y-up cameras render
  // scenes upside down. Default to -Y up; the U key flips for other sources.
  camera.up.set(0, -1, 0);
  camera.position.copy(center).add(new THREE.Vector3(0.4, 0.3, 1).normalize().multiplyScalar(radius * 1.8));
  camera.near = radius / 1000;
  camera.far = radius * 100;
  camera.updateProjectionMatrix();
  controls.update();
}

window.addEventListener("keydown", (e) => {
  if (e.key === "u" || e.key === "U") {
    camera.up.y *= -1;
    controls.update();
  }
});

function restartWorker(positions) {
  if (worker) worker.terminate();
  worker = new Worker("sort-worker.js");
  workerBusy = false;
  lastSortedView = new THREE.Matrix4().makeScale(0, 0, 0);
  worker.postMessage({ type: "init", positions }); // structured-clone copy
  worker.onmessage = (e) => {
    if (e.data.type !== "sorted" || !mesh) return;
    const attr = mesh.geometry.getAttribute("splatIndex");
    attr.array.set(e.data.indices);
    attr.needsUpdate = true;
    workerBusy = false;
  };
}

// Re-sort only when the camera actually moved and the worker is idle —
// sorting every frame with a still camera is pure waste.
function maybeSort() {
  if (!worker || !mesh || workerBusy) return;
  const view = camera.matrixWorldInverse;
  let drift = 0;
  for (let i = 0; i < 16; i++) drift += Math.abs(view.elements[i] - lastSortedView.elements[i]);
  if (drift < 0.01) return;
  workerBusy = true;
  lastSortedView.copy(view);
  worker.postMessage({ type: "sort", view: new Float32Array(view.elements) });
}

// ---------------------------------------------------------------------------
// Scene fetch, drag & drop, resize, render loop
// ---------------------------------------------------------------------------

function showMessage(text) {
  message.classList.toggle("hidden", !text);
  message.textContent = text || "";
}

async function fetchScene(name) {
  const url = `scenes/${encodeURIComponent(name)}.splat`;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    loadBuffer(name, await res.arrayBuffer());
  } catch (err) {
    showMessage(
      `Could not load ${url} (${err.message}).\n` +
      `Run: uv run gsplat-pipeline view --scene ${name}\n` +
      `…or drag & drop any .splat file here.`
    );
  }
}

document.body.addEventListener("dragover", (e) => {
  e.preventDefault();
  document.body.classList.add("dragover");
});
document.body.addEventListener("dragleave", () => document.body.classList.remove("dragover"));
document.body.addEventListener("drop", async (e) => {
  e.preventDefault();
  document.body.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) loadBuffer(file.name.replace(/\.splat$/, ""), await file.arrayBuffer());
});

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

let frames = 0;
let fps = 0;
let lastFpsT = performance.now();

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  if (mesh) {
    const size = renderer.getSize(new THREE.Vector2()).multiplyScalar(renderer.getPixelRatio());
    mesh.material.uniforms.uViewport.value.copy(size);
    // Focal length in px from the projection matrix: P[0][0] = 2*fx/width.
    mesh.material.uniforms.uFocal.value.set(
      (camera.projectionMatrix.elements[0] * size.x) / 2,
      (camera.projectionMatrix.elements[5] * size.y) / 2
    );
    maybeSort();
  }

  renderer.render(scene, camera);

  frames++;
  const now = performance.now();
  if (now - lastFpsT >= 500) {
    fps = (frames * 1000) / (now - lastFpsT);
    frames = 0;
    lastFpsT = now;
    overlay.textContent =
      `scene:  ${sceneName}\nsplats: ${splatCount.toLocaleString()}\nfps:    ${fps.toFixed(0)}`;
  }
}
animate();

const params = new URLSearchParams(location.search);
const requested = params.get("scene");
if (requested) {
  fetchScene(requested);
} else {
  // No ?scene given: fall back to the first entry of scenes/index.json
  // (written by the `view` CLI, and by hand for static-hosted galleries).
  fetch("scenes/index.json")
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error("no manifest"))))
    .then((list) => {
      if (Array.isArray(list) && list.length) fetchScene(list[0]);
      else throw new Error("empty manifest");
    })
    .catch(() =>
      showMessage("No ?scene=<name> given — drag & drop a .splat file to view it.")
    );
}
