# Gaussian Splat Viewer

Interactive 3D Gaussian Splatting scenes in the browser — no plugins, no build
step. Rendered with Three.js instanced quads, EWA covariance projection, and a
Web Worker depth sort.

**Live:** https://dlux2015.github.io/gsplat-viewer-site/

Controls: drag to orbit · wheel to zoom · right-drag to pan · drag & drop any
`.splat` file onto the page to view it.

## Embed in your own site

```html
<iframe src="https://dlux2015.github.io/gsplat-viewer-site/?scene=fox"
        width="800" height="500" style="border:0" allowfullscreen></iframe>
```

## Add a scene

1. Produce a `scene.splat` with [gsplat-pipeline](../../) (`run all --scene NAME`).
2. Copy it to `scenes/<name>.splat`.
3. Add `<name>` to `scenes/index.json` (first entry = default scene).
4. Commit and push — GitHub Pages redeploys automatically.

Scenes are the antimatter15 `.splat` format (32 bytes/gaussian), so files from
other 3DGS tools work too.

## Credits

- Viewer built with [Three.js](https://threejs.org); splat math follows the
  [INRIA 3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
  rasterizer and [antimatter15/splat](https://github.com/antimatter15/splat).
- Demo scene reconstructed from the fox capture in
  [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp) sample data.
