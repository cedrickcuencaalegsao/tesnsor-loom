import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { NetworkSnapshot } from "../types/type";

interface Props {
  snapshot: NetworkSnapshot | null;
}

const HEIGHT = 480;
const LAYER_GAP = 14;
const MAX_EXTENT = 36; // max size of a layer's node grid, in scene units
const MAX_SPACING = 2.6;

const COLOR = {
  surface: "#F8FAFC",
  secondary: "#4A5568",
  node: "#6984A9",
  accent: "#A0D585",
  negative: "#EF4444",
};

function layerName(layer: number, layerCount: number): string {
  if (layer === 0) return "Input";
  if (layer === layerCount - 1) return "Output";
  return `Hidden ${layer}`;
}

function maxAbs(values: number[]): number {
  let m = 0;
  for (const v of values) m = Math.max(m, Math.abs(v));
  return m > 0 ? m : 1;
}

/** Position of node `i` in a layer of `count` nodes: a centred grid on the Y/Z plane. */
function nodePosition(
  layer: number,
  layerCount: number,
  i: number,
  count: number,
): { pos: THREE.Vector3; spacing: number } {
  const cols = Math.ceil(Math.sqrt(count));
  const rows = Math.ceil(count / cols);
  const spacing = Math.min(MAX_SPACING, MAX_EXTENT / Math.max(cols, rows, 1));
  const col = i % cols;
  const row = Math.floor(i / cols);
  const x = (layer - (layerCount - 1) / 2) * LAYER_GAP;
  const y = (row - (rows - 1) / 2) * spacing;
  const z = (col - (cols - 1) / 2) * spacing;
  return { pos: new THREE.Vector3(x, y, z), spacing };
}

function makeLabel(text: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 96;
  const ctx = canvas.getContext("2d")!;
  ctx.font = "600 40px 'Plus Jakarta Sans', sans-serif";
  ctx.fillStyle = COLOR.secondary;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 256, 48);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }),
  );
  sprite.scale.set(10, 10 * (96 / 512), 1);
  sprite.renderOrder = 10;
  return sprite;
}

function disposeGroup(group: THREE.Group) {
  group.traverse((obj) => {
    const o = obj as THREE.Mesh;
    o.geometry?.dispose();
    const mat = o.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
    else if (mat) {
      (mat as THREE.SpriteMaterial).map?.dispose();
      mat.dispose();
    }
  });
  group.clear();
}

/** Live 3D network: each layer is a grid of spheres, layers are spaced along X.
 *  Edge opacity = |weight|, green = positive, red = negative.
 *  Node colour intensity = mean activation. Drag to orbit, scroll to zoom. */
export function NetworkVisualizer({ snapshot }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    graph: THREE.Group;
    layoutKey: string;
  } | null>(null);
  const [autoRotate, setAutoRotate] = useState(false);

  // One-time renderer / camera / controls setup.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, HEIGHT);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(COLOR.surface);

    const camera = new THREE.PerspectiveCamera(50, mount.clientWidth / HEIGHT, 0.1, 1000);
    camera.position.set(0, 20, 60);

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(20, 40, 30);
    scene.add(sun);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const graph = new THREE.Group();
    scene.add(graph);

    sceneRef.current = { renderer, scene, camera, controls, graph, layoutKey: "" };

    let frame = 0;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(loop);
    };
    loop();

    const observer = new ResizeObserver(() => {
      const w = mount.clientWidth;
      renderer.setSize(w, HEIGHT);
      camera.aspect = w / HEIGHT;
      camera.updateProjectionMatrix();
    });
    observer.observe(mount);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      disposeGroup(graph);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (sceneRef.current) {
      sceneRef.current.controls.autoRotate = autoRotate;
      sceneRef.current.controls.autoRotateSpeed = 1.2;
    }
  }, [autoRotate]);

  // Rebuild the graph whenever the snapshot changes (camera is preserved).
  useEffect(() => {
    const ctx = sceneRef.current;
    if (!ctx || !snapshot) return;
    const { graph, camera, controls } = ctx;
    disposeGroup(graph);

    const sizes = snapshot.layer_sizes;
    const layerCount = sizes.length;

    // --- edges: one LineSegments with RGBA vertex colours ---
    const edgeMax = maxAbs(snapshot.edges.flatMap((b) => b.weights));
    const positions: number[] = [];
    const colors: number[] = [];
    const pos = new THREE.Color(COLOR.accent);
    const neg = new THREE.Color(COLOR.negative);

    snapshot.edges.forEach((block, k) => {
      const fromCount = sizes[k];
      const toCount = sizes[k + 1];
      const from: THREE.Vector3[] = [];
      const to: THREE.Vector3[] = [];
      for (let i = 0; i < fromCount; i++) from.push(nodePosition(k, layerCount, i, fromCount).pos);
      for (let i = 0; i < toCount; i++) to.push(nodePosition(k + 1, layerCount, i, toCount).pos);

      for (let r = 0; r < block.rows; r++) {
        for (let c = 0; c < block.cols; c++) {
          const w = block.weights[r * block.cols + c];
          const strength = Math.abs(w) / edgeMax;
          const alpha = 0.04 + 0.7 * strength;
          const col = w >= 0 ? pos : neg;
          const a = from[r];
          const b = to[c];
          positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
          colors.push(col.r, col.g, col.b, alpha, col.r, col.g, col.b, alpha);
        }
      }
    });

    if (positions.length) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 4));
      const lines = new THREE.LineSegments(
        geo,
        new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }),
      );
      graph.add(lines);
    }

    // --- nodes: one InstancedMesh per layer, colour = activation ---
    const base = new THREE.Color(COLOR.node);
    const dim = new THREE.Color(COLOR.surface);
    const tmp = new THREE.Color();
    const matrix = new THREE.Matrix4();

    sizes.forEach((count, l) => {
      const activity = snapshot.node_activity[l] ?? [];
      const layerMax = maxAbs(activity);
      const { spacing } = nodePosition(l, layerCount, 0, count);
      const radius = Math.max(0.15, Math.min(0.9, spacing * 0.3));

      const mesh = new THREE.InstancedMesh(
        new THREE.SphereGeometry(radius, 20, 14),
        new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.05 }),
        count,
      );
      for (let i = 0; i < count; i++) {
        const { pos: p } = nodePosition(l, layerCount, i, count);
        matrix.setPosition(p);
        mesh.setMatrixAt(i, matrix);
        const level = Math.abs(activity[i] ?? 0) / layerMax;
        tmp.copy(dim).lerp(base, 0.25 + 0.75 * level);
        mesh.setColorAt(i, tmp);
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      graph.add(mesh);

      // layer label below the grid
      const rows = Math.ceil(count / Math.ceil(Math.sqrt(count)));
      const label = makeLabel(`${layerName(l, layerCount)} (${count})`);
      label.position.set(
        (l - (layerCount - 1) / 2) * LAYER_GAP,
        -((rows - 1) / 2) * spacing - radius - 3,
        0,
      );
      graph.add(label);
    });

    // Re-frame the camera only when the architecture changes.
    const key = sizes.join("-");
    if (ctx.layoutKey !== key) {
      ctx.layoutKey = key;
      const biggest = Math.max(...sizes, 1);
      const extent = Math.min(MAX_EXTENT, Math.ceil(Math.sqrt(biggest)) * MAX_SPACING);
      const span = Math.max((layerCount - 1) * LAYER_GAP, extent) || 20;
      const dist = span * 1.1 + 15;
      camera.position.set(dist * 0.35, dist * 0.35, dist);
      controls.target.set(0, 0, 0);
      controls.update();
    }
  }, [snapshot]);

  // The mount element must always be rendered so the one-time three.js setup can find it.
  return (
    <figure className="m-0">
      <div className="relative">
        <div
          ref={mountRef}
          className="w-full overflow-hidden border border-secondary/35"
          style={{ height: HEIGHT }}
          role="img"
          aria-label={`3D neural network at epoch ${snapshot?.epoch ?? 0}`}
        />
        {!snapshot && (
          <p className="absolute inset-0 flex items-center justify-center text-sm text-secondary">
            The network appears here once training starts.
          </p>
        )}
      </div>
      <figcaption className="mt-2 flex items-center justify-between gap-4 text-xs text-secondary">
        <span>
          {snapshot ? `Epoch ${snapshot.epoch}. ` : ""}Green = positive weight, red = negative,
          more opaque = larger. Node colour = mean activation. Drag to rotate, scroll to zoom,
          right-drag to pan.
        </span>
        <label className="flex shrink-0 items-center gap-1">
          <input
            type="checkbox"
            checked={autoRotate}
            onChange={(e) => setAutoRotate(e.target.checked)}
          />
          Auto-rotate
        </label>
      </figcaption>
    </figure>
  );
}