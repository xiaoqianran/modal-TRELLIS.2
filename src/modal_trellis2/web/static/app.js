import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const form = document.querySelector("#form");
const fileInput = document.querySelector("#file");
const drop = document.querySelector("#drop");
const preview = document.querySelector("#preview");
const previewWrap = document.querySelector("#preview-wrap");
const dropCopy = document.querySelector("#drop-copy");
const go = document.querySelector("#go");
const hint = document.querySelector("#hint");
const dryRun = document.querySelector("#dry-run");
const statusDot = document.querySelector(".dot");
const statusLabel = document.querySelector("#status-label");
const stageLabel = document.querySelector("#stage-label");
const download = document.querySelector("#download");
const host = document.querySelector("#canvas-host");
const queuePanel = document.querySelector("#queue-panel");
const queueList = document.querySelector("#queue-list");
const queueSummary = document.querySelector("#queue-summary");
const MAX_LIVE_BATCH = 20;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.08, 40);
camera.position.set(2.1, 1.45, 2.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0.15, 0);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const table = new THREE.Mesh(
  new THREE.CylinderGeometry(1.15, 1.15, 0.04, 64),
  new THREE.MeshStandardMaterial({ color: 0x2c261f, roughness: 0.82, metalness: 0.08 }),
);
table.position.y = -0.42;
scene.add(table);

const ring = new THREE.Mesh(
  new THREE.TorusGeometry(1.16, 0.012, 12, 80),
  new THREE.MeshStandardMaterial({ color: 0xc56a3a, roughness: 0.35, metalness: 0.55 }),
);
ring.rotation.x = Math.PI / 2;
ring.position.y = -0.39;
scene.add(ring);

const grid = new THREE.GridHelper(6, 24, 0x6fb8ae, 0x3a3530);
grid.position.y = -0.44;
grid.material.transparent = true;
grid.material.opacity = 0.28;
scene.add(grid);

scene.add(new THREE.HemisphereLight(0xf2ebe0, 0x2c261f, 1.1));
const key = new THREE.DirectionalLight(0xfff4e8, 1.4);
key.position.set(2.4, 3.2, 1.4);
scene.add(key);

const loader = new GLTFLoader();
let currentRoot = null;
let previewUrl = null;
let frame = 0;

function resize() {
  const { clientWidth, clientHeight } = host;
  renderer.setSize(clientWidth, clientHeight, false);
  camera.aspect = clientWidth / Math.max(clientHeight, 1);
  camera.updateProjectionMatrix();
}

function tick() {
  frame = requestAnimationFrame(tick);
  ring.rotation.z += 0.003;
  controls.update();
  renderer.render(scene, camera);
}

function setMode() {
  const live = !dryRun.checked;
  statusDot.dataset.mode = live ? "live" : "dry";
  statusLabel.textContent = live ? "official TRELLIS.2-4B" : "dry-run";
}

function showPreview(file) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  previewWrap.hidden = false;
  dropCopy.hidden = true;
}

function disposeRoot(root) {
  root.traverse((object) => {
    if (!object.isMesh) return;
    object.geometry?.dispose();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (!material) continue;
      for (const value of Object.values(material)) {
        if (value?.isTexture) value.dispose();
      }
      material.dispose?.();
    }
  });
}

function setModel(url) {
  if (currentRoot) {
    scene.remove(currentRoot);
    disposeRoot(currentRoot);
    currentRoot = null;
  }
  loader.load(url, (gltf) => {
    currentRoot = gltf.scene;
    const box = new THREE.Box3().setFromObject(currentRoot);
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    const center = box.getCenter(new THREE.Vector3());
    currentRoot.position.sub(center);
    currentRoot.scale.setScalar(1.15 / size);
    currentRoot.position.y += 0.08;
    scene.add(currentRoot);
  });
}

drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("is-over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("is-over"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("is-over");
  const files = Array.from(event.dataTransfer?.files || []);
  if (!files.length) return;
  const transfer = new DataTransfer();
  for (const file of files) transfer.items.add(file);
  fileInput.files = transfer.files;
  showPreview(files[0]);
  renderQueue(files);
});
fileInput.addEventListener("change", () => {
  const files = Array.from(fileInput.files || []);
  if (files.length) showPreview(files[0]);
  renderQueue(files);
});
dryRun.addEventListener("change", setMode);

function renderQueue(files) {
  queueList.replaceChildren();
  queuePanel.hidden = files.length === 0;
  queueSummary.textContent = `0 / ${files.length}`;
  for (const [index, file] of files.entries()) {
    const item = document.createElement("li");
    item.dataset.index = String(index);
    item.dataset.state = "queued";
    const name = document.createElement("span");
    name.textContent = file.name;
    const status = document.createElement("small");
    status.textContent = "queued";
    item.append(name, status);
    queueList.appendChild(item);
  }
}

function setQueueState(index, state, detail = state) {
  const item = queueList.querySelector(`[data-index="${index}"]`);
  if (!item) return;
  item.dataset.state = state;
  const status = item.querySelector("small");
  if (status) status.textContent = detail;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const files = Array.from(fileInput.files || []);
  if (!files.length) {
    hint.textContent = "先放至少一张图。";
    return;
  }
  if (!dryRun.checked && files.length > MAX_LIVE_BATCH) {
    hint.textContent = `live 队列一次最多 ${MAX_LIVE_BATCH} 张，避免意外持续占用 GPU。`;
    return;
  }

  renderQueue(files);
  go.disabled = true;
  let completed = 0;
  let failed = 0;

  // Deliberately serial: each request awaits completion before the next one starts,
  // so Modal can reuse the one warm GPU instead of receiving a parallel burst.
  for (const [index, file] of files.entries()) {
    setQueueState(index, "running", "running");
    hint.textContent = dryRun.checked
      ? `dry-run ${index + 1}/${files.length}：${file.name}`
      : `TRELLIS.2 ${index + 1}/${files.length}：${file.name}`;

    const body = new FormData();
    body.set("image", file);
    body.set("seed", document.querySelector("#seed").value);
    body.set("pipeline", document.querySelector("#pipeline").value);
    body.set("dry_run", dryRun.checked ? "true" : "false");

    try {
      const response = await fetch("/api/generate", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "generation failed");

      completed += 1;
      setQueueState(index, "completed", `${payload.glb_size_bytes} B`);
      setModel(payload.asset_url);
      download.href = payload.asset_url;
      download.hidden = false;
      stageLabel.textContent = `${payload.id} · ${payload.glb_size_bytes} B`;
    } catch (error) {
      failed += 1;
      setQueueState(index, "failed", error.message || "failed");
    } finally {
      queueSummary.textContent = `${completed + failed} / ${files.length}`;
    }
  }

  go.disabled = false;
  hint.textContent = failed
    ? `队列完成：${completed} 成功，${failed} 失败。`
    : `队列完成：${completed}/${files.length}。连续任务已按顺序提交。`;
});

window.addEventListener("resize", resize);
setMode();
resize();
tick();

fetch("/api/meta")
  .then((response) => response.json())
  .then((meta) => {
    dryRun.checked = meta.dry_run;
    setMode();
  })
  .catch(() => {});

if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  cancelAnimationFrame(frame);
  renderer.render(scene, camera);
}
