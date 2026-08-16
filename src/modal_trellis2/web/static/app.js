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
  statusLabel.textContent = live ? "live GPU" : "dry-run";
}

function showPreview(file) {
  const url = URL.createObjectURL(file);
  preview.src = url;
  previewWrap.hidden = false;
  dropCopy.hidden = true;
}

function setModel(url) {
  if (currentRoot) {
    scene.remove(currentRoot);
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
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  showPreview(file);
});
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) showPreview(file);
});
dryRun.addEventListener("change", setMode);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) {
    hint.textContent = "先放一张图。";
    return;
  }
  const body = new FormData();
  body.set("image", file);
  body.set("seed", document.querySelector("#seed").value);
  body.set("pipeline", document.querySelector("#pipeline").value);
  body.set("dry_run", dryRun.checked ? "true" : "false");
  go.disabled = true;
  hint.textContent = dryRun.checked ? "本地立方体上转台……" : "正在叫 Modal GPU……";
  try {
    const response = await fetch("/api/generate", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "generation failed");
    setModel(payload.asset_url);
    download.href = payload.asset_url;
    download.hidden = false;
    stageLabel.textContent = `${payload.id} · ${payload.glb_size_bytes} B`;
    hint.textContent = payload.dry_run
      ? "这是 tinted cube，用来先把上传/回传/可视化跑通。"
      : "TRELLIS.2 网格已回到转台上。";
  } catch (error) {
    hint.textContent = error.message;
  } finally {
    go.disabled = false;
  }
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
