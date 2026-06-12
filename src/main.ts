import "./style.css";
import { Model, SPHERE_RADIUS } from "./model";
import { Controller } from "./controller";
import { Renderer, WebGPUNotSupportedError } from "./view/renderer";
import { createCube } from "./view/geometry";
import { createLandTest, loadLandSdf, type LandTest } from "./view/landsdf";
import { loadCities } from "./view/cities";
import { loadBorders } from "./view/borders";

async function main(): Promise<void> {
  const loading = document.querySelector<HTMLElement>("#loading")!;
  const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
  const tooltip = document.querySelector<HTMLElement>("#tooltip")!;
  const populationSelect = document.querySelector<HTMLSelectElement>("#population-select")!;
  const lineWidthSlider = document.querySelector<HTMLInputElement>("#line-width-slider")!;
  const lineWidthLabel = document.querySelector<HTMLElement>("#line-width-value")!;
  const weightSlider = document.querySelector<HTMLInputElement>("#weight-slider")!;
  const weightLabel = document.querySelector<HTMLElement>("#weight-value")!;
  const centeringSlider = document.querySelector<HTMLInputElement>("#centering-slider")!;
  const centeringLabel = document.querySelector<HTMLElement>("#centering-value")!;
  const bordersCheckbox = document.querySelector<HTMLInputElement>("#borders-checkbox")!;
  const themeSelect = document.querySelector<HTMLSelectElement>("#theme-select")!;
  const colorControls = document.querySelector<HTMLElement>("#color-controls")!;

  const model = new Model();

  let renderer: Renderer;
  try {
    renderer = await Renderer.create(canvas, tooltip);
  } catch (e) {
    if (e instanceof WebGPUNotSupportedError) {
      loading.remove();
      canvas.replaceWith(
        Object.assign(document.createElement("div"), {
          className: "error",
          textContent: e.message,
        }),
      );
      return;
    }
    throw e;
  }

  const controller = new Controller(
    canvas,
    model,
    populationSelect,
    lineWidthSlider,
    lineWidthLabel,
    weightSlider,
    weightLabel,
    centeringSlider,
    centeringLabel,
    bordersCheckbox,
    themeSelect,
    colorControls,
  );

  // Both fetches start now; the frame loop (and the loading cover's removal)
  // waits for them so the first visible frame is the complete scene.
  const sdfPromise = loadLandSdf("/data/land-sdf.bin");
  const citiesPromise = loadCities("/data/cities.bin");
  const bordersPromise = loadBorders("/data/borders.bin");

  // The SDF must be in place before the sphere and Voronoi bind groups
  // capture it; on failure the renderer's all-land fallback keeps rendering,
  // and the all-land hover test stays consistent with it.
  let landTest: LandTest = () => true;
  try {
    const sdf = await sdfPromise;
    renderer.setLandSdf(sdf.faceSize, sdf.data);
    landTest = createLandTest(sdf.faceSize, sdf.data);
  } catch (e) {
    console.error("Land SDF load failed:", e);
  }

  // The sphere matches the line/city geometry radius exactly (the shader
  // offsets only its depth, so layering works without parallax); the cube
  // proxy is padded so its faces aren't tangent to the sphere.
  renderer.setSphere(createCube(1.001 * SPHERE_RADIUS), SPHERE_RADIUS);

  // On failure the cover still comes down so the bare globe is visible
  // rather than an eternal loading screen.
  try {
    const cities = await citiesPromise;
    renderer.setCities(cities);
    controller.setPickingData(cities, landTest, () => renderer.pickPositions);
  } catch (e) {
    console.error("Cities load failed:", e);
  }

  try {
    const borders = await bordersPromise;
    renderer.setBorders(borders);
  } catch (e) {
    console.error("Borders load failed:", e);
  }

  let framesRendered = 0;
  const frame = () => {
    // The second callback runs once the first complete frame (scene and
    // Voronoi compute share one submission) has been presented.
    if (framesRendered === 1) loading.remove();
    framesRendered++;
    controller.update();
    renderer.render(model);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main();
