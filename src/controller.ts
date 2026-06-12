import { type ColorKey, FIELD_OF_VIEW, Model, SPHERE_RADIUS } from "./model";
import { THEMES } from "./themes";
import { Cities, visibleCityCount } from "./view/cities";
import { latLonToXyz } from "./view/geometry";
import type { LandTest } from "./view/landsdf";
import type { RelaxedPositions } from "./view/renderer";
import { CAP_RADIUS } from "./view/voronoi";

const POPULATION_OPTIONS = [
  5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000,
  1_000_000, 2_500_000, 5_000_000, 10_000_000,
];

// One settings row per scene color; the background has no alpha slider
// because the canvas is opaque and its alpha could never show.
const COLOR_SETTINGS: { key: ColorKey; label: string; alpha: boolean }[] = [
  { key: "background", label: "Background", alpha: false },
  { key: "water", label: "Water", alpha: true },
  { key: "land", label: "Land", alpha: true },
  { key: "highlight", label: "Highlight", alpha: true },
  { key: "city", label: "Cities", alpha: true },
  { key: "voronoi", label: "Voronoi", alpha: true },
  { key: "countryBorder", label: "Country borders", alpha: true },
  { key: "stateBorder", label: "State borders", alpha: true },
];

// Surface point to hold fixed under the cursor while zooming in, with the
// screen position it should stay under.
interface ZoomAnchor {
  longitude: number;
  latitude: number;
  x: number;
  y: number;
}

export class Controller {
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private zoomAnchor: ZoomAnchor | null = null;
  // Last known cursor position, for re-picking while the camera animates
  // under a stationary cursor.
  private cursorX = 0;
  private cursorY = 0;
  private cursorOnCanvas = false;
  private canvas: HTMLCanvasElement;
  private model: Model;
  private cities: Cities | null = null;
  private isLand: LandTest | null = null;
  private relaxedPositions: () => RelaxedPositions = () => null;

  constructor(
    canvas: HTMLCanvasElement,
    model: Model,
    populationSelect: HTMLSelectElement,
    lineWidthSlider: HTMLInputElement,
    lineWidthLabel: HTMLElement,
    weightSlider: HTMLInputElement,
    weightLabel: HTMLElement,
    centeringSlider: HTMLInputElement,
    centeringLabel: HTMLElement,
    bordersCheckbox: HTMLInputElement,
    themeSelect: HTMLSelectElement,
    colorControls: HTMLElement,
  ) {
    this.canvas = canvas;
    this.model = model;
    canvas.addEventListener("pointerdown", this.onPointerDown);
    canvas.addEventListener("pointermove", this.onPointerMove);
    canvas.addEventListener("pointerup", this.onPointerUp);
    canvas.addEventListener("pointerleave", () => {
      this.cursorOnCanvas = false;
      model.setHoveredCity(null);
    });
    canvas.addEventListener("wheel", this.onWheel, { passive: false });

    for (const population of POPULATION_OPTIONS) {
      const option = document.createElement("option");
      option.value = String(population);
      option.textContent = formatPopulation(population);
      populationSelect.append(option);
    }
    // The model is the source of truth for the initial threshold.
    populationSelect.value = String(model.minPopulation);
    populationSelect.addEventListener("change", () => {
      model.setMinPopulation(Number(populationSelect.value));
      // The hovered index points into the old visible prefix.
      model.setHoveredCity(null);
    });

    const updateLineWidthLabel = () => {
      lineWidthLabel.textContent = model.lineWidth.toFixed(2);
    };
    lineWidthSlider.value = String(model.lineWidth);
    updateLineWidthLabel();
    lineWidthSlider.addEventListener("input", () => {
      model.setLineWidth(Number(lineWidthSlider.value));
      updateLineWidthLabel();
    });

    const updateWeightLabel = () => {
      weightLabel.textContent = model.populationWeight.toFixed(2);
    };
    weightSlider.value = String(model.populationWeight);
    updateWeightLabel();
    weightSlider.addEventListener("input", () => {
      model.setPopulationWeight(Number(weightSlider.value));
      updateWeightLabel();
      // The cell under the cursor may have changed shape.
      model.setHoveredCity(null);
    });

    bordersCheckbox.checked = model.showBorders;
    bordersCheckbox.addEventListener("change", () => {
      model.setShowBorders(bordersCheckbox.checked);
    });

    const updateCenteringLabel = () => {
      centeringLabel.textContent = String(model.lloydIterations);
    };
    centeringSlider.value = String(model.lloydIterations);
    updateCenteringLabel();
    centeringSlider.addEventListener("input", () => {
      model.setLloydIterations(Number(centeringSlider.value));
      updateCenteringLabel();
      model.setHoveredCity(null);
    });

    const colorRefreshers: (() => void)[] = [];
    for (const { key, label, alpha } of COLOR_SETTINGS) {
      const { row, refresh } = buildColorRow(model, key, label, alpha);
      colorControls.append(row);
      colorRefreshers.push(refresh);
    }

    // The dropdown is a one-way applicator: it overwrites every color (and
    // the rows below follow), while manual tweaks leave it untouched.
    for (const theme of THEMES) {
      const option = document.createElement("option");
      option.value = theme.name;
      option.textContent = theme.name;
      themeSelect.append(option);
    }
    themeSelect.value = THEMES[0].name;
    themeSelect.addEventListener("change", () => {
      const theme = THEMES.find((t) => t.name === themeSelect.value);
      if (!theme) return;
      for (const { key } of COLOR_SETTINGS) {
        model.setColor(key, [...theme.colors[key]]);
      }
      for (const refresh of colorRefreshers) refresh();
    });
  }

  // Hover picking is a no-op until both the city arrays and the land test
  // arrive. relaxedPositions supplies the virtual sites whenever Lloyd
  // relaxation is active, so picking matches the rendered diagram.
  public setPickingData(
    cities: Cities,
    isLand: LandTest,
    relaxedPositions: () => RelaxedPositions,
  ): void {
    this.cities = cities;
    this.isLand = isLand;
    this.relaxedPositions = relaxedPositions;
  }

  private onPointerDown = (e: PointerEvent): void => {
    this.dragging = true;
    // The drag owns the rotation now; don't fight it to hold the anchor.
    this.zoomAnchor = null;
    this.canvas.setPointerCapture(e.pointerId);
    this.lastX = e.offsetX;
    this.lastY = e.offsetY;
  };

  private onPointerMove = (e: PointerEvent): void => {
    this.cursorX = e.offsetX;
    this.cursorY = e.offsetY;
    this.cursorOnCanvas = true;
    if (this.dragging) {
      const k = this.anglePerPixel();
      this.model.rotateBy(
        -(e.offsetX - this.lastX) * k,
        (e.offsetY - this.lastY) * k,
      );
      this.lastX = e.offsetX;
      this.lastY = e.offsetY;
    }
    this.model.setHoveredCity(this.pick(e.offsetX, e.offsetY));
  };

  // Rotation per pixel that matches the apparent surface speed at the
  // globe's center, so free rotation feels like dragging the surface.
  private anglePerPixel(): number {
    return (
      ((2 * Math.tan(FIELD_OF_VIEW / 2)) / this.canvas.clientHeight) *
      ((this.model.camera.distance - SPHERE_RADIUS) / SPHERE_RADIUS)
    );
  }

  private onPointerUp = (e: PointerEvent): void => {
    this.dragging = false;
    this.canvas.releasePointerCapture(e.pointerId);
  };

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    if (e.deltaY < 0) {
      // Zooming in: hold the surface point under the cursor (land or water)
      // in place. Off the globe, fall through to a straight zoom.
      const hit = this.raycastSphere(e.offsetX, e.offsetY);
      this.zoomAnchor = hit && {
        longitude: Math.atan2(-hit[2], hit[0]),
        latitude: Math.asin(hit[1] / SPHERE_RADIUS),
        x: e.offsetX,
        y: e.offsetY,
      };
    } else {
      // Zooming out is always straight out.
      this.zoomAnchor = null;
    }
    this.model.zoomBy(Math.pow(1.001, e.deltaY));
  };

  // Advances the camera animation; called once per frame before rendering.
  public update(): void {
    if (!this.model.step()) {
      this.zoomAnchor = null;
      return;
    }
    const anchor = this.zoomAnchor;
    if (anchor) {
      // Shift the view so the anchor lands back under its screen position.
      // Exact in longitude; latitude is approximate but converges as it is
      // re-applied every frame of the animation.
      const hit = this.raycastSphere(anchor.x, anchor.y);
      if (hit) {
        this.model.offsetBy(
          anchor.longitude - Math.atan2(-hit[2], hit[0]),
          anchor.latitude - Math.asin(hit[1] / SPHERE_RADIUS),
        );
      }
    }
    // The camera moved under a possibly stationary cursor; keep the tooltip
    // matched to whatever is under it now.
    this.model.setHoveredCity(
      this.cursorOnCanvas ? this.pick(this.cursorX, this.cursorY) : null,
    );
  }

  // Nearest intersection of the cursor ray with the globe, or null off it.
  private raycastSphere(x: number, y: number): [number, number, number] | null {
    const { longitude, latitude, distance } = this.model.camera;
    const [ex, ey, ez] = latLonToXyz(latitude, longitude, distance);

    // Camera basis matching the renderer's lookAt(eye, origin, +Y); latitude
    // is clamped short of the poles, so forward is never parallel to +Y.
    const fl = Math.hypot(ex, ey, ez);
    const fx = -ex / fl;
    const fy = -ey / fl;
    const fz = -ez / fl;
    const rl = Math.hypot(fz, fx);
    const rx = -fz / rl;
    const rz = fx / rl;
    const ux = -rz * fy;
    const uy = rz * fx - rx * fz;
    const uz = rx * fy;

    // Cursor ray through the perspective projection used for rendering.
    const tanHalf = Math.tan(FIELD_OF_VIEW / 2);
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    const px = ((2 * x) / width - 1) * tanHalf * (width / height);
    const py = (1 - (2 * y) / height) * tanHalf;
    let dx = fx + px * rx + py * ux;
    let dy = fy + py * uy;
    let dz = fz + px * rz + py * uz;
    const dl = Math.hypot(dx, dy, dz);
    dx /= dl;
    dy /= dl;
    dz /= dl;

    // Nearest ray/sphere intersection; same quadratic as sphere.wgsl.
    const od = ex * dx + ey * dy + ez * dz;
    const discriminant = od * od - (ex * ex + ey * ey + ez * ez - SPHERE_RADIUS * SPHERE_RADIUS);
    if (discriminant < 0) return null;
    const t = -od - Math.sqrt(discriminant);
    return [ex + t * dx, ey + t * dy, ez + t * dz];
  }

  // The city whose Voronoi cell is under the cursor: cast the cursor ray onto
  // the sphere and take the nearest visible site (a Voronoi cell is exactly
  // the nearest-site region). Null off the globe or over water.
  private pick(x: number, y: number): number | null {
    if (!this.cities || !this.isLand) return null;
    const hit = this.raycastSphere(x, y);
    if (!hit) return null;
    const [hx, hy, hz] = hit;
    if (!this.isLand([hx, hy, hz])) return null;

    const k = visibleCityCount(this.cities.populations, this.model.minPopulation);
    // Under Lloyd relaxation the diagram is built from the virtual sites; use
    // them only when the readback covers the current visible prefix (it is
    // transiently stale across threshold changes and in-flight readbacks).
    const relaxed = this.relaxedPositions();
    const positions =
      relaxed !== null && relaxed.count === k ? relaxed.data : this.cities.positions;
    const alpha = this.model.weightAlpha;
    const logPop = this.cities.logPopulations;
    let best = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < k; i++) {
      const score =
        hx * positions[i * 4] +
        hy * positions[i * 4 + 1] +
        hz * positions[i * 4 + 2] +
        alpha * logPop[i];
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
    if (best === -1) {
      return null;
    }
    // Cells are truncated to the initial cap; outside it nothing is rendered.
    const inCap =
      hx * positions[best * 4] + hy * positions[best * 4 + 1] + hz * positions[best * 4 + 2] >=
      Math.cos(CAP_RADIUS);
    return inCap ? best : null;
  }
}

function formatPopulation(population: number): string {
  return population >= 1_000_000 ? `${population / 1_000_000}M` : `${population / 1_000}k`;
}

// The unorm color buffers take channel values raw, so hex <-> float is a
// straight division by 255 with no color-space conversion.
function rgbToHex(color: number[]): string {
  return (
    "#" +
    color
      .slice(0, 3)
      .map((c) => Math.round(c * 255).toString(16).padStart(2, "0"))
      .join("")
  );
}

function hexToRgb(hex: string): number[] {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
}

// A row in the Colors section: label, swatch, and (except for the background)
// an alpha slider with a numeric readout. refresh re-reads the model into the
// inputs after something else (a theme preset) changes the color.
function buildColorRow(
  model: Model,
  key: ColorKey,
  label: string,
  alpha: boolean,
): { row: HTMLElement; refresh: () => void } {
  const row = document.createElement("div");
  row.className = "color-control";

  const text = document.createElement("span");
  text.className = "color-label";
  text.textContent = label;
  row.append(text);

  const swatch = document.createElement("input");
  swatch.type = "color";
  swatch.addEventListener("input", () => {
    model.setColor(key, [...hexToRgb(swatch.value), model.colors[key][3]]);
  });
  row.append(swatch);

  let refresh = () => {
    swatch.value = rgbToHex(model.colors[key]);
  };

  if (alpha) {
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "1";
    slider.step = "0.01";
    const value = document.createElement("span");
    slider.addEventListener("input", () => {
      model.setColor(key, [...model.colors[key].slice(0, 3), Number(slider.value)]);
      value.textContent = model.colors[key][3].toFixed(2);
    });
    row.append(slider, value);
    const refreshColor = refresh;
    refresh = () => {
      refreshColor();
      slider.value = String(model.colors[key][3]);
      value.textContent = model.colors[key][3].toFixed(2);
    };
  }

  refresh();
  return { row, refresh };
}
