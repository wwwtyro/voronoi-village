import { THEMES } from "./themes";

export const SPHERE_RADIUS = 1;
export const FIELD_OF_VIEW = Math.PI / 4;
export const MAX_LLOYD_ITERATIONS = 10;

// Power-diagram weight scale at the top of the population-weight slider, in
// dot-product (cosine) units. Sized so the slider's upper range lets
// megacities swallow neighbors out to a few degrees: the boundary between
// sites theta apart shifts by roughly delta-weight / theta.
const WEIGHT_ALPHA_MAX = 0.01;

const MAX_LATITUDE = (89.9 * Math.PI) / 180;
const MIN_DISTANCE = 1.02;
const MAX_DISTANCE = 10;

// Per-frame easing factor for the camera animation.
const SMOOTHING = 0.25;

export type ColorKey =
  | "background"
  | "water"
  | "land"
  | "highlight"
  | "city"
  | "voronoi"
  | "countryBorder"
  | "stateBorder";

export class Model {
  // Boots fully zoomed out over the equator at the prime meridian; the
  // animation swoops from here to the target on the first frames.
  public camera = {
    longitude: 0,
    latitude: 0,
    distance: MAX_DISTANCE,
  };

  // Where the camera animation is headed; camera eases toward this per
  // frame. Starts centered on the contiguous USA (39.8°N, 98.6°W).
  public target = {
    longitude: (-98.6 * Math.PI) / 180,
    latitude: (39.8 * Math.PI) / 180,
    distance: 2,
  };

  public minPopulation = 50_000;

  // Rendered line width in CSS pixels.
  public lineWidth = 1.5;

  // Whether country and state borders are drawn.
  public showBorders = true;

  // Every scene color as RGBA in [0, 1], starting from the first theme.
  // The background's alpha is fixed: the canvas is opaque, so it could never
  // show.
  public colors: Record<ColorKey, number[]> = Object.fromEntries(
    Object.entries(THEMES[0].colors).map(([key, value]) => [key, [...value]]),
  ) as Record<ColorKey, number[]>;

  // Population-weight slider position in [0, 1]; see weightAlpha.
  public populationWeight = 0;

  // Lloyd relaxation steps applied to the virtual Voronoi sites.
  public lloydIterations = 0;

  // Index into the city arrays of the municipality under the cursor, or null
  // when the cursor is off the globe, over water, or nothing has loaded yet.
  public hoveredCity: number | null = null;

  // The power-diagram weight scale: quadratic in the slider for fine control
  // at the low end, where small weights already reshape dense regions.
  public get weightAlpha(): number {
    return this.populationWeight ** 2 * WEIGHT_ALPHA_MAX;
  }

  public rotateBy(dLongitude: number, dLatitude: number): void {
    this.target.longitude += dLongitude;
    this.target.latitude = clamp(this.target.latitude + dLatitude, -MAX_LATITUDE, MAX_LATITUDE);
  }

  // Shift the view immediately, bypassing the easing: zoom anchoring applies
  // per-frame corrections that must land this frame. Camera and target move
  // together so any in-flight rotation animation is preserved.
  public offsetBy(dLongitude: number, dLatitude: number): void {
    this.camera.longitude += dLongitude;
    this.target.longitude += dLongitude;
    this.camera.latitude = clamp(this.camera.latitude + dLatitude, -MAX_LATITUDE, MAX_LATITUDE);
    this.target.latitude = clamp(this.target.latitude + dLatitude, -MAX_LATITUDE, MAX_LATITUDE);
  }

  public zoomBy(factor: number): void {
    this.target.distance = clamp(this.target.distance * factor, MIN_DISTANCE, MAX_DISTANCE);
  }

  // One frame of the camera animation; returns whether it is still running.
  // Snap thresholds end the animation crisply: ~0.0006° is far subpixel at
  // the closest zoom, as is 1e-4 distance units.
  public step(): boolean {
    let animating = false;
    const ease = (key: "longitude" | "latitude" | "distance", epsilon: number) => {
      const remaining = this.target[key] - this.camera[key];
      if (Math.abs(remaining) < epsilon) {
        this.camera[key] = this.target[key];
      } else {
        this.camera[key] += SMOOTHING * remaining;
        animating = true;
      }
    };
    ease("longitude", 1e-5);
    ease("latitude", 1e-5);
    ease("distance", 1e-4);
    // Keep longitude bounded; shifting camera and target by the same whole
    // number of turns is invisible.
    if (Math.abs(this.target.longitude) > 2 * Math.PI) {
      const turns = 2 * Math.PI * Math.trunc(this.target.longitude / (2 * Math.PI));
      this.target.longitude -= turns;
      this.camera.longitude -= turns;
    }
    return animating;
  }

  public setMinPopulation(value: number): void {
    this.minPopulation = Math.max(0, value);
  }

  public setShowBorders(value: boolean): void {
    this.showBorders = value;
  }

  public setColor(key: ColorKey, value: number[]): void {
    this.colors[key] = value.map((c) => clamp(c, 0, 1));
  }

  public setLineWidth(value: number): void {
    this.lineWidth = Math.max(0.05, value);
  }

  public setPopulationWeight(value: number): void {
    this.populationWeight = clamp(value, 0, 1);
  }

  public setLloydIterations(value: number): void {
    this.lloydIterations = clamp(Math.round(value), 0, MAX_LLOYD_ITERATIONS);
  }

  public setHoveredCity(value: number | null): void {
    this.hoveredCity = value;
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
