import type { ColorKey } from "./model";

export interface Theme {
  name: string;
  colors: Record<ColorKey, number[]>;
}

// Hex + alpha, converted to RGBA floats below. The unorm color buffers take
// channel values raw, so the conversion is a straight division by 255.
type ThemeSpec = Record<ColorKey, [string, number]>;

// Scene mapping by emphasis: terrain is canvas, data is ink. Each theme's
// neutral surface ladder paints the terrain — background = theme background,
// water = the adjacent dark panel tone, land = the selection/surface neutral,
// borders = the deepest near-black — and its accents are reserved for the
// data: voronoi = the signature accent, city = the brightest accent for
// maximum pop, highlight = a secondary accent. The first entry is the
// startup default.
const THEME_SPECS: { name: string; spec: ThemeSpec }[] = [
  {
    name: "Monokai",
    spec: {
      background: ["#272822", 1],
      water: ["#1e1f1c", 0.8],
      land: ["#49483e", 0.8],
      highlight: ["#66d9ef", 0.8],
      city: ["#e6db74", 1],
      voronoi: ["#f92672", 1],
      countryBorder: ["#181913", 0.75],
      stateBorder: ["#181913", 0.45],
    },
  },
  {
    name: "Solarized Dark",
    spec: {
      background: ["#002b36", 1],
      water: ["#073642", 0.8],
      land: ["#586e75", 0.8],
      highlight: ["#2aa198", 0.8],
      city: ["#fdf6e3", 1],
      voronoi: ["#268bd2", 1],
      countryBorder: ["#001f27", 0.75],
      stateBorder: ["#001f27", 0.45],
    },
  },
  {
    // The ladder inverts on the cream background: water and land step down
    // in lightness, and the data accents are dark and saturated instead of
    // bright.
    name: "Solarized Light",
    spec: {
      background: ["#fdf6e3", 1],
      water: ["#eee8d5", 0.8],
      land: ["#93a1a1", 0.8],
      highlight: ["#2aa198", 0.8],
      city: ["#cb4b16", 1],
      voronoi: ["#268bd2", 1],
      countryBorder: ["#657b83", 0.75],
      stateBorder: ["#657b83", 0.45],
    },
  },
  {
    name: "Dracula",
    spec: {
      background: ["#282a36", 1],
      water: ["#21222c", 0.8],
      land: ["#44475a", 0.8],
      highlight: ["#6272a4", 0.8],
      city: ["#f1fa8c", 1],
      voronoi: ["#bd93f9", 1],
      countryBorder: ["#191a21", 0.75],
      stateBorder: ["#191a21", 0.45],
    },
  },
  {
    name: "Nord",
    spec: {
      background: ["#2e3440", 1],
      water: ["#3b4252", 0.8],
      land: ["#4c566a", 0.8],
      highlight: ["#5e81ac", 0.8],
      city: ["#ebcb8b", 1],
      voronoi: ["#88c0d0", 1],
      countryBorder: ["#242933", 0.75],
      stateBorder: ["#242933", 0.45],
    },
  },
  {
    name: "Gruvbox",
    spec: {
      background: ["#282828", 1],
      water: ["#32302f", 0.8],
      land: ["#504945", 0.8],
      highlight: ["#83a598", 0.8],
      city: ["#fabd2f", 1],
      voronoi: ["#fe8019", 1],
      countryBorder: ["#1d2021", 0.75],
      stateBorder: ["#1d2021", 0.45],
    },
  },
  {
    name: "Catppuccin Mocha",
    spec: {
      background: ["#1e1e2e", 1],
      water: ["#181825", 0.8],
      land: ["#45475a", 0.8],
      highlight: ["#b4befe", 0.8],
      city: ["#f9e2af", 1],
      voronoi: ["#cba6f7", 1],
      countryBorder: ["#11111b", 0.75],
      stateBorder: ["#11111b", 0.45],
    },
  },
  {
    name: "Tokyo Night",
    spec: {
      background: ["#1a1b26", 1],
      water: ["#16161e", 0.8],
      land: ["#414868", 0.8],
      highlight: ["#bb9af7", 0.8],
      city: ["#e0af68", 1],
      voronoi: ["#7aa2f7", 1],
      countryBorder: ["#15161e", 0.75],
      stateBorder: ["#15161e", 0.45],
    },
  },
  {
    name: "One Dark",
    spec: {
      background: ["#282c34", 1],
      water: ["#21252b", 0.8],
      land: ["#3e4451", 0.8],
      highlight: ["#c678dd", 0.8],
      city: ["#e5c07b", 1],
      voronoi: ["#61afef", 1],
      countryBorder: ["#181a1f", 0.75],
      stateBorder: ["#181a1f", 0.45],
    },
  },
];

function toRgba([hex, alpha]: [string, number]): number[] {
  return [...[1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255), alpha];
}

export const THEMES: Theme[] = THEME_SPECS.map(({ name, spec }) => ({
  name,
  colors: Object.fromEntries(
    Object.entries(spec).map(([key, value]) => [key, toRgba(value)]),
  ) as Record<ColorKey, number[]>,
}));
