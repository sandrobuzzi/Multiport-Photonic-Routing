"use strict";

const elements = Object.fromEntries([
  "mode", "point-count", "point-size", "point-size-value", "opacity",
  "opacity-value", "gamma", "gamma-value", "data-url", "load-data",
  "download-csv", "plot-title", "plot-subtitle", "status"
].map(id => [id, document.getElementById(id)]));

const titles = {
  "experiment-output": ["Measured outputs", "Fig. 4a · RGB encodes the three measured output channels"],
  "model-output": ["Fitted model", "Fig. 4b · RGB encodes the three fitted output channels"],
  "experiment-extinction": ["Measured extinction", "Fig. 4c · RGB intensity encodes channel-selective extinction"],
  "model-extinction": ["Model extinction", "Fitted channel-selective extinction across the voltage cube"]
};

let payload = null;
let decoded = null;
let displayed = null;

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("error", isError);
}

function decodeArray(spec) {
  const binary = atob(spec.base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  if (bytes.byteLength !== spec.length * 8) throw new Error("Unexpected float64 array length");
  return new Float64Array(bytes.buffer);
}

function decodeGroup(group) {
  return Object.fromEntries(Object.entries(group).map(([key, spec]) => [key, decodeArray(spec)]));
}

async function ungzipJson(response) {
  if (!response.ok) throw new Error(`Data request failed (${response.status})`);
  const compressed = await response.arrayBuffer();
  const bytes = new Uint8Array(compressed);
  let text;
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    if (!("DecompressionStream" in window)) throw new Error("This browser cannot decompress gzip data");
    const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
    text = await new Response(stream).text();
  } else {
    text = new TextDecoder().decode(compressed);
  }
  return JSON.parse(text);
}

function range(values) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const value of values) {
    if (value < lo) lo = value;
    if (value > hi) hi = value;
  }
  return [lo, hi];
}

function normalization() {
  const result = [];
  for (const channel of ["out1", "out2", "out3"]) {
    const [elo, ehi] = range(decoded.experiment[channel]);
    const [mlo, mhi] = range(decoded.model[channel]);
    result.push([Math.min(elo, mlo), Math.max(ehi, mhi)]);
  }
  return result;
}

function normalizedOutputs(group, indices, limits) {
  return indices.map(i => {
    const r = limits[0][1] > limits[0][0] ? Math.max(0, Math.min(1, (group.out1[i] - limits[0][0]) / (limits[0][1] - limits[0][0]))) : 0.5;
    const g = limits[1][1] > limits[1][0] ? Math.max(0, Math.min(1, (group.out2[i] - limits[1][0]) / (limits[1][1] - limits[1][0]))) : 0.5;
    const b = limits[2][1] > limits[2][0] ? Math.max(0, Math.min(1, (group.out3[i] - limits[2][0]) / (limits[2][1] - limits[2][0]))) : 0.5;
    return [r, g, b];
  });
}

function extinction(group, indices) {
  const channelRanges = [range(group.out1), range(group.out2), range(group.out3)];
  const raw = Array.from({length: group.out1.length}, (_, i) => {
    const n = [group.out1[i], group.out2[i], group.out3[i]].map((value, channel) => {
      const [lo, hi] = channelRanges[channel];
      return hi > lo ? (value - lo) / (hi - lo) : 0.5;
    });
    const db = n.map(value => 10 * Math.log10(value + 1e-6));
    return [
      Math.max(db[0] - Math.max(db[1], db[2]), 0),
      Math.max(db[1] - Math.max(db[0], db[2]), 0),
      Math.max(db[2] - Math.max(db[0], db[1]), 0)
    ];
  });
  let scale = 0;
  for (const row of raw) for (const value of row) scale = Math.max(scale, value);
  // Display mapping follows the paper: P1/CH2 red, P2/CH3 green, P3/CH4 blue.
  if (scale === 0) return indices.map(() => [0, 0, 0]);
  return indices.map(i => raw[i].map(value => value / scale));
}

function sampledIndices(length) {
  const requested = elements["point-count"].value === "all"
    ? length : Math.min(Number(elements["point-count"].value), length);
  if (requested === length) return Array.from({length}, (_, i) => i);
  return Array.from({length: requested}, (_, i) => Math.floor(i * (length - 1) / (requested - 1)));
}

function rgbString(rgb, gamma) {
  const values = rgb.map(value => Math.round(255 * Math.pow(Math.max(0, Math.min(1, value)), gamma)));
  return `rgb(${values[0]},${values[1]},${values[2]})`;
}

function render() {
  if (!decoded || typeof Plotly === "undefined") return;
  const mode = elements.mode.value;
  const isModel = mode.startsWith("model");
  const group = isModel ? decoded.model : decoded.experiment;
  const indices = sampledIndices(group.v1.length);
  const colors = mode.endsWith("extinction")
    ? extinction(group, indices)
    : normalizedOutputs(group, indices, decoded.normalization);
  const gamma = Number(elements.gamma.value);
  const colorStrings = colors.map(color => rgbString(color, gamma));
  const x = indices.map(i => group.v1[i]);
  const y = indices.map(i => group.v2[i]);
  const z = indices.map(i => group.v3[i]);
  const customdata = indices.map(i => [group.out1[i], group.out2[i], group.out3[i]]);

  displayed = {group, indices, x, y, z};
  const [title, subtitle] = titles[mode];
  elements["plot-title"].textContent = title;
  elements["plot-subtitle"].textContent = `${subtitle} · ${indices.length.toLocaleString()} of ${group.v1.length.toLocaleString()} points`;

  const trace = {
    type: "scatter3d",
    mode: "markers",
    x, y, z, customdata,
    marker: {
      color: colorStrings,
      size: Number(elements["point-size"].value),
      opacity: Number(elements.opacity.value),
      line: {width: 0}
    },
    hovertemplate: [
      "V1 %{x:.3f} V", "V2 %{y:.3f} V", "V3 %{z:.3f} V",
      "D1 %{customdata[0]:.5g}", "D2 %{customdata[1]:.5g}",
      "D3 %{customdata[2]:.5g}<extra></extra>"
    ].join("<br>")
  };
  const layout = {
    margin: {l: 0, r: 0, t: 10, b: 0},
    paper_bgcolor: "rgba(0,0,0,0)",
    scene: {
      xaxis: {title: "V1 (V)", gridcolor: "#dfe5eb", zeroline: false},
      yaxis: {title: "V2 (V)", gridcolor: "#dfe5eb", zeroline: false},
      zaxis: {title: "V3 (V)", gridcolor: "#dfe5eb", zeroline: false},
      camera: {eye: {x: 1.45, y: 1.45, z: 1.1}},
      aspectmode: "cube"
    },
    font: {family: "Inter, system-ui, sans-serif", color: "#17243b"}
  };
  Plotly.react("plot", [trace], layout, {responsive: true, displaylogo: false, toImageButtonOptions: {format: "png", scale: 2}});
  setStatus("Ready");
}

async function loadData() {
  try {
    setStatus("Loading");
    const url = elements["data-url"].value.trim();
    payload = await ungzipJson(await fetch(url, {mode: "cors"}));
    decoded = {
      experiment: decodeGroup(payload.experiment),
      model: decodeGroup(payload.model),
      routingStates: decodeGroup(payload.routing_states)
    };
    decoded.normalization = normalization();
    render();
  } catch (error) {
    console.error(error);
    setStatus("Load failed", true);
    elements["plot-subtitle"].textContent = error.message;
  }
}

function downloadCsv() {
  if (!displayed) return;
  const rows = ["v1_v,v2_v,v3_v,output1_v,output2_v,output3_v"];
  for (let j = 0; j < displayed.indices.length; j += 1) {
    const i = displayed.indices[j];
    rows.push([
      displayed.group.v1[i], displayed.group.v2[i], displayed.group.v3[i],
      displayed.group.out1[i], displayed.group.out2[i], displayed.group.out3[i]
    ].join(","));
  }
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([rows.join("\n")], {type: "text/csv"}));
  link.download = `${elements.mode.value}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

for (const id of ["mode", "point-count", "point-size", "opacity", "gamma"]) {
  elements[id].addEventListener("input", () => {
    if (id === "point-size") elements["point-size-value"].value = elements[id].value;
    if (id === "opacity") elements["opacity-value"].value = elements[id].value;
    if (id === "gamma") elements["gamma-value"].value = elements[id].value;
    render();
  });
}
elements["load-data"].addEventListener("click", loadData);
elements["download-csv"].addEventListener("click", downloadCsv);

const queryData = new URLSearchParams(window.location.search).get("data");
if (queryData) elements["data-url"].value = queryData;
window.addEventListener("load", loadData);
