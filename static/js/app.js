// UN Voting Intelligence Dashboard frontend

const state = {
  startYear: 2010,
  endYear: 2020,
  numClusters: 10,
  networkLayout: "force",
  projectionMethod: "pca",
  similarityThreshold: 0.65,
  showNetworkLabels: false,
  activeTab: "story",
  dashboardLoadedFor: null,
  profileCountry: "USA",
  lastProfile: null,
  lastCoalitionTopic: null,
  lastNewsletterMarkdown: null,
  lastNewsletterText: null,
  lastNewsletterSubject: null,
  lastArchiveParams: "",
  knownEvents: null,
  issueCatalog: [],
  issueIndex: [],
  lastSummary: null,
  lastInsights: [],
  lastClustering: null,
  lastSoftPower: null,
  lastPrediction: null,
  lastTraining: null,
  methodsMeta: null,
  countryNames: {},
};

// Colorblind-safe palette (Okabe-Ito), single source of truth for Plotly
// traces. Matches the CSS --ok / --bad and the newsletter SVG colors. Blue vs
// vermillion replaces the old green/red, which was unreadable for red-green
// colour deficiency.
const PALETTE = {
  yes: "#0072b2", // aligned / convergence (blue)
  no: "#d55e00", // opposed / divergence (vermillion)
  abstain: "#999999", // grey
  // Diverging scale for the alignment map: opposed (vermillion) → neutral → aligned (blue)
  diverging: [
    [0, "#d55e00"],
    [0.5, "#f2f2f2"],
    [1, "#0072b2"],
  ],
};

let softPowerToken = 0;
let softPowerAbortController = null;
let trainAbortController = null;

function getErrorMessage(error) {
  return (
    error?.response?.data?.error ||
    error?.message ||
    "Request failed. Please try again."
  );
}

function toFiniteNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clearNode(element) {
  if (!element) return;
  element.innerHTML = "";
}

function setLoading(element, message) {
  if (!element) return;
  clearNode(element);
  const loader = document.createElement("div");
  loader.className = "loading-overlay";
  loader.textContent = message;
  element.appendChild(loader);
}

function showErrorElement(element, message) {
  if (!element) return;
  clearNode(element);
  const panel = document.createElement("div");
  panel.className = "error-message";
  panel.textContent = message;
  element.appendChild(panel);
}

function debounce(fn, waitMs) {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), waitMs);
  };
}

function requireLibraries() {
  if (typeof axios === "undefined") {
    throw new Error("Axios failed to load");
  }
}

function requirePlotly() {
  if (typeof Plotly === "undefined") {
    throw new Error("Plotly failed to load");
  }
}

function normalizeCodeInput(value) {
  return String(value || "")
    .trim()
    .toUpperCase()
    .slice(0, 3);
}

function reduceState(prev, action) {
  switch (action.type) {
    case "SET_RANGE":
      return {
        ...prev,
        startYear:
          action.startYear !== undefined ? action.startYear : prev.startYear,
        endYear: action.endYear !== undefined ? action.endYear : prev.endYear,
      };
    case "SET_CLUSTERS":
      return { ...prev, numClusters: action.numClusters };
    case "SET_LAYOUT":
      return { ...prev, networkLayout: action.networkLayout };
    case "SET_PROJECTION":
      return { ...prev, projectionMethod: action.projectionMethod };
    case "SET_THRESHOLD":
      return { ...prev, similarityThreshold: action.similarityThreshold };
    case "SET_LABELS":
      return { ...prev, showNetworkLabels: action.showNetworkLabels };
    case "SET_TAB":
      return { ...prev, activeTab: action.activeTab };
    default:
      return prev;
  }
}

function writeHashState() {
  const params = new URLSearchParams();
  params.set("tab", state.activeTab);
  params.set("country", state.profileCountry);
  params.set("start", String(state.startYear));
  params.set("end", String(state.endYear));
  params.set("k", String(state.numClusters));
  params.set("layout", state.networkLayout);
  params.set("projection", state.projectionMethod);
  params.set("threshold", String(state.similarityThreshold));
  params.set("labels", state.showNetworkLabels ? "1" : "0");

  const hash = params.toString();
  if (window.location.hash.slice(1) !== hash) {
    history.replaceState(null, "", `#${hash}`);
  }
}

function applyHashState() {
  if (!window.location.hash || window.location.hash.length <= 1) return;
  const params = new URLSearchParams(window.location.hash.slice(1));

  // A missing hash parameter must keep the current value: Number(null) is 0,
  // which used to send start_year=0 on every deep link without a window.
  const hashNumber = (key, fallback) => {
    const raw = params.get(key);
    return raw === null || raw.trim() === "" ? fallback : toFiniteNumber(raw, fallback);
  };
  const start = hashNumber("start", state.startYear);
  const end = hashNumber("end", state.endYear);
  const k = hashNumber("k", state.numClusters);
  const threshold = hashNumber("threshold", state.similarityThreshold);
  const allowedTabs = new Set([
    "story",
    "lenses",
    "profile",
    "map",
    "pivotality",
    "abstention",
    "coalition",
    "newsletter",
    "drift",
    "dashboard",
    "network",
    "softpower",
    "bloc",
    "predictions",
    "compare",
  ]);
  const requestedTab = params.get("tab") || state.activeTab;
  const tab = allowedTabs.has(requestedTab) ? requestedTab : state.activeTab;
  const country = normalizeCodeInput(params.get("country") || state.profileCountry);
  if (country.length === 3) state.profileCountry = country;
  const layout = params.get("layout") || state.networkLayout;
  const projection = params.get("projection") || state.projectionMethod;
  const labels = params.get("labels") === "1";

  Object.assign(
    state,
    reduceState(state, {
      type: "SET_RANGE",
      startYear: start,
      endYear: end,
    }),
  );
  Object.assign(state, reduceState(state, { type: "SET_CLUSTERS", numClusters: k }));
  Object.assign(
    state,
    reduceState(state, { type: "SET_THRESHOLD", similarityThreshold: threshold }),
  );
  Object.assign(
    state,
    reduceState(state, { type: "SET_LAYOUT", networkLayout: layout }),
  );
  Object.assign(
    state,
    reduceState(state, { type: "SET_PROJECTION", projectionMethod: projection }),
  );
  Object.assign(
    state,
    reduceState(state, { type: "SET_LABELS", showNetworkLabels: labels }),
  );
  Object.assign(state, reduceState(state, { type: "SET_TAB", activeTab: tab }));
}

function syncControlsFromState() {
  const start = document.getElementById("startYear");
  const end = document.getElementById("endYear");
  const clusters = document.getElementById("numClusters");
  const clusterVal = document.getElementById("clusterVal");
  const layout = document.getElementById("networkLayout");
  const projection = document.getElementById("projectionMethod");
  const threshold = document.getElementById("similarityThreshold");
  const labels = document.getElementById("showNetworkLabels");

  if (start) start.value = String(state.startYear);
  if (end) end.value = String(state.endYear);
  if (clusters) clusters.value = String(state.numClusters);
  if (clusterVal) clusterVal.textContent = String(state.numClusters);
  if (layout) layout.value = state.networkLayout;
  if (projection) projection.value = state.projectionMethod;
  if (threshold) threshold.value = String(state.similarityThreshold);
  if (labels) labels.checked = state.showNetworkLabels;
}

function updateMethodPanel(meta) {
  if (!meta) return;
  state.methodsMeta = meta;

  const target = document.getElementById("methodsContent");
  if (!target) return;

  const selected = meta.selected_window || {};
  const methods = meta.methods || {};
  const caveats = meta.caveats || [];

  clearNode(target);

  const rows = [
    `Window: ${selected.start_year ?? "-"} to ${selected.end_year ?? "-"}`,
    `Vote Encoding: yes=${methods.vote_encoding?.yes}, abstain=${methods.vote_encoding?.abstain}, no=${methods.vote_encoding?.no}`,
    `Similarity: ${methods.similarity || "-"}`,
    `Clustering: ${methods.clustering || "-"}`,
    `Projection: ${methods.projection || "-"}`,
    `Soft Power Label: ${methods.soft_power_label || "-"}`,
  ];

  rows.forEach((line) => {
    const row = document.createElement("div");
    row.textContent = line;
    target.appendChild(row);
  });

  if (caveats.length) {
    caveats.slice(0, 3).forEach((item) => {
      const caveat = document.createElement("div");
      caveat.className = "text-muted";
      caveat.textContent = `Caveat: ${item}`;
      target.appendChild(caveat);
    });
  }
}

function maybeUpdateMeta(payload) {
  const meta = payload?.meta || null;
  if (meta) {
    updateMethodPanel(meta);
  }
}

function csvEscape(value) {
  const str = String(value ?? "");
  if (str.includes(",") || str.includes('"') || str.includes("\n")) {
    return `"${str.replaceAll('"', '""')}"`;
  }
  return str;
}

function downloadCsv(filename, rows) {
  const content = rows.map((row) => row.map(csvEscape).join(",")).join("\n");
  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function pollJob(jobId, options = {}) {
  const intervalMs = options.intervalMs || 1000;
  const timeoutMs = options.timeoutMs || 180000;
  const startedAt = Date.now();
  const signal = options.signal;

  const assertNotAborted = () => {
    if (signal?.aborted) {
      throw new DOMException("Job polling cancelled", "AbortError");
    }
  };

  const waitWithAbort = (ms) =>
    new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (signal) {
          signal.removeEventListener("abort", onAbort);
        }
        resolve();
      }, ms);

      const onAbort = () => {
        clearTimeout(timer);
        signal.removeEventListener("abort", onAbort);
        reject(new DOMException("Job polling cancelled", "AbortError"));
      };

      if (signal) {
        signal.addEventListener("abort", onAbort, { once: true });
      }
    });

  while (Date.now() - startedAt < timeoutMs) {
    assertNotAborted();
    const response = await axios.get(`/api/jobs/${jobId}`, { signal });
    const job = response.data;
    if (typeof options.onProgress === "function") {
      options.onProgress(job);
    }

    if (job.status === "completed") {
      return job.result;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "Background job failed");
    }

    await waitWithAbort(intervalMs);
  }

  throw new Error("Background job timed out");
}

function renderPlot(targetId, payload, margin = { l: 45, r: 20, t: 42, b: 42 }) {
  requirePlotly();
  const target = document.getElementById(targetId);
  if (!target) return;

  clearNode(target);
  const layout = { ...(payload.layout || {}) };
  layout.autosize = true;
  layout.margin = { ...margin, ...(layout.margin || {}) };

  Plotly.newPlot(target, payload.data, layout, {
    responsive: true,
    displayModeBar: false,
  });
}

function renderClusters(data) {
  const container = document.getElementById("clusterResults");
  if (!container) return;

  clearNode(container);
  const clusters = data?.clusters || {};
  const entries = Object.entries(clusters).sort((a, b) => b[1].length - a[1].length);

  if (!entries.length) {
    showErrorElement(container, "No clusters were generated for this period.");
    return;
  }

  if (data?.stability) {
    const s = data.stability;
    const card = document.createElement("div");
    card.className = "comparison-meta";
    if (s.available) {
      card.textContent = `Stability (bootstrapped): ARI ${toFiniteNumber(
        s.ari_mean,
        0,
      ).toFixed(3)} +/- ${toFiniteNumber(s.ari_std, 0).toFixed(
        3,
      )}, NMI ${toFiniteNumber(s.nmi_mean, 0).toFixed(3)} +/- ${toFiniteNumber(
        s.nmi_std,
        0,
      ).toFixed(3)} over ${toFiniteNumber(s.n_effective, 0)} runs.`;
    } else {
      card.textContent = `Stability metrics unavailable: ${s.reason || "insufficient data"}`;
    }
    container.appendChild(card);
  }

  const grid = document.createElement("div");
  grid.className = "cluster-grid";

  const labels = data?.cluster_labels || {};
  entries.forEach(([clusterId, members]) => {
    const card = document.createElement("article");
    card.className = "cluster-card";

    const labelInfo = labels[clusterId];
    const title = document.createElement("div");
    title.className = "cluster-title";
    title.textContent = labelInfo?.label || `Cluster ${toFiniteNumber(clusterId, 0) + 1}`;

    const count = document.createElement("div");
    count.className = "cluster-count";
    if (labelInfo?.signature_topic) {
      count.innerHTML = `${members.length} members · <span class="cluster-signature">${labelInfo.signature_topic}</span>`;
    } else {
      count.textContent = `${members.length} members`;
    }

    const list = document.createElement("div");
    list.className = "cluster-members";

    members.forEach((member) => {
      const pill = document.createElement("div");
      pill.className = "cluster-member";
      pill.textContent = String(member);
      list.appendChild(pill);
    });

    card.append(title, count, list);
    grid.appendChild(card);
  });

  container.appendChild(grid);
}

async function loadDataSummary() {
  try {
    requireLibraries();
    const response = await axios.get("/api/data/summary");
    const data = response.data;
    state.lastSummary = data;

    document.getElementById("totalVotes").textContent = toFiniteNumber(
      data.total_votes,
    ).toLocaleString();
    document.getElementById("totalCountries").textContent = toFiniteNumber(
      data.countries,
    ).toLocaleString();
    document.getElementById("totalResolutions").textContent = toFiniteNumber(
      data.resolutions,
    ).toLocaleString();

    const minYear = data?.year_range?.min ?? "-";
    const maxYear = data?.year_range?.max ?? "-";
    document.getElementById("yearRange").textContent = `${minYear} - ${maxYear}`;

    const start = document.getElementById("startYear");
    const end = document.getElementById("endYear");
    const trainEnd = document.getElementById("trainEndYear");
    const testStart = document.getElementById("testStartYear");

    if (start && end && Number.isFinite(minYear) && Number.isFinite(maxYear)) {
      start.min = minYear;
      start.max = maxYear;
      end.min = minYear;
      end.max = maxYear;
      trainEnd.min = minYear;
      trainEnd.max = maxYear;
      testStart.min = minYear;
      testStart.max = maxYear;
    }

    maybeUpdateMeta(data);
  } catch (error) {
    console.error("Summary load failed", error);
  }
}

async function loadMethods() {
  try {
    requireLibraries();
    const response = await axios.get("/api/methods");
    updateMethodPanel(response.data);
  } catch (error) {
    console.error("Methods load failed", error);
  }
}

async function loadClustering() {
  const container = document.getElementById("clusterResults");
  setLoading(container, "Computing voting blocs and stability diagnostics...");

  try {
    requireLibraries();
    const response = await axios.post("/api/analysis/clustering", {
      start_year: state.startYear,
      end_year: state.endYear,
      num_clusters: state.numClusters,
      include_stability: true,
    });
    state.lastClustering = response.data;
    renderClusters(response.data);
    maybeUpdateMeta(response.data);
  } catch (error) {
    showErrorElement(container, `Clustering failed: ${getErrorMessage(error)}`);
  }
}

async function loadPCAPlot() {
  const container = document.getElementById("pcaPlot");
  setLoading(
    container,
    `Projecting voting space (${state.projectionMethod.toUpperCase()})...`,
  );

  try {
    requireLibraries();
    const response = await axios.post("/api/visualization/pca", {
      start_year: state.startYear,
      end_year: state.endYear,
      projection: state.projectionMethod,
    });

    if (!response.data?.data) {
      throw new Error("Invalid PCA payload");
    }
    renderPlot("pcaPlot", response.data);
    maybeUpdateMeta(response.data);
  } catch (error) {
    showErrorElement(
      container,
      `Projection failed: ${getErrorMessage(error)}`,
    );
  }
}

async function loadIssueTimeline() {
  const container = document.getElementById("issueTimeline");
  setLoading(container, "Loading issue frequencies...");

  try {
    requireLibraries();
    const response = await axios.post("/api/visualization/issue-timeline", {
      start_year: state.startYear,
      end_year: state.endYear,
      top_n: 10,
    });

    if (!response.data?.data) {
      throw new Error("Invalid issue timeline payload");
    }
    renderPlot("issueTimeline", response.data, { l: 60, r: 240, t: 50, b: 50 });
    maybeUpdateMeta(response.data);
  } catch (error) {
    showErrorElement(container, `Issue timeline failed: ${getErrorMessage(error)}`);
  }
}

async function loadBlocTimeline() {
  if (state.activeTab !== "bloc") return;

  const container = document.getElementById("blocTimelinePlot");
  setLoading(container, "Building bloc membership timeline...");

  try {
    requireLibraries();
    const response = await axios.post("/api/analysis/bloc-timeline", {
      start_year: state.startYear,
      end_year: state.endYear,
      window: 5,
      num_clusters: state.numClusters,
    });

    if (!response.data?.data) {
      throw new Error("Invalid bloc timeline payload");
    }

    renderPlot("blocTimelinePlot", response.data, {
      l: 20,
      r: 20,
      t: 44,
      b: 28,
    });
    maybeUpdateMeta(response.data);
  } catch (error) {
    showErrorElement(container, `Bloc timeline failed: ${getErrorMessage(error)}`);
  }
}

async function loadNetworkGraph() {
  if (state.activeTab !== "network") return;

  const container = document.getElementById("networkGraph");
  const button = document.getElementById("updateNetworkBtn");
  setLoading(container, "Building similarity network...");

  if (button) {
    button.disabled = true;
    button.textContent = "Updating...";
  }

  try {
    requireLibraries();
    const response = await axios.post("/api/visualization/network", {
      start_year: state.startYear,
      end_year: state.endYear,
      layout: state.networkLayout,
      threshold: state.similarityThreshold,
      show_labels: state.showNetworkLabels,
    });

    if (!response.data?.data) {
      throw new Error("Invalid network payload");
    }

    const layout = { ...(response.data.layout || {}) };
    layout.autosize = true;
    layout.margin = { l: 20, r: 20, t: 40, b: 20 };

    requirePlotly();
    clearNode(container);
    Plotly.newPlot(container, response.data.data, layout, {
      responsive: true,
      displayModeBar: true,
    });
    maybeUpdateMeta(response.data);
  } catch (error) {
    showErrorElement(container, `Network failed: ${getErrorMessage(error)}`);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "Update Network";
    }
  }
}

function renderSoftPowerTable(data) {
  const container = document.getElementById("softPowerTable");
  if (!container) return;
  clearNode(container);

  const scores = data?.scores || {};
  const entries = Object.entries(scores);
  if (!entries.length) {
    showErrorElement(container, "No soft power data available.");
    return;
  }

  const table = document.createElement("table");
  table.className = "data-table";

  const thead = document.createElement("thead");
  thead.innerHTML = "<tr><th>Rank</th><th>Country</th><th>Score</th></tr>";

  const tbody = document.createElement("tbody");
  let rank = 1;

  entries.forEach(([country, rawScore]) => {
    const score = Math.max(0, Math.min(1, toFiniteNumber(rawScore, 0)));

    const row = document.createElement("tr");
    const rankCell = document.createElement("td");
    rankCell.textContent = String(rank++);

    const countryCell = document.createElement("td");
    countryCell.textContent = country;

    const scoreCell = document.createElement("td");
    const barWrap = document.createElement("div");
    barWrap.style.display = "flex";
    barWrap.style.alignItems = "center";
    barWrap.style.gap = "8px";

    const track = document.createElement("div");
    track.style.flexGrow = "1";
    track.style.height = "8px";
    track.style.borderRadius = "99px";
    track.style.background = "#e4edf0";

    const fill = document.createElement("div");
    fill.style.height = "100%";
    fill.style.width = `${(score * 100).toFixed(1)}%`;
    fill.style.borderRadius = "99px";
    fill.style.background = "linear-gradient(90deg, #0e6f82, #1f8ea5)";

    track.appendChild(fill);

    const value = document.createElement("span");
    value.textContent = score.toFixed(4);

    barWrap.append(track, value);
    scoreCell.appendChild(barWrap);

    row.append(rankCell, countryCell, scoreCell);
    tbody.appendChild(row);
  });

  table.append(thead, tbody);
  container.appendChild(table);
}

async function loadSoftPower() {
  if (state.activeTab !== "softpower") return;

  const table = document.getElementById("softPowerTable");
  const trends = document.getElementById("softPowerTrends");
  const currentToken = ++softPowerToken;
  if (softPowerAbortController) {
    softPowerAbortController.abort();
  }
  softPowerAbortController = new AbortController();
  const signal = softPowerAbortController.signal;

  setLoading(table, "Computing soft power scores...");
  setLoading(trends, "Preparing trend computation...");

  try {
    requireLibraries();

    const scoreResp = await axios.post("/api/analysis/soft-power", {
      start_year: state.startYear,
      end_year: state.endYear,
    }, { signal });

    if (currentToken !== softPowerToken) return;

    state.lastSoftPower = scoreResp.data;
    renderSoftPowerTable(scoreResp.data);
    maybeUpdateMeta(scoreResp.data);

    const jobResp = await axios.post("/api/jobs/soft-power-trends", {
      start_year: state.startYear,
      end_year: state.endYear,
    }, { signal });

    const trendsPayload = await pollJob(jobResp.data.job_id, {
      onProgress: (job) => {
        if (currentToken !== softPowerToken) return;
        const pct = Math.round(toFiniteNumber(job.progress, 0) * 100);
        setLoading(trends, `${job.message || "Computing trends"} (${pct}%)`);
      },
      timeoutMs: 240000,
      signal,
    });

    if (currentToken !== softPowerToken) return;

    renderPlot("softPowerTrends", trendsPayload, {
      l: 50,
      r: 18,
      t: 42,
      b: 46,
    });
    maybeUpdateMeta(trendsPayload);
  } catch (error) {
    if (error?.name === "AbortError") {
      return;
    }
    if (currentToken !== softPowerToken) return;
    showErrorElement(table, `Soft power failed: ${getErrorMessage(error)}`);
    showErrorElement(trends, `Trend rendering failed: ${getErrorMessage(error)}`);
  } finally {
    if (softPowerAbortController?.signal === signal) {
      softPowerAbortController = null;
    }
  }
}

function renderIssueIndex() {
  const container = document.getElementById("issueIndex");
  if (!container) return;
  clearNode(container);

  if (!state.issueIndex.length) return;

  state.issueIndex.slice(0, 6).forEach((topic) => {
    const row = document.createElement("div");
    row.className = "issue-index-row";
    row.textContent = `${topic.topic} (${topic.count})`;
    container.appendChild(row);
  });
}

function renderIssueOptions(filterText = "") {
  const select = document.getElementById("predictionIssue");
  const meta = document.getElementById("issueSearchMeta");
  if (!select) return;

  const query = filterText.trim().toLowerCase();
  let filtered = state.issueCatalog;
  if (query) {
    filtered = state.issueCatalog.filter((issue) =>
      issue.toLowerCase().includes(query),
    );
  }
  filtered = filtered.slice(0, 300);

  clearNode(select);
  filtered.forEach((issue) => {
    const option = document.createElement("option");
    option.value = issue;
    option.textContent = issue;
    select.appendChild(option);
  });

  if (meta) {
    meta.textContent = `Showing ${filtered.length} of ${state.issueCatalog.length} issues`;
  }
}

async function loadIssues() {
  try {
    requireLibraries();
    const response = await axios.get("/api/prediction/issues");

    state.issueCatalog = response.data?.issues || [];
    state.issueIndex = response.data?.issue_index || [];

    renderIssueOptions("");
    renderIssueIndex();
    maybeUpdateMeta(response.data);
  } catch (error) {
    console.error("Issue load failed", error);
  }
}

function showFieldError(fieldId, message) {
  const field = document.getElementById(fieldId);
  if (!field || !field.parentElement) return;

  const existing = field.parentElement.querySelector(".field-error");
  if (existing) existing.remove();

  const msg = document.createElement("div");
  msg.className = "field-error";
  msg.textContent = message;
  field.parentElement.appendChild(msg);

  field.addEventListener(
    "input",
    () => {
      msg.remove();
    },
    { once: true },
  );
}

function renderCalibrationDiagnostics(diagnostics) {
  const host = document.getElementById("calibrationResults");
  if (!host) return;

  if (!diagnostics || !Array.isArray(diagnostics.reliability_bins)) {
    host.classList.remove("active");
    clearNode(host);
    return;
  }

  const bins = diagnostics.reliability_bins;
  if (!bins.length) {
    host.classList.remove("active");
    clearNode(host);
    return;
  }

  host.classList.add("active");
  clearNode(host);

  requirePlotly();
  const x = bins.map((b) => toFiniteNumber(b.mean_confidence, 0));
  const y = bins.map((b) => toFiniteNumber(b.empirical_accuracy, 0));
  const size = bins.map((b) => Math.max(8, toFiniteNumber(b.count, 0) / 50));

  Plotly.newPlot(
    host,
    [
      {
        x,
        y,
        mode: "markers+lines",
        marker: { size, color: "#0e6f82", opacity: 0.9 },
        line: { color: "#1f8ea5", width: 2 },
        name: "Observed",
      },
      {
        x: [0, 1],
        y: [0, 1],
        mode: "lines",
        line: { color: "#c64141", dash: "dash" },
        name: "Perfect calibration",
      },
    ],
    {
      title: "Reliability Curve",
      margin: { t: 40, l: 42, r: 18, b: 42 },
      xaxis: { title: "Predicted confidence", range: [0, 1] },
      yaxis: { title: "Empirical accuracy", range: [0, 1] },
      autosize: true,
      showlegend: true,
    },
    { displayModeBar: false, responsive: true },
  );
}

function renderTrainingResult(data) {
  const container = document.getElementById("trainResults");
  clearNode(container);

  const accuracy = Number(data.accuracy);
  const accuracyText = Number.isFinite(accuracy)
    ? `${(accuracy * 100).toFixed(1)}%`
    : String(data.accuracy ?? "N/A");

  const card = document.createElement("div");
  card.className = "success-message";

  const title = document.createElement("h4");
  title.textContent = data.cached ? "Model Ready (Cached)" : "Model Trained";

  const p1 = document.createElement("p");
  p1.textContent = `Accuracy: ${accuracyText}`;

  const p2 = document.createElement("p");
  p2.textContent = `Training Samples: ${toFiniteNumber(data.train_samples, 0)}`;

  const p3 = document.createElement("p");
  p3.textContent = `Test Samples: ${toFiniteNumber(data.test_samples, 0)}`;

  const diagnostics = data.diagnostics || {};
  const brier = diagnostics.brier_score;
  if (Number.isFinite(brier)) {
    const p4 = document.createElement("p");
    p4.textContent = `Brier Score: ${brier.toFixed(4)}`;
    card.appendChild(p4);
  }

  const baselines = diagnostics.baselines || {};
  const baselineRows = [];
  if (Number.isFinite(baselines.majority_vote_accuracy)) {
    baselineRows.push(
      `Majority baseline: ${(baselines.majority_vote_accuracy * 100).toFixed(1)}%`,
    );
  }
  if (Number.isFinite(baselines.country_prior_accuracy)) {
    baselineRows.push(
      `Country-prior baseline: ${(baselines.country_prior_accuracy * 100).toFixed(1)}%`,
    );
  }

  card.append(title, p1, p2, p3);
  baselineRows.forEach((line) => {
    const row = document.createElement("p");
    row.textContent = line;
    card.appendChild(row);
  });

  container.appendChild(card);

  renderCalibrationDiagnostics(diagnostics);
}

async function trainModel() {
  const trainEndInput = document.getElementById("trainEndYear");
  const testStartInput = document.getElementById("testStartYear");
  const button = document.getElementById("trainModelBtn");
  const container = document.getElementById("trainResults");
  const predictBtn = document.getElementById("predictBtn");

  const trainEnd = toFiniteNumber(trainEndInput.value, NaN);
  const testStart = toFiniteNumber(testStartInput.value, NaN);

  if (!Number.isFinite(trainEnd) || !Number.isFinite(testStart)) {
    showFieldError("trainEndYear", "Provide valid numeric years.");
    return;
  }
  if (testStart <= trainEnd) {
    showFieldError("testStartYear", "Test year must be greater than train year.");
    return;
  }

  setLoading(container, "Queueing model training...");
  button.disabled = true;
  if (predictBtn) predictBtn.disabled = true;

  if (trainAbortController) {
    trainAbortController.abort();
  }
  trainAbortController = new AbortController();
  const signal = trainAbortController.signal;

  const diagnosticsHost = document.getElementById("calibrationResults");
  if (diagnosticsHost) {
    diagnosticsHost.classList.remove("active");
    clearNode(diagnosticsHost);
  }

  try {
    requireLibraries();
    const queueResp = await axios.post("/api/jobs/train-model", {
      train_end: trainEnd,
      test_start: testStart,
    }, { signal });

    const result = await pollJob(queueResp.data.job_id, {
      onProgress: (job) => {
        const pct = Math.round(toFiniteNumber(job.progress, 0) * 100);
        setLoading(container, `${job.message || "Training predictor"} (${pct}%)`);
      },
      timeoutMs: 300000,
      signal,
    });

    state.lastTraining = result;
    renderTrainingResult(result);

    if (predictBtn) {
      predictBtn.disabled = false;
      predictBtn.title = "Predict using trained model";
    }

    maybeUpdateMeta(result);
  } catch (error) {
    if (error?.name === "AbortError") {
      return;
    }
    showErrorElement(container, `Training failed: ${getErrorMessage(error)}`);
  } finally {
    button.disabled = false;
    if (trainAbortController?.signal === signal) {
      trainAbortController = null;
    }
  }
}

async function predictVote() {
  const container = document.getElementById("predictionResults");
  const button = document.getElementById("predictBtn");
  const issue = document.getElementById("predictionIssue").value;
  const trainEnd = toFiniteNumber(document.getElementById("trainEndYear").value, NaN);
  const testStart = toFiniteNumber(document.getElementById("testStartYear").value, NaN);

  if (!issue) {
    showErrorElement(container, "Select an issue before predicting.");
    return;
  }

  setLoading(container, "Predicting vote distribution...");
  button.disabled = true;

  try {
    requireLibraries();
    requirePlotly();

    const response = await axios.post("/api/prediction/predict", {
      issue,
      train_end: trainEnd,
      test_start: testStart,
      prediction_year: state.endYear,
    });

    const summary = response.data?.summary || [];
    state.lastPrediction = response.data;

    const labels = summary.map((entry) => String(entry.Vote));
    const values = summary.map((entry) => toFiniteNumber(entry.Count, 0));

    clearNode(container);
    Plotly.newPlot(
      container,
      [
        {
          x: labels,
          y: values,
          type: "bar",
          marker: {
            color: labels.map((label) => {
              if (label === "Yes") return PALETTE.yes;
              if (label === "No") return PALETTE.no;
              return "#1f8ea5";
            }),
          },
        },
      ],
      {
        title: `Predicted Vote Distribution`,
        margin: { t: 44, l: 44, r: 18, b: 44 },
        autosize: true,
      },
      { displayModeBar: false, responsive: true },
    );

    maybeUpdateMeta(response.data);
  } catch (error) {
    showErrorElement(container, `Prediction failed: ${getErrorMessage(error)}`);
  } finally {
    button.disabled = false;
  }
}

async function compareCountries() {
  const container = document.getElementById("compareResults");
  const button = document.getElementById("compareBtn");
  const countryAInput = document.getElementById("countryA");
  const countryBInput = document.getElementById("countryB");

  const countryA = resolveCountryCode(countryAInput.value);
  const countryB = resolveCountryCode(countryBInput.value);

  if (!countryA) {
    showFieldError("countryA", "Pick a country from the list, or type its ISO-3 code");
    return;
  }
  if (!countryB) {
    showFieldError("countryB", "Pick a country from the list, or type its ISO-3 code");
    return;
  }
  countryAInput.value = nameFor(countryA);
  countryBInput.value = nameFor(countryB);

  setLoading(container, "Comparing country voting behavior...");
  button.disabled = true;

  try {
    requireLibraries();
    const response = await axios.post("/api/analysis/compare", {
      country_a: countryA,
      country_b: countryB,
      start_year: state.startYear,
      end_year: state.endYear,
    });

    const data = response.data;
    clearNode(container);

    const summary = document.createElement("div");
    summary.className = "comparison-meta";

    const pct = (toFiniteNumber(data.similarity, 0) * 100).toFixed(1);
    summary.textContent = `Similarity: ${pct}% (${countryA} vs ${countryB}, ${state.startYear}-${state.endYear})`;

    const list = document.createElement("div");
    list.className = "anomaly-list";

    (data.anomalies || []).forEach((anomaly) => {
      const item = document.createElement("article");
      item.className = "anomaly-item";

      const top = document.createElement("div");
      top.className = "anomaly-top";

      const year = document.createElement("span");
      const anomalyDate = anomaly.date ? new Date(anomaly.date) : null;
      year.textContent = anomalyDate ? String(anomalyDate.getFullYear()) : "Unknown year";

      const delta = document.createElement("span");
      delta.className = "anomaly-delta";
      delta.textContent = `Delta ${toFiniteNumber(anomaly.similarity_delta, 0).toFixed(2)}`;

      top.append(year, delta);

      const issue = document.createElement("p");
      issue.textContent = anomaly.issue || "Unknown issue";

      const detail = document.createElement("p");
      detail.className = "text-muted";
      detail.textContent = `${countryA}: ${anomaly.vote_a} vs ${countryB}: ${anomaly.vote_b}`;

      item.append(top, issue, detail);
      list.appendChild(item);
    });

    if (!list.children.length) {
      const empty = document.createElement("div");
      empty.className = "text-muted";
      empty.textContent = "No major anomalies were detected for this period.";
      list.appendChild(empty);
    }

    container.append(summary, list);
    maybeUpdateMeta(data);
  } catch (error) {
    showErrorElement(container, `Comparison failed: ${getErrorMessage(error)}`);
  } finally {
    button.disabled = false;
  }
}

function getIconForType(type) {
  if (type === "success") return "chart-line";
  if (type === "warning") return "triangle-exclamation";
  return "circle-info";
}

async function loadInsights() {
  const container = document.getElementById("insights-container");
  if (!container) return;

  try {
    requireLibraries();
    const response = await axios.post("/api/insights", {
      start_year: state.startYear,
      end_year: state.endYear,
    });
    const insights = response.data?.insights || [];
    state.lastInsights = insights;

    clearNode(container);

    insights.forEach((insight) => {
      const card = document.createElement("article");
      card.className = `insight-card ${insight.type || "info"}`;

      const iconWrap = document.createElement("div");
      iconWrap.className = "insight-icon";
      const icon = document.createElement("i");
      icon.className = `fas fa-${getIconForType(insight.type)}`;
      iconWrap.appendChild(icon);

      const content = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = insight.title || "Insight";
      const text = document.createElement("p");
      text.textContent = insight.text || "";

      content.append(title, text);
      card.append(iconWrap, content);
      container.appendChild(card);
    });

    maybeUpdateMeta(response.data);
  } catch (error) {
    console.error("Insight load failed", error);
  }
}

function exportData(type) {
  if (type === "analysis") {
    const rows = [["Field", "Value"]];

    rows.push(["Start Year", state.startYear]);
    rows.push(["End Year", state.endYear]);
    rows.push(["Clusters", state.numClusters]);
    rows.push(["Network Layout", state.networkLayout]);
    rows.push(["Similarity Threshold", state.similarityThreshold]);

    if (state.lastSummary) {
      rows.push(["Total Votes", state.lastSummary.total_votes]);
      rows.push(["Countries", state.lastSummary.countries]);
      rows.push(["Resolutions", state.lastSummary.resolutions]);
    }

    if (state.lastClustering?.stability?.available) {
      const s = state.lastClustering.stability;
      rows.push(["Stability ARI Mean", s.ari_mean]);
      rows.push(["Stability ARI Std", s.ari_std]);
      rows.push(["Stability NMI Mean", s.nmi_mean]);
      rows.push(["Stability NMI Std", s.nmi_std]);
      rows.push(["Stability Runs", s.n_effective]);
    }

    state.lastInsights.forEach((insight, idx) => {
      rows.push([`Insight ${idx + 1} - ${insight.title}`, insight.text]);
    });

    downloadCsv("analysis_summary.csv", rows);
    return;
  }

  if (!state.lastPrediction?.summary?.length) {
    alert("Run a prediction first to export prediction output.");
    return;
  }

  const rows = [["Vote", "Count"]];
  state.lastPrediction.summary.forEach((row) => {
    rows.push([row.Vote, row.Count]);
  });

  if (Array.isArray(state.lastPrediction.details) && state.lastPrediction.details.length) {
    rows.push([]);
    rows.push(["Country", "Predicted Vote"]);
    state.lastPrediction.details.forEach((row) => {
      rows.push([row.Country, row["Predicted Vote"]]);
    });
  }

  downloadCsv("prediction_results.csv", rows);
}

const COALITION_TIER_ORDER = [
  "Champion supporter",
  "Reliable supporter",
  "Leans supporter",
  "Fence-sitter",
  "Leans opposed",
  "Reliable opposed",
  "Champion opposed",
];

function renderCoalition(report) {
  const headline = document.getElementById("coalitionHeadline");
  const tallyEl = document.getElementById("coalitionTally");
  const tiersEl = document.getElementById("coalitionTiers");
  const samplesEl = document.getElementById("coalitionSamples");
  if (!report) return;

  if (headline) {
    headline.innerHTML = "";
    const story = document.createElement("p");
    story.className = "profile-story";
    if (report.matched_resolutions === 0) {
      story.innerHTML = `No resolutions matched <strong>"${report.topic}"</strong> in ${report.window?.start_year}–${report.window?.end_year}. Try a different keyword.`;
    } else {
      const fence = (report.tiers?.["Fence-sitter"] || []).length;
      story.innerHTML =
        `On <strong>"${report.topic}"</strong> across <strong>${report.matched_resolutions}</strong> matched resolutions, ` +
        `<strong>${report.predicted_tally?.yes ?? 0}</strong> states would back a Yes vote, ` +
        `<strong>${report.predicted_tally?.no ?? 0}</strong> would oppose, ` +
        `and <strong>${report.predicted_tally?.abstain ?? 0}</strong> would abstain. ` +
        `<strong>${fence}</strong> fence-sitters are the lobbying targets.`;
    }
    headline.appendChild(story);
  }

  if (tallyEl) {
    clearNode(tallyEl);
    if (report.matched_resolutions > 0) {
      const tally = report.predicted_tally || {};
      const total = (tally.yes || 0) + (tally.no || 0) + (tally.abstain || 0) + (tally.no_history || 0);
      const segments = [
        { label: "Yes", value: tally.yes || 0, cls: "tally-yes" },
        { label: "Abstain", value: tally.abstain || 0, cls: "tally-abstain" },
        { label: "No", value: tally.no || 0, cls: "tally-no" },
        { label: "No history", value: tally.no_history || 0, cls: "tally-unknown" },
      ];
      const bar = document.createElement("div");
      bar.className = "coalition-tally-bar";
      segments.forEach((seg) => {
        if (seg.value === 0) return;
        const part = document.createElement("div");
        part.className = `coalition-tally-bar__seg ${seg.cls}`;
        part.style.flexGrow = String(seg.value);
        part.title = `${seg.label}: ${seg.value} (${total ? Math.round((seg.value / total) * 100) : 0}%)`;
        part.textContent = `${seg.label} ${seg.value}`;
        bar.appendChild(part);
      });
      tallyEl.appendChild(bar);
    }
  }

  if (tiersEl) {
    clearNode(tiersEl);
    COALITION_TIER_ORDER.forEach((tier) => {
      const members = report.tiers?.[tier] || [];
      if (members.length === 0) return;
      const card = document.createElement("article");
      card.className = `coalition-tier coalition-tier--${tier.toLowerCase().replace(/\W+/g, "-")}`;
      const header = document.createElement("header");
      header.className = "coalition-tier__header";
      header.innerHTML = `<h4>${tier}</h4><span class="coalition-tier__count">${members.length}</span>`;
      card.appendChild(header);

      const list = document.createElement("ul");
      list.className = "coalition-tier__list";
      members.slice(0, 25).forEach((row) => {
        const li = document.createElement("li");
        const yes = row.n_yes;
        const no = row.n_no;
        const ab = row.n_abstain;
        const lean = row.mean_vote >= 0 ? `+${row.mean_vote.toFixed(2)}` : row.mean_vote.toFixed(2);
        const profileLink = `<button class="btn btn-link coalition-profile" data-code="${row.country}" title="Open profile for ${row.name}">profile →</button>`;
        li.innerHTML =
          `<span class="coalition-row__name"><strong>${row.name}</strong> <span class="text-muted">(${row.country})</span></span>` +
          `<span class="coalition-row__lean">${lean}</span>` +
          `<span class="coalition-row__breakdown text-muted">${yes}Y / ${no}N / ${ab}A · n=${row.n_votes}</span>` +
          `<span class="coalition-row__action">${profileLink}</span>`;
        list.appendChild(li);
      });
      card.appendChild(list);
      if (members.length > 25) {
        const more = document.createElement("p");
        more.className = "text-muted coalition-tier__more";
        more.textContent = `+${members.length - 25} more`;
        card.appendChild(more);
      }
      tiersEl.appendChild(card);
    });

    tiersEl.querySelectorAll(".coalition-profile").forEach((btn) => {
      btn.addEventListener("click", () => {
        const code = btn.getAttribute("data-code");
        if (!code) return;
        state.profileCountry = code;
        const input = document.getElementById("profileCountry");
        if (input) input.value = nameFor(code);
        activateTab("profile");
        writeHashState();
      });
    });
  }

  if (samplesEl) {
    clearNode(samplesEl);
    const samples = report.sample_resolutions || [];
    if (!samples.length) {
      samplesEl.innerHTML = '<p class="text-muted">No sample resolutions to show.</p>';
    } else {
      const ul = document.createElement("ul");
      ul.className = "coalition-samples";
      samples.forEach((s) => {
        const li = document.createElement("li");
        li.innerHTML = `<span class="text-muted">${s.year ?? ""}</span> · ${s.title || "(untitled)"}`;
        ul.appendChild(li);
      });
      samplesEl.appendChild(ul);
    }
  }

  maybeUpdateMeta(report);
}

async function loadCoalition() {
  const headline = document.getElementById("coalitionHeadline");
  const topic = document.getElementById("coalitionTopic")?.value?.trim();
  if (!topic) {
    if (headline) showErrorElement(headline, "Enter a topic to search.");
    return;
  }
  setLoading(headline, `Computing coalition for "${topic}"…`);
  try {
    const params = new URLSearchParams({
      topic,
      start_year: String(state.startYear),
      end_year: String(state.endYear),
    });
    const response = await axios.get(`/api/coalition?${params.toString()}`);
    renderCoalition(response.data);
  } catch (error) {
    showErrorElement(headline, getErrorMessage(error));
  }
}

function setupNewsletterControls() {
  const btn = document.getElementById("newsletterComposeBtn");
  if (btn) btn.addEventListener("click", composeNewsletter);

  const wireCopy = (id, getText, label) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("click", async () => {
      const txt = getText();
      if (!txt) return;
      try {
        await navigator.clipboard.writeText(txt);
        el.textContent = "Copied ✓";
        setTimeout(() => (el.textContent = label), 1500);
      } catch (e) {
        console.error("Copy failed", e);
      }
    });
  };
  wireCopy("newsletterCopyMd", () => state.lastNewsletterMarkdown || "", "Copy Markdown");
  wireCopy("newsletterCopyText", () => state.lastNewsletterText || "", "Copy plain text");
  wireCopy("newsletterCopySubject", () => state.lastNewsletterSubject || "", "Copy subject line");

  const archiveBtn = document.getElementById("newsletterArchiveBtn");
  if (archiveBtn) {
    archiveBtn.addEventListener("click", async () => {
      archiveBtn.disabled = true;
      const original = archiveBtn.textContent;
      archiveBtn.textContent = "Archiving…";
      try {
        const params = state.lastArchiveParams;
        if (!params) {
          archiveBtn.textContent = "Compose first";
          setTimeout(() => (archiveBtn.textContent = original), 1500);
          return;
        }
        await axios.post(`/api/newsletter/archive?${params}`);
        archiveBtn.textContent = "Archived ✓";
        await loadArchiveList();
        setTimeout(() => (archiveBtn.textContent = original), 1800);
      } catch (e) {
        archiveBtn.textContent = "Failed";
        console.error("Archive failed", e);
        setTimeout(() => (archiveBtn.textContent = original), 1800);
      } finally {
        archiveBtn.disabled = false;
      }
    });
  }
}

async function loadArchiveList() {
  const target = document.getElementById("newsletterArchiveList");
  if (!target) return;
  try {
    const response = await axios.get("/api/newsletter/archive");
    const items = response.data?.editions || [];
    clearNode(target);
    if (items.length === 0) {
      target.innerHTML = '<p class="text-muted">No archived editions yet. Compose one above and click "Archive this edition".</p>';
      return;
    }
    const table = document.createElement("table");
    table.className = "data-table";
    table.innerHTML =
      "<thead><tr><th>Date</th><th>№</th><th>Country</th><th>Headline</th><th>Files</th></tr></thead>";
    const tbody = document.createElement("tbody");
    items.slice(0, 25).forEach((e) => {
      const tr = document.createElement("tr");
      const country = e.country_focus
        ? `<span class="percentile-chip">${e.country_focus}</span>`
        : '<span class="text-muted">global</span>';
      const links = ["md", "html", "txt", "json"]
        .filter((f) => e.formats?.[f])
        .map((f) => `<a class="btn btn-link" target="_blank" rel="noopener" href="/api/newsletter/archive/${e.year}/${e.slug}.${f}">.${f}</a>`)
        .join(" ");
      tr.innerHTML =
        `<td>${e.edition_date || ""}</td>` +
        `<td>${e.edition_number ?? ""}</td>` +
        `<td>${country}</td>` +
        `<td>${(e.headline || "").slice(0, 100)}</td>` +
        `<td>${links}</td>`;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    target.appendChild(table);
  } catch (e) {
    showErrorElement(target, getErrorMessage(e));
  }
}

async function composeNewsletter() {
  const yrRaw = (document.getElementById("newsletterYear")?.value || "").trim();
  const yr = yrRaw ? toFiniteNumber(yrRaw, state.endYear) : null;
  const bw = toFiniteNumber(document.getElementById("newsletterBaseline")?.value, 3);
  const topics = (document.getElementById("newsletterTopics")?.value || "").trim();
  const countryRaw = (document.getElementById("newsletterCountry")?.value || "").trim();
  const country = resolveCountryCode(countryRaw);
  if (countryRaw && !country) {
    showFieldError("newsletterCountry", "Pick a country from the list, or type its ISO-3 code");
    return;
  }
  const params = new URLSearchParams({ baseline_window: String(bw) });
  if (yr) params.set("recent_year", String(yr));
  if (topics) params.set("topics", topics);
  if (country && country.length === 3) params.set("country", country);
  state.lastArchiveParams = params.toString();

  const frame = document.getElementById("newsletterFrame");
  if (frame) frame.srcdoc = '<div style="padding:24px;font-family:sans-serif;color:#456783">Composing edition…</div>';

  // Set up the format-specific URLs.
  const baseUrl = `/api/newsletter/weekly?${params.toString()}`;
  const htmlUrl = baseUrl + "&format=html";
  const mdUrl = baseUrl + "&format=markdown";
  const txtUrl = baseUrl + "&format=text";
  const dlHtml = document.getElementById("newsletterDownloadHtml");
  const dlMd = document.getElementById("newsletterDownloadMd");
  const dlTxt = document.getElementById("newsletterDownloadText");
  const openHtml = document.getElementById("newsletterOpenHtml");
  const editionYear = yr || "latest";
  if (dlHtml) { dlHtml.href = htmlUrl; dlHtml.download = `weekly-atlas-${editionYear}.html`; }
  if (dlMd) { dlMd.href = mdUrl; dlMd.download = `weekly-atlas-${editionYear}.md`; }
  if (dlTxt) { dlTxt.href = txtUrl; dlTxt.download = `weekly-atlas-${editionYear}.txt`; }
  if (openHtml) openHtml.href = htmlUrl;

  try {
    // JSON fetch gives us both the structured data and the markdown for copy.
    const response = await axios.get(baseUrl + "&format=json");
    state.lastNewsletterMarkdown = response.data?.markdown || "";
    // Show edition number / dateline + email subject in the masthead-info blocks.
    const masthead = document.getElementById("newsletterMastheadInfo");
    const subjectInfo = document.getElementById("newsletterSubjectInfo");
    if (masthead && response.data) {
      const d = response.data;
      masthead.textContent =
        `Edition №${d.edition_number} · ${d.dateline} · ${d.byline} · ${d.period_label}`;
      state.lastNewsletterSubject = d.email_subject || "";
      // The server may have auto-picked the year; name the downloads after it.
      if (d.recent_year) {
        if (dlHtml) dlHtml.download = `weekly-atlas-${d.recent_year}.html`;
        if (dlMd) dlMd.download = `weekly-atlas-${d.recent_year}.md`;
        if (dlTxt) dlTxt.download = `weekly-atlas-${d.recent_year}.txt`;
      }
      if (subjectInfo) {
        subjectInfo.innerHTML = `Subject: <strong>${(d.email_subject || "").replace(/</g,"&lt;")}</strong> <span class="text-muted">(slug: ${d.edition_slug})</span>`;
      }
    }
    // Fetch text for copy-as-plain-text.
    try {
      const txtResp = await axios.get(txtUrl, { responseType: "text" });
      state.lastNewsletterText = txtResp.data;
    } catch (_) { /* non-fatal */ }
    // Now fetch the HTML and put it in the iframe via srcdoc.
    const htmlResp = await axios.get(htmlUrl, { responseType: "text" });
    if (frame) frame.srcdoc = htmlResp.data;
    maybeUpdateMeta(response.data);
  } catch (error) {
    if (frame) {
      frame.srcdoc =
        '<div style="padding:24px;font-family:sans-serif;color:#c64141">' +
        `Failed to compose: ${getErrorMessage(error).replace(/</g, "&lt;")}</div>`;
    }
  }
}

function setupCoalitionControls() {
  const button = document.getElementById("coalitionRunBtn");
  const input = document.getElementById("coalitionTopic");
  if (button) button.addEventListener("click", loadCoalition);
  if (input) {
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        loadCoalition();
      }
    });
  }
  document.querySelectorAll("[data-coalition-topic]").forEach((chip) => {
    chip.addEventListener("click", () => {
      if (input) input.value = chip.getAttribute("data-coalition-topic") || "";
      loadCoalition();
    });
  });
}

function renderDriftFeed(targetEl, drifts, options = {}) {
  if (!targetEl) return;
  clearNode(targetEl);
  if (!drifts || drifts.length === 0) {
    targetEl.innerHTML = '<p class="text-muted">No drifts available — try a different year or a wider baseline.</p>';
    return;
  }
  const focusCountry = options.focusCountry || null;

  drifts.forEach((drift) => {
    const card = document.createElement("article");
    const deltaPts = drift.delta * 100;
    const direction = deltaPts >= 0 ? "up" : "down";
    card.className = `drift-card drift-card--${direction}`;

    const arrow = deltaPts >= 0 ? "▲" : "▼";
    const sign = deltaPts >= 0 ? "+" : "";
    const headerLabel =
      focusCountry && drift.country_b === focusCountry
        ? `${nameFor(drift.country_b)} ↔ ${nameFor(drift.country_a)}`
        : `${nameFor(drift.country_a)} ↔ ${nameFor(drift.country_b)}`;

    const topicNames = (drift.driving_topics || [])
      .slice(0, 3)
      .map((t) => `<span class="drift-topic">${t.topic}</span>`)
      .join(" ");
    const topicLine = topicNames
      ? `<div class="drift-driver"><span class="drift-driver__label">Driven by:</span> ${topicNames}</div>`
      : `<div class="drift-driver text-muted">No dominant topic — split votes scattered across issues.</div>`;

    card.innerHTML = `
      <header class="drift-card__header">
        <span class="drift-card__arrow">${arrow}</span>
        <h4 class="drift-card__pair">${headerLabel}</h4>
        <span class="drift-card__delta">${sign}${deltaPts.toFixed(0)} pts</span>
      </header>
      <div class="drift-card__numbers">
        <span class="drift-num drift-num--baseline">${(drift.baseline_agreement * 100).toFixed(0)}%</span>
        <span class="drift-num__arrow">→</span>
        <span class="drift-num drift-num--recent">${(drift.recent_agreement * 100).toFixed(0)}%</span>
        <span class="drift-num__context text-muted">(${drift.n_baseline_votes} baseline / ${drift.n_recent_votes} recent votes)</span>
      </div>
      ${topicLine}
      <footer class="drift-card__footer">
        <button class="btn btn-link drift-compare" data-a="${drift.country_a}" data-b="${drift.country_b}">Inspect splits →</button>
      </footer>
    `;
    targetEl.appendChild(card);
  });

  targetEl.querySelectorAll(".drift-compare").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.preventDefault();
      const a = btn.getAttribute("data-a");
      const b = btn.getAttribute("data-b");
      const aInput = document.getElementById("countryA");
      const bInput = document.getElementById("countryB");
      if (aInput) aInput.value = nameFor(a);
      if (bInput) bInput.value = nameFor(b);
      activateTab("compare");
      const compareBtn = document.getElementById("compareBtn");
      if (compareBtn) compareBtn.click();
    });
  });
}

async function loadKnownEvents() {
  try {
    const response = await axios.get("/api/events");
    state.knownEvents = response.data?.events || [];
  } catch (error) {
    console.warn("Known events failed to load", error);
    state.knownEvents = [];
  }
}

async function loadDriftDigest() {
  const target = document.getElementById("driftDigest");
  if (!target) return;
  setLoading(target, "Composing digest…");
  const recentYearRaw = (document.getElementById("driftRecentYear")?.value || "").trim();
  const recentYear = recentYearRaw ? toFiniteNumber(recentYearRaw, state.endYear) : null;
  const baselineWindow = toFiniteNumber(
    document.getElementById("driftBaselineWindow")?.value,
    5,
  );
  try {
    const params = new URLSearchParams({
      baseline_window: String(baselineWindow),
      top: "6",
    });
    if (recentYear) params.set("recent_year", String(recentYear));
    const response = await axios.get(`/api/drift/digest?${params.toString()}`);
    clearNode(target);
    const digestEl = document.createElement("div");
    digestEl.className = "drift-digest";
    const text = String(response.data?.digest || "").trim();
    text.split(/\n\n+/).forEach((para) => {
      const p = document.createElement("p");
      p.textContent = para;
      digestEl.appendChild(p);
    });
    target.appendChild(digestEl);
  } catch (error) {
    showErrorElement(target, getErrorMessage(error));
  }
}

async function loadDriftFeed() {
  const target = document.getElementById("driftFeed");
  const windowEl = document.getElementById("driftWindow");
  if (!target) return;
  setLoading(target, "Computing alignment drifts…");

  const recentYearRaw = (document.getElementById("driftRecentYear")?.value || "").trim();
  const recentYear = recentYearRaw ? toFiniteNumber(recentYearRaw, state.endYear) : null;
  const baselineWindow = toFiniteNumber(
    document.getElementById("driftBaselineWindow")?.value,
    5,
  );
  const direction = document.getElementById("driftDirection")?.value || "all";

  try {
    const params = new URLSearchParams({
      baseline_window: String(baselineWindow),
      direction,
      top: "12",
    });
    if (recentYear) params.set("recent_year", String(recentYear));
    const response = await axios.get(`/api/drift?${params.toString()}`);
    const payload = response.data;
    if (windowEl) {
      const baseline = payload.baseline_window || {};
      windowEl.textContent =
        `Comparing ${payload.recent_year} against baseline ${baseline.start}–${baseline.end} ` +
        `(${payload.drifts?.length || 0} drifts shown).`;
    }
    renderDriftFeed(target, payload.drifts || []);
    maybeUpdateMeta(payload);
    loadDriftDigest();
  } catch (error) {
    showErrorElement(target, getErrorMessage(error));
  }
}

function profileCountryLabel(profile) {
  if (!profile) return "";
  return profile.country_name
    ? `${profile.country_name} (${profile.country})`
    : profile.country;
}

function neighbourLabel(row) {
  return row?.name ? `${row.name} (${row.country})` : row?.country || "";
}

function openCompareWithPeer(peerCode) {
  if (!peerCode) return;
  const a = document.getElementById("countryA");
  const b = document.getElementById("countryB");
  if (a) a.value = nameFor(state.profileCountry);
  if (b) b.value = nameFor(peerCode);
  activateTab("compare");
  const btn = document.getElementById("compareBtn");
  if (btn) btn.click();
}

function renderProfile(profile) {
  if (!profile) return;
  state.lastProfile = profile;

  const headline = document.getElementById("profileHeadline");
  if (headline) {
    const totals = profile.totals || {};
    const win = profile.window || {};
    const label = profileCountryLabel(profile);
    const topAlly = profile.top_allies?.[0];
    const topOpp = profile.top_opponents?.[0];
    headline.innerHTML = "";
    const story = document.createElement("p");
    story.className = "profile-story";
    const allyText = topAlly
      ? `closest to <strong>${neighbourLabel(topAlly)}</strong> (${(topAlly.similarity * 100).toFixed(0)}% aligned)`
      : "no close allies in this window";
    const oppText = topOpp
      ? `most opposed to <strong>${neighbourLabel(topOpp)}</strong> (${(topOpp.similarity * 100).toFixed(0)}%)`
      : "";
    story.innerHTML =
      `Between <strong>${win.start_year}</strong> and <strong>${win.end_year}</strong>, ` +
      `<strong>${label}</strong> cast <strong>${totals.votes_cast ?? 0}</strong> votes ` +
      `(${totals.yes ?? 0} Yes / ${totals.no ?? 0} No / ${totals.abstain ?? 0} Abstain). ` +
      `${allyText}${oppText ? "; " + oppText : ""}.`;
    headline.appendChild(story);
  }

  // Bloc-alignment strip — one comparable headline number per reference bloc.
  const blocStrip = document.getElementById("profileBlocStrip");
  if (blocStrip) {
    clearNode(blocStrip);
    const blocs = profile.bloc_alignment || {};
    const order = ["Western", "Non-aligned", "Eastern"];
    order.forEach((blocName) => {
      const data = blocs[blocName];
      if (!data) return;
      const tile = document.createElement("div");
      tile.className = `bloc-tile bloc-tile--${blocName.toLowerCase().replace(/\W+/g, "-")}`;
      const pct =
        data.alignment == null ? "—" : `${(data.alignment * 100).toFixed(0)}%`;
      tile.innerHTML =
        `<div class="bloc-tile__label">${blocName}</div>` +
        `<div class="bloc-tile__value">${pct}</div>` +
        `<div class="bloc-tile__meta text-muted">${data.n_pairs ?? 0} peers</div>`;
      blocStrip.appendChild(tile);
    });
  }

  const renderList = (elementId, rows) => {
    const target = document.getElementById(elementId);
    if (!target) return;
    clearNode(target);
    if (!rows || rows.length === 0) {
      target.innerHTML = '<p class="text-muted">No data for this window.</p>';
      return;
    }
    const table = document.createElement("table");
    table.className = "data-table profile-neighbours";
    const thead = document.createElement("thead");
    thead.innerHTML = `<tr><th>#</th><th>Country</th><th>Alignment</th><th></th></tr>`;
    table.appendChild(thead);
    const tbody = document.createElement("tbody");
    rows.forEach((row, idx) => {
      const tr = document.createElement("tr");
      tr.className = "neighbour-row";
      const pct = (row.similarity * 100).toFixed(1) + "%";
      const label = row.name ? `${row.name} <span class="text-muted">(${row.country})</span>` : row.country;
      const percentileChip = Number.isFinite(row.percentile)
        ? `<span class="percentile-chip" title="${row.n_overlap || 0} shared votes">p${Math.round(row.percentile)}</span>`
        : "";
      const compareBtn = `<button class="btn btn-link compare-link" data-peer="${row.country}" title="Compare ${state.profileCountry} vs ${row.country}">Compare →</button>`;
      tr.innerHTML = `<td>${idx + 1}</td><td><strong>${label}</strong> ${percentileChip}</td><td>${pct}</td><td>${compareBtn}</td>`;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    target.appendChild(table);

    target.querySelectorAll(".compare-link").forEach((btn) => {
      btn.addEventListener("click", (event) => {
        event.preventDefault();
        openCompareWithPeer(btn.getAttribute("data-peer"));
      });
    });
  };
  renderList("profileAllies", profile.top_allies);
  renderList("profileOpponents", profile.top_opponents);

  // P5 alignment plot.
  const plotEl = document.getElementById("profileP5");
  if (plotEl && typeof Plotly !== "undefined") {
    const traces = [];
    const palette = {
      USA: "#3b82f6",
      GBR: "#10b981",
      FRA: "#8b5cf6",
      RUS: "#ef4444",
      CHN: "#f59e0b",
    };
    const seriesMap = profile.p5_alignment || {};
    Object.entries(seriesMap).forEach(([code, points]) => {
      if (!Array.isArray(points) || points.length === 0) return;
      if (code === profile.country) return; // skip self-line, always 1.0
      traces.push({
        x: points.map((p) => p.year),
        y: points.map((p) => p.agreement),
        mode: "lines+markers",
        name: code,
        line: { color: palette[code] || "#6b7280", width: 2 },
        hovertemplate: `%{x}: %{y:.0%} agreement<br>n=%{customdata}`,
        customdata: points.map((p) => p.n_votes),
      });
    });
    const shapes = [];
    const annotations = [];
    const years = Object.values(profile.p5_alignment || {})
      .flat()
      .map((p) => p.year);
    if (years.length && Array.isArray(state.knownEvents)) {
      const yMin = Math.min(...years);
      const yMax = Math.max(...years);
      state.knownEvents
        .filter((e) => e.year >= yMin && e.year <= yMax)
        .forEach((ev) => {
          shapes.push({
            type: "line",
            x0: ev.year,
            x1: ev.year,
            yref: "paper",
            y0: 0,
            y1: 1,
            line: { color: "rgba(248, 113, 113, 0.55)", width: 1, dash: "dot" },
          });
          annotations.push({
            x: ev.year,
            y: 1.02,
            yref: "paper",
            text: ev.label,
            showarrow: false,
            font: { size: 9, color: "rgba(252, 165, 165, 0.85)" },
            textangle: -45,
            xanchor: "left",
          });
        });
    }
    const layout = {
      margin: { t: 50, r: 10, b: 40, l: 50 },
      yaxis: { title: "Agreement", range: [0, 1], tickformat: ",.0%" },
      xaxis: { title: "Year" },
      legend: { orientation: "h", y: -0.2 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      shapes,
      annotations,
    };
    Plotly.react(plotEl, traces, layout, { displayModeBar: false, responsive: true });
  }

  // Per-country drift strip — what shifted most for this country this year?
  const driftStrip = document.getElementById("profileDrift");
  if (driftStrip) {
    renderDriftFeed(driftStrip, profile.drift_alerts || [], {
      focusCountry: profile.country,
    });
  }

  // Splits table.
  const splitsEl = document.getElementById("profileSplits");
  if (splitsEl) {
    clearNode(splitsEl);
    const splits = profile.biggest_divergences || [];
    if (splits.length === 0) {
      splitsEl.innerHTML = '<p class="text-muted">No opposite-direction votes with the top ally in this window.</p>';
    } else {
      const table = document.createElement("table");
      table.className = "data-table";
      const peer = splits[0].peer;
      table.innerHTML =
        `<thead><tr><th>Date</th><th>Resolution</th><th>Issue</th>` +
        `<th>${profile.country}</th><th>${peer}</th></tr></thead>`;
      const tbody = document.createElement("tbody");
      splits.forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML =
          `<td>${row.date ? String(row.date).slice(0, 10) : "—"}</td>` +
          `<td>${row.resolution || "—"}</td>` +
          `<td>${row.issue || "—"}</td>` +
          `<td><strong>${row.vote_self}</strong></td>` +
          `<td>${row.vote_peer}</td>`;
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      splitsEl.appendChild(table);
    }
  }

  maybeUpdateMeta(profile);
}

async function loadCountryProfile() {
  const code = normalizeCodeInput(state.profileCountry);
  if (code.length !== 3) return;
  loadCountryStory(code);
  const headline = document.getElementById("profileHeadline");
  setLoading(headline, `Loading profile for ${code}…`);
  try {
    const params = new URLSearchParams({
      start_year: String(state.startYear),
      end_year: String(state.endYear),
    });
    const response = await axios.get(`/api/country/${code}/profile?${params.toString()}`);
    renderProfile(response.data);
  } catch (error) {
    showErrorElement(headline, getErrorMessage(error));
  }
}

// Load only what the open tab needs. The dashboard's four analytics used to
// run at every startup whatever tab was showing; now they wait to be looked at.
async function runAnalysis() {
  try {
    if (state.activeTab === "story") {
      await loadBigPicture();
    }
    if (state.activeTab === "lenses") {
      await loadLenses();
    }
    if (state.activeTab === "profile") {
      await loadCountryProfile();
    }
    if (state.activeTab === "dashboard") {
      await loadDashboard(true);
    }
    // Deep links (#tab=drift, #tab=map …) land here with skipLoad set, so
    // every tab that fetches on activation must also fetch on first run.
    if (state.activeTab === "drift") {
      loadDriftFeed();
    }
    if (state.activeTab === "map") {
      loadAlignmentMap();
    }
    if (state.activeTab === "pivotality") {
      loadPivotality();
    }
    if (state.activeTab === "abstention") {
      loadAbstention();
    }
    if (state.activeTab === "coalition" && !state.lastCoalitionTopic) {
      loadCoalition();
    }
    if (state.activeTab === "newsletter") {
      if (!state.lastNewsletterMarkdown) composeNewsletter();
      loadArchiveList();
    }
    if (state.activeTab === "network") {
      await loadNetworkGraph();
    }
    if (state.activeTab === "softpower") {
      await loadSoftPower();
    }
    if (state.activeTab === "bloc") {
      await loadBlocTimeline();
    }
  } catch (error) {
    console.error("Analysis run failed", error);
  }
}

// Clustering, PCA, issue timeline and the insight cards — once per window,
// sequentially to avoid the backend's analysis-slot limit (429 busy).
async function loadDashboard(force = false) {
  const key = `${state.startYear}-${state.endYear}-${state.numClusters}-${state.projectionMethod}`;
  if (!force && state.dashboardLoadedFor === key) return;
  state.dashboardLoadedFor = key;
  await loadClustering();
  await loadPCAPlot();
  await loadIssueTimeline();
  await loadInsights();
}

// ── Country-name lookup (decode ISO-3 codes on axes / hovers / keys) ────────

async function loadCountryNames() {
  try {
    const res = await axios.get("/api/countries");
    state.countryNames = res.data?.countries || {};
  } catch (error) {
    state.countryNames = {};
  }
}

function nameFor(code) {
  return (state.countryNames && state.countryNames[code]) || code;
}

// Country inputs accept either an ISO-3 code ("USA") or a display name
// ("United States", case-insensitive; a unique prefix or substring is enough).
// Returns the ISO-3 code, or "" when nothing matches. Falls back to plain
// code handling if the name list never loaded.
function resolveCountryCode(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";
  const names = state.countryNames || {};
  const entries = Object.entries(names);
  const upper = raw.toUpperCase();
  if (/^[A-Z]{3}$/.test(upper) && (!entries.length || names[upper])) return upper;
  const needle = raw.toLowerCase();
  const exact = entries.find(([, n]) => n.toLowerCase() === needle);
  if (exact) return exact[0];
  const prefix = entries.filter(([, n]) => n.toLowerCase().startsWith(needle));
  if (prefix.length === 1) return prefix[0][0];
  const within = entries.filter(([, n]) => n.toLowerCase().includes(needle));
  return within.length === 1 ? within[0][0] : "";
}

// Fill the shared <datalist> every country input points at: value = display
// name (what the browser inserts), label = ISO-3 code.
function populateCountryDatalist() {
  const list = document.getElementById("countryOptions");
  if (!list) return;
  clearNode(list);
  Object.entries(state.countryNames || {})
    .sort((a, b) => a[1].localeCompare(b[1]))
    .forEach(([code, name]) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.label = code;
      list.appendChild(opt);
    });
}

// Inputs start life holding bare codes (defaults / URL hash); show names once
// the code → name map has loaded.
function syncCountryInputs() {
  const profile = document.getElementById("profileCountry");
  if (profile) profile.value = nameFor(state.profileCountry);
  const map = document.getElementById("mapCountry");
  if (map) map.value = nameFor(resolveCountryCode(map.value) || "USA");
}

// Break a long category label at word boundaries into at most `maxLines`
// lines (Plotly renders <br> in tick labels) instead of truncating it.
function wrapLabel(text, width = 32, maxLines = 2) {
  const lines = [];
  let line = "";
  String(text || "").split(/\s+/).forEach((word) => {
    if (line && `${line} ${word}`.length > width) {
      lines.push(line);
      line = word;
    } else {
      line = line ? `${line} ${word}` : word;
    }
  });
  if (line) lines.push(line);
  if (lines.length > maxLines) {
    const kept = lines.slice(0, maxLines);
    kept[maxLines - 1] = `${kept[maxLines - 1]}…`;
    return kept.join("<br>");
  }
  return lines.join("<br>");
}

// ── Alignment map · Pivotality · Abstention (new analytical views) ──────────

async function loadAlignmentMap() {
  const host = document.getElementById("alignmentMap");
  if (!host) return;
  const input = document.getElementById("mapCountry");
  const typed = (input?.value || "").trim();
  const code = typed ? resolveCountryCode(typed) : "USA";
  if (!code) {
    showFieldError("mapCountry", "Pick a country from the list, or type its ISO-3 code");
    return;
  }
  if (input) input.value = nameFor(code);
  setLoading(host, `Mapping voting alignment with ${nameFor(code)}…`);
  try {
    requirePlotly();
    const params = new URLSearchParams({
      start_year: String(state.startYear),
      end_year: String(state.endYear),
    });
    const res = await axios.get(`/api/country/${code}/alignment-map?${params.toString()}`);
    const points = res.data?.points || [];
    if (!points.length) {
      showErrorElement(host, "No alignment data for this country / window.");
      return;
    }
    maybeUpdateMeta(res.data);
    const trace = {
      type: "choropleth",
      locationmode: "ISO-3",
      locations: points.map((p) => p.code),
      z: points.map((p) => p.alignment),
      text: points.map((p) => p.name),
      zmin: -1,
      zmax: 1,
      zmid: 0,
      colorscale: PALETTE.diverging,
      colorbar: { title: "Alignment", tickformat: ".1f", thickness: 14 },
      hovertemplate: "%{text}<br>alignment %{z:.2f}<extra></extra>",
      marker: { line: { color: "#ffffff", width: 0.4 } },
    };
    // The selected country has no alignment with itself, so it would render as
    // no-data white; paint it ink-dark on top so the anchor is unmistakable.
    const selected = {
      type: "choropleth",
      locationmode: "ISO-3",
      locations: [code],
      z: [1],
      zmin: 0,
      zmax: 1,
      text: [`${res.data.country_name || nameFor(code)} (selected)`],
      colorscale: [
        [0, "#0b2238"],
        [1, "#0b2238"],
      ],
      showscale: false,
      hovertemplate: "%{text}<extra></extra>",
      marker: { line: { color: "#ffffff", width: 1.2 } },
    };
    clearNode(host);
    Plotly.newPlot(
      host,
      [trace, selected],
      {
        autosize: true,
        title: {
          text: `Voting alignment with ${res.data.country_name || code} · ${res.data.start_year}–${res.data.end_year}`,
          font: { size: 15 },
        },
        geo: {
          showframe: false,
          showcoastlines: true,
          coastlinecolor: "#cfd8df",
          projection: { type: "natural earth" },
          bgcolor: "rgba(0,0,0,0)",
        },
        margin: { l: 6, r: 6, t: 46, b: 6 },
        paper_bgcolor: "rgba(0,0,0,0)",
      },
      // topojsonURL points Plotly at the self-hosted world geometry
      // (static/vendor/world_110m.json) instead of cdn.plot.ly — no external
      // runtime dependency, so the basemap can't silently go blank.
      { responsive: true, displayModeBar: false, topojsonURL: "/static/vendor/" },
    );
  } catch (error) {
    showErrorElement(host, getErrorMessage(error));
  }
}

async function loadPivotality() {
  const host = document.getElementById("pivotalityChart");
  const summary = document.getElementById("pivotalitySummary");
  if (!host) return;
  setLoading(host, "Finding swing votes on the closest resolutions…");
  if (summary) clearNode(summary);
  try {
    requirePlotly();
    if (!Object.keys(state.countryNames).length) await loadCountryNames();
    const res = await axios.post("/api/analysis/pivotality", {
      start_year: state.startYear,
      end_year: state.endYear,
      // UNGA votes are rarely knife-edge: at a 15-25% margin only a handful
      // qualify and every country ties. 0.5 = "winning side under 75%", i.e.
      // genuinely DIVIDED votes — ~135 of them in 2010-2020 — which gives a
      // real ranking of who lands on the prevailing side.
      margin_threshold: 0.5,
    });
    maybeUpdateMeta(res.data);
    const scores = res.data?.pivotality_scores || {};
    const contested = Math.max(1, toFiniteNumber(res.data?.contested_count, 0));
    const rows = Object.entries(scores)
      .map(([code, v]) => {
        const wins = toFiniteNumber(v.pivotality_index ?? v.swing_votes, 0);
        return { code, wins, share: (100 * wins) / contested };
      })
      .sort((a, b) => b.wins - a.wins)
      .slice(0, 20)
      .reverse();
    if (summary) {
      summary.textContent =
        `${res.data.contested_count} divided resolutions (winning side under 75%) ` +
        `out of ${res.data.total_resolutions} in ${state.startYear}–${state.endYear}. ` +
        `Each dot is the share of those a country landed on the prevailing side of. ` +
        `The axis starts near the lowest of the top 20, so read positions, not lengths.`;
    }
    if (!rows.length) {
      showErrorElement(host, "No contested resolutions in this window.");
      return;
    }
    // Dot plot, not bars: the top 20 sit within a few points of each other, so
    // bars from zero all look identical. A narrowed axis is honest for dots
    // (position encodes the value) where it would mislead for bars (length).
    const lowest = Math.min(...rows.map((r) => r.share));
    const axisStart = Math.max(0, Math.floor(lowest / 5) * 5 - 5);
    clearNode(host);
    Plotly.newPlot(
      host,
      [
        {
          type: "scatter",
          mode: "markers+text",
          x: rows.map((r) => r.share),
          y: rows.map((r) => nameFor(r.code)),
          text: rows.map((r) => `${r.share.toFixed(0)}%`),
          textposition: "middle right",
          textfont: { size: 11, color: "#456783" },
          cliponaxis: false,
          customdata: rows.map((r) => [r.code, r.wins, contested]),
          marker: { color: PALETTE.yes, size: 11 },
          hovertemplate:
            "%{y} (%{customdata[0]})<br>%{customdata[1]} of %{customdata[2]} divided votes (%{x:.0f}%)<extra></extra>",
        },
      ],
      {
        autosize: true,
        xaxis: {
          title: "Share of divided votes on the prevailing side (%)",
          range: [axisStart, 101],
          ticksuffix: "%",
          automargin: true,
          gridcolor: "#ece8e0",
          zeroline: false,
        },
        yaxis: { automargin: true, gridcolor: "#f3f1ec" },
        margin: { l: 10, r: 48, t: 16, b: 56 },
        showlegend: false,
      },
      { responsive: true, displayModeBar: false },
    );
  } catch (error) {
    showErrorElement(host, getErrorMessage(error));
  }
}

async function loadAbstention() {
  const countriesHost = document.getElementById("abstentionCountries");
  const topicsHost = document.getElementById("abstentionTopics");
  if (!countriesHost) return;
  setLoading(countriesHost, "Measuring fence-sitting…");
  if (topicsHost) clearNode(topicsHost);
  try {
    requirePlotly();
    if (!Object.keys(state.countryNames).length) await loadCountryNames();
    const res = await axios.post("/api/analysis/abstention", {
      start_year: state.startYear,
      end_year: state.endYear,
    });
    maybeUpdateMeta(res.data);

    const rates = res.data?.global_rates || {};
    const countryRows = Object.entries(rates)
      .map(([code, v]) => ({ code, rate: toFiniteNumber(v.abstention_rate, 0) }))
      .sort((a, b) => b.rate - a.rate)
      .slice(0, 20)
      .reverse();
    clearNode(countriesHost);
    if (countryRows.length) {
      Plotly.newPlot(
        countriesHost,
        [
          {
            type: "bar",
            orientation: "h",
            x: countryRows.map((r) => r.rate * 100),
            y: countryRows.map((r) => nameFor(r.code)),
            customdata: countryRows.map((r) => r.code),
            marker: { color: PALETTE.abstain },
            hovertemplate: "%{y} (%{customdata})<br>%{x:.0f}% abstained<extra></extra>",
          },
        ],
        {
          autosize: true,
          xaxis: { title: "Abstention rate (%)", automargin: true },
          yaxis: { automargin: true },
          margin: { l: 10, r: 20, t: 16, b: 56 },
        },
        { responsive: true, displayModeBar: false },
      );
    } else {
      showErrorElement(countriesHost, "No abstention data in this window.");
    }

    const topics = res.data?.top_strategic_topics || {};
    const topicRows = Object.entries(topics)
      .map(([topic, v]) => ({ topic, rate: toFiniteNumber(v.abstention_rate, 0) }))
      .sort((a, b) => b.rate - a.rate)
      .slice(0, 12)
      .reverse();
    if (topicsHost && topicRows.length) {
      clearNode(topicsHost);
      Plotly.newPlot(
        topicsHost,
        [
          {
            type: "bar",
            orientation: "h",
            x: topicRows.map((r) => r.rate * 100),
            y: topicRows.map((r) => wrapLabel(r.topic, 30)),
            text: topicRows.map((r) => r.topic),
            textposition: "none",
            marker: { color: "#d38b2a" },
            hovertemplate: "%{text}<br>%{x:.0f}% abstained<extra></extra>",
          },
        ],
        {
          autosize: true,
          xaxis: { title: "Abstention rate (%)", automargin: true },
          yaxis: { automargin: true },
          margin: { l: 10, r: 20, t: 16, b: 56 },
        },
        { responsive: true, displayModeBar: false },
      );
    }
  } catch (error) {
    showErrorElement(countriesHost, getErrorMessage(error));
  }
}

function activateTab(tabId, options = {}) {
  const skipLoad = Boolean(options.skipLoad);
  const tabs = Array.from(document.querySelectorAll(".tab-link"));
  const panes = Array.from(document.querySelectorAll(".tab-pane"));

  tabs.forEach((tab) => {
    const active = tab.dataset.tab === tabId;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  });

  panes.forEach((pane) => {
    const active = pane.id === tabId;
    pane.classList.toggle("active", active);
    pane.hidden = !active;
  });

  Object.assign(state, reduceState(state, { type: "SET_TAB", activeTab: tabId }));
  writeHashState();

  if (!skipLoad && tabId === "story") {
    loadBigPicture();
  }
  if (!skipLoad && tabId === "lenses") {
    loadLenses();
  }
  if (!skipLoad && tabId === "dashboard") {
    loadDashboard();
  }
  if (!skipLoad && tabId === "profile") {
    loadCountryProfile();
  }
  if (!skipLoad && tabId === "coalition") {
    if (!state.lastCoalitionTopic) loadCoalition();
  }
  if (!skipLoad && tabId === "newsletter") {
    if (!state.lastNewsletterMarkdown) composeNewsletter();
    loadArchiveList();
  }
  if (!skipLoad && tabId === "drift") {
    loadDriftFeed();
  }
  if (!skipLoad && tabId === "network") {
    loadNetworkGraph();
  }
  if (!skipLoad && tabId === "softpower") {
    loadSoftPower();
  }
  if (!skipLoad && tabId === "bloc") {
    loadBlocTimeline();
  }
  if (!skipLoad && tabId === "map") {
    loadAlignmentMap();
  }
  if (!skipLoad && tabId === "pivotality") {
    loadPivotality();
  }
  if (!skipLoad && tabId === "abstention") {
    loadAbstention();
  }
}

function setupTabs() {
  const tabs = Array.from(document.querySelectorAll(".tab-link"));
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        const dir = event.key === "ArrowRight" ? 1 : -1;
        const next = (index + dir + tabs.length) % tabs.length;
        tabs[next].focus();
        activateTab(tabs[next].dataset.tab);
        return;
      }
      if (event.key === "Home") {
        event.preventDefault();
        tabs[0].focus();
        activateTab(tabs[0].dataset.tab);
        return;
      }
      if (event.key === "End") {
        event.preventDefault();
        const last = tabs[tabs.length - 1];
        last.focus();
        activateTab(last.dataset.tab);
        return;
      }
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activateTab(tab.dataset.tab);
      }
    });
  });
}

function setupMobileControls() {
  const toggleBtn = document.getElementById("controlToggleBtn");
  const panel = document.getElementById("controlPanel");
  if (!toggleBtn || !panel) return;

  const applyResponsiveState = () => {
    const mobile = window.matchMedia("(max-width: 740px)").matches;
    if (!mobile) {
      panel.classList.remove("collapsed");
      toggleBtn.setAttribute("aria-expanded", "true");
      return;
    }
    if (!panel.classList.contains("collapsed")) {
      panel.classList.add("collapsed");
      toggleBtn.setAttribute("aria-expanded", "false");
    }
  };

  toggleBtn.addEventListener("click", () => {
    panel.classList.toggle("collapsed");
    const expanded = !panel.classList.contains("collapsed");
    toggleBtn.setAttribute("aria-expanded", String(expanded));
  });

  window.addEventListener("resize", debounce(applyResponsiveState, 120));
  applyResponsiveState();
}

function setupDriftControls() {
  const button = document.getElementById("driftLoadBtn");
  if (button) button.addEventListener("click", loadDriftFeed);
  ["driftRecentYear", "driftBaselineWindow", "driftDirection"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("change", () => {
      if (state.activeTab === "drift") loadDriftFeed();
    });
  });
}

function setupProfileControls() {
  const input = document.getElementById("profileCountry");
  const loadBtn = document.getElementById("profileLoadBtn");
  if (input) input.value = nameFor(state.profileCountry);

  const submit = () => {
    const code = input ? resolveCountryCode(input.value) : state.profileCountry;
    if (!code) {
      showFieldError("profileCountry", "Pick a country from the list, or type its ISO-3 code");
      return;
    }
    state.profileCountry = code;
    if (input) input.value = nameFor(code);
    writeHashState();
    loadCountryProfile();
  };

  if (loadBtn) loadBtn.addEventListener("click", submit);
  const shareBtn = document.getElementById("profileShareBtn");
  if (shareBtn) {
    shareBtn.addEventListener("click", async () => {
      const code = normalizeCodeInput(state.profileCountry);
      const url = `${window.location.origin}${window.location.pathname}#tab=profile&country=${code}`;
      try {
        await navigator.clipboard.writeText(url);
        shareBtn.textContent = "Link copied";
        setTimeout(() => {
          shareBtn.textContent = "Share this country";
        }, 1500);
      } catch (error) {
        // Clipboard access can be refused; show the link inline instead of a modal.
        showFieldError("profileCountry", `Copy this link: ${url}`);
      }
    });
  }
  if (input) {
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        submit();
      }
    });
  }
  document.querySelectorAll("[data-profile-suggest]").forEach((chip) => {
    chip.addEventListener("click", () => {
      const code = normalizeCodeInput(chip.getAttribute("data-profile-suggest"));
      state.profileCountry = code;
      if (input) input.value = nameFor(code);
      writeHashState();
      loadCountryProfile();
    });
  });
}

function setupEventListeners() {
  const debouncedRun = debounce(runAnalysis, 800);

  document.getElementById("startYear").addEventListener("change", (event) => {
    Object.assign(
      state,
      reduceState(state, {
        type: "SET_RANGE",
        startYear: toFiniteNumber(event.target.value, state.startYear),
      }),
    );
    writeHashState();
    debouncedRun();
  });

  document.getElementById("endYear").addEventListener("change", (event) => {
    Object.assign(
      state,
      reduceState(state, {
        type: "SET_RANGE",
        endYear: toFiniteNumber(event.target.value, state.endYear),
      }),
    );
    writeHashState();
    debouncedRun();
  });

  const clusterInput = document.getElementById("numClusters");
  clusterInput.addEventListener("change", (event) => {
    Object.assign(
      state,
      reduceState(state, {
        type: "SET_CLUSTERS",
        numClusters: toFiniteNumber(event.target.value, state.numClusters),
      }),
    );
    document.getElementById("clusterVal").textContent = String(state.numClusters);
    writeHashState();
    debouncedRun();
  });

  const layoutInput = document.getElementById("networkLayout");
  layoutInput.addEventListener("change", async (event) => {
    Object.assign(
      state,
      reduceState(state, {
        type: "SET_LAYOUT",
        networkLayout: String(event.target.value),
      }),
    );
    writeHashState();
    if (state.activeTab === "network") {
      await loadNetworkGraph();
    }
  });

  const projectionInput = document.getElementById("projectionMethod");
  projectionInput.addEventListener("change", async (event) => {
    Object.assign(
      state,
      reduceState(state, {
        type: "SET_PROJECTION",
        projectionMethod: String(event.target.value || "pca"),
      }),
    );
    writeHashState();
    await loadPCAPlot();
  });

  const thresholdInput = document.getElementById("similarityThreshold");
  thresholdInput.addEventListener("change", async (event) => {
    Object.assign(
      state,
      reduceState(state, {
        type: "SET_THRESHOLD",
        similarityThreshold: toFiniteNumber(
          event.target.value,
          state.similarityThreshold,
        ),
      }),
    );
    writeHashState();
    if (state.activeTab === "network") {
      await loadNetworkGraph();
    }
  });

  const labelToggle = document.getElementById("showNetworkLabels");
  labelToggle.addEventListener("change", async (event) => {
    Object.assign(
      state,
      reduceState(state, {
        type: "SET_LABELS",
        showNetworkLabels: Boolean(event.target.checked),
      }),
    );
    writeHashState();
    if (state.activeTab === "network") {
      await loadNetworkGraph();
    }
  });

  const issueSearchInput = document.getElementById("predictionIssueSearch");
  const debouncedIssueFilter = debounce(() => {
    renderIssueOptions(issueSearchInput.value);
  }, 150);
  issueSearchInput.addEventListener("input", debouncedIssueFilter);

  const updateButton = document.getElementById("updateNetworkBtn");
  updateButton.addEventListener("click", loadNetworkGraph);

  document.getElementById("trainModelBtn").addEventListener("click", trainModel);
  document.getElementById("predictBtn").addEventListener("click", predictVote);
  document.getElementById("compareBtn").addEventListener("click", compareCountries);

  document
    .getElementById("exportAnalysisBtn")
    .addEventListener("click", () => exportData("analysis"));
  document
    .getElementById("exportPredictionBtn")
    .addEventListener("click", () => exportData("prediction"));

  const shareBtn = document.getElementById("shareViewBtn");
  if (shareBtn) {
    shareBtn.addEventListener("click", async () => {
      writeHashState();
      const href = window.location.href;
      try {
        await navigator.clipboard.writeText(href);
        const originalText = shareBtn.textContent;
        shareBtn.textContent = "Copied";
        setTimeout(() => {
          shareBtn.textContent = originalText || "Share View";
        }, 1200);
      } catch (error) {
        alert(href);
      }
    });
  }

  const reportBtn = document.getElementById("downloadReportBtn");
  if (reportBtn) {
    reportBtn.addEventListener("click", async () => {
      reportBtn.disabled = true;
      try {
        const response = await axios.post(
          "/api/report",
          {
            start_year: state.startYear,
            end_year: state.endYear,
            format: "pdf",
          },
          { responseType: "blob" },
        );
        const blob = new Blob([response.data], { type: "application/pdf" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `un-voting-report-${state.startYear}-${state.endYear}.pdf`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      } catch (error) {
        alert(`Report export failed: ${getErrorMessage(error)}`);
      } finally {
        reportBtn.disabled = false;
      }
    });
  }
}

function setupMapControls() {
  const btn = document.getElementById("mapLoadBtn");
  if (btn) btn.addEventListener("click", loadAlignmentMap);
  const input = document.getElementById("mapCountry");
  if (input) {
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        loadAlignmentMap();
      }
    });
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  try {
    requireLibraries();
    state.startYear = toFiniteNumber(document.getElementById("startYear").value, 2010);
    state.endYear = toFiniteNumber(document.getElementById("endYear").value, 2020);
    state.numClusters = toFiniteNumber(document.getElementById("numClusters").value, 10);
    state.networkLayout = document.getElementById("networkLayout").value;
    state.projectionMethod = document.getElementById("projectionMethod").value || "pca";
    state.similarityThreshold = toFiniteNumber(
      document.getElementById("similarityThreshold").value,
      0.65,
    );
    state.showNetworkLabels = Boolean(
      document.getElementById("showNetworkLabels").checked,
    );

    applyHashState();
    syncControlsFromState();

    setupEventListeners();
    setupProfileControls();
    setupMapControls();
    setupStoryControls();
    setupDriftControls();
    setupCoalitionControls();
    setupNewsletterControls();
    setupTabs();
    setupMobileControls();
    activateTab(state.activeTab, { skipLoad: true });

    await loadDataSummary();
    await Promise.all([loadIssues(), loadMethods(), loadKnownEvents(), loadCountryNames()]);
    populateCountryDatalist();
    syncCountryInputs();
    await runAnalysis();
    writeHashState();

    window.addEventListener("hashchange", async () => {
      applyHashState();
      syncControlsFromState();
      activateTab(state.activeTab, { skipLoad: true });
      await runAnalysis();
    });
  } catch (error) {
    console.error("Dashboard initialization failed", error);
  }
});

// ── The Big Picture (whole-record story) ────────────────────────────────────
// Six readings of the entire roll-call record. Each card gets a headline
// finding computed from the numbers, the chart, the server's takeaway, and
// its caveat, so the finding never drifts from the data behind it.

const STORY = {
  loaded: false,
  loading: null,
  anchor: "USA",
  landmarks: [],
  activeLandmark: null,
  emergency: [],
};

// One colour per theme (Okabe-Ito, its yellow swapped for a gold that
// survives on white) and one per UN regional group.
const THEME_COLORS = {
  "Israel & Palestine": "#d55e00",
  "Decolonization & self-determination": "#e69f00",
  "Nuclear weapons & disarmament": "#0072b2",
  "Human rights": "#cc79a7",
  "Development, economy & environment": "#009e73",
  "UN institutions, budget & law": "#56b4e9",
  "Peace, security & conflicts": "#8a6d00",
  Other: "#b8b8b8",
};
const REGION_ORDER = [
  "Western Europe & Others",
  "Eastern Europe",
  "Latin America & Caribbean",
  "Africa",
  "Asia-Pacific",
  "Other",
];
const REGION_COLORS = {
  "Western Europe & Others": "#0072b2",
  "Eastern Europe": "#56b4e9",
  "Latin America & Caribbean": "#009e73",
  Africa: "#e69f00",
  "Asia-Pacific": "#d55e00",
  Other: "#999999",
};
const PLOT_CONFIG = { responsive: true, displayModeBar: false };

function storyLayout(overrides = {}) {
  const base = {
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Avenir Next, Trebuchet MS, Gill Sans, sans-serif", size: 12, color: "#23425f" },
    margin: { l: 10, r: 16, t: 12, b: 48 },
    hovermode: "closest",
    hoverlabel: { bgcolor: "#0b2238", bordercolor: "#0b2238", font: { color: "#ffffff", size: 12 } },
    legend: { orientation: "h", x: 0, y: 1.14, font: { size: 11 } },
    xaxis: { gridcolor: "#ece8e0", zeroline: false, automargin: true },
    yaxis: { gridcolor: "#ece8e0", zeroline: false, automargin: true },
  };
  const out = { ...base, ...overrides };
  out.xaxis = { ...base.xaxis, ...(overrides.xaxis || {}) };
  out.yaxis = { ...base.yaxis, ...(overrides.yaxis || {}) };
  if (overrides.margin) out.margin = { ...base.margin, ...overrides.margin };
  if (overrides.legend) out.legend = { ...base.legend, ...overrides.legend };
  return out;
}

function storyEl(cardId, role) {
  return document.querySelector(`#${cardId} [data-role="${role}"]`);
}

function setStoryText(cardId, role, text) {
  const el = storyEl(cardId, role);
  if (el) el.textContent = text || "";
}

function storyEscape(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function pct(value, digits = 0) {
  return `${(value * 100).toFixed(digits)}%`;
}

function movingSum(values, width = 3) {
  const half = Math.floor(width / 2);
  return values.map((_, i) => {
    let sum = 0;
    for (let j = i - half; j <= i + half; j += 1) {
      if (j >= 0 && j < values.length) sum += values[j];
    }
    return sum;
  });
}

// Dotted guide lines plus hoverable markers for the annotated events, so a
// reader can tie a kink in a line to 1956, 1991 or 2022 without a legend.
function eventShapes(minYear, maxYear, top = 100) {
  const raw = state.knownEvents;
  const events = (Array.isArray(raw) ? raw : (raw && raw.events) || []).filter(
    (e) => e.year >= minYear && e.year <= maxYear,
  );
  return {
    shapes: events.map((e) => ({
      type: "line", x0: e.year, x1: e.year, y0: 0, y1: 1, yref: "paper",
      line: { color: "#d8d3c8", width: 1, dash: "dot" },
    })),
    trace: {
      type: "scatter", mode: "markers", name: "events", showlegend: false,
      x: events.map((e) => e.year), y: events.map(() => top),
      marker: { color: "#b9b2a4", size: 7, symbol: "diamond" },
      text: events.map((e) => `${e.year}: ${e.label}`),
      hovertemplate: "%{text}<extra></extra>",
    },
  };
}

async function loadBigPicture(force = false) {
  if (STORY.loading) return STORY.loading;
  if (STORY.loaded && !force) return null;
  STORY.loading = (async () => {
    try {
      requirePlotly();
      ["storyAgenda", "storyDivision", "storyAlignment", "storyScatter", "storyRecurring"].forEach((id) => {
        const host = storyEl(id, "chart");
        if (host) setLoading(host, "Reading the whole record…");
      });
      const requests = {
        agenda: axios.get("/api/story/agenda"),
        division: axios.get("/api/story/division"),
        alignment: axios.get(`/api/story/alignment?anchor=${STORY.anchor}`),
        scatter: axios.get("/api/story/scatter"),
        landmarks: axios.get("/api/story/landmarks"),
        cuba: axios.get("/api/story/recurring/cuba"),
        calendar: axios.get("/api/story/calendar"),
        emergency: axios.get("/api/story/emergency"),
      };
      const cardFor = {
        agenda: ["storyAgenda", "chart"], division: ["storyDivision", "chart"],
        alignment: ["storyAlignment", "chart"], scatter: ["storyScatter", "chart"],
        landmarks: ["storyLandmarks", "map"], cuba: ["storyRecurring", "chart"],
        emergency: ["storyEmergency", "chart"],
      };
      const keys = Object.keys(requests);
      const settled = await Promise.allSettled(keys.map((k) => requests[k]));
      const data = {};
      settled.forEach((result, i) => {
        const key = keys[i];
        if (result.status === "fulfilled") {
          data[key] = result.value.data;
        } else if (cardFor[key]) {
          const host = storyEl(...cardFor[key]);
          if (host) showErrorElement(host, getErrorMessage(result.reason));
        }
      });
      if (data.agenda) renderStoryAgenda(data.agenda);
      if (data.division) renderStoryDivision(data.division);
      if (data.alignment) renderStoryAlignment(data.alignment);
      if (data.scatter) renderStoryScatter(data.scatter);
      if (data.landmarks) renderStoryLandmarks(data.landmarks.landmarks || []);
      if (data.cuba) renderStoryRecurring(data.cuba);
      renderStoryStats(data);
      if (data.calendar) renderStoryCalendar(data.calendar);
      if (data.emergency) renderStoryEmergency(data.emergency.sessions || []);
      STORY.loaded = true;
    } catch (error) {
      console.error("Big Picture failed", error);
    } finally {
      STORY.loading = null;
    }
  })();
  return STORY.loading;
}

function renderStoryStats(data) {
  const el = document.getElementById("storyStats");
  if (!el) return;
  clearNode(el);
  const tiles = [];
  if (data.agenda) {
    const total = data.agenda.totals.reduce((a, b) => a + b, 0);
    const years = data.agenda.years;
    tiles.push([total.toLocaleString(), `recorded votes, ${years[0]}–${years[years.length - 1]}`]);
  }
  if (data.division) {
    const last = data.division.series.find((r) => r.year === data.division.last_full_year);
    const max = data.division.series.reduce((a, b) => (b.votes > a.votes ? b : a));
    if (last) {
      tiles.push([
        last.votes.toLocaleString(),
        last.year === max.year
          ? `recorded votes in ${last.year}, the most in any year on record`
          : `recorded votes in ${last.year} (the record is ${max.votes} in ${max.year})`,
      ]);
    }
  }
  if (data.alignment && data.alignment.anchor === "USA") {
    const last = data.alignment.series.find((r) => r.year === data.alignment.last_full_year);
    if (last && last.agreement != null) {
      tiles.push([pct(last.agreement), `of other members' votes sided with the United States in ${last.year}`]);
    }
  }
  if (data.scatter) {
    const [a1, a2] = data.scatter.anchors.map((a) => a.toLowerCase());
    const closer = data.scatter.points.filter((p) => p[a2] > p[a1]).length;
    tiles.push([
      `${closer} of ${data.scatter.points.length}`,
      `members voted more often with ${data.scatter.anchor_names[1]} than with ${data.scatter.anchor_names[0]}, ${data.scatter.window[0]}–${data.scatter.window[1]}`,
    ]);
  }
  tiles.forEach(([value, label]) => {
    const tile = document.createElement("div");
    tile.className = "stat-tile";
    const v = document.createElement("div");
    v.className = "stat-tile__value";
    v.textContent = value;
    const l = document.createElement("div");
    l.className = "stat-tile__label";
    l.textContent = label;
    tile.appendChild(v);
    tile.appendChild(l);
    el.appendChild(tile);
  });
}

function renderStoryAgenda(d) {
  const host = storyEl("storyAgenda", "chart");
  clearNode(host);
  const traces = d.themes.map((theme) => ({
    type: "scatter", mode: "lines", stackgroup: "agenda", groupnorm: "percent",
    x: d.years, y: movingSum(d.counts[theme]), name: theme,
    line: { width: 0.6, color: THEME_COLORS[theme] || "#999" },
    fillcolor: THEME_COLORS[theme] || "#999",
    hovertemplate: `${theme}: %{y:.0f}%<extra></extra>`,
  }));
  Plotly.newPlot(host, traces, storyLayout({
    xaxis: { dtick: 10, range: [d.years[0], d.years[d.years.length - 1]] },
    yaxis: { title: "Share of recorded votes", ticksuffix: "%", range: [0, 100] },
    legend: { y: -0.12, x: 0 },
    margin: { t: 8, b: 96 },
    hovermode: "x unified",
  }), PLOT_CONFIG);

  const decades = Object.keys(d.decade_shares).sort();
  const lastDecade = decades[decades.length - 1];
  const shares = d.decade_shares[lastDecade];
  const ranked = Object.entries(shares).filter(([t]) => t !== "Other").sort((a, b) => b[1] - a[1]);
  const decol = "Decolonization & self-determination";
  const decolThen = d.decade_shares["1960"] ? d.decade_shares["1960"][decol] : null;
  let finding = `In the ${lastDecade}s the Assembly's recorded votes are mostly about ${ranked[0][0].toLowerCase()} and ${ranked[1][0].toLowerCase()}`;
  if (decolThen != null && decolThen > 0.25 && shares[decol] < 0.1) {
    finding += "; the colonial questions that once filled the agenda have nearly disappeared";
  }
  setStoryText("storyAgenda", "finding", finding);
  setStoryText("storyAgenda", "takeaway", d.takeaway);
  setStoryText("storyAgenda", "caveat", d.caveat);
}

function renderStoryDivision(d) {
  const host = storyEl("storyDivision", "chart");
  clearNode(host);
  const years = d.series.map((r) => r.year);
  const agree = d.series.map((r) => (r.agreement == null ? null : r.agreement * 100));
  const divided = d.series.map((r) => (r.divided_share == null ? null : r.divided_share * 100));
  const ev = eventShapes(years[0], years[years.length - 1]);
  Plotly.newPlot(host, [
    { type: "scatter", mode: "lines", x: years, y: agree, name: "Agreement between members", line: { color: "#0072b2", width: 2.5 }, hovertemplate: "%{y:.0f}% agreement<extra></extra>" },
    { type: "scatter", mode: "lines", x: years, y: divided, name: "Divided votes (one in ten dissented)", line: { color: "#d55e00", width: 2, dash: "dot" }, hovertemplate: "%{y:.0f}% of votes divided<extra></extra>" },
    ev.trace,
  ], storyLayout({
    shapes: ev.shapes,
    xaxis: { dtick: 10 },
    yaxis: { ticksuffix: "%", range: [0, 104] },
    hovermode: "x unified",
  }), PLOT_CONFIG);

  const valid = d.series.filter((r) => r.agreement != null);
  const peak = valid.reduce((a, b) => (b.agreement > a.agreement ? b : a));
  const low = valid.reduce((a, b) => (b.agreement < a.agreement ? b : a));
  const last = d.series.find((r) => r.year === d.last_full_year) || valid[valid.length - 1];
  setStoryText(
    "storyDivision", "finding",
    `The room was most united in ${peak.year} and most split in ${low.year}; in ${last.year} two members taking a side agreed ${pct(last.agreement)} of the time`,
  );
  setStoryText("storyDivision", "takeaway", d.takeaway);
  setStoryText("storyDivision", "caveat", d.caveat);
}

function renderStoryAlignment(d) {
  const host = storyEl("storyAlignment", "chart");
  clearNode(host);
  const years = d.series.map((r) => r.year);
  const agree = d.series.map((r) => (r.agreement == null ? null : r.agreement * 100));
  const isolated = d.series.map((r) => r.isolated_votes);
  const ev = eventShapes(years[0], years[years.length - 1]);
  Plotly.newPlot(host, [
    { type: "bar", x: years, y: isolated, name: "Votes cast with two or fewer others", yaxis: "y2", marker: { color: "#e8dfcd" }, hovertemplate: "%{y} isolated votes<extra></extra>" },
    { type: "scatter", mode: "lines", x: years, y: agree, name: `Members siding with ${d.anchor_name}`, line: { color: "#0b2238", width: 2.5 }, hovertemplate: `%{y:.0f}% sided with ${d.anchor_name}<extra></extra>` },
    ev.trace,
  ], storyLayout({
    shapes: ev.shapes,
    xaxis: { dtick: 10 },
    yaxis: { ticksuffix: "%", range: [0, 104] },
    yaxis2: { overlaying: "y", side: "right", showgrid: false, rangemode: "tozero", title: { text: "isolated votes", font: { size: 11 } } },
    hovermode: "x unified",
    margin: { r: 56 },
  }), PLOT_CONFIG);

  const valid = d.series.filter((r) => r.agreement != null);
  const peak = valid.reduce((a, b) => (b.agreement > a.agreement ? b : a));
  const last = d.series.find((r) => r.year === d.last_full_year) || valid[valid.length - 1];
  setStoryText(
    "storyAlignment", "finding",
    `Other members sided with ${d.anchor_name} ${pct(last.agreement)} of the time in ${last.year}, against ${pct(peak.agreement)} at the peak in ${peak.year}`,
  );
  setStoryText("storyAlignment", "takeaway", d.takeaway);

  const list = storyEl("storyAlignment", "list");
  clearNode(list);
  if (d.recent_isolated && d.recent_isolated.length) {
    const title = document.createElement("div");
    title.className = "story-list__title";
    title.textContent = `Latest votes ${d.anchor_name} cast with two or fewer companions`;
    list.appendChild(title);
    const ul = document.createElement("ul");
    d.recent_isolated.slice(0, 6).forEach((r) => {
      const li = document.createElement("li");
      li.textContent = `${r.year} · ${r.title} `;
      const span = document.createElement("span");
      span.textContent = r.companions === 0 ? "(alone)" : `(with ${r.companions} other${r.companions === 1 ? "" : "s"})`;
      li.appendChild(span);
      ul.appendChild(li);
    });
    list.appendChild(ul);
  }
}

function setupStoryControls() {
  const recurring = document.getElementById("storyRecurringKey");
  if (recurring) {
    recurring.addEventListener("change", async () => {
      const host = storyEl("storyRecurring", "chart");
      if (host) setLoading(host, "Loading the series…");
      try {
        const res = await axios.get(`/api/story/recurring/${recurring.value}`);
        renderStoryRecurring(res.data);
      } catch (error) {
        if (host) showErrorElement(host, getErrorMessage(error));
      }
    });
  }
  const emergency = document.getElementById("storyEmergencySession");
  if (emergency) {
    emergency.addEventListener("change", () => renderEmergencySession(Number(emergency.value)));
  }
  const select = document.getElementById("storyAnchor");
  if (!select) return;
  select.addEventListener("change", async () => {
    STORY.anchor = select.value;
    const host = storyEl("storyAlignment", "chart");
    if (host) setLoading(host, "Recomputing…");
    try {
      const res = await axios.get(`/api/story/alignment?anchor=${STORY.anchor}`);
      renderStoryAlignment(res.data);
    } catch (error) {
      if (host) showErrorElement(host, getErrorMessage(error));
    }
  });
}

function renderStoryScatter(d) {
  const host = storyEl("storyScatter", "chart");
  clearNode(host);
  const [a1, a2] = d.anchors.map((a) => a.toLowerCase());
  const [n1, n2] = d.anchor_names;
  const byRegion = {};
  d.points.forEach((p) => {
    (byRegion[p.region] = byRegion[p.region] || []).push(p);
  });
  const traces = REGION_ORDER.filter((r) => byRegion[r]).map((region) => {
    const pts = byRegion[region];
    return {
      type: "scatter", mode: "markers", name: region,
      x: pts.map((p) => p[a1] * 100), y: pts.map((p) => p[a2] * 100),
      text: pts.map((p) => p.name),
      marker: { color: REGION_COLORS[region], size: 9, opacity: 0.85, line: { color: "#fff", width: 0.8 } },
      hovertemplate: `<b>%{text}</b><br>with ${n1}: %{x:.0f}%<br>with ${n2}: %{y:.0f}%<extra></extra>`,
    };
  });
  const baseKey1 = `${a1}_base`;
  const baseKey2 = `${a2}_base`;
  const movers = d.points
    .filter((p) => p[baseKey1] != null)
    .map((p) => ({ ...p, dist: Math.hypot(p[a1] - p[baseKey1], p[a2] - p[baseKey2]) }))
    .sort((x, y) => y.dist - x.dist)
    .slice(0, 12);
  const annotations = movers.map((p) => ({
    x: p[a1] * 100, y: p[a2] * 100, ax: p[baseKey1] * 100, ay: p[baseKey2] * 100,
    xref: "x", yref: "y", axref: "x", ayref: "y",
    showarrow: true, arrowhead: 3, arrowsize: 1, arrowwidth: 1.2, arrowcolor: "#8a94a0", text: "",
  }));
  const labelled = new Set([...movers.map((p) => p.code), "ISR", "GBR", "DEU", "IND", "BRA", "ZAF", "TUR", "JPN", "SAU", "UKR"]);
  const labels = d.points.filter((p) => labelled.has(p.code));
  traces.push({
    type: "scatter", mode: "text", showlegend: false, hoverinfo: "skip",
    x: labels.map((p) => p[a1] * 100), y: labels.map((p) => p[a2] * 100),
    text: labels.map((p) => p.name), textposition: "top center", textfont: { size: 10, color: "#456783" },
  });
  Plotly.newPlot(host, traces, storyLayout({
    annotations,
    shapes: [{ type: "line", x0: 0, y0: 0, x1: 100, y1: 100, line: { color: "#c9c4b8", width: 1, dash: "dash" } }],
    xaxis: { title: `Voted with ${n1} (% of votes where both took a side)`, range: [0, 102], ticksuffix: "%" },
    yaxis: { title: `Voted with ${n2}`, range: [0, 102], ticksuffix: "%" },
    legend: { y: -0.16, x: 0 },
    margin: { t: 8, b: 84 },
  }), PLOT_CONFIG);

  const closer = d.points.filter((p) => p[a2] > p[a1]).length;
  setStoryText(
    "storyScatter", "finding",
    `${closer} of ${d.points.length} members voted more often with ${n2} than with ${n1} in ${d.window[0]}–${d.window[1]}`,
  );
  setStoryText("storyScatter", "takeaway", d.takeaway);
  setStoryText("storyScatter", "caveat", d.caveat);
}

function renderStoryLandmarks(items) {
  STORY.landmarks = items;
  const list = storyEl("storyLandmarks", "list");
  clearNode(list);
  items.forEach((item) => {
    const total = (item.yes || 0) + (item.no || 0) + (item.abstain || 0) || 1;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "landmark";
    btn.setAttribute("role", "listitem");
    btn.dataset.rcid = String(item.rcid);
    btn.innerHTML =
      `<div class="landmark__year">${item.year} · ${storyEscape(item.symbol)}</div>` +
      `<div class="landmark__label">${storyEscape(item.label)}</div>` +
      `<div class="tally" aria-hidden="true">` +
      `<span class="yes" style="width:${(100 * item.yes) / total}%"></span>` +
      `<span class="abstain" style="width:${(100 * item.abstain) / total}%"></span>` +
      `<span class="no" style="width:${(100 * item.no) / total}%"></span></div>` +
      `<div class="landmark__tally">${item.yes} for · ${item.no} against · ${item.abstain} abstaining</div>`;
    btn.addEventListener("click", () => selectLandmark(item.rcid));
    list.appendChild(btn);
  });
  const preferred = items.find((i) => i.symbol === "A/RES/ES-11/1") || items[items.length - 1];
  if (preferred) selectLandmark(preferred.rcid);
}

async function selectLandmark(rcid) {
  STORY.activeLandmark = rcid;
  document.querySelectorAll("#storyLandmarks .landmark").forEach((b) => {
    b.classList.toggle("active", Number(b.dataset.rcid) === rcid);
  });
  const head = storyEl("storyLandmarks", "detail");
  const host = storyEl("storyLandmarks", "map");
  const item = STORY.landmarks.find((i) => i.rcid === rcid);
  if (host) setLoading(host, "Mapping the vote…");
  try {
    const res = await axios.get(`/api/story/resolution/${rcid}/map`);
    const d = res.data;
    const t = d.tally;
    if (head) {
      head.innerHTML =
        `<h4>${storyEscape(item ? item.label : d.title)}</h4>` +
        `<p class="meta">${storyEscape(d.symbol)} · ${storyEscape(d.date)} · ${storyEscape(d.title)}</p>` +
        `<p class="why">${storyEscape(item ? item.why : "")}</p>` +
        `<div class="vote-legend"><span class="yes"><i></i>${t.yes} for</span>` +
        `<span class="no"><i></i>${t.no} against</span>` +
        `<span class="abstain"><i></i>${t.abstain} abstained</span>` +
        `<span class="absent"><i></i>${t.absent} absent or not voting</span>` +
        `<span>Blank: not a member at the time</span></div>`;
      if (d.regions && d.regions.length) {
        const rows = d.regions.map((r) => {
          const total = r.yes + r.no + r.abstain + r.absent || 1;
          return `<tr><th scope="row">${storyEscape(r.region)}</th><td>` +
            `<div class="tally" aria-hidden="true"><span class="yes" style="width:${(100 * r.yes) / total}%"></span>` +
            `<span class="abstain" style="width:${(100 * r.abstain) / total}%"></span>` +
            `<span class="no" style="width:${(100 * r.no) / total}%"></span></div></td>` +
            `<td class="num">${r.yes}–${r.no}–${r.abstain}</td></tr>`;
        }).join("");
        head.innerHTML +=
          `<table class="region-table"><caption>How the regional groups voted (for–against–abstained)</caption>` +
          `<tbody>${rows}</tbody></table>` +
          (d.region_note ? `<p class="why">${storyEscape(d.region_note)}</p>` : "");
      }
    }
    const code = { yes: 3, abstain: 2, no: 1, absent: 0 };
    const label = { yes: "For", no: "Against", abstain: "Abstained", absent: "Absent / not voting" };
    clearNode(host);
    Plotly.newPlot(host, [{
      type: "choropleth", locationmode: "ISO-3",
      locations: d.votes.map((v) => v.code),
      z: d.votes.map((v) => code[v.vote]),
      text: d.votes.map((v) => `${v.name}: ${label[v.vote]}`),
      zmin: 0, zmax: 3, showscale: false,
      colorscale: [
        [0, "#d0d0d0"], [0.25, "#d0d0d0"],
        [0.25, "#d55e00"], [0.5, "#d55e00"],
        [0.5, "#d38b2a"], [0.75, "#d38b2a"],
        [0.75, "#0072b2"], [1, "#0072b2"],
      ],
      hovertemplate: "%{text}<extra></extra>",
      marker: { line: { color: "#ffffff", width: 0.4 } },
    }], storyLayout({
      geo: { showframe: false, showcoastlines: true, coastlinecolor: "#cfd8df", projection: { type: "natural earth" }, bgcolor: "rgba(0,0,0,0)" },
      margin: { l: 0, r: 0, t: 0, b: 0 },
    }), { ...PLOT_CONFIG, topojsonURL: "/static/vendor/" });
  } catch (error) {
    if (host) showErrorElement(host, getErrorMessage(error));
  }
}

function renderStoryRecurring(d) {
  const host = storyEl("storyRecurring", "chart");
  clearNode(host);
  const select = document.getElementById("storyRecurringKey");
  if (select && d.available && select.options.length !== d.available.length) {
    clearNode(select);
    d.available.forEach((a) => {
      const opt = document.createElement("option");
      opt.value = a.key;
      opt.textContent = a.label;
      select.appendChild(opt);
    });
  }
  if (select) select.value = d.key;
  const years = d.series.map((s) => s.year);
  const bar = (key, name, color) => ({
    type: "bar", x: years, y: d.series.map((s) => s[key]), name, marker: { color },
    hovertemplate: `${name}: %{y}<extra></extra>`,
  });
  Plotly.newPlot(host, [bar("yes", "For", "#0072b2"), bar("abstain", "Abstained", "#d38b2a"), bar("no", "Against", "#d55e00")], storyLayout({
    barmode: "stack", bargap: 0.25,
    xaxis: { dtick: 5 },
    yaxis: { title: "members" },
    hovermode: "x unified",
  }), PLOT_CONFIG);
  setStoryText("storyRecurring", "caption", d.why);
  const first = d.series[0];
  const last = d.series[d.series.length - 1];
  const peak = d.series.reduce((a, b) => (b.yes > a.yes ? b : a));
  setStoryText(
    "storyRecurring", "finding",
    `${d.label}: ${first.yes} supporters in ${first.year}, ${peak.yes} at the ${peak.year} peak, ${last.yes} in ${last.year}`,
  );
  setStoryText("storyRecurring", "takeaway", d.takeaway);
}


// ── The long view (whole-record trajectory on the Country Profile) ──────────

async function loadCountryStory(code) {
  const host = storyEl("profileStory", "chart");
  if (!host) return;
  setLoading(host, "Reading the whole record…");
  try {
    const res = await axios.get(`/api/story/country/${code}`);
    renderCountryStory(res.data);
  } catch (error) {
    showErrorElement(host, getErrorMessage(error));
  }
}

function renderCountryStory(d) {
  const host = storyEl("profileStory", "chart");
  clearNode(host);
  const years = d.series.map((r) => r.year);
  const colors = { usa: "#0072b2", rus: "#d55e00", chn: "#8a6d00" };
  const line = (key, name, color, dash) => ({
    type: "scatter", mode: "lines", name, connectgaps: false,
    x: years, y: d.series.map((r) => (r[key] == null ? null : r[key] * 100)),
    line: { color, width: key === "with_majority" ? 1.5 : 2.2, dash },
    hovertemplate: `${name}: %{y:.0f}%<extra></extra>`,
  });
  const traces = [];
  d.anchors.forEach((a) => {
    const key = a.toLowerCase();
    if (d.series.some((r) => r[key] != null)) {
      traces.push(line(key, `with ${d.anchor_names[a] || a}`, colors[key] || "#999999"));
    }
  });
  traces.push(line("with_majority", "on the winning side", "#456783", "dot"));
  const first = Math.max(years[0], d.first_year - 1);
  const ev = eventShapes(first, years[years.length - 1]);
  traces.push(ev.trace);
  Plotly.newPlot(host, traces, storyLayout({
    shapes: ev.shapes,
    xaxis: { dtick: 10, range: [first, years[years.length - 1]] },
    yaxis: { ticksuffix: "%", range: [0, 104] },
    hovermode: "x unified",
  }), PLOT_CONFIG);

  let finding = `${d.name} has cast recorded votes since ${d.first_year}`;
  const latest = d.latest;
  if (latest) {
    const scored = d.anchors
      .map((a) => ({ name: d.anchor_names[a] || a, value: latest[a.toLowerCase()] }))
      .filter((x) => x.value != null)
      .sort((x, y) => y.value - x.value);
    if (scored.length >= 2) {
      finding = `In ${latest.year} ${d.name} voted most often with ${scored[0].name} (${pct(scored[0].value)}) and least with ${scored[scored.length - 1].name} (${pct(scored[scored.length - 1].value)})`;
    } else if (scored.length === 1) {
      finding = `In ${latest.year} ${d.name} sided with ${scored[0].name} ${pct(scored[0].value)} of the time`;
    }
  }
  setStoryText("profileStory", "finding", finding);
  setStoryText("profileStory", "takeaway", d.takeaway);
  setStoryText("profileStory", "caveat", d.caveat);
}


function renderStoryCalendar(d) {
  const el = document.getElementById("storyCalendar");
  if (!el || !d || !d.note) return;
  clearNode(el);
  const strong = document.createElement("strong");
  strong.textContent = "Where the session stands. ";
  el.appendChild(strong);
  el.appendChild(document.createTextNode(d.note));
}


// ── The emergency special sessions ──────────────────────────────────────────

function renderStoryEmergency(sessions) {
  STORY.emergency = sessions;
  const select = document.getElementById("storyEmergencySession");
  if (select) {
    clearNode(select);
    sessions.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = String(s.number);
      opt.textContent = `ES-${s.number} · ${s.label}`;
      select.appendChild(opt);
    });
  }
  const preferred = sessions.filter((s) => s.count >= 2).sort((a, b) => b.number - a.number)[0] || sessions[sessions.length - 1];
  if (preferred) {
    if (select) select.value = String(preferred.number);
    renderEmergencySession(preferred.number);
  }
}

function renderEmergencySession(number) {
  const session = STORY.emergency.find((s) => s.number === number);
  const host = storyEl("storyEmergency", "chart");
  if (!session || !host) return;
  clearNode(host);
  const labels = session.votes.map((v) => `${v.symbol.replace("A/RES/", "")} · ${v.date}`);
  const bar = (key, name, color) => ({
    type: "bar", name, x: labels, y: session.votes.map((v) => v[key]), marker: { color },
    customdata: session.votes.map((v) => v.title),
    hovertemplate: `${name}: %{y}<br>%{customdata}<extra></extra>`,
  });
  Plotly.newPlot(host, [bar("yes", "For", "#0072b2"), bar("abstain", "Abstained", "#d38b2a"), bar("no", "Against", "#d55e00")], storyLayout({
    barmode: "stack", bargap: 0.3,
    xaxis: { type: "category", tickangle: session.votes.length > 8 ? -35 : 0, tickfont: { size: 10 } },
    yaxis: { title: "members" },
    hovermode: "x unified",
    margin: { b: session.votes.length > 8 ? 110 : 60 },
  }), PLOT_CONFIG);
  const latest = session.votes[session.votes.length - 1];
  setStoryText(
    "storyEmergency", "finding",
    `ES-${session.number}, ${session.label}: ${session.count} recorded vote${session.count === 1 ? "" : "s"}; the latest passed ${latest.yes} to ${latest.no} on ${latest.date}`,
  );
  setStoryText("storyEmergency", "takeaway", session.takeaway);
}


// ── Through which lens? (competing IR theories, year by year) ───────────────

const LENS = { loaded: false, loading: null, data: null };
const LENS_COLORS = {
  realism: "#0b2238",
  liberalism: "#0072b2",
  world_systems: "#d55e00",
  constructivism: "#009e73",
  feminism: "#cc79a7",
};
const PARTITION_LABELS = {
  alliance: "Treaty alliances (realism)",
  tier: "Income tiers (world-systems)",
  region: "Regional groups (constructivism)",
};
const PARTITION_COLORS = { alliance: "#0b2238", tier: "#d55e00", region: "#009e73" };

function pctOrNull(v) {
  return v == null ? null : v * 100;
}

// Sentences quote the last full year; a partial current year still plots.
function lensAnchorIndex(d) {
  const i = d.last_full_year ? d.years.indexOf(d.last_full_year) : -1;
  return i >= 0 ? i : d.years.length - 1;
}

async function loadLenses(force = false) {
  if (LENS.loading) return LENS.loading;
  if (LENS.loaded && !force) return null;
  LENS.loading = (async () => {
    try {
      requirePlotly();
      ["lensOrganise", "lensHomeTurf", "lensFingerprints", "lensCascades", "lensNorthSouth", "lensCohesion"].forEach((id) => {
        const host = storyEl(id, "chart");
        if (host) setLoading(host, "Reading the whole record through five lenses… (the first load takes a moment)");
      });
      const res = await axios.get("/api/story/lenses");
      LENS.data = res.data;
      renderLensOrganise(res.data);
      renderLensHomeTurf(res.data);
      renderLensFingerprints(res.data);
      renderLensEras(res.data);
      renderLensCascades(res.data);
      renderLensNorthSouth(res.data);
      renderLensCohesion(res.data);
      renderLensMethods(res.data);
      LENS.loaded = true;
    } catch (error) {
      const host = storyEl("lensOrganise", "chart");
      if (host) showErrorElement(host, getErrorMessage(error));
      console.error("Lenses failed", error);
    } finally {
      LENS.loading = null;
    }
  })();
  return LENS.loading;
}

function partitionLine(d, scheme, source, dash) {
  const rows = source[scheme];
  return {
    type: "scatter", mode: "lines", name: PARTITION_LABELS[scheme], connectgaps: false,
    x: d.years, y: rows.map((p) => pctOrNull(p.adjusted)),
    line: { color: PARTITION_COLORS[scheme], width: 2.2, dash },
    hovertemplate: `${PARTITION_LABELS[scheme]}: %{y:.0f}%<extra></extra>`,
  };
}

function decadeMeans(years, values) {
  const out = {};
  years.forEach((y, i) => {
    const d = Math.floor(y / 10) * 10;
    if (values[i] == null) return;
    (out[d] = out[d] || []).push(values[i]);
  });
  return Object.fromEntries(Object.entries(out).map(([d, v]) => [d, v.reduce((a, b) => a + b, 0) / v.length]));
}

function renderLensOrganise(d) {
  const host = storyEl("lensOrganise", "chart");
  clearNode(host);
  const ev = eventShapes(d.years[0], d.years[d.years.length - 1]);
  Plotly.newPlot(host, [
    partitionLine(d, "region", d.partitions),
    partitionLine(d, "alliance", d.partitions),
    partitionLine(d, "tier", d.partitions),
    ev.trace,
  ], storyLayout({
    shapes: ev.shapes,
    xaxis: { dtick: 10 },
    yaxis: { title: "Voting variance explained beyond chance", ticksuffix: "%", range: [0, 100] },
    hovermode: "x unified",
  }), PLOT_CONFIG);

  const means = {};
  ["region", "alliance", "tier"].forEach((s) => {
    means[s] = decadeMeans(d.years, d.partitions[s].map((p) => p.adjusted));
  });
  const decades = Object.keys(means.region).sort();
  const regionWins = decades.filter((dec) => means.region[dec] >= (means.alliance[dec] || 0) && means.region[dec] >= (means.tier[dec] || 0)).length;
  const alliancePeak = d.years.reduce((best, y, i) => {
    const v = d.partitions.alliance[i].adjusted;
    return v != null && v > best.v ? { y, v } : best;
  }, { y: null, v: -1 });
  const last = lensAnchorIndex(d);
  setStoryText(
    "lensOrganise", "finding",
    regionWins === decades.length
      ? `The UN's own regional groups have predicted votes better than treaty alliances or wealth in every decade`
      : `The UN's regional groups predicted votes better than alliances or wealth in ${regionWins} of ${decades.length} decades`,
  );
  setStoryText(
    "lensOrganise", "takeaway",
    `Treaty alliances explained the most in ${alliancePeak.y} (${(alliancePeak.v * 100).toFixed(0)}% beyond chance) and ${(d.partitions.alliance[last].adjusted * 100).toFixed(0)}% in ${d.years[last]}; ` +
    `regional groups ${(d.partitions.region[last].adjusted * 100).toFixed(0)}% and income tiers ${(d.partitions.tier[last].adjusted * 100).toFixed(0)}% in ${d.years[last]}.`,
  );
}

function renderLensHomeTurf(d) {
  const host = storyEl("lensHomeTurf", "chart");
  clearNode(host);
  const label = { alliance: "Alliances on security items", tier: "Income tiers on economic items", region: "Regional groups on rights items" };
  const smooth = d.home_turf_smoothed || {};
  const traces = ["region", "alliance", "tier"].map((s) => ({
    type: "scatter", mode: "lines", name: label[s], connectgaps: false,
    x: d.years, y: (smooth[s] || d.home_turf[s].map((p) => p.adjusted)).map(pctOrNull),
    customdata: d.home_turf[s].map((p) => (p.adjusted == null ? "–" : (p.adjusted * 100).toFixed(0) + "%")),
    line: { color: PARTITION_COLORS[s], width: 2, shape: "spline", smoothing: 0.6 },
    hovertemplate: `${label[s]}: %{y:.0f}% (this year alone %{customdata})<extra></extra>`,
  }));
  Plotly.newPlot(host, traces, storyLayout({
    xaxis: { dtick: 10 },
    yaxis: { title: "Variance explained beyond chance, 5-year average", ticksuffix: "%", range: [0, 100] },
    hovermode: "x unified",
  }), PLOT_CONFIG);
  const peak = (s) => d.years.reduce((best, y, i) => {
    const v = d.home_turf[s][i].adjusted;
    return v != null && v > best.v ? { y, v } : best;
  }, { y: null, v: -1 });
  const a = peak("alliance"), t = peak("tier"), r = peak("region");
  setStoryText("lensHomeTurf", "finding", `On security items alliances explained up to ${(a.v * 100).toFixed(0)}% of the vote (${a.y}); on economic items income tiers up to ${(t.v * 100).toFixed(0)}% (${t.y}); on rights items regional groups up to ${(r.v * 100).toFixed(0)}% (${r.y})`);
  setStoryText("lensHomeTurf", "takeaway", "Peaks are single years; lines average five. Gaps are stretches with fewer than five recorded votes on the theme.");
}

function renderLensFingerprints(d) {
  const host = storyEl("lensFingerprints", "chart");
  clearNode(host);
  const smooth = d.indices_smoothed || d.indices;
  const traces = d.lenses.map((l) => ({
    type: "scatter", mode: "lines", name: l.label, connectgaps: false,
    x: d.years, y: smooth[l.key].map((v) => (v == null ? null : v * 100)),
    customdata: d.indices[l.key].map((v) => (v == null ? "–" : (v * 100).toFixed(0))),
    line: { color: LENS_COLORS[l.key], width: l.key === "feminism" ? 1.4 : 2.2, dash: l.key === "feminism" ? "dot" : "solid", shape: "spline", smoothing: 0.6 },
    hovertemplate: `${l.label}: %{y:.0f} (this year alone %{customdata})<extra></extra>`,
  }));
  const ev = eventShapes(d.years[0], d.years[d.years.length - 1]);
  traces.push(ev.trace);
  Plotly.newPlot(host, traces, storyLayout({
    shapes: ev.shapes,
    xaxis: { dtick: 10 },
    yaxis: { title: "Fingerprint index, 5-year average (0–100)", range: [0, 104] },
    hovermode: "x unified",
  }), PLOT_CONFIG);

  const strip = storyEl("lensFingerprints", "eras");
  clearNode(strip);
  d.eras.forEach((e) => {
    const cell = document.createElement("div");
    cell.className = "era-strip__cell";
    cell.style.background = LENS_COLORS[e.top] || "#999";
    cell.innerHTML = `<b>${e.label}</b>${storyEscape(e.top_label)}${e.close ? `<small> · ${storyEscape(e.close_label)} close</small>` : ""}`;
    cell.title = e.ranking.map((r) => `${r.label} ${(r.score * 100).toFixed(0)}`).join(" · ");
    strip.appendChild(cell);
  });
  const counts = {};
  d.eras.forEach((e) => { counts[e.top_label] = (counts[e.top_label] || 0) + 1; });
  const ranked = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const latest = d.eras[d.eras.length - 1];
  setStoryText("lensFingerprints", "finding", `${ranked[0][0]} has the most pronounced fingerprint in ${ranked[0][1]} of ${d.eras.length} decades; the ${latest.label} read as ${latest.top_label.toLowerCase()}`);
  setStoryText("lensFingerprints", "takeaway", `Ranking in the ${latest.label}: ` + latest.ranking.map((r) => `${r.label} ${(r.score * 100).toFixed(0)}`).join(", ") + ".");
  setStoryText("lensFingerprints", "caveat", d.caveats[2]);
}

function renderLensEras(d) {
  const host = storyEl("lensEras", "table");
  clearNode(host);
  const rows = d.eras.map((e) => {
    const chip = `<span class="lens-chip" style="background:${LENS_COLORS[e.top] || "#999"}">${storyEscape(e.top_label)}</span>` + (e.close ? ` <small class="lens-close">${storyEscape(e.close_label)} close behind</small>` : "");
    const facts = e.facts.map((f) => storyEscape(f.charAt(0).toUpperCase() + f.slice(1))).join(". ");
    return `<tr><td class="decade">${e.label}</td><td>${chip}</td><td>${facts}.</td></tr>`;
  }).join("");
  host.innerHTML = `<table class="era-table"><thead><tr><th>Decade</th><th>Most pronounced lens</th><th>By deed</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderLensCascades(d) {
  const host = storyEl("lensCascades", "chart");
  clearNode(host);
  const palette = ["#009e73", "#0072b2", "#e69f00", "#cc79a7", "#56b4e9"];
  const traces = d.cascades.map((c, i) => ({
    type: "scatter", mode: "lines+markers", name: c.label,
    x: c.series.map((p) => p.year), y: c.series.map((p) => p.support * 100),
    line: { color: palette[i % palette.length], width: 2 }, marker: { size: 5 },
    hovertemplate: `${c.label}: %{y:.0f}% in favour<extra>%{x}</extra>`,
  }));
  Plotly.newPlot(host, traces, storyLayout({
    xaxis: { dtick: 5 },
    yaxis: { title: "Members in favour", ticksuffix: "%", range: [0, 104] },
    legend: { y: -0.2, x: 0 },
    margin: { b: 90 },
  }), PLOT_CONFIG);
}

function renderLensNorthSouth(d) {
  const select = document.getElementById("lensMarker");
  if (select && select.options.length !== d.north_south_markers.length) {
    clearNode(select);
    d.north_south_markers.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.key;
      opt.textContent = m.label;
      select.appendChild(opt);
    });
    select.addEventListener("change", () => renderLensNorthSouth(LENS.data));
  }
  const key = select && select.value ? select.value : d.north_south_markers[0]?.key;
  const marker = d.north_south_markers.find((m) => m.key === key) || d.north_south_markers[0];
  const host = storyEl("lensNorthSouth", "chart");
  clearNode(host);
  if (!marker) return;
  const years = marker.series.map((s) => s.year);
  const bar = (k, name, color) => ({
    type: "bar", x: years, y: marker.series.map((s) => s[k]), name, marker: { color },
    hovertemplate: `${name}: %{y}<extra>%{x}</extra>`,
  });
  Plotly.newPlot(host, [bar("yes", "For", "#0072b2"), bar("abstain", "Abstained", "#d38b2a"), bar("no", "Against", "#d55e00")], storyLayout({
    barmode: "stack", bargap: 0.25,
    xaxis: { dtick: 5 },
    yaxis: { title: "members" },
    hovermode: "x unified",
  }), PLOT_CONFIG);
}

function renderLensCohesion(d) {
  const host = storyEl("lensCohesion", "chart");
  clearNode(host);
  const spec = [
    ["soviet_led", "Soviet / Russian-led camp", "#d55e00", "solid"],
    ["us_led", "US-led camp", "#0b2238", "solid"],
    ["core", "Core (high income)", "#8a6d00", "dot"],
    ["periphery", "Periphery (low income)", "#e69f00", "dot"],
    ["regions", "Regional groups (average)", "#009e73", "dash"],
  ];
  const traces = spec.map(([k, name, color, dash]) => ({
    type: "scatter", mode: "lines", name, connectgaps: false,
    x: d.years, y: d.cohesion[k].map(pctOrNull),
    line: { color, width: 2, dash },
    hovertemplate: `${name}: %{y:.0f}%<extra></extra>`,
  }));
  const ev = eventShapes(d.years[0], d.years[d.years.length - 1]);
  traces.push(ev.trace);
  Plotly.newPlot(host, traces, storyLayout({
    shapes: ev.shapes,
    xaxis: { dtick: 10 },
    yaxis: { title: "Within-bloc agreement", ticksuffix: "%", range: [50, 102] },
    hovermode: "x unified",
    legend: { y: -0.16, x: 0 },
    margin: { b: 84 },
  }), PLOT_CONFIG);
  const i1985 = d.years.indexOf(1985);
  const last = lensAnchorIndex(d);
  const sov = i1985 >= 0 ? d.cohesion.soviet_led[i1985] : null;
  setStoryText(
    "lensCohesion", "finding",
    sov != null
      ? `In 1985 the Soviet camp voted together ${(sov * 100).toFixed(0)}% of the time and the US-led camp ${(d.cohesion.us_led[i1985] * 100).toFixed(0)}%; in ${d.years[last]} the periphery is the most disciplined bloc at ${(d.cohesion.periphery[last] * 100).toFixed(0)}%`
      : "Bloc discipline over the record",
  );
  setStoryText("lensCohesion", "takeaway", `US-led camp ${(d.cohesion.us_led[last] * 100).toFixed(0)}%, Russian-led camp ${(d.cohesion.soviet_led[last] * 100).toFixed(0)}%, core ${(d.cohesion.core[last] * 100).toFixed(0)}%, regional groups ${(d.cohesion.regions[last] * 100).toFixed(0)}% in ${d.years[last]}.`);
}

function renderLensMethods(d) {
  const host = storyEl("lensMethods", "methods");
  clearNode(host);
  const names = {
    explained_variance: "Explained variance",
    cohesion: "Cohesion",
    index: "Fingerprint index",
    alliance_camps: "Alliance camps",
    tiers: "Income tiers",
    feminist_cohort: "Feminist-policy cohort",
  };
  const dl = Object.entries(d.definitions).map(([k, v]) => `<dt>${storyEscape(names[k] || k)}</dt><dd>${storyEscape(v)}</dd>`).join("");
  const ul = d.caveats.map((c) => `<li>${storyEscape(c)}</li>`).join("");
  host.innerHTML = `<div class="methods"><dl>${dl}</dl><ul>${ul}</ul></div>`;
}
