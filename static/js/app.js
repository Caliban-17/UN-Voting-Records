// UN Voting Intelligence Dashboard frontend

const state = {
  startYear: 2010,
  endYear: 2020,
  numClusters: 10,
  networkLayout: "force",
  projectionMethod: "pca",
  similarityThreshold: 0.65,
  showNetworkLabels: false,
  activeTab: "profile",
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

  const start = toFiniteNumber(params.get("start"), state.startYear);
  const end = toFiniteNumber(params.get("end"), state.endYear);
  const k = toFiniteNumber(params.get("k"), state.numClusters);
  const threshold = toFiniteNumber(params.get("threshold"), state.similarityThreshold);
  const allowedTabs = new Set([
    "profile",
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
              if (label === "Yes") return "#2a9d57";
              if (label === "No") return "#c64141";
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

  const countryA = normalizeCodeInput(countryAInput.value);
  const countryB = normalizeCodeInput(countryBInput.value);
  countryAInput.value = countryA;
  countryBInput.value = countryB;

  if (!/^[A-Z]{3}$/.test(countryA)) {
    showFieldError("countryA", "Use a 3-letter country code (e.g., USA)");
    return;
  }
  if (!/^[A-Z]{3}$/.test(countryB)) {
    showFieldError("countryB", "Use a 3-letter country code (e.g., RUS)");
    return;
  }

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
        if (input) input.value = code;
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
  const yr = toFiniteNumber(document.getElementById("newsletterYear")?.value, state.endYear);
  const bw = toFiniteNumber(document.getElementById("newsletterBaseline")?.value, 3);
  const topics = (document.getElementById("newsletterTopics")?.value || "").trim();
  const country = normalizeCodeInput(document.getElementById("newsletterCountry")?.value || "");
  const params = new URLSearchParams({
    recent_year: String(yr),
    baseline_window: String(bw),
  });
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
  if (dlHtml) { dlHtml.href = htmlUrl; dlHtml.download = `weekly-atlas-${yr}.html`; }
  if (dlMd) { dlMd.href = mdUrl; dlMd.download = `weekly-atlas-${yr}.md`; }
  if (dlTxt) { dlTxt.href = txtUrl; dlTxt.download = `weekly-atlas-${yr}.txt`; }
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
      focusCountry && drift.country_a === focusCountry
        ? `${drift.country_a} ↔ ${drift.country_b}`
        : focusCountry && drift.country_b === focusCountry
          ? `${drift.country_b} ↔ ${drift.country_a}`
          : `${drift.country_a} ↔ ${drift.country_b}`;

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
      if (aInput) aInput.value = a;
      if (bInput) bInput.value = b;
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
  const recentYear = toFiniteNumber(
    document.getElementById("driftRecentYear")?.value,
    state.endYear,
  );
  const baselineWindow = toFiniteNumber(
    document.getElementById("driftBaselineWindow")?.value,
    5,
  );
  try {
    const params = new URLSearchParams({
      recent_year: String(recentYear),
      baseline_window: String(baselineWindow),
      top: "6",
    });
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

  const recentYear = toFiniteNumber(
    document.getElementById("driftRecentYear")?.value,
    state.endYear,
  );
  const baselineWindow = toFiniteNumber(
    document.getElementById("driftBaselineWindow")?.value,
    5,
  );
  const direction = document.getElementById("driftDirection")?.value || "all";

  try {
    const params = new URLSearchParams({
      recent_year: String(recentYear),
      baseline_window: String(baselineWindow),
      direction,
      top: "12",
    });
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
  if (a) a.value = state.profileCountry;
  if (b) b.value = peerCode;
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

async function runAnalysis() {
  try {
    // Profile is the primary landing view — always refresh it so the headline
    // story matches the current year range, even when the user is on another tab.
    await loadCountryProfile();

    // Run core analytics sequentially to avoid backend slot contention (429 busy).
    await loadClustering();
    await loadPCAPlot();
    await loadIssueTimeline();
    await loadInsights();

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
  if (input) input.value = state.profileCountry;

  const submit = () => {
    const code = normalizeCodeInput(input ? input.value : state.profileCountry);
    if (code.length !== 3) return;
    state.profileCountry = code;
    if (input) input.value = code;
    writeHashState();
    loadCountryProfile();
  };

  if (loadBtn) loadBtn.addEventListener("click", submit);
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
      if (input) input.value = code;
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

  ["countryA", "countryB"].forEach((fieldId) => {
    const field = document.getElementById(fieldId);
    field.addEventListener("input", () => {
      field.value = normalizeCodeInput(field.value);
    });
  });

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
    setupDriftControls();
    setupCoalitionControls();
    setupNewsletterControls();
    setupTabs();
    setupMobileControls();
    activateTab(state.activeTab, { skipLoad: true });

    await loadDataSummary();
    await Promise.all([loadIssues(), loadMethods(), loadKnownEvents()]);
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
