// ReviewGuard Enterprise Client Engine
let modelWeights = null;
let sampleData = [];

// Cliché Phrases
const COMMON_FAKE_PHRASES = [
  "best product ever",
  "highly recommend",
  "sponsored",
  "discount for review",
  "free sample",
  "life changing",
  "five stars"
];

// Presets
const PRESETS = {
  p1: "Best product ever!!! Totally changed my life. HIGHLY RECOMMEND!!!",
  p2: "Use my discount code for more!! Sponsored but opinions are my own",
  p3: "I got a free sample for my honest review and wow this is amazing",
  p4: "The build quality is decent for the price, but the battery drains fast.",
  p5: "Arrived on time and works as expected. Packaging could be better."
};

const POS_WORDS = new Set(["best", "great", "amazing", "love", "good", "excellent", "comfortable", "responsive", "perfect", "honest", "nice", "wonderful"]);
const NEG_WORDS = new Set(["bad", "worst", "drains", "fast", "terrible", "poor", "horrible", "defect", "broken", "awful"]);

function cleanText(s) {
  if (!s) return "";
  let text = String(s).toLowerCase();
  text = text.replace(/https?:\/\/\S+|www\.\S+/g, " ");
  text = text.replace(/<.*?>/g, " ");
  text = text.replace(/[^a-zA-Z\s]/g, " ");
  text = text.replace(/\s+/g, " ").trim();
  return text;
}

function extractNumericFeatures(s) {
  const text = String(s || "");
  const words = text.split(/\s+/).filter(w => w.length > 0);
  
  const exclamations = (text.match(/!/g) || []).length;
  const allCapsTokens = words.filter(w => {
    const cleanW = w.replace(/[^a-zA-Z]/g, "");
    return cleanW.length > 3 && cleanW === cleanW.toUpperCase();
  }).length;
  
  const lowerText = text.toLowerCase();
  let repeatedPhrases = 0;
  COMMON_FAKE_PHRASES.forEach(p => {
    if (lowerText.includes(p)) repeatedPhrases++;
  });
  
  let posCount = 0;
  let negCount = 0;
  words.forEach(w => {
    const clean = w.toLowerCase().replace(/[^a-z]/g, "");
    if (POS_WORDS.has(clean)) posCount++;
    if (NEG_WORDS.has(clean)) negCount++;
  });
  const sentiment = words.length > 0 ? (posCount - negCount) / Math.max(1, posCount + negCount) : 0.0;
  const charLength = text.length;
  const uniqueWords = new Set(words.map(w => w.toLowerCase()));
  const uniqueRatio = words.length > 0 ? uniqueWords.size / words.length : 0.0;

  return {
    sentiment,
    exclamation_count: exclamations,
    all_caps_tokens: allCapsTokens,
    repeated_phrases: repeatedPhrases,
    char_length: charLength,
    unique_word_ratio: uniqueRatio
  };
}

function predictReview(text, threshold = 0.5) {
  const numFeats = extractNumericFeatures(text);
  
  if (!modelWeights || !modelWeights.vocabulary) {
    let score = 0.05;
    if (numFeats.repeated_phrases > 0) score += 0.45 * numFeats.repeated_phrases;
    if (numFeats.exclamation_count >= 2) score += 0.25;
    if (numFeats.all_caps_tokens >= 1) score += 0.2;
    const clamped = Math.min(Math.max(score, 0.01), 0.99);
    return { prob: clamped, isFake: clamped >= threshold, features: numFeats, z: 0 };
  }
  
  const cleaned = cleanText(text);
  const words = cleaned.split(" ").filter(w => w.length > 0);
  
  const ngrams = [];
  for (let i = 0; i < words.length; i++) {
    ngrams.push(words[i]);
    if (i < words.length - 1) {
      ngrams.push(`${words[i]} ${words[i+1]}`);
    }
  }
  
  const tf = {};
  ngrams.forEach(ng => {
    tf[ng] = (tf[ng] || 0) + 1;
  });
  
  // Scikit-Learn TfidfVectorizer L2 normalization calculation
  const vec = {};
  let normSq = 0.0;
  Object.keys(tf).forEach(term => {
    if (modelWeights.vocabulary.hasOwnProperty(term)) {
      const idx = modelWeights.vocabulary[term];
      const idfVal = modelWeights.idf[idx];
      const val = tf[term] * idfVal;
      vec[term] = { idx, val };
      normSq += val * val;
    }
  });
  
  const norm = Math.sqrt(normSq);
  let z = modelWeights.intercept || 0;
  
  // TF-IDF dot product with L2 normalized vector
  Object.keys(vec).forEach(term => {
    const item = vec[term];
    const normedVal = norm > 0 ? item.val / norm : 0;
    const weight = modelWeights.coef[item.idx];
    z += normedVal * weight;
  });
  
  // Numeric features contribution scaled by StandardScaler scale_
  const numWeights = modelWeights.coef.slice(-6);
  const scale = modelWeights.scale || [0.164, 2.58, 0.458, 0.601, 8.087, 0.073];
  
  const numVals = [
    numFeats.sentiment,
    numFeats.exclamation_count,
    numFeats.all_caps_tokens,
    numFeats.repeated_phrases,
    numFeats.char_length,
    numFeats.unique_word_ratio
  ];
  
  for (let i = 0; i < 6; i++) {
    const scaled = scale[i] > 0 ? numVals[i] / scale[i] : numVals[i];
    z += scaled * (numWeights[i] || 0);
  }
  
  const prob = 1 / (1 + Math.exp(-z));
  const isFake = prob >= threshold;
  
  return {
    prob: Math.min(Math.max(prob, 0.001), 0.999),
    isFake,
    features: numFeats,
    z
  };
}

function highlightSuspicious(text) {
  if (!text) return "No review text entered.";
  let safe = String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
    
  COMMON_FAKE_PHRASES.forEach(phrase => {
    const reg = new RegExp(`(${phrase})`, "gi");
    safe = safe.replace(reg, '<span class="highlight-cliche">$1</span>');
  });
  
  const tokens = safe.split(/\s+/);
  const processedTokens = tokens.map(t => {
    const clean = t.replace(/[^a-zA-Z]/g, "");
    if (clean.length > 3 && clean === clean.toUpperCase() && !t.includes("class=")) {
      return `<span class="highlight-caps">${t}</span>`;
    }
    return t;
  });
  
  return processedTokens.join(" ");
}

document.addEventListener("DOMContentLoaded", async () => {
  try {
    const res = await fetch("./model_weights.json");
    if (res.ok) {
      modelWeights = await res.json();
    }
  } catch (err) {
    console.warn("Model weights fetch warning:", err);
  }

  try {
    const sampleRes = await fetch("./reviews_sample.csv");
    if (sampleRes.ok) {
      const csvText = await sampleRes.text();
      parseSampleCSV(csvText);
    }
  } catch (err) {
    console.warn("Sample CSV fetch warning:", err);
  }

  // Navigation Tabs
  const navLinks = document.querySelectorAll(".nav-link");
  navLinks.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetView = link.getAttribute("data-view");
      
      navLinks.forEach(n => n.classList.remove("active"));
      link.classList.add("active");
      
      document.querySelectorAll(".tab-content").forEach(tab => tab.classList.remove("active"));
      const activeTab = document.getElementById(targetView);
      if (activeTab) activeTab.classList.add("active");
      
      const titles = {
        "tab-analyzer": ["Review Inspection", "Real-time linguistic, behavioral, and statistical classification"],
        "tab-batch": ["Batch Pipeline", "Bulk CSV verification and distribution analytics"],
        "tab-metrics": ["Evaluation Metrics", "Model diagnostic curves and statistical benchmarks"],
        "tab-dataset": ["Dataset Explorer", "Explore and filter baseline ground-truth records"],
        "tab-architecture": ["Model Specs", "Pipeline parameters and heuristic vectorization details"]
      };
      
      if (titles[targetView]) {
        document.getElementById("page-title").textContent = titles[targetView][0];
        document.getElementById("page-desc").textContent = titles[targetView][1];
      }
    });
  });

  // Presets
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const pKey = btn.getAttribute("data-preset");
      if (PRESETS[pKey]) {
        document.getElementById("review-input").value = PRESETS[pKey];
        runLiveAnalysis();
      }
    });
  });

  // Threshold slider live inspection
  const thresholdSlider = document.getElementById("threshold-slider");
  const thresholdVal = document.getElementById("threshold-val");
  if (thresholdSlider) {
    thresholdSlider.addEventListener("input", (e) => {
      thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
      runLiveAnalysis();
    });
  }

  const btnAnalyze = document.getElementById("btn-analyze");
  if (btnAnalyze) {
    btnAnalyze.addEventListener("click", runLiveAnalysis);
  }
  
  // Batch Threshold slider
  const batchThresholdSlider = document.getElementById("batch-threshold");
  const batchThresholdVal = document.getElementById("batch-threshold-val");
  if (batchThresholdSlider && batchThresholdVal) {
    batchThresholdSlider.addEventListener("input", (e) => {
      batchThresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  // Batch CSV Handlers
  const batchFileInput = document.getElementById("batch-file-input");
  const btnBatchSample = document.getElementById("btn-batch-sample");
  const btnRunBatch = document.getElementById("btn-run-batch");
  
  if (btnBatchSample) {
    btnBatchSample.addEventListener("click", () => {
      if (sampleData && sampleData.length > 0) {
        window.batchLoadedRows = sampleData;
        populateColumnSelect(sampleData);
        document.getElementById("batch-status-msg").innerHTML = `<span style="color: var(--status-verified)">✓ Loaded ${sampleData.length} records ready for batch analysis</span>`;
        if (btnRunBatch) btnRunBatch.disabled = false;
      } else {
        document.getElementById("batch-status-msg").innerHTML = `<span style="color: var(--status-flagged)">Loading sample data... please try in 1s.</span>`;
      }
    });
  }

  if (batchFileInput) {
    batchFileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        const textContent = evt.target.result;
        const parsedRows = parseCSVFull(textContent);
        if (parsedRows.length > 0) {
          window.batchLoadedRows = parsedRows;
          populateColumnSelect(parsedRows);
          document.getElementById("batch-status-msg").innerHTML = `<span style="color: var(--status-verified)">✓ Uploaded ${parsedRows.length} rows from "${file.name}"</span>`;
          if (btnRunBatch) btnRunBatch.disabled = false;
        } else {
          document.getElementById("batch-status-msg").innerHTML = `<span style="color: var(--status-flagged)">Could not parse records from CSV.</span>`;
        }
      };
      reader.readAsText(file);
    });
  }

  if (btnRunBatch) {
    btnRunBatch.addEventListener("click", runBatchInference);
  }

  const dSearch = document.getElementById("dataset-search");
  const dFilter = document.getElementById("dataset-filter-label");
  if (dSearch) dSearch.addEventListener("input", filterDatasetTable);
  if (dFilter) dFilter.addEventListener("change", filterDatasetTable);

  runLiveAnalysis();
});

function runLiveAnalysis() {
  const inputEl = document.getElementById("review-input");
  if (!inputEl) return;
  const text = inputEl.value;
  const threshold = parseFloat(document.getElementById("threshold-slider")?.value || 0.5);
  const result = predictReview(text, threshold);
  
  const verdictBox = document.getElementById("verdict-box");
  const verdictTag = document.getElementById("verdict-tag");
  const meterFill = document.getElementById("meter-fill");
  const verdictScore = document.getElementById("verdict-score");
  const verdictDesc = document.getElementById("verdict-desc");
  
  if (verdictBox && verdictTag && meterFill && verdictScore && verdictDesc) {
    if (result.isFake) {
      verdictBox.className = "verdict-box flagged";
      verdictTag.className = "verdict-tag flagged";
      verdictTag.textContent = "FLAGGED: SUSPICIOUS";
      meterFill.className = "meter-fill flagged";
      verdictDesc.textContent = "High risk score detected from linguistic markers and repeated promotional phrasing.";
    } else {
      verdictBox.className = "verdict-box verified";
      verdictTag.className = "verdict-tag verified";
      verdictTag.textContent = "VERIFIED: AUTHENTIC";
      meterFill.className = "meter-fill verified";
      verdictDesc.textContent = "Linguistic markers and balanced sentiment indicate an organic customer review.";
    }
    
    const pct = (result.prob * 100).toFixed(1);
    verdictScore.textContent = `${pct}%`;
    meterFill.style.width = `${pct}%`;
  }
  
  const sentEl = document.getElementById("metric-sentiment");
  const capsEl = document.getElementById("metric-caps");
  const excEl = document.getElementById("metric-exclamations");
  const cliEl = document.getElementById("metric-cliches");
  const hlEl = document.getElementById("highlight-box");

  if (sentEl) sentEl.textContent = result.features.sentiment.toFixed(2);
  if (capsEl) capsEl.textContent = result.features.all_caps_tokens;
  if (excEl) excEl.textContent = result.features.exclamation_count;
  if (cliEl) cliEl.textContent = result.features.repeated_phrases;
  if (hlEl) hlEl.innerHTML = highlightSuspicious(text);
}

function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      inQuotes = !inQuotes;
    } else if (c === ',' && !inQuotes) {
      result.push(current.trim().replace(/^"|"$/g, ''));
      current = '';
    } else {
      current += c;
    }
  }
  result.push(current.trim().replace(/^"|"$/g, ''));
  return result;
}

function parseCSVFull(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
  if (lines.length < 2) return [];
  const headers = parseCSVLine(lines[0]);
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    if (values.length > 0) {
      const row = {};
      headers.forEach((h, idx) => {
        row[h] = values[idx] || "";
      });
      rows.push(row);
    }
  }
  return rows;
}

function parseSampleCSV(text) {
  sampleData = parseCSVFull(text);
  if (sampleData.length === 0) {
    const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
    for (let i = 1; i < lines.length; i++) {
      const values = parseCSVLine(lines[i]);
      if (values.length >= 2) {
        sampleData.push({ text: values[0], label: values[1] });
      }
    }
  }
  renderDatasetTable(sampleData);
}

function populateColumnSelect(rows) {
  if (!rows || rows.length === 0) return;
  const colSelect = document.getElementById("batch-col-select");
  if (!colSelect) return;
  colSelect.innerHTML = "";
  const keys = Object.keys(rows[0]);
  keys.forEach(k => {
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = k;
    if (k.toLowerCase().includes("text") || k.toLowerCase().includes("review")) {
      opt.selected = true;
    }
    colSelect.appendChild(opt);
  });
}

function runBatchInference() {
  const rows = window.batchLoadedRows;
  if (!rows || rows.length === 0) {
    alert("Please upload a CSV or load the sample benchmark dataset first.");
    return;
  }
  
  const colSelect = document.getElementById("batch-col-select");
  const col = colSelect ? colSelect.value : Object.keys(rows[0])[0];
  const thr = parseFloat(document.getElementById("batch-threshold")?.value || 0.5);
  
  let fakeCount = 0;
  let realCount = 0;
  let totalProb = 0;
  
  const results = rows.map(r => {
    const text = String(r[col] || Object.values(r)[0] || "");
    const pred = predictReview(text, thr);
    if (pred.isFake) fakeCount++;
    else realCount++;
    totalProb += pred.prob;
    return {
      ...r,
      _review_text: text,
      predicted_label: pred.isFake ? "FLAGGED" : "AUTHENTIC",
      fake_probability: (pred.prob * 100).toFixed(1) + "%",
      _prob_num: pred.prob
    };
  });
  
  window.lastBatchResults = results;
  
  const totalEl = document.getElementById("batch-total-count");
  const fakeEl = document.getElementById("batch-fake-count");
  const realEl = document.getElementById("batch-real-count");
  const avgEl = document.getElementById("batch-avg-prob");
  
  if (totalEl) totalEl.textContent = results.length;
  if (fakeEl) fakeEl.textContent = fakeCount;
  if (realEl) realEl.textContent = realCount;
  if (avgEl) avgEl.textContent = ((totalProb / results.length) * 100).toFixed(1) + "%";
  
  renderBatchTable(results);
  const dash = document.getElementById("batch-results-dashboard");
  if (dash) dash.style.display = "block";
}

function renderBatchTable(data) {
  const tbody = document.getElementById("batch-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";
  
  data.slice(0, 100).forEach(row => {
    const tr = document.createElement("tr");
    const isFake = row.predicted_label === "FLAGGED";
    const text = row._review_text || row.text || Object.values(row)[0];
    tr.innerHTML = `
      <td style="max-width: 500px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.85rem;">${escapeHtml(text)}</td>
      <td><span class="verdict-tag ${isFake ? 'flagged' : 'verified'}">${row.predicted_label}</span></td>
      <td style="font-family: var(--font-mono); font-weight: 600;">${row.fake_probability}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderDatasetTable(data) {
  const tbody = document.getElementById("dataset-table-body");
  if (!tbody || !data) return;
  tbody.innerHTML = "";
  
  const totalEl = document.getElementById("dataset-total-count");
  const realEl = document.getElementById("dataset-real-count");
  const fakeEl = document.getElementById("dataset-fake-count");
  
  let real = 0, fake = 0;
  data.forEach(r => {
    const label = String(r.label || "").toUpperCase();
    if (label === "FAKE" || label === "1") fake++;
    else real++;
  });
  
  if (totalEl) totalEl.textContent = data.length;
  if (realEl) realEl.textContent = real;
  if (fakeEl) fakeEl.textContent = fake;
  
  data.forEach(r => {
    const tr = document.createElement("tr");
    const isFake = String(r.label || "").toUpperCase() === "FAKE";
    tr.innerHTML = `
      <td>${escapeHtml(r.text || "")}</td>
      <td><span class="verdict-tag ${isFake ? 'flagged' : 'verified'}">${isFake ? 'SYNTHETIC' : 'AUTHENTIC'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function filterDatasetTable() {
  const q = (document.getElementById("dataset-search")?.value || "").toLowerCase();
  const filterLabel = document.getElementById("dataset-filter-label")?.value || "ALL";
  
  const filtered = sampleData.filter(r => {
    const text = String(r.text || "").toLowerCase();
    const label = String(r.label || "").toUpperCase();
    const matchText = text.includes(q);
    const matchLabel = filterLabel === "ALL" || (filterLabel === "REAL" && (label === "REAL" || label === "0")) || (filterLabel === "FAKE" && (label === "FAKE" || label === "1"));
    return matchText && matchLabel;
  });
  
  const tbody = document.getElementById("dataset-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";
  filtered.forEach(r => {
    const tr = document.createElement("tr");
    const isFake = String(r.label || "").toUpperCase() === "FAKE";
    tr.innerHTML = `
      <td>${escapeHtml(r.text || "")}</td>
      <td><span class="verdict-tag ${isFake ? 'flagged' : 'verified'}">${isFake ? 'SYNTHETIC' : 'AUTHENTIC'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
