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
  let text = s.toLowerCase();
  text = text.replace(/https?:\/\/\S+|www\.\S+/g, " ");
  text = text.replace(/<.*?>/g, " ");
  text = text.replace(/[^a-zA-Z\s]/g, " ");
  text = text.replace(/\s+/g, " ").trim();
  return text;
}

function extractNumericFeatures(s) {
  const text = s || "";
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
  if (!modelWeights) return { prob: 0.5, isFake: false, features: extractNumericFeatures(text) };
  
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
  
  let z = modelWeights.intercept;
  
  Object.keys(tf).forEach(term => {
    if (modelWeights.vocabulary.hasOwnProperty(term)) {
      const idx = modelWeights.vocabulary[term];
      const idfVal = modelWeights.idf[idx];
      const tfidfVal = tf[term] * idfVal;
      const weight = modelWeights.coef[idx];
      z += tfidfVal * weight;
    }
  });
  
  const numFeats = extractNumericFeatures(text);
  const numWeights = modelWeights.coef.slice(-6);
  
  z += numFeats.sentiment * numWeights[0];
  z += numFeats.exclamation_count * numWeights[1];
  z += numFeats.all_caps_tokens * numWeights[2];
  z += numFeats.repeated_phrases * numWeights[3];
  z += (numFeats.char_length / 100) * numWeights[4];
  z += numFeats.unique_word_ratio * numWeights[5];
  
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
  let safe = text
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
    modelWeights = await res.json();
  } catch (err) {
    console.warn("Model weights load status:", err);
  }

  try {
    const sampleRes = await fetch("./reviews_sample.csv");
    const csvText = await sampleRes.text();
    parseSampleCSV(csvText);
  } catch (err) {
    console.warn("Sample CSV load status:", err);
  }

  // Navigation Links
  const navLinks = document.querySelectorAll(".nav-link");
  navLinks.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetView = link.getAttribute("data-view");
      
      navLinks.forEach(n => n.classList.remove("active"));
      link.classList.add("active");
      
      document.querySelectorAll(".tab-content").forEach(tab => tab.classList.remove("active"));
      document.getElementById(targetView).classList.add("active");
      
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

  // Threshold Slider
  const thresholdSlider = document.getElementById("threshold-slider");
  const thresholdVal = document.getElementById("threshold-val");
  thresholdSlider.addEventListener("input", (e) => {
    thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  document.getElementById("btn-analyze").addEventListener("click", runLiveAnalysis);
  runLiveAnalysis();

  // Batch CSV
  const batchFileInput = document.getElementById("batch-file-input");
  const btnBatchSample = document.getElementById("btn-batch-sample");
  const btnRunBatch = document.getElementById("btn-run-batch");
  
  btnBatchSample.addEventListener("click", () => {
    if (sampleData.length > 0) {
      window.batchLoadedRows = sampleData;
      populateColumnSelect(sampleData);
      document.getElementById("batch-status-msg").innerHTML = `<span style="color: var(--status-verified)">✓ Loaded ${sampleData.length} records</span>`;
      btnRunBatch.disabled = false;
    }
  });

  batchFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      const lines = evt.target.result.split(/\r?\n/).filter(l => l.trim().length > 0);
      const rows = [];
      const headers = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, ""));
      for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        if (values.length >= headers.length) {
          const row = {};
          headers.forEach((h, idx) => row[h] = values[idx]);
          rows.push(row);
        }
      }
      window.batchLoadedRows = rows;
      populateColumnSelect(rows);
      document.getElementById("batch-status-msg").innerHTML = `<span style="color: var(--status-verified)">✓ Loaded ${rows.length} rows</span>`;
      btnRunBatch.disabled = false;
    };
    reader.readAsText(file);
  });

  btnRunBatch.addEventListener("click", runBatchInference);
  document.getElementById("dataset-search").addEventListener("input", filterDatasetTable);
  document.getElementById("dataset-filter-label").addEventListener("change", filterDatasetTable);
});

function runLiveAnalysis() {
  const text = document.getElementById("review-input").value;
  const threshold = parseFloat(document.getElementById("threshold-slider").value);
  const result = predictReview(text, threshold);
  
  const verdictBox = document.getElementById("verdict-box");
  const verdictTag = document.getElementById("verdict-tag");
  const meterFill = document.getElementById("meter-fill");
  const verdictScore = document.getElementById("verdict-score");
  const verdictDesc = document.getElementById("verdict-desc");
  
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
  
  document.getElementById("metric-sentiment").textContent = result.features.sentiment.toFixed(2);
  document.getElementById("metric-caps").textContent = result.features.all_caps_tokens;
  document.getElementById("metric-exclamations").textContent = result.features.exclamation_count;
  document.getElementById("metric-cliches").textContent = result.features.repeated_phrases;
  document.getElementById("highlight-box").innerHTML = highlightSuspicious(text);
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

function parseSampleCSV(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
  sampleData = [];
  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    if (values.length >= 2) {
      sampleData.push({ text: values[0], label: values[1] });
    }
  }
  renderDatasetTable(sampleData);
}

function populateColumnSelect(rows) {
  if (!rows || rows.length === 0) return;
  const colSelect = document.getElementById("batch-col-select");
  colSelect.innerHTML = "";
  Object.keys(rows[0]).forEach(k => {
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
  if (!rows) return;
  const col = document.getElementById("batch-col-select").value;
  const thr = parseFloat(document.getElementById("batch-threshold").value);
  
  let fakeCount = 0;
  let realCount = 0;
  let totalProb = 0;
  
  const results = rows.map(r => {
    const text = r[col] || "";
    const pred = predictReview(text, thr);
    if (pred.isFake) fakeCount++;
    else realCount++;
    totalProb += pred.prob;
    return {
      ...r,
      predicted_label: pred.isFake ? "FLAGGED" : "AUTHENTIC",
      fake_probability: (pred.prob * 100).toFixed(1) + "%"
    };
  });
  
  window.lastBatchResults = results;
  
  document.getElementById("batch-total-count").textContent = results.length;
  document.getElementById("batch-fake-count").textContent = fakeCount;
  document.getElementById("batch-real-count").textContent = realCount;
  document.getElementById("batch-avg-prob").textContent = ((totalProb / results.length) * 100).toFixed(1) + "%";
  
  renderBatchTable(results);
  document.getElementById("batch-results-dashboard").style.display = "block";
}

function renderBatchTable(data) {
  const tbody = document.getElementById("batch-table-body");
  tbody.innerHTML = "";
  data.slice(0, 50).forEach(row => {
    const tr = document.createElement("tr");
    const isFake = row.predicted_label === "FLAGGED";
    tr.innerHTML = `
      <td style="max-width: 480px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.85rem;">${escapeHtml(row.text || Object.values(row)[0])}</td>
      <td><span class="verdict-tag ${isFake ? 'flagged' : 'verified'}">${row.predicted_label}</span></td>
      <td style="font-family: var(--font-mono); font-weight: 600;">${row.fake_probability}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderDatasetTable(data) {
  const tbody = document.getElementById("dataset-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";
  document.getElementById("dataset-total-count").textContent = data.length;
  
  let real = 0, fake = 0;
  data.forEach(r => {
    if (r.label.toUpperCase() === "FAKE") fake++;
    else real++;
  });
  document.getElementById("dataset-real-count").textContent = real;
  document.getElementById("dataset-fake-count").textContent = fake;
  
  data.forEach(r => {
    const tr = document.createElement("tr");
    const isFake = r.label.toUpperCase() === "FAKE";
    tr.innerHTML = `
      <td>${escapeHtml(r.text)}</td>
      <td><span class="verdict-tag ${isFake ? 'flagged' : 'verified'}">${isFake ? 'SYNTHETIC' : 'AUTHENTIC'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function filterDatasetTable() {
  const q = document.getElementById("dataset-search").value.toLowerCase();
  const filterLabel = document.getElementById("dataset-filter-label").value;
  
  const filtered = sampleData.filter(r => {
    const matchText = r.text.toLowerCase().includes(q);
    const matchLabel = filterLabel === "ALL" || (filterLabel === "REAL" && r.label.toUpperCase() === "REAL") || (filterLabel === "FAKE" && r.label.toUpperCase() === "FAKE");
    return matchText && matchLabel;
  });
  
  const tbody = document.getElementById("dataset-table-body");
  tbody.innerHTML = "";
  filtered.forEach(r => {
    const tr = document.createElement("tr");
    const isFake = r.label.toUpperCase() === "FAKE";
    tr.innerHTML = `
      <td>${escapeHtml(r.text)}</td>
      <td><span class="verdict-tag ${isFake ? 'flagged' : 'verified'}">${isFake ? 'SYNTHETIC' : 'AUTHENTIC'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
