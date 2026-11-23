const els = {
  pillGpu: document.getElementById("pill-gpu"),
  pillEngine: document.getElementById("pill-engine"),
  pillTta: document.getElementById("pill-tta"),
  modeSelect: document.getElementById("mode-select"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  btnBrowse: document.getElementById("btn-browse"),
  loadingBox: document.getElementById("loading-box"),
  errorBox: document.getElementById("error-box"),
  emptyState: document.getElementById("empty-state"),
  resultsContent: document.getElementById("results-content"),
  verdictBanner: document.getElementById("verdict-banner"),
  verdictBadge: document.getElementById("verdict-badge"),
  verdictTitle: document.getElementById("verdict-title"),
  verdictMeta: document.getElementById("verdict-meta"),
  gaugeValue: document.getElementById("gauge-value"),
  fakeProbText: document.getElementById("fake-prob-text"),
  fakeProbBar: document.getElementById("fake-prob-bar"),
  realProbText: document.getElementById("real-prob-text"),
  realProbBar: document.getElementById("real-prob-bar"),
  mNeural: document.getElementById("m-neural"),
  mFft: document.getElementById("m-fft"),
  mEla: document.getElementById("m-ela"),
  mSeam: document.getElementById("m-seam"),
  votesSection: document.getElementById("votes-section"),
  votesList: document.getElementById("votes-list"),
  videoTimelineSection: document.getElementById("video-timeline-section"),
  videoTimelineSub: document.getElementById("video-timeline-sub"),
  timelineBars: document.getElementById("timeline-bars"),
  imgPreview: document.getElementById("img-preview"),
  imgGradcam: document.getElementById("img-gradcam"),
  imgFft: document.getElementById("img-fft"),
  imgEla: document.getElementById("img-ela"),
  trainModal: document.getElementById("train-modal"),
  btnOpenTrain: document.getElementById("btn-open-train"),
  btnCloseTrain: document.getElementById("btn-close-train"),
  btnReloadEngine: document.getElementById("btn-reload-engine"),
};

async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    els.pillGpu.textContent = data.cuda_available
      ? `GPU: ${data.gpu_name} (${data.vram_gb} GB)`
      : "Device: CPU Mode";
    els.pillEngine.textContent = `Engine: ${data.active_engine}`;
    els.pillTta.textContent = `TTA: ${data.tta_views}x Multi-Crop Active`;
  } catch {
    els.pillGpu.textContent = "Status: Offline";
  }
}

function setLoading(isLoading) {
  els.loadingBox.classList.toggle("hidden", !isLoading);
  if (isLoading) {
    els.errorBox.classList.add("hidden");
  }
}

function showError(msg) {
  els.errorBox.textContent = msg;
  els.errorBox.classList.remove("hidden");
}

function renderResult(data) {
  els.emptyState.classList.add("hidden");
  els.resultsContent.classList.remove("hidden");

  const isFake = Boolean(data.is_fake);
  els.verdictBanner.classList.toggle("is-fake", isFake);
  els.verdictBanner.classList.toggle("is-real", !isFake);
  els.verdictBadge.textContent = isFake ? "DEEPFAKE DETECTED" : "AUTHENTIC MEDIA";
  els.verdictTitle.textContent = isFake
    ? "Synthetic / Manipulated Facial Artifacts Found"
    : "Consistent With Authentic Camera Capture";

  const isVideo = data.media_type === "video";
  const frameSource = isVideo ? data.peak_frame : data;

  const faceTag = frameSource.face_detected
    ? (frameSource.already_tight_crop ? "Tight face crop detected" : "Auto-cropped face (+22% margin)")
    : "Center fallback crop";

  els.verdictMeta.textContent = isVideo
    ? `${data.filename} · ${data.frames_analyzed} frames (${data.duration_sec}s) · Peak frame @ ${frameSource.peak_timestamp_sec}s · ${data.engine}`
    : `${data.filename || "Image"} · ${faceTag} · ${data.engine}`;

  els.gaugeValue.textContent = `${(data.confidence * 100).toFixed(1)}%`;

  const fakePct = (data.fake_probability * 100).toFixed(1);
  const realPct = (data.real_probability * 100).toFixed(1);
  els.fakeProbText.textContent = `${fakePct}%`;
  els.fakeProbBar.style.width = `${fakePct}%`;
  els.realProbText.textContent = `${realPct}%`;
  els.realProbBar.style.width = `${realPct}%`;

  const f = frameSource.forensics;
  els.mNeural.textContent = `${(f.neural_fake_score * 100).toFixed(1)}%`;
  els.mFft.textContent = `${(f.fft_spectral_score * 100).toFixed(1)}%`;
  els.mEla.textContent = `${(f.ela_compression_score * 100).toFixed(1)}%`;
  els.mSeam.textContent = `${(f.seam_consistency_score * 100).toFixed(1)}%`;

  if (Array.isArray(data.model_votes) && data.model_votes.length > 0) {
    els.votesSection.classList.remove("hidden");
    els.votesList.innerHTML = "";
    data.model_votes.forEach((v) => {
      const pct = (v.fake_probability * 100).toFixed(1);
      const isFakeVote = v.fake_probability >= 0.5;
      const row = document.createElement("div");
      row.className = "vote-row";
      row.innerHTML = `
        <div class="vote-header">
          <span>${v.name}</span>
          <strong>${pct}% Fake</strong>
        </div>
        <div class="vote-track">
          <div class="vote-fill" style="width: ${pct}%; background: ${isFakeVote ? "#ef4444" : "#10b981"}"></div>
        </div>
      `;
      els.votesList.appendChild(row);
    });
  } else {
    els.votesSection.classList.add("hidden");
  }

  if (isVideo && Array.isArray(data.timeline)) {
    els.videoTimelineSection.classList.remove("hidden");
    els.videoTimelineSub.textContent = `Showing fake probability across ${data.timeline.length} sampled frames (visuals below correspond to peak suspicious frame at ${frameSource.peak_timestamp_sec}s).`;
    els.timelineBars.innerHTML = "";
    data.timeline.forEach((pt) => {
      const col = document.createElement("div");
      col.className = "timeline-bar-col";
      const bar = document.createElement("div");
      bar.className = "timeline-bar";
      const pct = Math.max(6, Math.round(pt.fake_probability * 100));
      bar.style.height = `${pct}%`;
      bar.style.background = pt.fake_probability >= 0.5 ? "#ef4444" : "#10b981";
      bar.title = `Frame ${pt.frame_index} (${pt.timestamp_sec}s): ${(pt.fake_probability * 100).toFixed(1)}% Fake`;
      const ts = document.createElement("span");
      ts.className = "timeline-ts";
      ts.textContent = `${pt.timestamp_sec}s`;
      col.appendChild(bar);
      col.appendChild(ts);
      els.timelineBars.appendChild(col);
    });
  } else {
    els.videoTimelineSection.classList.add("hidden");
  }

  const v = frameSource.visuals;
  els.imgPreview.src = `data:image/png;base64,${v.annotated_preview_b64}`;
  els.imgGradcam.src = `data:image/png;base64,${v.gradcam_b64}`;
  els.imgFft.src = `data:image/png;base64,${v.fft_spectrum_b64}`;
  els.imgEla.src = `data:image/png;base64,${v.ela_map_b64}`;
}

async function uploadFile(file) {
  if (!file) return;
  const isVideo = file.type.startsWith("video/") || /\.(mp4|mov|avi|webm|mkv)$/i.test(file.name);
  const endpoint = isVideo ? "/api/analyze/video" : "/api/analyze/image";
  const form = new FormData();
  form.append("file", file);
  form.append("mode", els.modeSelect?.value || "ensemble");

  setLoading(true);
  try {
    const res = await fetch(endpoint, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Analysis failed");
    renderResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

async function runSample(sampleId) {
  setLoading(true);
  try {
    const mode = encodeURIComponent(els.modeSelect?.value || "ensemble");
    const res = await fetch(`/api/analyze/sample/${sampleId}?mode=${mode}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Sample analysis failed");
    renderResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

// Event Listeners
els.btnBrowse.addEventListener("click", (e) => {
  e.stopPropagation();
  els.fileInput.click();
});
els.dropzone.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", (e) => {
  if (e.target.files?.[0]) uploadFile(e.target.files[0]);
});

els.dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  els.dropzone.classList.add("dragover");
});
els.dropzone.addEventListener("dragleave", () => els.dropzone.classList.remove("dragover"));
els.dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  els.dropzone.classList.remove("dragover");
  if (e.dataTransfer.files?.[0]) uploadFile(e.dataTransfer.files[0]);
});

document.querySelectorAll("[data-sample]").forEach((btn) => {
  btn.addEventListener("click", () => runSample(btn.getAttribute("data-sample")));
});

els.btnOpenTrain.addEventListener("click", () => els.trainModal.classList.remove("hidden"));
els.btnCloseTrain.addEventListener("click", () => els.trainModal.classList.add("hidden"));
els.trainModal.querySelector(".modal-backdrop").addEventListener("click", () => {
  els.trainModal.classList.add("hidden");
});

els.btnReloadEngine.addEventListener("click", async () => {
  await fetch("/api/reload", { method: "POST" });
  await fetchStatus();
  els.trainModal.classList.add("hidden");
});

fetchStatus();
