// DocGuard Client-Side Controller

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const fileInfoCard = document.getElementById('fileInfoCard');
  const fileName = document.getElementById('fileName');
  const clearFileBtn = document.getElementById('clearFileBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const spinner = document.getElementById('spinner');
  const btnText = analyzeBtn.querySelector('.btn-text');
  const telemetryText = document.getElementById('telemetryText');

  const emptyState = document.getElementById('emptyState');
  const reportContent = document.getElementById('reportContent');
  const verdictBanner = document.getElementById('verdictBanner');
  const verdictBadge = document.getElementById('verdictBadge');
  const verdictTitle = document.getElementById('verdictTitle');
  const forgeryTypeLabel = document.getElementById('forgeryTypeLabel');
  const confidenceValue = document.getElementById('confidenceValue');

  const viewerImage = document.getElementById('viewerImage');
  const viewerCaption = document.getElementById('viewerCaption');
  const tabButtons = document.querySelectorAll('.tab-btn');

  const regionsSummaryText = document.getElementById('regionsSummaryText');
  const ocrSummaryText = document.getElementById('ocrSummaryText');
  const rationaleText = document.getElementById('rationaleText');
  const downloadPdfBtn = document.getElementById('downloadPdfBtn');
  const sampleButtons = document.querySelectorAll('.sample-btn');

  let currentFile = null;
  let currentAnalysisData = null;

  const CAPTIONS = {
    mask_overlay: 'Pixel-level tampering localization segmentation overlay with detected bounding boxes.',
    gradcam_overlay: 'Grad-CAM deep feature attribution heatmap highlighting discriminator attention.',
    clean_mask: 'Binary output mask (thresholded and morphologically filtered).',
    rectified: 'Document after boundary detection, 4-point perspective warp, and deskewing.',
    original: 'Raw unedited document input as received by the system.',
  };

  // 1. Fetch Backend Telemetry
  async function fetchTelemetry() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        const dev = data.gpu_name ? `${data.gpu_name} (CUDA)` : data.device.toUpperCase();
        telemetryText.textContent = `Online • ${dev}`;
      } else {
        telemetryText.textContent = 'Server Busy';
      }
    } catch (e) {
      telemetryText.textContent = 'Backend Offline';
    }
  }
  fetchTelemetry();

  // 2. Drag & Drop Handling
  dropZone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  function handleSelectedFile(file) {
    currentFile = file;
    fileName.textContent = file.name;
    fileInfoCard.classList.remove('hidden');
    dropZone.classList.add('hidden');
    analyzeBtn.disabled = false;
  }

  clearFileBtn.addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    fileInfoCard.classList.add('hidden');
    dropZone.classList.remove('hidden');
    analyzeBtn.disabled = true;
  });

  // 3. Analyze File Upload
  analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', currentFile);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Analysis request failed');
      }

      const data = await res.json();
      renderAnalysis(data);
    } catch (err) {
      alert(`Error during analysis: ${err.message}`);
    } finally {
      setLoading(false);
    }
  });

  // 4. Controlled Synthetic Test Buttons
  sampleButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      const type = btn.getAttribute('data-type');
      setLoading(true);

      try {
        const res = await fetch(`/api/sample?forgery_type=${type}`, { method: 'POST' });
        if (!res.ok) throw new Error('Failed to generate sample');
        const data = await res.json();
        renderAnalysis(data);
      } catch (err) {
        alert(`Failed to analyze sample: ${err.message}`);
      } finally {
        setLoading(false);
      }
    });
  });

  // 5. Render Results
  function renderAnalysis(data) {
    currentAnalysisData = data;
    emptyState.classList.add('hidden');
    reportContent.classList.remove('hidden');

    const isForged = data.is_forged;
    verdictBanner.className = `verdict-banner ${isForged ? 'forged' : 'authentic'}`;
    verdictBadge.textContent = isForged ? 'TAMPERING DETECTED' : 'AUTHENTICITY VERIFIED';
    verdictTitle.textContent = data.verdict;
    forgeryTypeLabel.textContent = `Identified Type: ${data.forgery_type}`;
    confidenceValue.textContent = `${data.confidence_pct}%`;

    // Populate Findings & Rationale
    regionsSummaryText.textContent = data.regions_summary;
    ocrSummaryText.textContent = data.ocr_summary;
    rationaleText.textContent = data.rationale;

    // PDF Download Link
    downloadPdfBtn.href = `/api/report/${data.report_id}`;

    // Set Default Tab
    switchView('mask_overlay');
  }

  // 6. View Switcher Tabs
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const view = btn.getAttribute('data-view');
      switchView(view);
    });
  });

  function switchView(viewName) {
    if (!currentAnalysisData || !currentAnalysisData.images_base64) return;

    tabButtons.forEach(b => b.classList.remove('active'));
    const activeBtn = document.querySelector(`.tab-btn[data-view="${viewName}"]`);
    if (activeBtn) activeBtn.classList.add('active');

    const b64 = currentAnalysisData.images_base64[viewName];
    if (b64) {
      viewerImage.src = `data:image/jpeg;base64,${b64}`;
      viewerCaption.textContent = CAPTIONS[viewName] || '';
    }
  }

  function setLoading(isLoading) {
    if (isLoading) {
      analyzeBtn.disabled = true;
      spinner.classList.remove('hidden');
      btnText.textContent = 'Forensic Neural Processing...';
    } else {
      analyzeBtn.disabled = !currentFile;
      spinner.classList.add('hidden');
      btnText.textContent = 'Execute Forensic Audit';
    }
  }
});
