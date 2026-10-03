/**
 * Conway's Game of Life — Interactive HPC Visualizer & Benchmark Suite
 * Matching exact HPC Seed 42 PRNG, zero-boundary rules, and 64-bit checksum.
 */

(function () {
  'use strict';

  // --- HPC Reference Baseline Table (from README.md Table §11) ---
  const HPC_BASELINES = {
    64: { targetGen: 20, living: 571, checksum: '2642065318239818179' },
    128: { targetGen: 100, living: null, checksum: null },
    256: { targetGen: 100, living: 5877, checksum: '2188031159639976069' },
    512: { targetGen: 100, living: 23852, checksum: '3838351066650152210' },
    1024: { targetGen: 100, living: 99296, checksum: '14789994132222743192' },
    2048: { targetGen: 100, living: 390059, checksum: '17074688330164608745' },
    4096: { targetGen: 100, living: 1584269, checksum: '15231916768216565527' }
  };

  const CHECKSUM_MULTIPLIER = 1315423911n;
  const MASK64 = 0xFFFFFFFFFFFFFFFFn;
  const RANDOM_SEED = 42;

  // --- Patterns for stamping ---
  const PATTERNS = {
    glider: [
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 1]
    ],
    pulsar: [
      [0,0,1,1,1,0,0,0,1,1,1,0,0],
      [0,0,0,0,0,0,0,0,0,0,0,0,0],
      [1,0,0,0,0,1,0,1,0,0,0,0,1],
      [1,0,0,0,0,1,0,1,0,0,0,0,1],
      [1,0,0,0,0,1,0,1,0,0,0,0,1],
      [0,0,1,1,1,0,0,0,1,1,1,0,0],
      [0,0,0,0,0,0,0,0,0,0,0,0,0],
      [0,0,1,1,1,0,0,0,1,1,1,0,0],
      [1,0,0,0,0,1,0,1,0,0,0,0,1],
      [1,0,0,0,0,1,0,1,0,0,0,0,1],
      [1,0,0,0,0,1,0,1,0,0,0,0,1],
      [0,0,0,0,0,0,0,0,0,0,0,0,0],
      [0,0,1,1,1,0,0,0,1,1,1,0,0]
    ],
    lwss: [
      [0, 1, 0, 0, 1],
      [1, 0, 0, 0, 0],
      [1, 0, 0, 0, 1],
      [1, 1, 1, 1, 0]
    ],
    gosper: [
      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
      [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
      [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ],
    acorn: [
      [0, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0],
      [1, 1, 0, 0, 1, 1, 1]
    ]
  };

  // --- Themes ---
  const THEMES = [
    { id: 'theme-cyberpunk', name: 'Emerald', bodyClass: 'theme-cyberpunk' },
    { id: 'theme-plasma', name: 'Plasma', bodyClass: 'theme-plasma' },
    { id: 'theme-matrix', name: 'Matrix', bodyClass: 'theme-matrix' },
    { id: 'theme-amethyst', name: 'Amethyst', bodyClass: 'theme-amethyst' }
  ];
  let currentThemeIdx = 0;

  // --- App State ---
  let N = 256;
  let targetGenerations = 100;
  let currentGen = 0;
  let isRunning = false;
  let fpsLimit = 60;
  let selectedPattern = 'none';
  let colorMode = 'age'; // 'classic', 'age', 'monochrome'
  let showGrid = true;

  // Grids & Age Buffers
  let curGrid = new Uint8Array(N * N);
  let nextGrid = new Uint8Array(N * N);
  let cellAge = new Uint16Array(N * N);

  // History buffer for step-back (up to 60 steps)
  const MAX_HISTORY = 60;
  const historyStack = [];

  // Population history for chart
  const popHistory = [];
  const MAX_POP_HISTORY = 60;

  // Viewport & Pan/Zoom
  let zoom = 1.0;
  let panX = 0;
  let panY = 0;
  let isPanning = false;
  let isDrawing = false;
  let drawVal = 1;
  let dragStartX = 0;
  let dragStartY = 0;
  let panStartX = 0;
  let panStartY = 0;

  // Animation & Benchmarking timing
  let animId = null;
  let lastFrameTime = performance.now();
  let frameTimes = [];
  let lastGpsCalcTime = performance.now();
  let framesInSecond = 0;
  let currentGps = 0;

  // --- DOM Elements ---
  const lifeCanvas = document.getElementById('lifeCanvas');
  const ctx = lifeCanvas.getContext('2d');
  const canvasViewport = document.getElementById('canvasViewport');
  const minimapCanvas = document.getElementById('minimapCanvas');
  const minimapCtx = minimapCanvas.getContext('2d');
  const minimapRect = document.getElementById('minimapRect');
  const popChartCanvas = document.getElementById('popChart');
  const popChartCtx = popChartCanvas.getContext('2d');

  // Headers & Pills
  const simStatusPill = document.getElementById('simStatusPill');
  const simStatusText = document.getElementById('simStatusText');
  const validationBadge = document.getElementById('validationBadge');
  const validationText = document.getElementById('validationText');
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const themeNameLabel = document.getElementById('themeNameLabel');

  // Controls
  const gridSizeSelect = document.getElementById('gridSizeSelect');
  const targetGensInput = document.getElementById('targetGensInput');
  const set100Btn = document.getElementById('set100Btn');
  const set20Btn = document.getElementById('set20Btn');
  const colorModeSelect = document.getElementById('colorModeSelect');
  const resetSeedBtn = document.getElementById('resetSeedBtn');
  const clearGridBtn = document.getElementById('clearGridBtn');
  const speedSlider = document.getElementById('speedSlider');
  const speedValueLabel = document.getElementById('speedValueLabel');
  const copyCliCmdBtn = document.getElementById('copyCliCmdBtn');

  // Dock controls
  const playPauseBtn = document.getElementById('playPauseBtn');
  const playIcon = document.getElementById('playIcon');
  const pauseIcon = document.getElementById('pauseIcon');
  const stepForwardBtn = document.getElementById('stepForwardBtn');
  const stepBackBtn = document.getElementById('stepBackBtn');
  const fastForwardBtn = document.getElementById('fastForwardBtn');
  const dockGenRatio = document.getElementById('dockGenRatio');
  const dockProgressBar = document.getElementById('dockProgressBar');

  // Viewport Toolbar
  const zoomInBtn = document.getElementById('zoomInBtn');
  const zoomOutBtn = document.getElementById('zoomOutBtn');
  const zoomResetBtn = document.getElementById('zoomResetBtn');
  const zoomLevelLabel = document.getElementById('zoomLevelLabel');
  const gridToggleBtn = document.getElementById('gridToggleBtn');
  const hoverCoordsText = document.getElementById('hoverCoordsText');

  // Telemetry stats
  const statGen = document.getElementById('statGen');
  const statGenPercent = document.getElementById('statGenPercent');
  const statLiving = document.getElementById('statLiving');
  const statDensity = document.getElementById('statDensity');
  const statChecksum = document.getElementById('statChecksum');
  const statFrameTime = document.getElementById('statFrameTime');
  const statFps = document.getElementById('statFps');
  const statTotalCells = document.getElementById('statTotalCells');
  const statGridDim = document.getElementById('statGridDim');

  // Verification Box
  const refLivingVal = document.getElementById('refLivingVal');
  const refChecksumVal = document.getElementById('refChecksumVal');
  const verificationStatusRow = document.getElementById('verificationStatusRow');

  // Modals
  const benchmarkModalBtn = document.getElementById('benchmarkModalBtn');
  const benchmarkModal = document.getElementById('benchmarkModal');
  const closeBenchmarkModalBtn = document.getElementById('closeBenchmarkModalBtn');
  const helpModalBtn = document.getElementById('helpModalBtn');
  const helpModal = document.getElementById('helpModal');
  const closeHelpModalBtn = document.getElementById('closeHelpModalBtn');

  // Pattern chips
  const patternChips = document.querySelectorAll('.pattern-chip');

  // Offscreen canvas for fast pixel blitting
  let offCanvas = document.createElement('canvas');
  let offCtx = offCanvas.getContext('2d');
  let offImageData = null;

  // --- Seed 42 PRNG & Grid Initialization ---
  function initPRNGGrid() {
    let state = RANDOM_SEED;
    const total = N * N;
    curGrid = new Uint8Array(total);
    nextGrid = new Uint8Array(total);
    cellAge = new Uint16Array(total);
    historyStack.length = 0;
    popHistory.length = 0;

    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        // state = 1664525u * state + 1013904223u;
        state = (Math.imul(1664525, state) + 1013904223) >>> 0;
        const val = state >>> 31;
        const idx = r * N + c;
        curGrid[idx] = val;
        cellAge[idx] = val ? 1 : 0;
      }
    }

    currentGen = 0;
    resetOffscreenCanvas();
    recordHistory();
    updateTelemetry(0);
    render();
    renderMinimap();
    renderPopChart();
    updateVerification();
  }

  function clearGrid() {
    const total = N * N;
    curGrid.fill(0);
    nextGrid.fill(0);
    cellAge.fill(0);
    historyStack.length = 0;
    currentGen = 0;
    recordHistory();
    updateTelemetry(0);
    render();
    renderMinimap();
    renderPopChart();
    updateVerification();
  }

  function resetOffscreenCanvas() {
    offCanvas.width = N;
    offCanvas.height = N;
    offImageData = offCtx.createImageData(N, N);
  }

  // --- 64-Bit Checksum Calculation ---
  function computeChecksum() {
    let checksum = 0n;
    const total = N * N;
    for (let i = 0; i < total; i++) {
      checksum = (checksum * CHECKSUM_MULTIPLIER + BigInt(curGrid[i])) & MASK64;
    }
    return checksum;
  }

  function countLiving() {
    let living = 0;
    const total = N * N;
    for (let i = 0; i < total; i++) {
      living += curGrid[i];
    }
    return living;
  }

  // --- Generation Computation (Zero Boundary) ---
  function computeNextGen() {
    const t0 = performance.now();

    for (let r = 0; r < N; r++) {
      const rN = r * N;
      for (let c = 0; c < N; c++) {
        let neighbours = 0;

        for (let dr = -1; dr <= 1; dr++) {
          const nr = r + dr;
          if (nr < 0 || nr >= N) continue;
          const nrN = nr * N;

          for (let dc = -1; dc <= 1; dc++) {
            if (dr === 0 && dc === 0) continue;
            const nc = c + dc;
            if (nc < 0 || nc >= N) continue;
            neighbours += curGrid[nrN + nc];
          }
        }

        const alive = curGrid[rN + c];
        let nxt = 0;
        if (alive === 1 && (neighbours === 2 || neighbours === 3)) {
          nxt = 1;
        } else if (alive === 0 && neighbours === 3) {
          nxt = 1;
        }

        const idx = rN + c;
        nextGrid[idx] = nxt;
        if (nxt === 1) {
          cellAge[idx] = (alive === 1) ? Math.min(65535, cellAge[idx] + 1) : 1;
        } else {
          cellAge[idx] = 0;
        }
      }
    }

    // Buffer swap
    const temp = curGrid;
    curGrid = nextGrid;
    nextGrid = temp;

    currentGen++;
    recordHistory();

    const t1 = performance.now();
    const frameDuration = t1 - t0;
    updateTelemetry(frameDuration);

    // Check target generation auto-stop
    if (targetGenerations > 0 && currentGen >= targetGenerations) {
      if (isRunning) {
        pauseSimulation();
        simStatusText.textContent = `Completed (${targetGenerations} Gens)`;
        const ind = simStatusPill.querySelector('.status-indicator');
        if (ind) ind.className = 'status-indicator ready';
      }
    }

    render();
    renderMinimap();
    renderPopChart();
    updateVerification();
  }

  function recordHistory() {
    if (historyStack.length >= MAX_HISTORY) {
      historyStack.shift();
    }
    historyStack.push({
      gen: currentGen,
      grid: new Uint8Array(curGrid),
      age: new Uint16Array(cellAge)
    });
  }

  function stepBack() {
    if (historyStack.length > 1) {
      historyStack.pop(); // Remove current
      const prev = historyStack[historyStack.length - 1];
      curGrid.set(prev.grid);
      cellAge.set(prev.age);
      currentGen = prev.gen;
      updateTelemetry(0);
      render();
      renderMinimap();
      renderPopChart();
      updateVerification();
    }
  }

  // --- Rendering Engine ---
  function getPaletteColors() {
    const isPlasma = document.body.classList.contains('theme-plasma');
    const isMatrix = document.body.classList.contains('theme-matrix');
    const isAmethyst = document.body.classList.contains('theme-amethyst');

    if (isPlasma) {
      return {
        dead: [11, 17, 32],
        young: [56, 189, 248],
        mid: [6, 182, 212],
        old: [14, 116, 144],
        ancient: [2, 132, 199]
      };
    } else if (isMatrix) {
      return {
        dead: [3, 8, 4],
        young: [74, 222, 128],
        mid: [34, 197, 94],
        old: [22, 163, 74],
        ancient: [21, 128, 61]
      };
    } else if (isAmethyst) {
      return {
        dead: [18, 10, 30],
        young: [192, 132, 252],
        mid: [168, 85, 247],
        old: [147, 51, 234],
        ancient: [126, 34, 206]
      };
    }
    // Default Cyberpunk Emerald
    return {
      dead: [11, 17, 32],
      young: [52, 211, 153],
      mid: [16, 185, 129],
      old: [5, 150, 105],
      ancient: [6, 182, 212]
    };
  }

  function updateOffscreenImageData() {
    const data = offImageData.data;
    const colors = getPaletteColors();
    const total = N * N;

    for (let i = 0; i < total; i++) {
      const pIdx = i * 4;
      const alive = curGrid[i];
      if (alive === 0) {
        data[pIdx] = colors.dead[0];
        data[pIdx + 1] = colors.dead[1];
        data[pIdx + 2] = colors.dead[2];
        data[pIdx + 3] = 255;
      } else {
        if (colorMode === 'monochrome') {
          data[pIdx] = 255;
          data[pIdx + 1] = 255;
          data[pIdx + 2] = 255;
          data[pIdx + 3] = 255;
        } else if (colorMode === 'classic') {
          data[pIdx] = colors.young[0];
          data[pIdx + 1] = colors.young[1];
          data[pIdx + 2] = colors.young[2];
          data[pIdx + 3] = 255;
        } else {
          // 'age' Bioluminescent Heatmap
          const age = cellAge[i];
          if (age <= 1) {
            data[pIdx] = colors.young[0];
            data[pIdx + 1] = colors.young[1];
            data[pIdx + 2] = colors.young[2];
          } else if (age <= 5) {
            data[pIdx] = colors.mid[0];
            data[pIdx + 1] = colors.mid[1];
            data[pIdx + 2] = colors.mid[2];
          } else if (age <= 15) {
            data[pIdx] = colors.old[0];
            data[pIdx + 1] = colors.old[1];
            data[pIdx + 2] = colors.old[2];
          } else {
            data[pIdx] = colors.ancient[0];
            data[pIdx + 1] = colors.ancient[1];
            data[pIdx + 2] = colors.ancient[2];
          }
          data[pIdx + 3] = 255;
        }
      }
    }

    offCtx.putImageData(offImageData, 0, 0);
  }

  function resizeCanvasIfNeeded() {
    const rect = canvasViewport.getBoundingClientRect();
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);
    if (lifeCanvas.width !== w || lifeCanvas.height !== h) {
      lifeCanvas.width = w;
      lifeCanvas.height = h;
      fitToScreen();
    }
  }

  function fitToScreen() {
    const rect = canvasViewport.getBoundingClientRect();
    const margin = 32;
    const availW = rect.width - margin * 2;
    const availH = rect.height - margin * 2;
    const fitZoom = Math.min(availW / N, availH / N);

    zoom = Math.max(0.1, fitZoom);
    panX = (rect.width - N * zoom) / 2;
    panY = (rect.height - N * zoom) / 2;

    zoomLevelLabel.textContent = `${zoom.toFixed(2)}x`;
  }

  function render() {
    resizeCanvasIfNeeded();
    updateOffscreenImageData();

    const w = lifeCanvas.width;
    const h = lifeCanvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = false;

    // Draw background boundary
    ctx.fillStyle = '#050811';
    ctx.fillRect(panX, panY, N * zoom, N * zoom);

    // Draw simulation grid
    ctx.drawImage(offCanvas, 0, 0, N, N, panX, panY, N * zoom, N * zoom);

    // Draw grid lines when zoomed in sufficiently
    if (showGrid && zoom >= 4.0) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();

      const startCol = Math.max(0, Math.floor(-panX / zoom));
      const endCol = Math.min(N, Math.ceil((w - panX) / zoom));
      const startRow = Math.max(0, Math.floor(-panY / zoom));
      const endRow = Math.min(N, Math.ceil((h - panY) / zoom));

      for (let c = startCol; c <= endCol; c++) {
        const x = Math.round(panX + c * zoom);
        ctx.moveTo(x + 0.5, Math.max(panY, 0));
        ctx.lineTo(x + 0.5, Math.min(panY + N * zoom, h));
      }
      for (let r = startRow; r <= endRow; r++) {
        const y = Math.round(panY + r * zoom);
        ctx.moveTo(Math.max(panX, 0), y + 0.5);
        ctx.lineTo(Math.min(panX + N * zoom, w), y + 0.5);
      }
      ctx.stroke();
    }

    // Outer grid border glow
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
    ctx.lineWidth = 2;
    ctx.strokeRect(panX, panY, N * zoom, N * zoom);
  }

  // --- Minimap Rendering ---
  function renderMinimap() {
    minimapCtx.clearRect(0, 0, minimapCanvas.width, minimapCanvas.height);
    minimapCtx.imageSmoothingEnabled = false;
    minimapCtx.drawImage(offCanvas, 0, 0, N, N, 0, 0, minimapCanvas.width, minimapCanvas.height);

    // Viewport box in minimap
    const mapW = minimapCanvas.width;
    const mapH = minimapCanvas.height;
    const viewW = lifeCanvas.width;
    const viewH = lifeCanvas.height;

    const visibleLeft = Math.max(0, -panX / (N * zoom));
    const visibleTop = Math.max(0, -panY / (N * zoom));
    const visibleW = Math.min(1, viewW / (N * zoom));
    const visibleH = Math.min(1, viewH / (N * zoom));

    minimapRect.style.left = `${Math.min(mapW, Math.max(0, visibleLeft * mapW))}px`;
    minimapRect.style.top = `${Math.min(mapH, Math.max(0, visibleTop * mapH))}px`;
    minimapRect.style.width = `${Math.min(mapW, Math.max(8, visibleW * mapW))}px`;
    minimapRect.style.height = `${Math.min(mapH, Math.max(8, visibleH * mapH))}px`;
  }

  // --- Population Dynamics Sparkline Chart ---
  function renderPopChart() {
    const living = countLiving();
    popHistory.push(living);
    if (popHistory.length > MAX_POP_HISTORY) {
      popHistory.shift();
    }

    const w = popChartCanvas.width = popChartCanvas.parentElement.clientWidth || 240;
    const h = popChartCanvas.height = 90;
    popChartCtx.clearRect(0, 0, w, h);

    if (popHistory.length < 2) return;

    let minPop = Infinity;
    let maxPop = -Infinity;
    for (let i = 0; i < popHistory.length; i++) {
      if (popHistory[i] < minPop) minPop = popHistory[i];
      if (popHistory[i] > maxPop) maxPop = popHistory[i];
    }
    if (minPop === maxPop) {
      minPop = Math.max(0, minPop - 10);
      maxPop = maxPop + 10;
    }

    const padding = 12;
    const chartW = w - padding * 2;
    const chartH = h - padding * 2;

    // Gradient background
    const grad = popChartCtx.createLinearGradient(0, padding, 0, h - padding);
    grad.addColorStop(0, 'rgba(16, 185, 129, 0.35)');
    grad.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

    popChartCtx.beginPath();
    for (let i = 0; i < popHistory.length; i++) {
      const x = padding + (i / (popHistory.length - 1)) * chartW;
      const y = h - padding - ((popHistory[i] - minPop) / (maxPop - minPop)) * chartH;
      if (i === 0) popChartCtx.moveTo(x, y);
      else popChartCtx.lineTo(x, y);
    }
    popChartCtx.strokeStyle = '#10b981';
    popChartCtx.lineWidth = 2;
    popChartCtx.stroke();

    // Fill under line
    popChartCtx.lineTo(padding + chartW, h - padding);
    popChartCtx.lineTo(padding, h - padding);
    popChartCtx.closePath();
    popChartCtx.fillStyle = grad;
    popChartCtx.fill();

    // Current living label
    popChartCtx.fillStyle = '#94a3b8';
    popChartCtx.font = '10px "JetBrains Mono", monospace';
    popChartCtx.fillText(`Min: ${minPop.toLocaleString()} | Max: ${maxPop.toLocaleString()}`, padding, 10);
  }

  // --- Telemetry & Verification Updates ---
  function updateTelemetry(frameTimeMs) {
    const living = countLiving();
    const total = N * N;
    const density = ((living / total) * 100).toFixed(2);
    const checksum = computeChecksum();

    statGen.textContent = currentGen.toLocaleString();
    const pct = targetGenerations > 0 ? Math.min(100, Math.round((currentGen / targetGenerations) * 100)) : 0;
    statGenPercent.textContent = `${pct}% of target`;

    statLiving.textContent = living.toLocaleString();
    statDensity.textContent = `${density}% density`;

    statChecksum.textContent = checksum.toString();
    statTotalCells.textContent = total.toLocaleString();
    statGridDim.textContent = `${N} × ${N}`;

    statFrameTime.innerHTML = `${frameTimeMs.toFixed(2)} <small>ms</small>`;

    dockGenRatio.textContent = `${currentGen} / ${targetGenerations}`;
    dockProgressBar.style.width = `${pct}%`;

    // Calculate GPS (Generations Per Second)
    framesInSecond++;
    const now = performance.now();
    if (now - lastGpsCalcTime >= 1000) {
      currentGps = Math.round((framesInSecond * 1000) / (now - lastGpsCalcTime));
      framesInSecond = 0;
      lastGpsCalcTime = now;
      statFps.textContent = `${currentGps} GPS`;
    }
  }

  function updateVerification() {
    const base = HPC_BASELINES[N];
    if (!base || base.living === null) {
      refLivingVal.textContent = 'N/A';
      refChecksumVal.textContent = 'No baseline for this N';
      verificationStatusRow.innerHTML = `<span class="v-tag pending">Arbitrary grid dimension</span>`;
      validationBadge.className = 'validation-badge pending';
      validationText.textContent = 'No HPC Table Baseline';
      return;
    }

    refLivingVal.textContent = base.living.toLocaleString();
    refChecksumVal.textContent = base.checksum;

    const curChecksumStr = computeChecksum().toString();
    const curLiving = countLiving();

    if (currentGen < base.targetGen) {
      verificationStatusRow.innerHTML = `<span class="v-tag pending">Simulate to Gen ${base.targetGen} to verify</span>`;
      validationBadge.className = 'validation-badge pending';
      validationText.textContent = `Pending (Gen ${currentGen}/${base.targetGen})`;
    } else if (currentGen === base.targetGen) {
      if (curLiving === base.living && curChecksumStr === base.checksum) {
        verificationStatusRow.innerHTML = `<span class="v-tag matched">✓ Matches HPC Baseline 100%</span>`;
        validationBadge.className = 'validation-badge';
        validationText.textContent = 'Matches HPC Baseline';
      } else {
        verificationStatusRow.innerHTML = `<span class="v-tag pending" style="color: #f43f5e; background: rgba(244,63,94,0.15);">Mismatch detected</span>`;
        validationBadge.className = 'validation-badge mismatch';
        validationText.textContent = 'Result Diverged';
      }
    } else {
      verificationStatusRow.innerHTML = `<span class="v-tag pending">Past target (${currentGen} > ${base.targetGen})</span>`;
      validationBadge.className = 'validation-badge';
      validationText.textContent = 'Custom Run';
    }
  }

  // --- Simulation Loop ---
  function simulationLoop(timestamp) {
    if (!isRunning) return;

    const interval = fpsLimit >= 120 ? 0 : 1000 / fpsLimit;
    const elapsed = timestamp - lastFrameTime;

    if (elapsed >= interval) {
      lastFrameTime = timestamp - (elapsed % (interval || 1));
      computeNextGen();
    }

    if (isRunning) {
      animId = requestAnimationFrame(simulationLoop);
    }
  }

  function startSimulation() {
    if (isRunning) return;
    isRunning = true;
    playIcon.classList.add('hidden');
    pauseIcon.classList.remove('hidden');
    playPauseBtn.classList.add('active');

    simStatusText.textContent = 'Running';
    const ind = simStatusPill.querySelector('.status-indicator');
    if (ind) ind.className = 'status-indicator running';

    lastFrameTime = performance.now();
    animId = requestAnimationFrame(simulationLoop);
  }

  function pauseSimulation() {
    isRunning = false;
    if (animId) {
      cancelAnimationFrame(animId);
      animId = null;
    }
    playIcon.classList.remove('hidden');
    pauseIcon.classList.add('hidden');
    playPauseBtn.classList.remove('active');

    simStatusText.textContent = 'Paused';
    const ind = simStatusPill.querySelector('.status-indicator');
    if (ind) ind.className = 'status-indicator paused';
  }

  function togglePlayPause() {
    if (isRunning) {
      pauseSimulation();
    } else {
      startSimulation();
    }
  }

  // --- Pattern Stamping & Drawing ---
  function stampPattern(gridRow, gridCol, patternKey) {
    const pattern = PATTERNS[patternKey];
    if (!pattern) return;
    const pH = pattern.length;
    const pW = pattern[0].length;
    const startR = gridRow - Math.floor(pH / 2);
    const startC = gridCol - Math.floor(pW / 2);

    for (let r = 0; r < pH; r++) {
      const curR = startR + r;
      if (curR < 0 || curR >= N) continue;
      for (let c = 0; c < pW; c++) {
        const curC = startC + c;
        if (curC < 0 || curC >= N) continue;
        const val = pattern[r][c];
        const idx = curR * N + curC;
        curGrid[idx] = val;
        cellAge[idx] = val ? 1 : 0;
      }
    }
    recordHistory();
    updateTelemetry(0);
    render();
    renderMinimap();
    renderPopChart();
    updateVerification();
  }

  function setCell(gridRow, gridCol, val) {
    if (gridRow < 0 || gridRow >= N || gridCol < 0 || gridCol >= N) return;
    const idx = gridRow * N + gridCol;
    curGrid[idx] = val;
    cellAge[idx] = val ? 1 : 0;
    render();
  }

  function getGridCoords(clientX, clientY) {
    const rect = lifeCanvas.getBoundingClientRect();
    const canvasX = clientX - rect.left;
    const canvasY = clientY - rect.top;

    const col = Math.floor((canvasX - panX) / zoom);
    const row = Math.floor((canvasY - panY) / zoom);
    return { row, col };
  }

  // --- Mouse & Touch Event Listeners on Viewport ---
  canvasViewport.addEventListener('mousedown', (e) => {
    if (e.target.closest('.viewport-floating-toolbar') || e.target.closest('.minimap-container')) return;

    if (e.button === 1 || e.shiftKey || (e.button === 0 && e.altKey)) {
      // Pan
      isPanning = true;
      dragStartX = e.clientX;
      dragStartY = e.clientY;
      panStartX = panX;
      panStartY = panY;
      canvasViewport.style.cursor = 'grabbing';
      return;
    }

    if (e.button === 0) {
      const { row, col } = getGridCoords(e.clientX, e.clientY);
      if (selectedPattern !== 'none' && PATTERNS[selectedPattern]) {
        stampPattern(row, col, selectedPattern);
      } else {
        isDrawing = true;
        if (row >= 0 && row < N && col >= 0 && col < N) {
          const idx = row * N + col;
          drawVal = curGrid[idx] ? 0 : 1;
          setCell(row, col, drawVal);
        }
      }
    }
  });

  window.addEventListener('mousemove', (e) => {
    const { row, col } = getGridCoords(e.clientX, e.clientY);
    if (row >= 0 && row < N && col >= 0 && col < N) {
      hoverCoordsText.textContent = `(${row}, ${col})`;
    } else {
      hoverCoordsText.textContent = `(—, —)`;
    }

    if (isPanning) {
      const dx = e.clientX - dragStartX;
      const dy = e.clientY - dragStartY;
      panX = panStartX + dx;
      panY = panStartY + dy;
      render();
      renderMinimap();
      return;
    }

    if (isDrawing && selectedPattern === 'none') {
      if (row >= 0 && row < N && col >= 0 && col < N) {
        setCell(row, col, drawVal);
      }
    }
  });

  window.addEventListener('mouseup', () => {
    if (isPanning) {
      isPanning = false;
      canvasViewport.style.cursor = 'default';
    }
    if (isDrawing) {
      isDrawing = false;
      recordHistory();
      updateTelemetry(0);
      renderMinimap();
      renderPopChart();
      updateVerification();
    }
  });

  // Zooming via Wheel
  canvasViewport.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = lifeCanvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    const newZoom = Math.min(64.0, Math.max(0.1, zoom * zoomFactor));

    // Center zoom around mouse cursor
    panX = mouseX - (mouseX - panX) * (newZoom / zoom);
    panY = mouseY - (mouseY - panY) * (newZoom / zoom);
    zoom = newZoom;

    zoomLevelLabel.textContent = `${zoom.toFixed(2)}x`;
    render();
    renderMinimap();
  }, { passive: false });

  // --- Minimap dragging/navigation ---
  minimapCanvas.addEventListener('mousedown', (e) => {
    const rect = minimapCanvas.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) / minimapCanvas.width;
    const clickY = (e.clientY - rect.top) / minimapCanvas.height;

    panX = lifeCanvas.width / 2 - clickX * N * zoom;
    panY = lifeCanvas.height / 2 - clickY * N * zoom;
    render();
    renderMinimap();
  });

  // --- Toolbar & Controls Listeners ---
  zoomInBtn.addEventListener('click', () => {
    const rect = lifeCanvas.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const newZoom = Math.min(64.0, zoom * 1.3);
    panX = cx - (cx - panX) * (newZoom / zoom);
    panY = cy - (cy - panY) * (newZoom / zoom);
    zoom = newZoom;
    zoomLevelLabel.textContent = `${zoom.toFixed(2)}x`;
    render();
    renderMinimap();
  });

  zoomOutBtn.addEventListener('click', () => {
    const rect = lifeCanvas.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const newZoom = Math.max(0.1, zoom / 1.3);
    panX = cx - (cx - panX) * (newZoom / zoom);
    panY = cy - (cy - panY) * (newZoom / zoom);
    zoom = newZoom;
    zoomLevelLabel.textContent = `${zoom.toFixed(2)}x`;
    render();
    renderMinimap();
  });

  zoomResetBtn.addEventListener('click', () => {
    fitToScreen();
    render();
    renderMinimap();
  });

  gridToggleBtn.addEventListener('click', () => {
    showGrid = !showGrid;
    gridToggleBtn.classList.toggle('active', showGrid);
    render();
  });

  playPauseBtn.addEventListener('click', togglePlayPause);

  stepForwardBtn.addEventListener('click', () => {
    pauseSimulation();
    computeNextGen();
  });

  stepBackBtn.addEventListener('click', () => {
    pauseSimulation();
    stepBack();
  });

  fastForwardBtn.addEventListener('click', () => {
    pauseSimulation();
    for (let i = 0; i < 10; i++) {
      computeNextGen();
    }
  });

  resetSeedBtn.addEventListener('click', () => {
    pauseSimulation();
    initPRNGGrid();
    fitToScreen();
  });

  clearGridBtn.addEventListener('click', () => {
    pauseSimulation();
    clearGrid();
  });

  gridSizeSelect.addEventListener('change', (e) => {
    pauseSimulation();
    N = parseInt(e.target.value, 10);
    const base = HPC_BASELINES[N];
    if (base) {
      targetGenerations = base.targetGen;
      targetGensInput.value = targetGenerations;
    }
    initPRNGGrid();
    fitToScreen();
  });

  targetGensInput.addEventListener('change', (e) => {
    targetGenerations = Math.max(1, parseInt(e.target.value, 10) || 100);
    updateTelemetry(0);
    updateVerification();
  });

  set100Btn.addEventListener('click', () => {
    targetGenerations = 100;
    targetGensInput.value = 100;
    updateTelemetry(0);
    updateVerification();
  });

  set20Btn.addEventListener('click', () => {
    targetGenerations = 20;
    targetGensInput.value = 20;
    updateTelemetry(0);
    updateVerification();
  });

  colorModeSelect.addEventListener('change', (e) => {
    colorMode = e.target.value;
    render();
    renderMinimap();
  });

  speedSlider.addEventListener('input', (e) => {
    fpsLimit = parseInt(e.target.value, 10);
    speedValueLabel.textContent = fpsLimit >= 120 ? 'Uncapped' : `${fpsLimit} FPS`;
  });

  // Stamp pattern selection
  patternChips.forEach(chip => {
    chip.addEventListener('click', () => {
      patternChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      selectedPattern = chip.getAttribute('data-pattern');
    });
  });

  // CLI command copy
  if (copyCliCmdBtn) {
    copyCliCmdBtn.addEventListener('click', () => {
      navigator.clipboard.writeText('./game_of_life_terminal_vis 64 50 40').then(() => {
        const origHtml = copyCliCmdBtn.innerHTML;
        copyCliCmdBtn.innerHTML = '<span style="color:#10b981; font-size:10px;">✓ Copied</span>';
        setTimeout(() => {
          copyCliCmdBtn.innerHTML = origHtml;
        }, 1800);
      });
    });
  }

  // Theme toggle
  themeToggleBtn.addEventListener('click', () => {
    THEMES.forEach(t => document.body.classList.remove(t.bodyClass));
    currentThemeIdx = (currentThemeIdx + 1) % THEMES.length;
    const nextTheme = THEMES[currentThemeIdx];
    document.body.classList.add(nextTheme.bodyClass);
    themeNameLabel.textContent = nextTheme.name;
    render();
    renderMinimap();
    renderPopChart();
  });

  // Modals
  benchmarkModalBtn.addEventListener('click', () => {
    benchmarkModal.classList.remove('hidden');
  });
  closeBenchmarkModalBtn.addEventListener('click', () => {
    benchmarkModal.classList.add('hidden');
  });

  helpModalBtn.addEventListener('click', () => {
    helpModal.classList.remove('hidden');
  });
  closeHelpModalBtn.addEventListener('click', () => {
    helpModal.classList.add('hidden');
  });

  [benchmarkModal, helpModal].forEach(modal => {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.classList.add('hidden');
      }
    });
  });

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

    switch (e.code) {
      case 'Space':
        e.preventDefault();
        togglePlayPause();
        break;
      case 'ArrowRight':
        e.preventDefault();
        pauseSimulation();
        computeNextGen();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        pauseSimulation();
        stepBack();
        break;
      case 'KeyR':
        e.preventDefault();
        pauseSimulation();
        initPRNGGrid();
        fitToScreen();
        break;
      case 'KeyC':
        e.preventDefault();
        pauseSimulation();
        clearGrid();
        break;
      case 'KeyF':
        e.preventDefault();
        fitToScreen();
        render();
        renderMinimap();
        break;
      case 'Equal':
      case 'NumpadAdd':
        e.preventDefault();
        zoomInBtn.click();
        break;
      case 'Minus':
      case 'NumpadSubtract':
        e.preventDefault();
        zoomOutBtn.click();
        break;
      case 'Escape':
        benchmarkModal.classList.add('hidden');
        helpModal.classList.add('hidden');
        break;
    }
  });

  window.addEventListener('resize', () => {
    resizeCanvasIfNeeded();
    render();
    renderMinimap();
    renderPopChart();
  });

  // --- Initial Launch ---
  initPRNGGrid();
  fitToScreen();
  render();
  renderMinimap();
  renderPopChart();
  updateVerification();

  console.log('Conway Game of Life Visualizer initialized with Seed 42 baseline.');
})();
