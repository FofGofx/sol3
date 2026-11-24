"use strict";

// The hero reports paper benchmarks. Sample measurements and prompts come from each clip's timing.json.
const models = [
  {
    id: "wan06", name: "Wan2.2 I2V", previewWidth: 960,
    paperAttention: "3.07×", paperDenoise: "1.88×",
    frames: 81, resolution: "1280 × 720",
    sampleAttention: "3.42×", sampleDenoise: "1.94×",
    prompt: "An unoccupied terracotta arcade at sunset. The camera advances slowly along the middle of the tiled passage through repeating warm arches. Nearby columns pass faster than the far archway. Straight architectural lines, textured clay, long stable shadows, a single continuous move."
  },
  {
    id: "h3c04", name: "H3-Base", previewWidth: 1008, audio: true,
    paperAttention: "2.27×", paperDenoise: "1.45×",
    frames: 124, resolution: "1344 × 768",
    sampleAttention: "2.37×", sampleDenoise: "1.47×",
    prompt: "One broad straight curtain of water seen dead front. A dark level basalt lip crosses the upper quarter edge to edge. A solid sheet of white falling water fills the middle of the picture and breaks into pale turquoise spray along the bottom edge. The water is visibly in constant downward motion; the rock ledge stays still. Cool overcast daylight, a locked-off camera, one uninterrupted nature-film shot, steady sound of rushing water."
  },
  {
    id: "sanac04", name: "SANA-WM Refiner", previewWidth: 960,
    paperAttention: "2.33×", paperDenoise: "1.36×",
    frames: 481, resolution: "1280 × 704",
    sampleAttention: "2.47×", sampleDenoise: "1.38×",
    prompt: "Starting with the supplied stone bridge, keep the bridge arch and its reflection visible as the camera follows its given path. The current breaks reflected daylight into fine moving streaks below the arch. Continuous landscape photography, stable masonry."
  },
  {
    id: "hyc02", name: "HunyuanVideo", previewWidth: 960,
    paperAttention: "2.84×", paperDenoise: "1.85×",
    frames: 129, resolution: "1280 × 720",
    sampleAttention: "2.90×", sampleDenoise: "1.87×",
    prompt: "A small classic sailboat with a single triangular ivory sail crosses left to right on dark ultramarine water. Medium-long level view with a low distant coastline; a narrow white wake steadily lengthens behind the hull. Clear morning light on the canvas, stable horizon and steady camera, one continuous shot."
  },
  {
    id: "ltxc02", name: "LTX-2.5", previewWidth: 960, audio: true,
    paperAttention: "2.58×", paperDenoise: "1.22×",
    frames: 361, resolution: "1536 × 1024",
    sampleAttention: "2.55×", sampleDenoise: "1.22×",
    prompt: "The camera looks straight down onto a flat honey-gold beach. Clear turquoise seawater fills the upper half of the image, dry ochre sand the lower half. A single long, nearly horizontal lace edge of white foam rolls toward the bottom of frame and gently draws back. Damp sand gleams where it passed. Bright coastal daylight, no people, footprints or objects. The aerial camera stays fixed without tilting; the only sound is the steady hiss of the surf."
  },
  {
    id: "fastc39", name: "FastH3", previewWidth: 1008, audio: true,
    paperAttention: "2.72×", paperDenoise: "1.50×",
    frames: 243, resolution: "1344 × 768",
    sampleAttention: "2.78×", sampleDenoise: "1.53×",
    prompt: "A full silver moon is centered in a deep midnight-blue sky and takes up roughly half the frame height. Dark lunar maria and illuminated craters are clear. A thin horizontal wisp of pale cloud drifts slowly from left to right across the lower third of the lunar disc, showing its movement against the craters. Locked telephoto nature cinematography, a single continuous shot, very faint night ambience."
  }
];

const mediaPath = (model, suffix) => "media/" + model.id + suffix;
const hero = document.querySelector(".hero");
const heroVideo = document.getElementById("hero-video");
const heroPlay = document.getElementById("hero-play");
const backdrop = document.querySelector(".hero-backdrop").getContext("2d");
heroVideo.addEventListener("timeupdate", () => {
  if (heroVisible && heroVideo.readyState >= 2) {
    backdrop.drawImage(heroVideo, 0, 0, backdrop.canvas.width, backdrop.canvas.height);
  }
});
const strip = document.getElementById("model-strip");
let heroIndex = 0;
let heroVisible = true;
let heroPaused = false;

function updatePlaybackButton() {
  const label = heroPaused ? "Play showcase" : "Pause showcase";
  heroPlay.setAttribute("aria-label", label);
  heroPlay.title = label;
  heroPlay.textContent = heroPaused ? "▶" : "Ⅱ";
}

function updateHeroPlayback() {
  if (heroPaused || !heroVisible || document.hidden) {
    heroVideo.pause();
    return;
  }
  heroVideo.play().catch(error => {
    if (error.name === "NotAllowedError") {
      heroPaused = true;
      updatePlaybackButton();
    }
  });
}

function selectModel(index) {
  heroIndex = index;
  backdrop.clearRect(0, 0, backdrop.canvas.width, backdrop.canvas.height);
  const model = models[index];
  const poster = mediaPath(model, ".jpg");
  hero.style.setProperty("--poster", "url('" + poster + "')");
  heroVideo.poster = poster;
  heroVideo.setAttribute("aria-label", model.name + ", Dense on the left and Sol³ on the right");
  Array.from(strip.children).forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
  heroVideo.src = mediaPath(model, ".mp4");
  heroVideo.load();
  updateHeroPlayback();
}

models.forEach((model, index) => {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "model-tab";
  button.innerHTML = "<span>" + model.name + "</span><strong>" + model.paperAttention +
    " <small>(" + model.paperDenoise + ")</small></strong>";
  button.setAttribute("aria-label", model.name + ", attention speedup " + model.paperAttention +
    ", denoise speedup " + model.paperDenoise);
  button.addEventListener("click", () => selectModel(index));
  strip.append(button);
});
heroVideo.addEventListener("loadedmetadata", () => {
  if (heroIndex === 0) heroVideo.currentTime = 1.25;
});
heroVideo.addEventListener("ended", () => selectModel((heroIndex + 1) % models.length));
heroPlay.addEventListener("click", () => {
  heroPaused = !heroPaused;
  updatePlaybackButton();
  updateHeroPlayback();
});
new IntersectionObserver(([entry]) => {
  heroVisible = entry.isIntersecting;
  updateHeroPlayback();
}).observe(hero);
selectModel(0);

const method = document.getElementById("routing");
let methodVisible = false;
function notifyMethod() {
  method.contentWindow.postMessage({ type: "sol3-method-visibility", visible: methodVisible && !document.hidden }, "*");
}
window.addEventListener("message", event => {
  if (event.source !== method.contentWindow || event.data?.type !== "sol3-method-height") return;
  const height = event.data.height;
  if (Number.isFinite(height) && height > 0) method.style.height = Math.ceil(height) + "px";
});
method.addEventListener("load", notifyMethod);
new IntersectionObserver(([entry]) => {
  methodVisible = entry.intersectionRatio >= 0.2;
  notifyMethod();
}, { threshold: [0, 0.2] }).observe(method);

// One audio element guarantees that at most one variant on the page can be heard.
const speaker = new Audio();
let audibleCard = null;
function setAudio(card, variant) {
  speaker.pause();
  if (audibleCard) {
    audibleCard.track = "off";
    audibleCard.updateAudioControls();
  }
  audibleCard = variant === "off" ? null : card;
  if (!audibleCard) return;
  card.track = variant;
  card.updateAudioControls();
  speaker.src = mediaPath(card.model, "-" + variant + ".m4a");
  speaker.currentTime = card.video.currentTime;
  // Start in the user gesture, including on mobile browsers.
  speaker.play().catch(error => {
    if (error.name !== "AbortError" && audibleCard === card && card.track === variant) setAudio(card, "off");
  });
}
function alignAudio(card) {
  if (audibleCard !== card || speaker.readyState < 1 || speaker.seeking) return;
  const time = card.video.currentTime;
  if (Math.abs(speaker.currentTime - time) > 0.055 && time < speaker.duration) speaker.currentTime = time;
  if (speaker.paused && !card.video.paused && card.video.readyState >= 3 && time < speaker.duration - 0.08) {
    speaker.play().catch(() => {});
  }
}
speaker.addEventListener("loadedmetadata", () => { if (audibleCard) alignAudio(audibleCard); });

const mutedIcon = '<path d="M11 5 6 9H3v6h3l5 4zM16 9l6 6m0-6-6 6"/>';
const soundIcon = '<path d="M11 5 6 9H3v6h3l5 4zM15 8c3 2 3 6 0 8m3-11c5 4 5 10 0 14"/>';
function addSoundControls(card) {
  const { panel, model } = card;
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "sound-button";
  toggle.setAttribute("aria-expanded", "false");
  const menu = document.createElement("div");
  menu.className = "track-menu";
  menu.id = "tracks-" + model.id;
  menu.hidden = true;
  toggle.setAttribute("aria-controls", menu.id);
  function showMenu(open) {
    menu.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
  }
  toggle.addEventListener("click", () => showMenu(menu.hidden));
  for (const [variant, label] of [["off", "Muted"], ["fa2", "Dense"], ["sol3", "Sol³"]]) {
    const option = document.createElement("button");
    option.type = "button";
    option.dataset.track = variant;
    option.textContent = label;
    option.addEventListener("click", () => {
      setAudio(card, variant);
      showMenu(false);
      toggle.focus({ preventScroll: true });
    });
    menu.append(option);
  }
  document.addEventListener("pointerdown", event => {
    if (!menu.hidden && !menu.contains(event.target) && !toggle.contains(event.target)) showMenu(false);
  });
  panel.addEventListener("keydown", event => {
    if (event.key === "Escape" && !menu.hidden) { showMenu(false); toggle.focus(); }
  });
  card.updateAudioControls = () => {
    const label = model.name + " audio: " + ({off:"muted", fa2:"Dense", sol3:"Sol³"})[card.track];
    toggle.setAttribute("aria-label", label);
    toggle.title = label;
    toggle.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true">' + (card.track === "off" ? mutedIcon : soundIcon) + "</svg>";
    for (const option of menu.children) option.setAttribute("aria-pressed", String(option.dataset.track === card.track));
  };
  panel.append(toggle, menu);
  card.updateAudioControls();
}

const cards = new Map();
const gallery = document.getElementById("gallery");
const visibleCards = new IntersectionObserver(entries => {
  for (const entry of entries) {
    const card = cards.get(entry.target);
    card.visible = entry.isIntersecting;
    card.updatePlayback();
  }
});

function createCard(model) {
  const article = document.createElement("article");
  article.className = "card";
  article.innerHTML = `<h3>${model.name}</h3>
    <div class="compare">
      <video muted playsinline loop preload="none"></video><canvas aria-hidden="true"></canvas>
      <span class="side-tag left">Dense</span><span class="side-tag right">Sol³</span>
      <span class="divider"></span><input class="wipe-input" type="range" min="0" max="100" step="0.1" value="50">
    </div>
    <dl class="clip-stats">
      <div><dt>Frames</dt><dd>${model.frames}</dd></div>
      <div><dt>Resolution</dt><dd>${model.resolution}</dd></div>
      <div><dt>Attn speedup</dt><dd>${model.sampleAttention}</dd></div>
      <div><dt>Denoise speedup</dt><dd>${model.sampleDenoise}</dd></div>
    </dl>
    <details class="clip-prompt"><summary>Prompt</summary><p></p></details>`;
  article.querySelector(".clip-prompt p").textContent = model.prompt;
  const panel = article.querySelector(".compare");
  const video = article.querySelector("video");
  const canvas = article.querySelector("canvas");
  const range = article.querySelector("input");
  range.setAttribute("aria-label", model.name + " comparison divider, percent Dense");
  const ctx = canvas.getContext("2d", { alpha: false });
  const poster = new Image();
  const card = { panel, video, model, track: "off", visible: false };
  function paint() {
    const playingFrame = video.readyState >= 2;
    if (!playingFrame && !poster.naturalWidth) return;
    const frame = playingFrame ? video : poster;
    const w = playingFrame ? video.videoWidth / 2 : poster.naturalWidth / 2;
    const h = playingFrame ? video.videoHeight : poster.naturalHeight;
    // Fit both views to the same 16:9 crop before applying the divider.
    const scale = Math.max(canvas.width / w, canvas.height / h);
    const cropWidth = canvas.width / scale;
    const cropHeight = canvas.height / scale;
    const x = (w - cropWidth) / 2;
    const y = (h - cropHeight) / 2;
    const dest = Math.round(canvas.width * Number(range.value) / 100);
    const cut = dest / scale;
    if (dest > 0) ctx.drawImage(frame, x, y, cut, cropHeight, 0, 0, dest, canvas.height);
    if (dest < canvas.width) ctx.drawImage(frame, w + x + cut, y, cropWidth - cut, cropHeight,
      dest, 0, canvas.width - dest, canvas.height);
  }
  new ResizeObserver(() => {
    canvas.width = Math.min(model.previewWidth, Math.max(1, Math.round(panel.clientWidth * devicePixelRatio)));
    canvas.height = Math.max(1, Math.round(canvas.width * 9 / 16));
    paint();
  }).observe(panel);
  poster.onload = paint;
  poster.src = mediaPath(model, ".jpg");
  range.addEventListener("input", () => {
    panel.style.setProperty("--split", range.value + "%");
    paint();
  });
  let callback = null;
  const hasFrameCallback = typeof video.requestVideoFrameCallback === "function";
  function drawFrame() {
    callback = null;
    paint();
    alignAudio(card);
    if (!video.paused && card.visible) queueFrame();
  }
  function queueFrame() {
    if (callback !== null) return;
    callback = hasFrameCallback ? video.requestVideoFrameCallback(drawFrame) : requestAnimationFrame(drawFrame);
  }
  video.addEventListener("playing", () => {
    queueFrame();
    if (audibleCard === card) speaker.play().catch(() => {});
  });
  video.addEventListener("waiting", () => { if (audibleCard === card) speaker.pause(); });
  video.addEventListener("seeked", () => { paint(); alignAudio(card); });
  card.updatePlayback = () => {
    if (card.visible && !document.hidden) {
      if (!video.hasAttribute("src")) video.src = mediaPath(model, ".mp4");
      video.play().catch(() => {});
    } else {
      video.pause();
      if (audibleCard === card) setAudio(card, "off");
      if (callback !== null) {
        if (hasFrameCallback) video.cancelVideoFrameCallback(callback);
        else cancelAnimationFrame(callback);
        callback = null;
      }
    }
  };
  if (model.audio) addSoundControls(card);
  cards.set(panel, card);
  gallery.append(article);
  visibleCards.observe(panel);
}
models.forEach(createCard);
document.addEventListener("visibilitychange", () => {
  updateHeroPlayback();
  notifyMethod();
  for (const card of cards.values()) card.updatePlayback();
});
