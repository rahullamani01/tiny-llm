const els = {
  messages: document.getElementById("messages"),
  welcome: document.getElementById("welcome"),
  input: document.getElementById("input"),
  composer: document.getElementById("composer"),
  send: document.getElementById("send"),
  stop: document.getElementById("stop"),
  status: document.getElementById("status"),
  dot: document.getElementById("dot"),
  newChat: document.getElementById("new-chat"),
  settings: document.getElementById("settings"),
  settingsToggle: document.getElementById("settings-toggle"),
  closeSettings: document.getElementById("close-settings"),
  backdrop: document.getElementById("backdrop"),
  system: document.getElementById("system"),
  temperature: document.getElementById("temperature"),
  temperatureValue: document.getElementById("temperature-value"),
  topK: document.getElementById("top-k"),
  topKValue: document.getElementById("top-k-value"),
  maxTokens: document.getElementById("max-tokens"),
  kvCache: document.getElementById("kv-cache"),
};

const DEFAULTS = {
  system: "You are a helpful, concise assistant.",
  temperature: 0.8,
  topK: 40,
  maxTokens: 256,
  useCache: true,
};

const state = { messages: [], streaming: null };
const STORAGE_KEY = "tiny-llm-settings";

function settings() {
  return {
    system: els.system.value.trim() || DEFAULTS.system,
    temperature: Number(els.temperature.value),
    topK: Number(els.topK.value),
    maxTokens: Math.max(1, Number(els.maxTokens.value) || DEFAULTS.maxTokens),
    useCache: els.kvCache.checked,
  };
}

function loadSettings() {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch (_) {
    saved = {};
  }
  els.system.value = saved.system ?? DEFAULTS.system;
  els.temperature.value = saved.temperature ?? DEFAULTS.temperature;
  els.topK.value = saved.topK ?? DEFAULTS.topK;
  els.maxTokens.value = saved.maxTokens ?? DEFAULTS.maxTokens;
  els.kvCache.checked = saved.useCache ?? DEFAULTS.useCache;
  updateLabels();
}

function saveSettings() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings()));
}

function updateLabels() {
  els.temperatureValue.textContent = Number(els.temperature.value).toFixed(2);
  els.topKValue.textContent = Number(els.topK.value) === 0 ? "off" : els.topK.value;
}

function setStatus(text, ok) {
  els.status.textContent = text;
  els.dot.classList.toggle("offline", !ok);
}

function setBusy(busy) {
  els.send.disabled = busy;
  els.stop.hidden = !busy;
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function addMessage(role, text) {
  els.welcome.hidden = true;
  const row = document.createElement("div");
  row.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "t";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  row.append(avatar, bubble);
  els.messages.append(row);
  scrollToBottom();
  return bubble;
}

function clearChat() {
  document.querySelectorAll(".message").forEach((node) => node.remove());
  state.messages = [];
  els.welcome.hidden = false;
  scrollToBottom();
}

function autoGrow() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 180)}px`;
}

async function readSSE(response, onEvent) {
  if (!response.body) throw new Error("streaming is not supported by this browser");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split;
    while ((split = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      for (const line of chunk.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        let event;
        try {
          event = JSON.parse(payload);
        } catch (error) {
          console.warn("bad SSE payload", payload, error);
          continue;
        }
        onEvent(event);
      }
    }
  }
}

async function sendMessage(raw) {
  const text = raw.trim();
  if (!text || state.streaming) return;

  const options = settings();
  state.messages.push({ role: "user", content: text });
  addMessage("user", text);

  els.input.value = "";
  autoGrow();
  setBusy(true);
  const bubble = addMessage("assistant", "");
  bubble.classList.add("streaming");

  const controller = new AbortController();
  state.streaming = controller;
  let full = "";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        messages: state.messages,
        system: options.system,
        temperature: options.temperature,
        top_k: options.topK === 0 ? null : options.topK,
        max_new_tokens: options.maxTokens,
        use_cache: options.useCache,
      }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(detail.detail || `request failed (${response.status})`);
    }

    await readSSE(response, (event) => {
      if (event.error) throw new Error(event.error);
      if (event.token) {
        full += event.token;
        bubble.textContent = full;
        scrollToBottom();
      }
      if (event.done) {
        full = event.content ?? full;
        bubble.textContent = full;
      }
    });

    if (full.trim()) {
      state.messages.push({ role: "assistant", content: full.trim() });
    } else {
      bubble.textContent = "[empty reply]";
    }
  } catch (error) {
    if (error.name === "AbortError") {
      if (full.trim()) state.messages.push({ role: "assistant", content: full.trim() });
      else bubble.remove();
    } else {
      bubble.classList.add("error");
      bubble.textContent = `Error: ${error.message}`;
    }
  } finally {
    bubble.classList.remove("streaming");
    setBusy(false);
    state.streaming = null;
    scrollToBottom();
  }
}

function openSettings(open) {
  els.settings.classList.toggle("open", open);
  els.backdrop.classList.toggle("show", open);
}

async function init() {
  loadSettings();
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("health check failed");
    const info = await response.json();
    const params = (info.parameters / 1e6).toFixed(2);
    setStatus(`${params}M params · context ${info.context}`, true);
  } catch (_) {
    setStatus("server offline", false);
  }
}

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(els.input.value);
});

els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.composer.requestSubmit();
  }
});

els.input.addEventListener("input", autoGrow);
els.stop.addEventListener("click", () => state.streaming?.abort());
els.newChat.addEventListener("click", () => {
  state.streaming?.abort();
  clearChat();
});
els.settingsToggle.addEventListener("click", () => openSettings(true));
els.closeSettings.addEventListener("click", () => openSettings(false));
els.backdrop.addEventListener("click", () => openSettings(false));

for (const input of [els.system, els.temperature, els.topK, els.maxTokens, els.kvCache]) {
  input.addEventListener("input", () => {
    updateLabels();
    saveSettings();
  });
}

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt || ""));
});

init();
