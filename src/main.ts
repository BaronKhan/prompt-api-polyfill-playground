// Minimal Prompt API playground.
//
// Behaviour:
//   1. If the browser exposes a native `LanguageModel` (the Prompt API, e.g.
//      Chrome or Edge with the on-device model flag enabled), use it directly.
//   2. Otherwise, load the experimental prompt-api-polyfill, which installs a
//      spec-compatible `window.LanguageModel` backed by a small model running
//      locally through Transformers.js (works in any modern browser).
//   3. Two hardcoded tools are exposed: get_current_date and search_docs (a
//      fake document search over synthetic data). When the native Prompt API
//      supports tools (behind a browser flag) they are passed as real tools the
//      model can call. Everywhere else (native without tool support, or the
//      polyfill, which has none) the same tools are driven by an emulated tool
//      loop: the model emits a TOOL_CALL line, we run the tool, feed the result
//      back as another turn, and the model reprocesses it into a natural-language
//      answer. This is the assistant-role tool-use emulation from the Prompt
//      API spec, generalised to arguments and multiple tools.

// Verbose debugging. Flip to true to trace session creation and the tool loop
// in the console.
const DEBUG = false;
function dbg(...args: unknown[]): void {
  if (DEBUG) console.log("[playground:debug]", ...args);
}

// ---------------------------------------------------------------------------
// Loose typings for the Prompt API surface we use (no official @types yet).
// ---------------------------------------------------------------------------
interface DownloadProgressEvent extends Event {
  loaded: number;
  total: number;
}

interface LanguageModelSession {
  prompt(input: string, options?: { signal?: AbortSignal }): Promise<string>;
  promptStreaming(
    input: string,
    options?: { signal?: AbortSignal }
  ): ReadableStream<string> | AsyncIterable<string>;
  clone?(): Promise<LanguageModelSession>;
  destroy?(): void;
}

interface LanguageModelTool {
  name: string;
  description: string;
  inputSchema: unknown;
  execute(args: unknown): Promise<string> | string;
}

interface LanguageModelExpected {
  type: string;
  languages?: string[];
}

interface LanguageModelStatic {
  availability(options?: unknown): Promise<string>;
  create(options?: {
    initialPrompts?: Array<{ role: string; content: string }>;
    expectedInputs?: LanguageModelExpected[];
    expectedOutputs?: LanguageModelExpected[];
    monitor?: (m: EventTarget) => void;
    signal?: AbortSignal;
  }): Promise<LanguageModelSession>;
}

declare global {
  interface Window {
    LanguageModel?: LanguageModelStatic;
    TRANSFORMERS_CONFIG?: Record<string, unknown>;
  }
}

// ---------------------------------------------------------------------------
// Configuration: the local model the polyfill downloads. We use the
// onnx-community Transformers.js builds, which ship WebGPU-optimised q4f16
// weights. Newer small instruct models (Qwen2.5, Llama-3.2) follow
// instructions far better than the older TinyLlama-1.1B, and the 0.5B Qwen is
// also faster, so it is the default.
// ---------------------------------------------------------------------------
interface ModelChoice {
  id: string; // Hugging Face repo id
  label: string; // shown in the picker and metrics
}

const MODELS: ModelChoice[] = [
  {
    id: "HuggingFaceTB/SmolLM2-360M-Instruct",
    label: "SmolLM2-360M-Instruct (smallest, fastest)",
  },
  {
    id: "onnx-community/Llama-3.2-1B-Instruct-ONNX",
    label: "Llama-3.2-1B-Instruct (balanced)",
  },
  {
    id: "onnx-community/Llama-3.2-3B-Instruct-ONNX",
    label: "Llama-3.2-3B-Instruct (best quality, needs a real GPU)",
  },
];

const DEFAULT_MODEL_ID = MODELS[0].id;

// Default user prompt shown in the textarea. Small models are easily confused
// about which tool to call, so the smallest model gets a simpler, more direct
// phrasing that reliably triggers search_docs. Larger models handle the richer
// "fetch and summarize" phrasing well.
const GENERAL_DEFAULT_PROMPT =
  "What were the follow ups from the offsite meeting notes?";
const MODEL_DEFAULT_PROMPTS: Record<string, string> = {
  "HuggingFaceTB/SmolLM2-360M-Instruct": "What did I miss regarding the offsite?",
};

function defaultPromptFor(modelId: string): string {
  return MODEL_DEFAULT_PROMPTS[modelId] ?? GENERAL_DEFAULT_PROMPT;
}

// Every prompt the app might auto-fill. We only swap the textarea when it still
// holds one of these, so a prompt the user typed themselves is never clobbered.
const KNOWN_DEFAULT_PROMPTS = new Set<string>([
  GENERAL_DEFAULT_PROMPT,
  ...Object.values(MODEL_DEFAULT_PROMPTS),
]);

function maybeSyncDefaultPrompt(modelId: string): void {
  const ta = document.getElementById("prompt") as HTMLTextAreaElement | null;
  if (!ta) return;
  if (KNOWN_DEFAULT_PROMPTS.has(ta.value.trim())) {
    ta.value = defaultPromptFor(modelId);
  }
}

// Auto-selected defaults per device. Without a GPU, pick the smallest/fastest
// model (WASM/CPU is slow, so size matters most). With WebGPU, pick a larger,
// higher-quality model since the GPU makes it fast.
const WASM_DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct";
const WEBGPU_DEFAULT_MODEL = "onnx-community/Llama-3.2-1B-Instruct-ONNX";

// True once the user manually changes the picker, so we stop auto-selecting.
let userPickedModel = false;

function selectedModel(): ModelChoice {
  const sel = document.getElementById("model") as HTMLSelectElement | null;
  const id = sel?.value || DEFAULT_MODEL_ID;
  return MODELS.find((m) => m.id === id) ?? MODELS[0];
}

// Presence of navigator.gpu is not enough: headless/older browsers expose the
// object but cannot return an adapter. Actually request one before committing
// to the WebGPU backend, otherwise fall back to WASM (CPU).
let webgpuAdapterInfo = "";

async function chooseTransformersDevice(): Promise<"webgpu" | "wasm"> {
  const gpu = (
    navigator as unknown as {
      gpu?: { requestAdapter?: () => Promise<unknown> };
    }
  ).gpu;
  if (!gpu?.requestAdapter) {
    webgpuAdapterInfo = "navigator.gpu is not present";
    return "wasm";
  }
  try {
    const adapter = (await gpu.requestAdapter()) as {
      info?: { vendor?: string; architecture?: string };
    } | null;
    if (adapter) {
      const info = adapter.info;
      webgpuAdapterInfo = info
        ? `${info.vendor ?? "?"} ${info.architecture ?? ""}`.trim()
        : "adapter found";
      return "webgpu";
    }
    webgpuAdapterInfo = "requestAdapter() returned null";
  } catch (e) {
    webgpuAdapterInfo = `requestAdapter() threw: ${(e as Error).message}`;
  }
  return "wasm";
}

let transformersDevice: "webgpu" | "wasm" = "wasm";
let transformersModel: ModelChoice = MODELS[0];

// Rough WebGPU throughput probe. A weak integrated GPU (for example Intel Gen 11
// on a Surface Laptop Go 2) reports a valid adapter but is slow for 1B+ models,
// so we run a small dependent multiply-add compute workload and time it. This is
// not a precise benchmark, just enough to flag clearly slow GPUs. Returns the
// elapsed milliseconds and a coarse tier, or null if it could not run.
interface GpuBenchmark {
  ms: number;
  tier: "fast" | "modest" | "slow";
}

async function benchmarkWebGpu(): Promise<GpuBenchmark | null> {
  const gpu = (navigator as unknown as { gpu?: unknown }).gpu as
    | {
        requestAdapter?: () => Promise<{
          requestDevice?: () => Promise<unknown>;
        } | null>;
      }
    | undefined;
  if (!gpu?.requestAdapter) return null;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let device: any = null;
  try {
    const adapter = await gpu.requestAdapter();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    device = await (adapter as any)?.requestDevice?.();
    if (!device) return null;

    const N = 1 << 20; // 1M elements
    // Keep the per-dispatch work small so a single command can never approach
    // the OS GPU watchdog (~2s) and trigger a TDR reset (which would lock up
    // WebGPU). This is ~8.6 GFLOP, tens of ms even on a weak integrated GPU.
    const ITER = 1 << 12; // 4096
    const STORAGE = 0x0080;
    const COPY_DST = 0x0008;

    const buffer = device.createBuffer({
      size: N * 4,
      usage: STORAGE | COPY_DST,
    });
    const module = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          let i = gid.x;
          if (i >= arrayLength(&data)) { return; }
          var x = data[i];
          for (var k = 0u; k < ${ITER}u; k = k + 1u) {
            x = x * 1.0000001 + 0.0000001;
          }
          data[i] = x;
        }
      `,
    });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } }],
    });

    const runOnce = async (): Promise<void> => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(N / 64));
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    };

    // Guards: bail out (return null, no warning) if the device is lost or the
    // whole probe takes too long, so a misbehaving GPU can never hang the page.
    const lost = device.lost?.then?.(() => "lost") as Promise<string> | undefined;
    const timeout = new Promise<string>((r) => setTimeout(() => r("timeout"), 4000));

    const measure = (async (): Promise<GpuBenchmark> => {
      await runOnce(); // warm-up (shader compile, first-run overhead)
      const runs: number[] = [];
      for (let i = 0; i < 3; i++) {
        const t0 = performance.now();
        await runOnce();
        runs.push(performance.now() - t0);
      }
      runs.sort((a, b) => a - b);
      const ms = Math.round(runs[1]); // median of 3
      const tier: GpuBenchmark["tier"] =
        ms < 25 ? "fast" : ms < 80 ? "modest" : "slow";
      return { ms, tier };
    })();

    const winner = await Promise.race(
      [measure, lost, timeout].filter(Boolean) as Array<
        Promise<GpuBenchmark | string>
      >
    );
    return typeof winner === "string" ? null : winner;
  } catch {
    return null;
  } finally {
    try {
      device?.destroy?.();
    } catch {
      /* ignore */
    }
  }
}

async function configureTransformersBackend(): Promise<void> {
  transformersDevice = await chooseTransformersDevice();

  // Auto-select the best model for the detected device, unless the user has
  // already made a manual choice via the picker.
  const sel = document.getElementById("model") as HTMLSelectElement | null;
  if (sel && !userPickedModel) {
    // On phones default to the smallest model regardless of WebGPU, since mobile
    // GPUs and memory are too limited for the larger models to run well. Detect a
    // real mobile device rather than a narrow window: prefer the UA-Client-Hints
    // mobile flag (Chromium, which is where the polyfill runs), and fall back to a
    // touch-primary device (coarse pointer, no hover).
    const uaMobile = (
      navigator as Navigator & { userAgentData?: { mobile?: boolean } }
    ).userAgentData?.mobile;
    const isMobile =
      uaMobile ?? window.matchMedia("(pointer: coarse) and (hover: none)").matches;
    const wanted = isMobile
      ? WASM_DEFAULT_MODEL
      : transformersDevice === "webgpu"
        ? WEBGPU_DEFAULT_MODEL
        : WASM_DEFAULT_MODEL;
    if (MODELS.some((m) => m.id === wanted)) sel.value = wanted;
    maybeSyncDefaultPrompt(sel.value);
  }

  transformersModel = selectedModel();
  console.log(
    `[playground] provider=polyfill model=${transformersModel.id} ` +
      `device=${transformersDevice} webgpu=${webgpuAdapterInfo}`
  );
  window.TRANSFORMERS_CONFIG = {
    apiKey: "dummy",
    modelName: transformersModel.id,
    device: transformersDevice,
    dtype: transformersDevice === "webgpu" ? "q4f16" : "q4",
  };
}

// Re-read the picker and update the polyfill config so a model change after
// page load actually takes effect on the next run. Device/dtype are unchanged.
function applySelectedPolyfillModel(): void {
  const chosen = selectedModel();
  if (chosen.id === transformersModel.id) return;
  transformersModel = chosen;
  const cfg = (window.TRANSFORMERS_CONFIG ?? {}) as Record<string, unknown>;
  cfg.modelName = chosen.id;
  window.TRANSFORMERS_CONFIG = cfg;
  const shortName = chosen.id.split("/").pop() ?? chosen.id;
  mModel.textContent =
    shortName + (transformersDevice === "webgpu" ? " (WebGPU)" : " (WASM/CPU)");
  console.log(`[playground] model switched to ${chosen.id}`);
}

// ---------------------------------------------------------------------------
// The hardcoded date tool.
// ---------------------------------------------------------------------------
function getCurrentDateString(): string {
  return new Date().toLocaleDateString(undefined, {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

const dateTool: LanguageModelTool = {
  name: "get_current_date",
  description:
    "Returns today's date as a human-readable string. Call this whenever the " +
    "user asks about the current date, today, or what day it is.",
  inputSchema: { type: "object", properties: {}, additionalProperties: false },
  execute: () => getCurrentDateString(),
};

// ---------------------------------------------------------------------------
// A fake "internal search" tool over synthetic data. This stands in for a real
// document-search or RAG backend. The data is entirely made up (no real people
// or content).
// ---------------------------------------------------------------------------
interface FakeDoc {
  source: string;
  date: string;
  text: string;
}

const FAKE_DOCS: FakeDoc[] = [
  {
    source: "Meeting notes: Q3 Offsite",
    date: "2026-07-31",
    text: "The Q3 team offsite was held on July 29 to 30 at the downtown office, room 34. Day one set the roadmap: ship the search rewrite this quarter and move the mobile app to Q4. The team agreed to cut two low-usage reports to free up capacity. Day two was planning: three new engineers start in October, and Priya now owns the onboarding guide. Follow-ups: file expense reports by August 8, and leads share headcount plans by August 15.",
  },
  {
    source: "Email from Priya Raman",
    date: "2026-08-11",
    text: "The design sync moved to Thursday at 3pm. We will review the draft spec, lock the navigation layout, and assign owners for the two open API tasks. Please read the spec beforehand and add comments on the error states.",
  },
  {
    source: "Chat: Platform channel",
    date: "2026-08-12",
    text: "The staging VPN outage is resolved. Root cause was an expired certificate; we added an alert so it will not recur. If you still cannot connect, reset your token at the internal vpn-reset page and restart the client.",
  },
  {
    source: "Email from Sam Okoro",
    date: "2026-08-13",
    text: "Budget review is done: the cloud spend for Q3 was approved with a 10 percent increase for the new GPU instances. The finance team asked us to tag all resources by project by the end of the month so the next review is faster.",
  },
];

function fakeSearch(query: string): string {
  const terms = query
    .toLowerCase()
    .split(/\W+/)
    .filter((w) => w.length > 2);
  const ranked = FAKE_DOCS.map((doc) => {
    const hay = (doc.source + " " + doc.text).toLowerCase();
    const score = terms.reduce((s, t) => s + (hay.includes(t) ? 1 : 0), 0);
    return { doc, score };
  })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 2);

  if (ranked.length === 0) return "No matching documents found.";
  return ranked
    .map((x) => `- (${x.doc.source}, ${x.doc.date}) ${x.doc.text}`)
    .join("\n");
}

const searchTool: LanguageModelTool = {
  name: "search_docs",
  description:
    "Searches the user's emails, chats and documents, which contain meeting " +
    "notes, offsites, plans, reminders, decisions, events and their dates, and " +
    "returns matching snippets. Use it for any question about the user's " +
    "schedule, meetings, or what happened and when.",
  inputSchema: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "Keywords to search for, taken from the user's question.",
      },
    },
    required: ["query"],
    additionalProperties: false,
  },
  execute: (args: unknown) => {
    const query = (args as { query?: string } | undefined)?.query ?? "";
    return fakeSearch(query);
  },
};

const TOOLS: LanguageModelTool[] = [dateTool, searchTool];

// Which tools are enabled is read from the checkboxes rendered in the UI.
function enabledTools(): LanguageModelTool[] {
  return TOOLS.filter((t) => {
    const cb = document.getElementById(
      `tool-${t.name}`
    ) as HTMLInputElement | null;
    return cb ? cb.checked : true;
  });
}

// Render a checkbox per tool so developers can see the tools the model is given
// and enable or disable each one. Built from the TOOLS array so it stays in sync.
// For search_docs, also list the synthetic corpus it searches so developers know
// what queries will return hits.
function renderToolToggles(): void {
  const container = document.getElementById("tools");
  if (!container) return;
  container.innerHTML = "";
  for (const tool of TOOLS) {
    const label = document.createElement("label");
    label.className = "checkbox";

    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = `tool-${tool.name}`;
    input.checked = tool.name !== "get_current_date";

    const text = document.createElement("span");
    const args = toolArgsHint(tool);
    const sig = args.startsWith("{}") ? "()" : `(${args})`;
    text.innerHTML =
      `<code>${tool.name}${sig}</code> ` +
      `<span class="tool-desc">${tool.description}</span>`;

    label.append(input, text);
    container.append(label);

    if (tool.name === "search_docs") {
      const details = document.createElement("details");
      details.className = "tool-data";
      const summary = document.createElement("summary");
      summary.textContent = `Synthetic documents searched (${FAKE_DOCS.length}) - all fake data`;
      const list = document.createElement("ul");
      for (const doc of FAKE_DOCS) {
        const li = document.createElement("li");
        li.textContent = `(${doc.source}, ${doc.date}) ${doc.text}`;
        list.append(li);
      }
      details.append(summary, list);
      container.append(details);
    }
  }
}

// A short "{...}" hint of a tool's arguments, derived from its inputSchema.
function toolArgsHint(tool: LanguageModelTool): string {
  const schema = tool.inputSchema as
    | { properties?: Record<string, { type?: string }> }
    | undefined;
  const props = schema?.properties ?? {};
  const keys = Object.keys(props);
  if (keys.length === 0) return "{} (takes no arguments)";
  return (
    "{" +
    keys.map((k) => `"${k}": ${props[k]?.type ?? "string"}`).join(", ") +
    "}"
  );
}

// System prompt: instructions only. The few-shot examples are provided as real
// conversation turns (see buildExampleTurns), not embedded as text, so small
// models do not echo "User:"/"Assistant:" lines back.
// The exact TOOL_CALL line for a tool, used to show the model precisely what to
// emit (small models copy the example format).
function toolCallExample(tool: LanguageModelTool): string {
  const schema = tool.inputSchema as
    | { properties?: Record<string, { type?: string }> }
    | undefined;
  const keys = Object.keys(schema?.properties ?? {});
  const args =
    keys.length === 0 ? "{}" : `{${keys.map((k) => `"${k}":"..."`).join(",")}}`;
  return `TOOL_CALL {"name":"${tool.name}","arguments":${args}}`;
}

function buildEmulatedSystemPrompt(
  basePrompt: string,
  tools: LanguageModelTool[]
): string {
  const base = basePrompt || "You are a concise, helpful assistant.";
  if (tools.length === 0) {
    return base + "\n\nYou have no tools available; answer questions directly.";
  }

  return [
    base,
    "",
    "You are connected to tools. You do NOT know anything about the user's " +
      "documents, emails, chats, meetings or events, and you must NOT guess or " +
      "make up an answer. To get information you MUST call a tool.",
    "",
    "To call a tool, output EXACTLY one line and nothing else, then stop:",
    'TOOL_CALL {"name":"<tool_name>","arguments":{ ... }}',
    "",
    "Do NOT write a line starting with TOOL_RESULT. Do NOT invent results or " +
      "answers. Only emit the TOOL_CALL and wait.",
    "",
    "Tools you can call:",
    ...tools.flatMap((t) => [
      `- ${t.name}: ${t.description}`,
      `  Call it as: ${toolCallExample(t)}`,
    ]),
    "",
    "After you emit a TOOL_CALL you will receive a message starting with " +
      "TOOL_RESULT that contains the real result. Only then, write a short " +
      "natural-language answer using it. Never show the raw TOOL_RESULT text.",
    "",
    "Choose search_docs for any question about the user's content (meetings, " +
      "offsites, plans, reminders, decisions, events or named topics). Choose " +
      "get_current_date only when the user asks for today's date. Build the " +
      "query from the user's own words.",
  ].join("\n");
}

// One short few-shot, as real turns, ending at the assistant's TOOL_CALL. We do
// NOT include the TOOL_RESULT turn or the final answer: small models copy those
// TOOL_RESULT lines verbatim instead of emitting a TOOL_CALL. The answer step is
// covered by the system-prompt instruction.
function buildExampleTurns(
  tools: LanguageModelTool[]
): Array<{ role: string; content: string }> {
  if (!tools.some((t) => t.name === "search_docs")) return [];
  return [
    { role: "user", content: "What did the team decide about the budget?" },
    {
      role: "assistant",
      content:
        'TOOL_CALL {"name":"search_docs","arguments":{"query":"budget decision"}}',
    },
  ];
}

interface ParsedToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

// Try to parse a JSON object, tolerating the sloppy JSON small models emit
// (single quotes, trailing commas).
function tryParseJson(s: string): Record<string, unknown> | null {
  try {
    return JSON.parse(s) as Record<string, unknown>;
  } catch {
    /* fall through */
  }
  try {
    const fixed = s.replace(/'/g, '"').replace(/,\s*([}\]])/g, "$1");
    return JSON.parse(fixed) as Record<string, unknown>;
  } catch {
    return null;
  }
}

// Extract the first tool call from a model reply. Small models produce sloppy
// JSON, so this is deliberately lenient: it brace-matches the first object and
// tries a tolerant JSON parse, then falls back to regex-extracting the tool name
// and a "query" argument. Known tool names are used to disambiguate.
function parseToolCall(text: string): ParsedToolCall | null {
  const markerIdx = text.search(/tool_call/i);
  if (markerIdx === -1) return null;
  const after = text.slice(markerIdx);
  const known = TOOLS.map((t) => t.name);

  const normalize = (
    name: unknown,
    args: unknown
  ): ParsedToolCall | null => {
    if (typeof name !== "string" || !name) return null;
    return {
      name,
      arguments:
        args && typeof args === "object"
          ? (args as Record<string, unknown>)
          : {},
    };
  };

  // 1. Brace-match the first { ... } and parse it (tolerantly).
  const start = after.indexOf("{");
  if (start !== -1) {
    let depth = 0;
    let end = -1;
    for (let i = start; i < after.length; i++) {
      if (after[i] === "{") depth++;
      else if (after[i] === "}" && --depth === 0) {
        end = i;
        break;
      }
    }
    if (end !== -1) {
      const obj = tryParseJson(after.slice(start, end + 1));
      if (obj && typeof obj.name === "string") {
        return normalize(obj.name, obj.arguments);
      }
    }
  }

  // 2. Regex fallback: pull out the tool name and an optional query.
  const nameMatch = after.match(/name"?\s*:\s*"?([a-z_][a-z0-9_]*)"?/i);
  let name = nameMatch?.[1];
  if (!name) {
    // No explicit name field: match any known tool name mentioned after the marker.
    name = known.find((n) => new RegExp(n, "i").test(after));
  }
  if (!name) return null;
  const queryMatch = after.match(/query"?\s*:\s*"([^"]*)"/i);
  return normalize(name, queryMatch ? { query: queryMatch[1] } : {});
}

// Small models often keep writing after their answer, inventing extra
// "User:"/"Assistant:" turns or another tool call, or emit a malformed TOOL_CALL
// that could not be parsed. Strip a leading role label and cut the reply at the
// first such marker so we show only the real answer.
function cleanReply(text: string): string {
  const t = text.trim().replace(/^(assistant|ai|model|bot)\s*[:>\-]\s*/i, "");
  const markers = [
    /(?:^|\n)\s*user\s*:/i,
    /(?:^|\n)\s*assistant\s*:/i,
    /(?:^|\n)\s*system\s*:/i,
    /(?:^|\n)\s*tool_result/i,
    /(?:^|\n)\s*tool_call/i, // a stray TOOL_CALL line means the answer has ended
  ];
  let cut = t.length;
  for (const m of markers) {
    const idx = t.search(m);
    if (idx !== -1 && idx < cut) cut = idx;
  }
  return t.slice(0, cut).trim();
}

// ---------------------------------------------------------------------------
// DOM helpers.
// ---------------------------------------------------------------------------
const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const statusEl = $("status");
const outputEl = $<HTMLPreElement>("output");
const runBtn = $<HTMLButtonElement>("run");
const stopBtn = $<HTMLButtonElement>("stop");
const promptEl = $<HTMLTextAreaElement>("prompt");
const systemEl = $<HTMLTextAreaElement>("system-prompt");
const mProvider = $("m-provider");
const mModel = $("m-model");
const mTools = $("m-tools");
const mTime = $("m-time");

function setStatus(text: string, kind: "pending" | "ok" | "warn"): void {
  statusEl.textContent = text;
  statusEl.className = `status status--${kind}`;
}

// ---------------------------------------------------------------------------
// Provider detection and setup.
// ---------------------------------------------------------------------------
type Provider = "native" | "polyfill";

let provider: Provider;

function forcePolyfillRequested(): boolean {
  const params = new URLSearchParams(location.search);
  return params.has("polyfill") && params.get("polyfill") !== "0";
}

async function setup(): Promise<void> {
  renderToolToggles();
  const forced = forcePolyfillRequested();
  const nativePresent = "LanguageModel" in window && !!window.LanguageModel;

  if (nativePresent && !forced) {
    provider = "native";
    // The model picker only affects the polyfill, so hide it on the native path.
    const modelRow = document.getElementById("model-row");
    if (modelRow) modelRow.hidden = true;
    mProvider.textContent = "Native Prompt API";
    // The Prompt API does not expose the model name, so we cannot show it. It is
    // internal to the browser (Edge: Aion-1.0-Instruct or Phi-4-mini; Chrome:
    // Gemini Nano).
    mModel.textContent = "browser built-in (name not exposed by the API)";
    mModel.title =
      "The Prompt API does not expose the model name. On Edge it is " +
      "Aion-1.0-Instruct or Phi-4-mini; on Chrome, Gemini Nano. See " +
      "edge://on-device-internals or chrome://on-device-internals.";
    mTools.textContent = "determined on first prompt";
    setStatus(
      "Native Prompt API detected. Using the browser's built-in on-device " +
        "model. Model download progress (if any) and tool support are shown on " +
        "your first prompt. Add ?polyfill=1 to the URL to force the polyfill " +
        "instead.",
      "ok"
    );
  } else {
    provider = "polyfill";
    mProvider.textContent = "prompt-api-polyfill";
    setStatus(
      forced && nativePresent
        ? "Forcing the polyfill (?polyfill=1). Loading..."
        : "Native Prompt API not found. Loading polyfill...",
      "pending"
    );
    await configureTransformersBackend();
    const shortName = transformersModel.id.split("/").pop() ?? transformersModel.id;
    mModel.textContent =
      shortName + (transformersDevice === "webgpu" ? " (WebGPU)" : " (WASM/CPU)");
    mTools.textContent = "emulated (polyfill has no native tools)";
    // Tell the polyfill to install itself even if a native LanguageModel exists.
    (window as unknown as Record<string, unknown>).__FORCE_PROMPT_API_POLYFILL__ =
      true;
    await import("prompt-api-polyfill");
    if (transformersDevice === "wasm") {
      setStatus(
        `WebGPU is unavailable (${webgpuAdapterInfo}), so the model runs on the ` +
          "CPU via WASM, which is very slow (tens of seconds per answer). For " +
          "fast inference, use a browser/GPU with WebGPU enabled. Polyfill " +
          "loaded; the model downloads on your first prompt.",
        "warn"
      );
    } else {
      setStatus(
        `Polyfill loaded on WebGPU (${webgpuAdapterInfo}). ${shortName} ` +
          "downloads locally on your first prompt (the first run can fetch " +
          "several hundred MB), then generation is GPU-accelerated.",
        "ok"
      );
      // Probe how fast the GPU actually is; warn on weak integrated GPUs. This
      // is awaited so its temporary WebGPU device is created and destroyed
      // before any model loads, avoiding a device-creation race that could hang
      // the GPU (seen with the fast-loading 0.5B model).
      dbg("running GPU benchmark before enabling prompts");
      await checkGpuPower();
      dbg("GPU benchmark done");
    }
  }
  runBtn.disabled = false;
  dbg("setup complete; run button enabled");
}

// Run the GPU benchmark and show a small note if the GPU looks too weak for
// larger local models.
async function checkGpuPower(): Promise<void> {
  const note = document.getElementById("gpu-note");
  const result = await benchmarkWebGpu();
  if (!result || !note) return;
  console.log(
    `[playground] GPU benchmark: ${result.ms} ms (${result.tier}), ${webgpuAdapterInfo}`
  );
  if (result.tier === "fast") return;

  const strength = result.tier === "slow" ? "quite slow" : "modest";
  note.textContent =
    `Heads up: this GPU looks ${strength} for local models ` +
    `(benchmark ${result.ms} ms on ${webgpuAdapterInfo}). Integrated GPUs like ` +
    "Intel Gen 11 struggle with 1B+ models. Prefer the smallest model " +
    "(Qwen2.5-0.5B); larger ones may be slow.";
  note.hidden = false;
}

// ---------------------------------------------------------------------------
// Model download progress. Wired identically for both providers: the native
// Prompt API and the polyfill both emit `downloadprogress` on the monitor
// EventTarget, with { loaded, total }. We normalise to a percentage and never
// let the reported figure jump backwards.
// ---------------------------------------------------------------------------
function makeMonitor(): (m: EventTarget) => void {
  let maxPct = 0;
  const label =
    provider === "native"
      ? "browser model"
      : (transformersModel.id.split("/").pop() ?? "model");
  return (m: EventTarget) => {
    m.addEventListener("downloadprogress", (e: Event) => {
      const ev = e as DownloadProgressEvent;
      const raw = ev.total ? (ev.loaded / ev.total) * 100 : 0;
      const pct = Math.min(100, Math.max(maxPct, Math.round(raw)));
      maxPct = pct;
      dbg("monitor: downloadprogress", { label, pct });
      if (pct >= 100) {
        setStatus(`${label} ready. Generating...`, "ok");
      } else {
        setStatus(`Downloading ${label}: ${pct}%`, "pending");
      }
    });
  };
}

// Create a session for the emulated tool loop. This drives the tools ourselves
// via the assistant-role TOOL_CALL protocol on whatever model is present (the
// native model when available, the polyfill model otherwise). It is used
// everywhere because current on-device models do not execute native tool calls
// well: create({ tools }) is accepted, but the model tends to echo the tool
// schema instead of calling the tool and answering. See the README note.
async function createSession(
  systemPrompt: string,
  tools: LanguageModelTool[]
): Promise<LanguageModelSession> {
  const LM = window.LanguageModel as LanguageModelStatic;
  dbg("createSession: calling LM.create()", {
    model: transformersModel.id,
    device: transformersDevice,
    tools: tools.map((t) => t.name),
  });
  const session = await LM.create({
    initialPrompts: [
      {
        role: "system",
        content: buildEmulatedSystemPrompt(systemPrompt, tools),
      },
      // Few-shot demonstrations as real turns, so the model learns the protocol
      // by example without echoing "User:"/"Assistant:" text back.
      ...buildExampleTurns(tools),
    ],
    // Declaring a text output language silences the "no output language
    // specified" warning from the native Prompt API.
    expectedInputs: [{ type: "text", languages: ["en"] }],
    expectedOutputs: [{ type: "text", languages: ["en"] }],
    monitor: makeMonitor(),
  });
  dbg("createSession: LM.create() resolved");
  return session;
}

// The polyfill (and native API) re-initialise the model on every create(), which
// on the polyfill means reloading the model into memory each prompt (slow, and
// it can exhaust memory). To avoid that, keep one base session per configuration
// and clone it per run, so the model loads once. Each run's clone starts from
// the same system + few-shot context without the previous run's turns.
let baseSession: LanguageModelSession | null = null;
let baseSessionKey = "";

async function getRunSession(
  systemPrompt: string,
  tools: LanguageModelTool[]
): Promise<{ session: LanguageModelSession; disposable: boolean }> {
  const key = [
    provider,
    provider === "polyfill" ? transformersModel.id : "native",
    systemPrompt,
    tools.map((t) => t.name).join(","),
  ].join("|");

  if (!baseSession || baseSessionKey !== key) {
    dbg("getRunSession: creating new base session", { key });
    baseSession?.destroy?.();
    baseSession = await createSession(systemPrompt, tools);
    baseSessionKey = key;
  } else {
    dbg("getRunSession: reusing cached base session");
  }

  if (baseSession.clone) {
    try {
      dbg("getRunSession: cloning base session");
      const cloned = await baseSession.clone();
      dbg("getRunSession: clone resolved");
      return { session: cloned, disposable: true };
    } catch (e) {
      dbg("getRunSession: clone failed, reusing base", e);
    }
  }
  return { session: baseSession, disposable: false };
}

// Drive the emulated tool loop: prompt, parse a TOOL_CALL, run the tool, feed
// the result back, repeat, then show the model's synthesised answer.
async function runEmulatedToolLoop(
  session: LanguageModelSession,
  userPrompt: string,
  tools: LanguageModelTool[],
  signal: AbortSignal
): Promise<void> {
  const trace: string[] = [];
  let lastResult = "";
  const showTrace = (note: string): void => {
    outputEl.textContent = trace.join("\n") + "\n\n" + note;
  };

  dbg("loop: session.prompt(userPrompt) start", { userPrompt });
  let reply = await session.prompt(userPrompt, { signal });
  dbg("loop: first reply received", { reply: reply.slice(0, 200) });

  // If a weak model did not emit a TOOL_CALL (it answered, refused, or
  // hallucinated a TOOL_RESULT), nudge it once with an explicit example.
  if (tools.length > 0 && !parseToolCall(reply)) {
    dbg("loop: no tool call on first reply, sending corrective nudge");
    setStatus("Asking the model to use a tool...", "pending");
    const example = toolCallExample(tools[0]);
    reply = await session.prompt(
      "You did not call a tool, and you must not answer from your own " +
        "knowledge. Emit ONLY a single TOOL_CALL line now and nothing else, " +
        `for example: ${example}`,
      { signal }
    );
    dbg("loop: reply after corrective nudge", { reply: reply.slice(0, 200) });
  }

  for (let step = 0; step < 3; step++) {
    const call = parseToolCall(reply);
    dbg("loop: parseToolCall", { step, call });
    if (!call) break;

    const tool = tools.find((t) => t.name === call.name);
    const result = tool
      ? String(await tool.execute(call.arguments))
      : `Unknown or disabled tool: ${call.name}`;
    lastResult = result;
    dbg("loop: tool executed", { name: call.name, result: result.slice(0, 120) });

    trace.push(
      `[tool] ${call.name}(${JSON.stringify(call.arguments)}) ->\n${result}`
    );
    showTrace("(model is reprocessing the tool result...)");

    setStatus("Reprocessing tool result...", "ok");
    // Let the model chain another tool if it still needs one, otherwise answer.
    dbg("loop: session.prompt(TOOL_RESULT) start", { step });
    reply = await session.prompt(
      `TOOL_RESULT: ${result}\n\nIf you still need more information, call ` +
        "another tool now. Otherwise, answer the user's question in plain " +
        "language using the results so far.",
      { signal }
    );
    dbg("loop: reply after TOOL_RESULT", { step, reply: reply.slice(0, 200) });
  }
  dbg("loop: finished", { finalReply: reply.slice(0, 200) });

  let answer = cleanReply(reply);
  if (!answer && lastResult) {
    // The model retrieved data but would not summarise it. Show the retrieved
    // result directly rather than a useless error (an always-retrieve fallback).
    answer = `(The model did not summarise the result. Retrieved:)\n${lastResult}`;
  } else if (!answer) {
    // No clean answer and nothing retrieved: show the model's raw output so the
    // developer can see what it produced, rather than hiding it.
    const raw = reply.trim();
    answer = raw
      ? `(No tool was called; raw model output:)\n${raw}`
      : "(The model returned an empty response. Small models can be unreliable; " +
        "try a larger model or a browser with the native Prompt API.)";
  }
  outputEl.textContent = trace.length
    ? trace.join("\n") + "\n\n" + answer
    : answer;
}

// ---------------------------------------------------------------------------
// Run a single prompt.
// ---------------------------------------------------------------------------
let currentController: AbortController | null = null;

async function run(): Promise<void> {
  const userPrompt = promptEl.value.trim();
  if (!userPrompt) return;
  dbg("run: clicked", { provider, model: transformersModel.id, userPrompt });

  runBtn.disabled = true;
  stopBtn.disabled = false;
  outputEl.textContent = "";
  mTime.textContent = "-";
  const started = performance.now();
  currentController = new AbortController();

  try {
    setStatus("Preparing session...", "pending");

    // On the polyfill path, apply the currently selected model (the picker can
    // change after load, so we re-read it and update the config each run).
    if (provider === "polyfill") {
      applySelectedPolyfillModel();
    }

    const tools = enabledTools();
    const { session, disposable } = await getRunSession(
      systemEl.value.trim(),
      tools
    );
    const loopLabel =
      provider === "native"
        ? "assistant-role tool loop (native model)"
        : "assistant-role tool loop (polyfill)";
    mTools.textContent =
      tools.length > 0
        ? `${loopLabel}; ${tools.map((t) => t.name).join(", ")}`
        : `${loopLabel}; no tools enabled`;

    setStatus("Generating...", "ok");
    await runEmulatedToolLoop(
      session,
      userPrompt,
      tools,
      currentController.signal
    );

    // Only dispose per-run clones; keep the cached base session for reuse.
    if (disposable) session.destroy?.();
    setStatus("Done.", "ok");
  } catch (err) {
    const e = err as Error;
    if (e.name === "AbortError") {
      setStatus("Stopped.", "warn");
    } else {
      setStatus(`Error: ${e.message}`, "warn");
      outputEl.textContent = `${outputEl.textContent}\n\n[error] ${e.message}`;
      console.error(e);
    }
  } finally {
    mTime.textContent = String(Math.round(performance.now() - started));
    runBtn.disabled = false;
    stopBtn.disabled = true;
    currentController = null;
  }
}

runBtn.addEventListener("click", () => void run());
stopBtn.addEventListener("click", () => currentController?.abort());

// Track manual model changes so device auto-selection does not override them.
document.getElementById("model")?.addEventListener("change", (e) => {
  userPickedModel = true;
  maybeSyncDefaultPrompt((e.target as HTMLSelectElement).value);
});

// ---------------------------------------------------------------------------
// Delete cached models. Transformers.js caches model weights in the browser
// (Cache Storage, and OPFS when cross-origin storage is used), so a downloaded
// model is reused and never re-fetched. This clears those caches, then reloads
// so the polyfill re-initialises from a clean slate.
// ---------------------------------------------------------------------------
async function deleteCachedModels(): Promise<void> {
  const deleteBtn = document.getElementById(
    "delete-model"
  ) as HTMLButtonElement | null;
  if (deleteBtn) deleteBtn.disabled = true;
  setStatus("Deleting cached model files...", "pending");

  let cachesCleared = 0;
  try {
    if ("caches" in window) {
      const keys = await caches.keys();
      for (const key of keys) {
        if (await caches.delete(key)) cachesCleared++;
      }
    }
  } catch (e) {
    console.warn("Cache Storage clear failed:", e);
  }

  // Origin Private File System (used by some Transformers.js storage paths).
  try {
    const storage = navigator.storage as unknown as {
      getDirectory?: () => Promise<{
        entries?: () => AsyncIterable<[string, unknown]>;
        removeEntry?: (name: string, opts?: { recursive?: boolean }) => Promise<void>;
      }>;
    };
    const root = await storage.getDirectory?.();
    if (root?.entries && root.removeEntry) {
      for await (const [name] of root.entries()) {
        await root.removeEntry(name, { recursive: true }).catch(() => {});
      }
    }
  } catch (e) {
    console.warn("OPFS clear failed:", e);
  }

  setStatus(
    `Cleared ${cachesCleared} cache store(s). Reloading to re-initialise...`,
    "ok"
  );
  setTimeout(() => location.reload(), 800);
}

document
  .getElementById("delete-model")
  ?.addEventListener("click", () => void deleteCachedModels());

void setup();

export {};
