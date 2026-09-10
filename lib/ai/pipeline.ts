import { jsonrepair } from "jsonrepair";
import { runLlm } from "./llm";
import { extractJson, isLikelyTruncatedJson } from "./json-util";
import { SYSTEM_PROMPT_DIGEST_EN, SYSTEM_PROMPT_DIGEST_ZH } from "./prompts";
import { REPORT_LOCALE } from "../sources/registry";
import type { Category, RawArticle } from "../sources/types";

const SYSTEM_PROMPT_DIGEST =
  REPORT_LOCALE === "en" ? SYSTEM_PROMPT_DIGEST_EN : SYSTEM_PROMPT_DIGEST_ZH;

export interface BriefItem {
  title: string;
  url: string;
  source: string;
  summary: string;
  importance: number;
}

export interface DailyReport {
  hero_headline: string;
  daily_overview: string;
  tech_briefs: BriefItem[];
  finance_briefs: BriefItem[];
  politics_briefs: BriefItem[];
  editor_note: string;
  keywords: string[];
  /** Optional trading-signals section, present when scripts/daily.ts ran successfully. */
  trading?: TradingSection;
}

import type { TickerAnalysis } from "../trading/signals";
import type { CryptoGlobalStats } from "../trading/coingecko";
import type { FearGreedSnapshot } from "../trading/fear-greed";
import type { TradingCommentary } from "./trading-commentary";

export interface TradingSection extends TradingCommentary {
  generated_at: string;
  tickers: TickerAnalysis[];
  crypto_fear_greed?: FearGreedSnapshot;
  crypto_global?: CryptoGlobalStats;
}

export interface ArticleInput extends RawArticle {
  source: string;
}

const PER_CATEGORY_LIMIT: Record<Category, number> = {
  tech: 25,
  finance: 20,
  politics: 15,
};

const MAX_AGE_DAYS = 14;

/**
 * Pick `limit` items from `items` so every source gets a fair shot.
 *
 * Why this exists: the previous `slice(0, limit)` honored insertion order,
 * which is the source-iteration order in daily.ts. That gave whichever
 * source came first 100% of the quota — e.g. all 25 tech slots filled by
 * Hacker News before GitHub Trending / Solidot / V2EX / 阮一峰 got a turn.
 *
 * Strategy: drop items older than MAX_AGE_DAYS, group by sourceId,
 * sort each bucket newest-first, then round-robin one item per source
 * until we hit the limit. Sources with fewer items naturally drop out
 * and others absorb the slack.
 */
function selectRoundRobin(
  items: ArticleInput[],
  limit: number,
): ArticleInput[] {
  const cutoff = Date.now() - MAX_AGE_DAYS * 86_400_000;
  const fresh = items.filter(
    (it) => !it.publishedAt || it.publishedAt.getTime() >= cutoff,
  );

  const bySource = new Map<string, ArticleInput[]>();
  for (const it of fresh) {
    const arr = bySource.get(it.sourceId) ?? [];
    arr.push(it);
    bySource.set(it.sourceId, arr);
  }
  for (const arr of bySource.values()) {
    arr.sort(
      (a, b) =>
        (b.publishedAt?.getTime() ?? 0) - (a.publishedAt?.getTime() ?? 0),
    );
  }

  const buckets = Array.from(bySource.values());
  const out: ArticleInput[] = [];
  let madeProgress = true;
  while (out.length < limit && madeProgress) {
    madeProgress = false;
    for (const b of buckets) {
      if (b.length === 0) continue;
      out.push(b.shift()!);
      madeProgress = true;
      if (out.length >= limit) break;
    }
  }
  return out;
}

type DigestScale = {
  /** Multiply PER_CATEGORY_LIMIT by this (0–1]. */
  inputScale: number;
  /** Cap excerpt chars per candidate. */
  excerptChars: number;
  /** Ask the model for a shorter overview / fewer briefs. */
  compact: boolean;
};

async function callOnce(
  userPayloadJson: string,
  scale: DigestScale,
): Promise<DailyReport> {
  // Claude Code CLI's built-in system prompt biases the model toward
  // conversational markdown output. Anchor the format expectation in the
  // user message (instruction recency wins) *and* explicitly demand every
  // schema field be populated — without this Sonnet has been observed to
  // emit a JSON shell with empty arrays to "satisfy" a JSON-only ask.
  const compactHint =
    REPORT_LOCALE === "en"
      ? scale.compact
        ? [
            "**Length budget (critical):** keep the whole JSON under ~2500 tokens.",
            "  - daily_overview: 80-120 words",
            "  - tech_briefs / finance_briefs: **exactly 3** each",
            "  - politics_briefs: **exactly 2**",
            "  - each brief summary: 1-2 short sentences",
            "  - editor_note: ≤40 words; keywords: 5",
            "",
          ]
        : []
      : scale.compact
        ? [
            "**长度预算（关键）**：整份 JSON 控制在约 2500 token 以内。",
            "  - daily_overview: 80-120 字",
            "  - tech_briefs / finance_briefs: **各恰好 3 条**",
            "  - politics_briefs: **恰好 2 条**",
            "  - 每条 summary: 1-2 句短句",
            "  - editor_note: ≤40 字；keywords: 5 个",
            "",
          ]
        : [];

  const userPrompt =
    REPORT_LOCALE === "en"
      ? [
          "**Output language: ENGLISH ONLY.** Every string value in the JSON — hero_headline, daily_overview, every brief's title/summary, editor_note, keywords — must be written entirely in English. No Chinese characters anywhere.",
          "",
          "Your task: generate today's daily brief from the candidate news below. **The response MUST be a single valid JSON object** — starts with `{`, ends with `}`, no markdown, no code fences, no explanations.",
          "",
          "The JSON must contain every field non-empty (briefs arrays per the system-prompt counts):",
          "  - hero_headline: 10-25 word headline of the day",
          scale.compact
            ? "  - daily_overview: **80-120 word** paragraph covering tech / finance / politics"
            : "  - daily_overview: **150-250 word** paragraph covering tech / finance / politics signals so a reader sees the whole picture at a glance",
          scale.compact
            ? "  - tech_briefs: **exactly 3** tech BriefItems"
            : "  - tech_briefs: **3-5** tech BriefItems",
          scale.compact
            ? "  - finance_briefs: **exactly 3** finance BriefItems"
            : "  - finance_briefs: **3-5** finance BriefItems",
          scale.compact
            ? "  - politics_briefs: **exactly 2** politics BriefItems"
            : "  - politics_briefs: **2-3** politics BriefItems",
          scale.compact
            ? "  - editor_note: ≤40 word editor's note"
            : "  - editor_note: 30-60 word editor's note",
          scale.compact ? "  - keywords: 5 keywords" : "  - keywords: 5-8 keywords",
          "",
          ...compactHint,
          "BriefItem fields: title, url (copied verbatim from candidate), source, summary, importance (1-10).",
          "**Quote rule (important!)**: For any quotation INSIDE a JSON string, use single quotes ' or curly quotes '\" — **never** raw double quotes \", which break JSON parsing.",
          "No trailing commas.",
          "",
          `Candidate news (JSON array, ${userPayloadJson.length} chars):`,
          userPayloadJson,
        ].join("\n")
      : [
          "你的任务：根据下方候选新闻，生成一份当日简报，**响应必须是一个合法 JSON 对象**——以 `{` 开头，以 `}` 结尾，不要 markdown / 不要代码围栏 / 不要任何解释。",
          "",
          "JSON 必须包含全部字段且不能为空（briefs 数组按 system prompt 规定的条数填充）：",
          "  - hero_headline: 10-25 字的当日一句话头条",
          scale.compact
            ? "  - daily_overview: **80-120 字** 的当日总览段落，一段话覆盖技术 / 财经 / 时政"
            : "  - daily_overview: **150-220 字** 的当日总览段落，一段话覆盖技术 / 财经 / 时政 的核心信号，让读者一眼抓住全貌",
          scale.compact
            ? "  - tech_briefs: **恰好 3 条** 科技 BriefItem"
            : "  - tech_briefs: **3-5 条** 科技 BriefItem",
          scale.compact
            ? "  - finance_briefs: **恰好 3 条** 财经 BriefItem"
            : "  - finance_briefs: **3-5 条** 财经 BriefItem",
          scale.compact
            ? "  - politics_briefs: **恰好 2 条** 时政 BriefItem"
            : "  - politics_briefs: **2-3 条** 时政 BriefItem",
          scale.compact
            ? "  - editor_note: ≤40 字的编辑短评"
            : "  - editor_note: 30-60 字的编辑短评",
          scale.compact ? "  - keywords: 5 个关键词" : "  - keywords: 5-8 个关键词",
          "",
          ...compactHint,
          "BriefItem 字段：title、url（必须从候选条目原样选取）、source、summary、importance(1-10)。",
          "**引号规则（重要！）**：JSON 字符串内的中文引用请使用**中文全角引号**「」或者 “”，**绝对不要**用英文双引号 \" —— 那会导致 JSON 解析失败。例：写 商务部回应「内卷」 而不是 商务部回应\"内卷\"。",
          "不要使用单引号、不要末尾多余逗号。",
          "",
          "候选新闻（JSON 数组，共 " + userPayloadJson.length + " 字符）：",
          userPayloadJson,
        ].join("\n");
  const { text } = await runLlm({
    systemPrompt: SYSTEM_PROMPT_DIGEST,
    userPrompt,
  });
  const cleaned = extractJson(text);
  if (isLikelyTruncatedJson(cleaned) || isLikelyTruncatedJson(text)) {
    try {
      const fs = await import("node:fs");
      fs.mkdirSync("logs", { recursive: true });
      const ts = new Date().toISOString().replace(/[:.]/g, "-");
      fs.writeFileSync(`logs/claude-raw-${ts}.txt`, text, "utf8");
      fs.writeFileSync(`logs/claude-cleaned-${ts}.txt`, cleaned, "utf8");
      console.warn(
        `[pipeline] truncated JSON detected; raw at logs/claude-raw-${ts}.txt`,
      );
    } catch {
      // best-effort logging
    }
    throw new Error(
      `Unexpected end of JSON input (truncated digest, cleanedChars=${cleaned.length})`,
    );
  }
  let parsed: Partial<DailyReport>;
  try {
    parsed = JSON.parse(cleaned) as Partial<DailyReport>;
  } catch (strictErr) {
    // LLMs routinely emit JSON with unescaped quotes inside Chinese
    // strings (e.g. 商务部回应"内卷"). jsonrepair fixes most of these
    // mechanically before we ever surface a failure.
    try {
      const repaired = jsonrepair(cleaned);
      parsed = JSON.parse(repaired) as Partial<DailyReport>;
      console.warn("[pipeline] JSON.parse failed but jsonrepair recovered");
    } catch {
      try {
        const fs = await import("node:fs");
        fs.mkdirSync("logs", { recursive: true });
        const ts = new Date().toISOString().replace(/[:.]/g, "-");
        fs.writeFileSync(`logs/claude-raw-${ts}.txt`, text, "utf8");
        fs.writeFileSync(`logs/claude-cleaned-${ts}.txt`, cleaned, "utf8");
        console.warn(
          `[pipeline] both JSON.parse and jsonrepair failed; raw at logs/claude-raw-${ts}.txt`,
        );
      } catch {
        // best-effort logging
      }
      throw strictErr;
    }
  }
  return {
    hero_headline: parsed.hero_headline ?? "",
    daily_overview: parsed.daily_overview ?? "",
    tech_briefs: parsed.tech_briefs ?? [],
    finance_briefs: parsed.finance_briefs ?? [],
    politics_briefs: parsed.politics_briefs ?? [],
    editor_note: parsed.editor_note ?? "",
    keywords: parsed.keywords ?? [],
  };
}

function buildPayload(
  articles: ArticleInput[],
  scale: DigestScale,
): string {
  const grouped: Record<Category, ArticleInput[]> = {
    tech: [],
    finance: [],
    politics: [],
  };
  for (const a of articles) grouped[a.category].push(a);

  const compact = (Object.keys(grouped) as Category[]).flatMap((c) =>
    selectRoundRobin(
      grouped[c],
      Math.max(4, Math.floor(PER_CATEGORY_LIMIT[c] * scale.inputScale)),
    ),
  );

  const userPayload = compact.map((a, i) => ({
    n: i + 1,
    title: a.title,
    url: a.url,
    source: a.source,
    category: a.category,
    excerpt: (a.excerpt ?? "").slice(0, scale.excerptChars),
    published: a.publishedAt?.toISOString() ?? "",
  }));
  return JSON.stringify(userPayload);
}

function isTruncationLike(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return /truncat|unexpected end of json|finish_reason=length/i.test(msg);
}

export async function generateDailyReport(
  articles: ArticleInput[],
): Promise<{ report: DailyReport; tokensUsed: number }> {
  // Progressive shrink: SenseNova flash-lite (and similar lite models) often
  // hit an output cap mid-JSON on the full digest. Same-prompt retry alone
  // does not help — reduce candidates + ask for a shorter schema instead.
  const attempts: DigestScale[] = [
    { inputScale: 1, excerptChars: 200, compact: false },
    { inputScale: 0.6, excerptChars: 120, compact: true },
    { inputScale: 0.35, excerptChars: 80, compact: true },
  ];

  let report: DailyReport | undefined;
  let lastErr: unknown;
  for (let i = 0; i < attempts.length; i++) {
    const scale = attempts[i];
    const userPayloadJson = buildPayload(articles, scale);
    try {
      if (i > 0) {
        console.warn(
          `[pipeline] digest attempt ${i + 1}/${attempts.length} ` +
            `(inputScale=${scale.inputScale}, excerpt=${scale.excerptChars}, compact=${scale.compact}, payloadChars=${userPayloadJson.length})`,
        );
      }
      report = await callOnce(userPayloadJson, scale);
      break;
    } catch (err) {
      lastErr = err;
      const truncation = isTruncationLike(err);
      console.warn(
        `[pipeline] digest attempt ${i + 1} failed${truncation ? " (truncation-like)" : ""}: ${
          err instanceof Error ? err.message : String(err)
        }`,
      );
      // Non-truncation errors still get one same-scale retry historically;
      // after that, continue shrinking only when it looks like truncation.
      if (!truncation && i === 0) {
        try {
          console.warn("[pipeline] non-truncation failure; one same-payload retry");
          report = await callOnce(userPayloadJson, scale);
          break;
        } catch (retryErr) {
          lastErr = retryErr;
          console.warn(
            `[pipeline] same-payload retry failed: ${
              retryErr instanceof Error ? retryErr.message : String(retryErr)
            }`,
          );
          // Fall through to smaller attempts in case the failure was actually
          // a truncated body that didn't match our heuristics.
        }
      }
    }
  }

  if (!report) {
    throw lastErr instanceof Error
      ? lastErr
      : new Error(String(lastErr ?? "digest generation failed"));
  }

  // Max subscription has no per-call token meter — we expose 0 for schema
  // compatibility; consumers should treat 0 as "metric not available".
  return { report, tokensUsed: 0 };
}
