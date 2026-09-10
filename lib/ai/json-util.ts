/**
 * Strip the LLM's chatty wrapping (markdown code fences, "Here is the JSON:"
 * preamble) and return just the payload between the first `{` and last `}`.
 * Does NOT validate parsability — callers still pipe the result through
 * JSON.parse with a jsonrepair fallback for unescaped-quote issues.
 */
export function extractJson(raw: string): string {
  let text = raw.trim();
  const fence = /^```(?:json)?\s*([\s\S]*?)\s*```$/.exec(text);
  if (fence) text = fence[1].trim();
  const firstBrace = text.indexOf("{");
  const lastBrace = text.lastIndexOf("}");
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    text = text.slice(firstBrace, lastBrace + 1);
  }
  return text;
}

/**
 * True when the model likely stopped mid-JSON (output truncation).
 * Brace balance ignores strings, so this is a heuristic — good enough
 * to decide "retry with a smaller ask" vs "retry the same prompt".
 */
export function isLikelyTruncatedJson(raw: string): boolean {
  const text = raw.trim();
  if (!text) return true;
  const firstBrace = text.indexOf("{");
  if (firstBrace === -1) return false;
  const slice = text.slice(firstBrace);
  if (!slice.includes("}")) return true;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = 0; i < slice.length; i++) {
    const ch = slice[i];
    if (inString) {
      if (escape) {
        escape = false;
      } else if (ch === "\\") {
        escape = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
    } else if (ch === "{") {
      depth++;
    } else if (ch === "}") {
      depth--;
      if (depth === 0 && i < slice.length - 1) {
        // Extra trailing junk after a complete object is fine.
        return false;
      }
    }
  }
  return depth !== 0 || inString;
}
