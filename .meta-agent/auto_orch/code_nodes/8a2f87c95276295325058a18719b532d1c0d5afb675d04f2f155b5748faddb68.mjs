/**
 * route_by_status — reads state/progress.json and deterministically routes.
 *
 * Completion gate: (healthy_streak >= 10 AND total_findings >= 20) OR iteration >= 20 → done
 * Otherwise:       label = status (if known), else attention_required
 * Missing/unreadable state → error
 */
export async function main(input, api) {
  let progress;
  try {
    progress = await api.state.readJson("state/progress.json");
  } catch (e) {
    return {
      action: "branch",
      label: "error",
      note: "Failed to read state/progress.json: " + (e && e.message ? e.message : String(e)),
    };
  }

  if (progress == null) {
    return {
      action: "branch",
      label: "error",
      note: "state/progress.json is missing or null",
    };
  }

  if (typeof progress !== "object" || Array.isArray(progress)) {
    return {
      action: "branch",
      label: "error",
      note: "state/progress.json is not a valid JSON object",
    };
  }

  // Coerce fields to numbers / strings defensively
  const healthy_streak = typeof progress.healthy_streak === "number" ? progress.healthy_streak : 0;
  const total_findings = typeof progress.total_findings === "number" ? progress.total_findings : 0;
  const iteration = typeof progress.iteration === "number" ? progress.iteration : 0;
  const status = typeof progress.status === "string" ? progress.status : null;

  const summary = { healthy_streak, total_findings, iteration, status };

  // ── Completion gate ──────────────────────────────────────────
  if ((healthy_streak >= 10 && total_findings >= 20) || iteration >= 20) {
    return { action: "branch", label: "done", data: summary };
  }

  // ── Route by status ──────────────────────────────────────────
  const knownLabels = new Set([
    "healthy",
    "stale",
    "pivot_required",
    "attention_required",
  ]);

  if (status && knownLabels.has(status)) {
    return { action: "branch", label: status, data: summary };
  }

  // status missing or unknown → attention_required
  return { action: "branch", label: "attention_required", data: summary };
}
