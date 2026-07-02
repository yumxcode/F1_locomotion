export async function main(input, api) {
  try {
    const taskDir = (input && typeof input.taskDir === "string" && input.taskDir.length > 0) ? input.taskDir : ".";
    const base = taskDir.replace(/\/+$/, "");
    const progressPath = `${base}/state/progress.json`;
    const evalPath = `${base}/state/iteration_eval.json`;

    const progress = await api.state.readJson(progressPath);
    if (!progress || typeof progress !== "object" || Array.isArray(progress)) {
      return { action: "branch", label: "error", note: "progress.json missing or not a JSON object" };
    }
    const evalResult = await api.state.readJson(evalPath);
    if (!evalResult || typeof evalResult !== "object" || Array.isArray(evalResult)) {
      return { action: "branch", label: "error", note: "iteration_eval.json missing or not a JSON object" };
    }

    const rawFindings = evalResult.new_findings_count;
    const newFindingsCount = Number.isFinite(Number(rawFindings)) ? Number(rawFindings) : 0;
    const rawDelta = evalResult.metric_delta;
    const metricDelta = Number.isFinite(Number(rawDelta)) ? Number(rawDelta) : 0;

    let staleCount = Number.isFinite(Number(progress.stale_count)) ? Number(progress.stale_count) : 0;
    const noNewFindings = !(newFindingsCount > 0);
    const metricRegressed = metricDelta < 0;
    if (noNewFindings || metricRegressed) {
      staleCount = staleCount + 1;
    } else {
      staleCount = Math.max(0, staleCount - 1);
    }

    const iteration = (Number.isFinite(Number(progress.iteration)) ? Number(progress.iteration) : 0) + 1;

    let status;
    if (staleCount >= 4) {
      status = "attention_required";
    } else if (staleCount >= 2) {
      status = "pivot_required";
    } else if (staleCount === 0) {
      status = "healthy";
    } else {
      status = "stale";
    }

    let healthyStreak = Number.isFinite(Number(progress.healthy_streak)) ? Number(progress.healthy_streak) : 0;
    if (status === "healthy") {
      healthyStreak = healthyStreak + 1;
    } else {
      healthyStreak = 0;
    }

    let lastDirection = progress.last_direction != null ? progress.last_direction : null;
    if (evalResult.direction_entry && evalResult.direction_entry.id != null) {
      lastDirection = evalResult.direction_entry.id;
    }

    let lastMetric = progress.last_metric != null ? progress.last_metric : null;
    if (evalResult.metric_value != null) {
      lastMetric = evalResult.metric_value;
    }

    const prevTotal = Number.isFinite(Number(progress.total_findings)) ? Number(progress.total_findings) : 0;
    const totalFindings = prevTotal + newFindingsCount;

    const updated = Object.assign({}, progress, {
      iteration: iteration,
      stale_count: staleCount,
      status: status,
      healthy_streak: healthyStreak,
      last_direction: lastDirection,
      last_metric: lastMetric,
      total_findings: totalFindings,
      updated_at: api.nowIso
    });

    const tempPath = `${base}/state/progress.json.tmp`;
    const payloadStr = JSON.stringify(updated, null, 2);
    try {
      await api.state.writeText(tempPath, payloadStr);
    } catch (_) {}
    await api.state.writeText(progressPath, payloadStr);

    return {
      action: "branch",
      label: "updated",
      note: "progress.json updated",
      data: {
        iteration: iteration,
        stale_count: staleCount,
        status: status,
        healthy_streak: healthyStreak,
        last_direction: lastDirection,
        last_metric: lastMetric,
        total_findings: totalFindings,
        new_findings_count: newFindingsCount,
        metric_delta: metricDelta,
        updated_at: api.nowIso
      }
    };
  } catch (e) {
    return { action: "branch", label: "error", note: "reduce_progress failed: " + String((e && e.message) ? e.message : e) };
  }
}
