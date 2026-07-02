export async function main(input, api) {
  const dir = (input && input.taskDir) ? input.taskDir : '.';

  const P = {
    findings:   `${dir}/state/findings.jsonl`,
    directions: `${dir}/state/directions_tried.json`,
    iterLog:    `${dir}/state/iteration_log.jsonl`,
    progress:   `${dir}/state/progress.json`,
    iterEval:   `${dir}/state/iteration_eval.json`,
    pivot:      `${dir}/state/pivot_plan.json`,
  };

  try {
    // --- Read required inputs (fail fast if missing) ---
    const iterEval = await api.state.readJson(P.iterEval);
    if (!iterEval) {
      return { action: 'branch', label: 'error', note: `missing or empty: ${P.iterEval}` };
    }

    const progress = await api.state.readJson(P.progress);
    if (!progress) {
      return { action: 'branch', label: 'error', note: `missing or empty: ${P.progress}` };
    }

    const directions = await api.state.readJson(P.directions);
    if (!directions) {
      return { action: 'branch', label: 'error', note: `missing or empty: ${P.directions}` };
    }

    // --- 1. Append each finding to findings.jsonl ---
    const findings = Array.isArray(iterEval.findings_to_append)
      ? iterEval.findings_to_append
      : [];
    for (const f of findings) {
      await api.state.appendJsonl(P.findings, f);
    }

    // --- 2. Append direction_entry to directions_tried.json + refresh last_updated ---
    if (iterEval.direction_entry) {
      if (!Array.isArray(directions.directions)) {
        directions.directions = [];
      }
      directions.directions.push(iterEval.direction_entry);
    }
    directions.last_updated = api.nowIso;

    // --- 5. Consume pivot_plan.json if present and non-empty ---
    let pivotConsumed = false;
    try {
      const pivot = await api.state.readJson(P.pivot);
      if (pivot && typeof pivot === 'object' && Object.keys(pivot).length > 0) {
        if (!Array.isArray(directions.pivots)) {
          directions.pivots = [];
        }
        directions.pivots.push({
          summary: pivot,
          consumed_at: api.nowIso,
        });
        pivotConsumed = true;
      }
    } catch (_) {
      // pivot_plan.json does not exist — nothing to consume
    }

    // Write directions_tried.json (directions push + pivots + last_updated)
    await api.state.writeJson(P.directions, directions);

    // --- 3. Append iteration_log_entry to iteration_log.jsonl ---
    if (iterEval.iteration_log_entry) {
      await api.state.appendJsonl(P.iterLog, iterEval.iteration_log_entry);
    }

    // --- 6. Validate progress.total_findings against findings.jsonl line count ---
    let lineCount = 0;
    try {
      const text = await api.state.readText(P.findings);
      if (text && text.trim()) {
        lineCount = text
          .trim()
          .split('\n')
          .filter(function (l) { return l.length > 0; })
          .length;
      }
    } catch (_) {
      // findings.jsonl may not yet exist
    }

    // Sync total_findings to the authoritative line count
    progress.total_findings = lineCount;

    // Atomic rewrite of progress.json
    await api.state.writeJson(P.progress, progress);

    // --- Clear pivot_plan.json (write empty object) if it was consumed ---
    if (pivotConsumed) {
      await api.state.writeJson(P.pivot, {});
    }

    return {
      action: 'branch',
      label: 'written',
      data: {
        findings_appended:   findings.length,
        direction_appended:   !!iterEval.direction_entry,
        pivot_consumed:       pivotConsumed,
        total_findings:       lineCount,
        directions_count:     Array.isArray(directions.directions) ? directions.directions.length : 0,
        timestamp:            api.nowIso,
      },
    };
  } catch (e) {
    return {
      action: 'branch',
      label: 'error',
      note: (e && e.message) ? e.message : String(e),
    };
  }
}
