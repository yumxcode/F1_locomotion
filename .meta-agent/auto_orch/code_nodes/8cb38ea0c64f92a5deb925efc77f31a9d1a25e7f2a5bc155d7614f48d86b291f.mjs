export async function main(input, api) {
  try {
    const taskDir = input && input.taskDir ? input.taskDir : '.';
    const S = taskDir.replace(/\/+$/, '') + '/state/';

    // ── Read core inputs ──
    const iterationEval = await api.state.readJson(S + 'iteration_eval.json');
    if (!iterationEval || typeof iterationEval !== 'object') {
      return { action: 'branch', label: 'error', note: 'state/iteration_eval.json missing or unreadable' };
    }
    const progress = await api.state.readJson(S + 'progress.json');
    if (!progress || typeof progress !== 'object') {
      return { action: 'branch', label: 'error', note: 'state/progress.json missing or unreadable' };
    }

    // ── 1. Append findings_to_append to findings.jsonl ──
    const findingsToAppend = Array.isArray(iterationEval.findings_to_append)
      ? iterationEval.findings_to_append
      : [];
    for (const f of findingsToAppend) {
      await api.state.appendJsonl(S + 'findings.jsonl', f);
    }

    // ── 2. Read / init directions_tried.json ──
    let directionsTried = await api.state.readJson(S + 'directions_tried.json');
    if (!directionsTried || typeof directionsTried !== 'object') {
      directionsTried = {};
    }
    if (!Array.isArray(directionsTried.directions)) directionsTried.directions = [];
    if (!Array.isArray(directionsTried.pivots)) directionsTried.pivots = [];

    let directionAppended = false;
    if (iterationEval.direction_entry) {
      directionsTried.directions.push(iterationEval.direction_entry);
      directionAppended = true;
    }

    // ── 5. Pivot handling: consume pivot_plan.json ──
    let pivotConsumed = false;
    const pivotPlan = await api.state.readJson(S + 'pivot_plan.json');
    if (pivotPlan && typeof pivotPlan === 'object' && Object.keys(pivotPlan).length > 0) {
      const pivotSummary = Object.assign({}, pivotPlan, { consumed_at: api.nowIso });
      directionsTried.pivots.push(pivotSummary);
      pivotConsumed = true;
      // Clear pivot_plan.json (write empty object)
      await api.state.writeJson(S + 'pivot_plan.json', {});
    }

    // Refresh last_updated and persist directions_tried.json
    directionsTried.last_updated = api.nowIso;
    await api.state.writeJson(S + 'directions_tried.json', directionsTried);

    // ── 3. Append iteration_log_entry to iteration_log.jsonl ──
    let logAppended = false;
    if (iterationEval.iteration_log_entry) {
      await api.state.appendJsonl(S + 'iteration_log.jsonl', iterationEval.iteration_log_entry);
      logAppended = true;
    }

    // ── 6. Verify total_findings against actual line count, then rewrite progress.json ──
    const findingsText = await api.state.readText(S + 'findings.jsonl');
    let findingsLineCount = 0;
    if (findingsText && typeof findingsText === 'string' && findingsText.trim().length > 0) {
      findingsLineCount = findingsText
        .trim()
        .split('\n')
        .filter(function (l) { return l.trim().length > 0; }).length;
    }

    const wasMismatch = progress.total_findings !== findingsLineCount;
    progress.total_findings = findingsLineCount;
    progress.last_updated = api.nowIso;

    // Atomic rewrite via writeJson (best available atomic primitive)
    await api.state.writeJson(S + 'progress.json', progress);

    return {
      action: 'branch',
      label: 'written',
      data: {
        findings_appended: findingsToAppend.length,
        direction_appended: directionAppended,
        log_appended: logAppended,
        pivot_consumed: pivotConsumed,
        total_findings: findingsLineCount,
        findings_count_corrected: wasMismatch,
        now_iso: api.nowIso
      }
    };
  } catch (e) {
    const msg = e && e.message ? e.message : String(e);
    return { action: 'branch', label: 'error', note: 'state_writer failed: ' + msg };
  }
}
