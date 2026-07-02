export async function main(input, api) {
  const taskDir = input && typeof input.taskDir === 'string' && input.taskDir.length > 0 ? input.taskDir : '.';
  const base = taskDir.replace(/\/+$/, '');
  const p = (rel) => base + '/' + String(rel).replace(/^\/+/, '');

  let progress, iterEval;
  try {
    progress = await api.state.readJson(p('state/progress.json'));
    iterEval = await api.state.readJson(p('state/iteration_eval.json'));
  } catch (e) {
    return { action: 'branch', label: 'error', note: 'failed to read input state: ' + (e && e.message ? e.message : String(e)) };
  }

  if (!progress || typeof progress !== 'object' || Array.isArray(progress)) {
    return { action: 'branch', label: 'error', note: 'progress.json is not a valid object' };
  }
  if (!iterEval || typeof iterEval !== 'object' || Array.isArray(iterEval)) {
    return { action: 'branch', label: 'error', note: 'iteration_eval.json is not a valid object' };
  }

  const num = (v, dflt) => (typeof v === 'number' && Number.isFinite(v) ? v : dflt);
  const findings = num(iterEval.new_findings_count, 0);
  const metricDelta = num(iterEval.metric_delta, 0);

  let staleCount = num(progress.stale_count, 0);
  if (findings <= 0 || metricDelta < 0) {
    staleCount += 1;
  } else {
    staleCount = Math.max(0, staleCount - 1);
  }

  let status;
  if (staleCount >= 4) {
    status = 'attention_required';
  } else if (staleCount >= 2) {
    status = 'pivot_required';
  } else if (staleCount === 0) {
    status = 'healthy';
  } else {
    status = 'stale';
  }

  const iteration = num(progress.iteration, 0) + 1;

  let healthyStreak = num(progress.healthy_streak, 0);
  healthyStreak = (status === 'healthy') ? healthyStreak + 1 : 0;

  const totalFindings = num(progress.total_findings, 0) + findings;

  const dirEntry = iterEval.direction_entry;
  const lastDirection = (dirEntry && dirEntry.id !== undefined && dirEntry.id !== null)
    ? dirEntry.id
    : (progress.last_direction !== undefined ? progress.last_direction : null);

  const lastMetric = (typeof iterEval.metric_value === 'number' && Number.isFinite(iterEval.metric_value))
    ? iterEval.metric_value
    : (progress.last_metric !== undefined ? progress.last_metric : null);

  const updatedAt = (typeof api.nowIso === 'string' && api.nowIso.length > 0) ? api.nowIso : null;

  const updated = Object.assign({}, progress, {
    iteration,
    stale_count: staleCount,
    status,
    healthy_streak: healthyStreak,
    total_findings: totalFindings,
    last_direction: lastDirection,
    last_metric: lastMetric,
    updated_at: updatedAt,
  });

  const tmpPath = p('state/.progress.json.tmp');
  const finalPath = p('state/progress.json');
  try {
    await api.state.writeJson(tmpPath, updated);
    await api.state.writeJson(finalPath, updated);
  } catch (e) {
    return { action: 'branch', label: 'error', note: 'failed to write progress.json: ' + (e && e.message ? e.message : String(e)) };
  }

  return {
    action: 'branch',
    label: 'updated',
    data: {
      iteration,
      stale_count: staleCount,
      status,
      healthy_streak: healthyStreak,
      total_findings: totalFindings,
      last_direction: lastDirection,
      last_metric: lastMetric,
      updated_at: updatedAt,
    },
  };
}
