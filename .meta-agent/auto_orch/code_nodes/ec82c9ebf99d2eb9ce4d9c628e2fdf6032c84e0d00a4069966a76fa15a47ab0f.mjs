export async function main(input, api) {
  if (!input || typeof input !== 'object') {
    return { action: 'branch', label: 'error', note: 'input missing or not an object' };
  }

  let progress;
  try {
    progress = await api.state.readJson('state/progress.json');
  } catch (e) {
    return { action: 'branch', label: 'error', note: 'failed to read state/progress.json: ' + (e && e.message ? e.message : String(e)) };
  }

  if (progress == null || typeof progress !== 'object' || Array.isArray(progress)) {
    return { action: 'branch', label: 'error', note: 'state/progress.json is missing or not a JSON object' };
  }

  const num = function (v) { var n = Number(v); return Number.isFinite(n) ? n : 0; };
  const healthy_streak = num(progress.healthy_streak);
  const total_findings = num(progress.total_findings);
  const iteration = num(progress.iteration);
  const status = progress.status;

  const doneByHealth = healthy_streak >= 10 && total_findings >= 20;
  const doneByIteration = iteration >= 20;

  if (doneByHealth || doneByIteration) {
    return {
      action: 'branch',
      label: 'done',
      data: {
        healthy_streak: healthy_streak,
        total_findings: total_findings,
        iteration: iteration,
        status: status,
        reason: doneByHealth ? 'health_criteria_met' : 'iteration_limit_reached'
      }
    };
  }

  const known = ['healthy', 'stale', 'pivot_required', 'attention_required'];
  if (typeof status === 'string' && known.indexOf(status) !== -1) {
    return {
      action: 'branch',
      label: status,
      data: { healthy_streak: healthy_streak, total_findings: total_findings, iteration: iteration, status: status }
    };
  }

  return {
    action: 'branch',
    label: 'attention_required',
    data: {
      healthy_streak: healthy_streak,
      total_findings: total_findings,
      iteration: iteration,
      status: status == null ? null : String(status),
      note: 'status missing or unknown'
    }
  };
}
