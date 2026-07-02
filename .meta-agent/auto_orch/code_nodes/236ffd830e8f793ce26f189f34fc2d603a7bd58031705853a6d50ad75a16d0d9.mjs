/**
 * error_writer — set orchestration to an unrecoverable error state.
 *
 * - Read state/progress.json
 * - Set status = 'error', refresh updated_at to UTC ISO (api.nowIso)
 * - Atomic write-back (writeJson)
 * - Append {timestamp, iteration, event:'error', summary} to state/iteration_log.jsonl
 * - Return OrchVerdict with label 'ok' (or 'error' if state IO fails)
 */
export async function main(input, api) {
  const taskDir = (input && typeof input.taskDir === 'string' && input.taskDir.length) ? input.taskDir : '.';
  const progressPath = `${taskDir}/state/progress.json`;
  const logPath = `${taskDir}/state/iteration_log.jsonl`;
  const nowIso = (api && typeof api.nowIso === 'string') ? api.nowIso : null;

  if (!api || !api.state || typeof api.state.readJson !== 'function' ||
      typeof api.state.writeJson !== 'function' ||
      typeof api.state.appendJsonl !== 'function') {
    return {
      action: 'branch',
      label: 'error',
      note: 'error_writer: state API unavailable'
    };
  }

  let progress;
  try {
    progress = await api.state.readJson(progressPath);
  } catch (e) {
    // progress.json missing/corrupt — still record the error with a minimal stub.
    progress = {};
  }
  if (progress === null || typeof progress !== 'object' || Array.isArray(progress)) {
    progress = {};
  }

  const previousStatus = progress.status;
  const iteration = (progress.iteration !== undefined && progress.iteration !== null)
    ? progress.iteration
    : null;

  // Mark error state and refresh timestamp.
  progress.status = 'error';
  if (nowIso) {
    progress.updated_at = nowIso;
  }

  try {
    // Atomic write-back (temp + rename is handled by the platform writeJson impl).
    await api.state.writeJson(progressPath, progress);
  } catch (e) {
    return {
      action: 'branch',
      label: 'error',
      note: `error_writer: failed to write progress.json (${e && e.message ? e.message : String(e)})`
    };
  }

  // Append the error event to the iteration log.
  const logEntry = {
    timestamp: nowIso,
    iteration: iteration,
    event: 'error',
    summary: 'orchestration node error'
  };
  try {
    await api.state.appendJsonl(logPath, logEntry);
  } catch (e) {
    return {
      action: 'branch',
      label: 'error',
      note: `error_writer: failed to append iteration_log.jsonl (${e && e.message ? e.message : String(e)})`
    };
  }

  return {
    action: 'branch',
    label: 'ok',
    data: {
      path: progressPath,
      previousStatus: previousStatus,
      status: 'error',
      updated_at: nowIso,
      iteration: iteration
    }
  };
}
