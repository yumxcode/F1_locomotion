export async function main(input, api) {
  try {
    const taskDir = (input && typeof input.taskDir === "string" && input.taskDir.length > 0) ? input.taskDir : ".";
    const progressPath = taskDir + "/state/progress.json";
    const logPath = taskDir + "/state/iteration_log.jsonl";

    // --- read existing progress (fail-soft: missing/unreadable => error verdict) ---
    let progress;
    try {
      progress = await api.state.readJson(progressPath);
    } catch (e) {
      progress = null;
    }
    if (progress == null || typeof progress !== "object" || Array.isArray(progress)) {
      return {
        action: "branch",
        label: "error",
        note: "state/progress.json missing or not a JSON object; cannot mark error"
      };
    }

    const nowIso = api.nowIso;
    const iteration = (progress.iteration != null) ? progress.iteration : 0;

    // --- mutate: status -> error, refresh timestamp ---
    progress.status = "error";
    progress.updated_at = nowIso;

    // --- atomic write back (temp + rename via writeJson; skip temp file if rename unsupported) ---
    const tmpPath = progressPath + ".tmp";
    const payload = JSON.stringify(progress, null, 2);
    let wroteAtomic = false;
    try {
      await api.state.writeText(tmpPath, payload);
      if (typeof api.state.rename === "function") {
        await api.state.rename(tmpPath, progressPath);
        wroteAtomic = true;
      }
    } catch (e) {
      wroteAtomic = false;
    }
    if (!wroteAtomic) {
      // fallback: direct write (writeJson already gives best-effort atomicity on platform)
      await api.state.writeJson(progressPath, progress);
      // best-effort cleanup of stray temp file
      try { await api.state.writeText(tmpPath, ""); } catch (_) {}
    }

    // --- append error event to iteration log ---
    await api.state.appendJsonl(logPath, {
      timestamp: nowIso,
      iteration: iteration,
      event: "error",
      summary: "orchestration node error"
    });

    return {
      action: "branch",
      label: "ok",
      data: {
        status: "error",
        iteration: iteration,
        updated_at: nowIso,
        progress_path: progressPath,
        log_path: logPath
      }
    };
  } catch (e) {
    return {
      action: "branch",
      label: "error",
      note: "error_writer failed: " + String(e && e.message ? e.message : e)
    };
  }
}
