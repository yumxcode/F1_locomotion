export async function main(input, api) {
  const inputs = ["state/progress.json","state/findings.jsonl","state/directions_tried.json"]
  const reportPath = "state/attention_required_report.md"
  const title = "Attention Required Report"
  const sections = []
  for (const path of inputs) {
    try {
      let value
      if (path.endsWith('.json')) value = await api.state.readJson(path)
      else value = await api.state.readText(path)
      sections.push('## ' + path + '\n\n' + formatValue(value))
    } catch (err) {
      sections.push('## ' + path + '\n\nUnavailable: ' + errorMessage(err))
    }
  }
  const body = '# ' + title + '\n\nGenerated at: ' + api.nowIso + '\n\n' + (sections.join('\n\n') || 'No declared inputs.')
  await api.state.writeText(reportPath, body + '\n')
  return { action: 'branch', label: 'ok', data: { reportPath } }
}

function formatValue(value) {
  if (typeof value === 'string') return value
  return '~~~json\n' + JSON.stringify(value, null, 2) + '\n~~~'
}

function errorMessage(err) {
  return err && err.message ? String(err.message) : String(err)
}
