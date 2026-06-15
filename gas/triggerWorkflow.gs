/**
 * GitHub Actions ワークフローをトリガーする
 *
 * パラメータ:
 *   workflow: 'friday-predict' | 'weekend' | 'sunday-results'
 *   mode: 'saturday' | 'sunday'（weekendの場合のみ）
 *
 * 事前準備:
 *   スクリプトプロパティに GITHUB_TOKEN を設定すること
 *   （リポジトリ keiba_ai に対する Actions: Read and write 権限の
 *    Fine-grained Personal Access Token）
 */
function triggerWorkflow(e) {
  const workflow = (e.parameter.workflow || '').toString();
  const mode = (e.parameter.mode || 'saturday').toString();
  const token = PropertiesService.getScriptProperties().getProperty('GITHUB_TOKEN');

  if (!token) {
    return jsonResponse({status: 'error', message: 'GITHUB_TOKEN未設定'});
  }

  const workflowFile = {
    'friday-predict': 'friday-predict.yml',
    'weekend': 'weekend.yml',
    'sunday-results': 'sunday-results.yml'
  }[workflow];

  if (!workflowFile) {
    return jsonResponse({status: 'error', message: '不明なworkflow: ' + workflow});
  }

  const url = 'https://api.github.com/repos/hanagenuku/keiba_ai/actions/workflows/'
              + workflowFile + '/dispatches';

  const payload = {ref: 'main'};
  if (workflow === 'weekend') {
    payload.inputs = {mode: mode};
  }

  const options = {
    method: 'post',
    headers: {
      'Authorization': 'token ' + token,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json'
    },
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  };

  const res = UrlFetchApp.fetch(url, options);
  const code = res.getResponseCode();

  if (code === 204) {
    return jsonResponse({status: 'ok', message: workflow + ' をトリガーしました'});
  } else {
    return jsonResponse({status: 'error', code: code, body: res.getContentText()});
  }
}

// doGet に分岐追加:
// if (action === 'trigger') return triggerWorkflow(e);
