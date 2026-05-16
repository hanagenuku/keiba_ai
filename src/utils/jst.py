import subprocess
from datetime import datetime, timezone, timedelta


def get_jst_now():
    """常に正確なJST時刻を返す"""
    try:
        result = subprocess.run(
            ['sudo', 'ntpdate', '-u', 'ntp.nict.jp'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            raise Exception('ntpdate failed')
    except Exception:
        pass
    JST = timezone(timedelta(hours=9))
    return datetime.now(JST)
