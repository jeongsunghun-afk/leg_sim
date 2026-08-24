#!/usr/bin/env python3
"""sysload.py — CPU·온도·프로세스별 부하. 의존성 없이 /proc·/sys 만 읽는다.

왜 따로 뺐는가: **GUI 와 텍스트 모니터 둘 다** 이걸 보여줘야 하는데, 양쪽에
복사하면 한쪽만 고쳐지고 조용히 갈라진다. 이 저장소가 반복해서 당한 버그다
(joint_map 규칙 복제 → calf 한계 어긋남 · calib_zero 환산식 복사본 stale).

왜 필요한가 — 500Hz 제어루프가 CPU·온도에 직접 물려 있다:
  · 열로 클럭이 내려가면 루프가 밀린다(실측 28~51ms 스톨 사례)
  · EtherCAT 동결 추적에서 "Emb 가 CPU 100% 로 살아 있었다" 가 핵심 증거였다
  · run_hw.sh 가 뷰어·모니터를 여럿 띄우는데 중복 기동이 루프를 민다
    (2026-08-21 monitor_plot 이 2개 쌓여 있었다)

쓰는 법:
    import sysload
    txt, sev = sysload.line()      # sev: 0 정상 · 1 주의 · 2 위험
⚠첫 호출은 CPU 를 '--' 로 낸다(델타를 재려면 표본 두 개가 필요하다).
"""
import os
import time

#   왜 GUI 에 넣는가: 500Hz 제어루프는 **CPU 와 온도에 직접 물려 있다**.
#     · Pi 가 열로 클럭을 내리면 루프가 밀린다(실측 28~51ms 스톨 사례가 있다)
#     · EtherCAT 동결을 쫓을 때 "Emb 가 CPU 100% 로 살아 있었다" 가 핵심 증거였는데,
#       그걸 보려면 그때마다 top 을 띄워야 했다. 상시로 보이게 한다.
#     · run_hw.sh 가 뷰어·모니터를 여럿 띄우는데, 중복 기동이 CPU 를 먹어 루프를 민다
#       (2026-08-21 monitor_plot 이 2개 쌓여 있었다). 그 상황이 바로 드러난다.
#   ★의존성 없이 /proc·/sys 만 읽는다. 없는 항목은 조용히 건너뛴다(WSL 등).
CLK = os.sysconf('SC_CLK_TCK') if hasattr(os, 'sysconf') else 100
_cpu_prev  = [None]          # (total, idle)
_prc_prev  = {}              # pid → (ticks, t)
_prc_scan  = [0.0, []]       # (마지막 스캔시각, [(label, pid)])
_thermal   = [None]          # 온도 파일 경로 목록(한 번만 찾는다)
_last_sys  = [0.0]
# 이름 → 표시라벨. 앞에서 먼저 맞는 것 하나만 잡는다(중복 기동은 개수로 드러낸다).
_WATCH = [('Emb', 'RobotEmbedded'), ('제어기', 'biped_deploy'), ('제어기', 'biped_emb.py'),
          ('GUI', 'teleop_gui_biped'), ('뷰어', 'biped_monitor'), ('모니터', 'monitor_')]


def _scan_pids():
    """감시 대상 프로세스의 pid. 5초에 한 번만 스캔한다.

    ★cmdline 만 보면 **bash 래퍼가 걸린다** — run_hw.sh 는
        bash -lc "... teleop_gui_biped.py ..."
      로 띄우므로 래퍼의 cmdline 에도 이름이 들어 있고, 래퍼는 CPU 를 안 쓰니
      늘 0% 로 보인다. (삭제된 '제어기 재시작' 버튼이 같은 함정을 기록해 뒀다:
      "명령줄 prefix 로는 sudo 래퍼를 못 잡는다" → 그래서 pgrep -x 를 썼다.)
    ⇒ **comm(프로세스 이름)** 으로 가른다:
        · 네이티브 바이너리(RobotEmbedded·biped_deploy·biped_monitor) → comm 이 곧 이름
        · 파이썬 스크립트 → comm 은 'python3' 이라 cmdline 을 봐야 한다.
          그때도 comm 이 python 계열일 것을 **요구**해서 셸 래퍼를 배제한다.
    ⚠comm 은 커널이 15자로 자른다. 지금 대상은 최장 13자라 안전하다.
    """
    if time.time() - _prc_scan[0] < 5.0:
        return _prc_scan[1]
    _prc_scan[0] = time.time()
    found, seen = [], set()
    try:
        for e in os.listdir('/proc'):
            if not e.isdigit():
                continue
            try:
                comm = open('/proc/%s/comm' % e).read().strip()
                cl = open('/proc/%s/cmdline' % e, 'rb').read().replace(b'\0', b' ').decode('utf8', 'replace')
            except Exception:
                continue
            for lab, needle in _WATCH:
                if (lab, needle) in seen:
                    continue
                if comm == needle or (comm.startswith('python') and needle in cl):
                    seen.add((lab, needle)); found.append((lab, int(e))); break
    except Exception:
        pass
    _prc_scan[1] = found
    return found


def _cpu_total_pct():
    """전체 CPU 사용률[%]. 두 번째 호출부터 값이 나온다."""
    try:
        f = open('/proc/stat').readline().split()[1:]
        v = [int(x) for x in f[:8]]
        tot, idle = sum(v), v[3] + v[4]
        pv = _cpu_prev[0]; _cpu_prev[0] = (tot, idle)
        if pv is None or tot <= pv[0]:
            return None
        return 100.0 * (1.0 - (idle - pv[1]) / float(tot - pv[0]))
    except Exception:
        return None


def _proc_pct(pid):
    try:
        st = open('/proc/%d/stat' % pid).read()
        fld = st[st.rindex(')') + 2:].split()      # ★comm 에 공백/괄호가 있어 rindex 로 자른다
        tk = int(fld[11]) + int(fld[12])           # utime + stime
        now = time.time(); pv = _prc_prev.get(pid); _prc_prev[pid] = (tk, now)
        if pv is None or now <= pv[1]:
            return None
        return 100.0 * (tk - pv[0]) / CLK / (now - pv[1])
    except Exception:
        _prc_prev.pop(pid, None)
        return None


def _temp_c():
    if _thermal[0] is None:
        c = []
        try:
            import glob as _g
            c = sorted(_g.glob('/sys/class/thermal/thermal_zone*/temp'))
        except Exception:
            pass
        _thermal[0] = c
    hi = None
    for f in _thermal[0]:
        try:
            v = int(open(f).read().strip()) / 1000.0
            if 0 < v < 200 and (hi is None or v > hi):
                hi = v
        except Exception:
            pass
    return hi


def _freq_ghz():
    try:
        v = max(int(open('/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq' % i).read())
                for i in range(os.cpu_count() or 1))
        return v / 1e6
    except Exception:
        return None



# 문턱 — 여기서만 고친다(GUI 색과 모니터 표시가 같은 기준을 쓰게).
WARN_C, CRIT_C = 70.0, 80.0
WARN_CPU, CRIT_CPU = 80.0, 92.0


def line(per_proc=True):
    """(표시문자열, 심각도 0/1/2). 1Hz 정도로 부르면 된다."""
    cpu, t = _cpu_total_pct(), _temp_c()
    parts = ['CPU %s' % ('--' if cpu is None else '%.0f%%' % cpu)]
    fq = _freq_ghz()
    if fq:
        parts.append('%.2fGHz' % fq)
    parts.append('온도 %s' % ('--' if t is None else '%.1f°C' % t))
    if per_proc:
        per = []
        for lab, pid in _scan_pids():
            v = _proc_pct(pid)
            if v is not None:
                per.append('%s %.0f%%' % (lab, v))
        if per:
            parts.append('│  ' + '  '.join(per))
    sev = 0
    if (t and t >= CRIT_C) or (cpu and cpu >= CRIT_CPU):
        sev = 2
    elif (t and t >= WARN_C) or (cpu and cpu >= WARN_CPU):
        sev = 1
    return '  '.join(parts), sev
