import os, subprocess, time, json, threading, sys
CPP='/home/jsh/simulation/biped/cpp'
def run(name, envx, seq_modes, T=26.0):
    cmd_p=f'/tmp/claude-1000/-home-jsh/e59e6f53-83a2-4bc0-a11a-74fa0df21fd7/scratchpad/cu_{name}.json'
    stt_p=f'/tmp/claude-1000/-home-jsh/e59e6f53-83a2-4bc0-a11a-74fa0df21fd7/scratchpad/su_{name}.json'
    env=dict(os.environ, RT_PRIO='0', GROUND_RATIO='0'); env.update(envx)
    seq=[0]
    def w(m):
        seq[0]+=1
        open(cmd_p+'.t','w').write(json.dumps({'mode':m,'seq':seq[0]}))
        os.replace(cmd_p+'.t',cmd_p)
    w('off')
    p=subprocess.Popen([CPP+'/build/biped_deploy','--mock','--cmd',cmd_p,'--state',stt_p,
        '--config','/home/jsh/simulation/biped/emb/config/biped_emb.yaml',
        '--mjcf','/home/jsh/simulation/biped/biped_from_quad.mjcf','--T',str(T)],
        cwd=CPP+'/build',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    t0=time.time(); stop=[False]
    def f():
        while not stop[0]:
            t=time.time()-t0
            m='off'
            for lim,mm in seq_modes:
                if t<lim: m=mm; break
            w(m); time.sleep(0.05)
    th=threading.Thread(target=f,daemon=True); th.start()
    out,_=p.communicate(timeout=T+30); stop[0]=True
    return out
# A: ch5(HR_thigh) 불응답 주입 → home 이동 중 감지 → stand 거부
out=run('A', {'FAULT_STUCK_CH':'5','FAULT_AT_S':'3'},
        [(2,'hold'),(14,'home'),(20,'stand'),(99,'hold')])
det='토크 불응답 의심 ch5' in out
blk='토크 불응답 채널이 있다' in out
print('[A 불응답 주입]', '✅' if (det and blk) else '❌', '감지', det, '· stand 차단', blk)
for l in out.splitlines():
    if '불응답' in l or 'STOP' in l: print('   ', l.strip()[:150])
# B: 정상 — 오탐 없어야 (같은 시퀀스, 주입 없음)
out=run('B', {}, [(2,'hold'),(14,'home'),(20,'stand'),(99,'hold')])
fp='불응답' in out
print('[B 정상 오탐]', '✅ 없음' if not fp else '❌ 오탐 발생')
sys.exit(0 if (det and blk and not fp) else 1)
