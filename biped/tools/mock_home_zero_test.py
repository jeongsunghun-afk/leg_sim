import os, subprocess, time, json, threading, sys
CPP='/home/jsh/simulation/biped/cpp'
def run(name, envx=None, T=13.0):
    cmd_p=f'/tmp/claude-1000/-home-jsh/e59e6f53-83a2-4bc0-a11a-74fa0df21fd7/scratchpad/cmdh_{name}.json'
    stt_p=f'/tmp/claude-1000/-home-jsh/e59e6f53-83a2-4bc0-a11a-74fa0df21fd7/scratchpad/stth_{name}.json'
    env=dict(os.environ, RT_PRIO='0'); env.update(envx or {})
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
            w('hold' if t<2 else 'home'); time.sleep(0.05)
    th=threading.Thread(target=f,daemon=True); th.start()
    out,_=p.communicate(timeout=T+30); stop[0]=True
    q=json.load(open(stt_p))['q_leg_deg']
    ban=[l for l in out.splitlines() if 'home →' in l or 'HOME_DEG' in l]
    return q, ban
q,ban=run('geo')
print('GEO 배너:', ban[0].strip() if ban else '없음')
print('GEO 최종 q:', ['%.1f'%v for v in q])
ok1='Qhome8' in (ban[0] if ban else '') and abs(q[1]-11.63)<0.6 and abs(q[2]+38.45)<0.6
q,ban=run('env', {'HOME_DEG':'0,5,-10,3,0,5,-10,3'})
print('ENV 배너:', ' | '.join(b.strip() for b in ban[:2]))
print('ENV 최종 q:', ['%.1f'%v for v in q])
ok2=abs(q[1]-5)<0.6 and abs(q[2]+10)<0.6 and abs(q[3]-3)<0.6
print('결과:', 'GEO', '✅' if ok1 else '❌', '/ ENV', '✅' if ok2 else '❌')
sys.exit(0 if ok1 and ok2 else 1)
