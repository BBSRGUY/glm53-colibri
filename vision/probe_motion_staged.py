import base64,json,time,urllib.request
Q=("Step 1: say whether the black ball is closer to the RED or BLUE square in Frame 1. "
   "Step 2: same for Frame 4. Step 3: from those two, state which square it moves FROM "
   "and which it moves TOWARD.")
def run(path,label):
    b64=base64.b64encode(open(path,"rb").read()).decode()
    body=json.dumps({"model":"glm-5.3-flash-colibri","max_tokens":90,"temperature":0,
      "messages":[{"role":"user","content":[
        {"type":"video_url","video_url":{"url":"data:video/mp4;base64,"+b64}},
        {"type":"text","text":Q}]}]}).encode()
    t=time.time()
    r=urllib.request.urlopen(urllib.request.Request(
      "http://127.0.0.1:8111/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=5400)
    d=json.loads(r.read()); c=d.get("choices")
    print(f"\n--- {label} [{time.time()-t:.0f}s] ---")
    print("A:", c[0]["message"]["content"] if c else d, flush=True)
run("motion_lr.mp4","LR  truth: moves RED -> BLUE")
run("motion_rl.mp4","RL  truth: moves BLUE -> RED")
