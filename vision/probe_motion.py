import base64,json,sys,time,urllib.request,urllib.error
Q=("A black ball moves between a red square and a blue square. "
   "Does the ball move FROM the red square TOWARD the blue square, "
   "or FROM the blue square TOWARD the red square? Answer with one of those two.")
def run(path,label):
    b64=base64.b64encode(open(path,"rb").read()).decode()
    body=json.dumps({"model":"glm-5.3-flash-colibri","max_tokens":40,"temperature":0,
      "messages":[{"role":"user","content":[
        {"type":"video_url","video_url":{"url":"data:video/mp4;base64,"+b64}},
        {"type":"text","text":Q}]}]}).encode()
    t=time.time()
    try:
        r=urllib.request.urlopen(urllib.request.Request(
          "http://127.0.0.1:8111/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=5400)
        d=json.loads(r.read()); c=d.get("choices")
        print(f"\n--- {label}  [{time.time()-t:.0f}s, prompt {d.get('usage',{}).get('prompt_tokens')}] ---")
        print("A:", c[0]["message"]["content"] if c else d, flush=True)
    except urllib.error.HTTPError as e:
        print(label,"HTTP",e.code,e.read().decode()[:200])
run("motion_lr.mp4","LEFT->RIGHT  (truth: red -> blue)")
run("motion_rl.mp4","RIGHT->LEFT  (truth: blue -> red)")
