import base64, json, sys, time, urllib.request
q = sys.argv[1] if len(sys.argv) > 1 else "Two shapes are shown: a red circle and a blue square. Which corner is the red circle in?"
b64 = base64.b64encode(open("spatial.png","rb").read()).decode()
body = json.dumps({
  "model":"glm-5.3-flash-colibri","max_tokens":60,"temperature":0,
  "messages":[{"role":"user","content":[
      {"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}},
      {"type":"text","text":q}]}]}).encode()
t=time.time()
r=urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8111/v1/chat/completions", body, {"Content-Type":"application/json"}), timeout=1800)
d=json.loads(r.read()); e=time.time()-t
c=d.get("choices")
print(f"[{e:.1f}s] prompt_tokens={d.get('usage',{}).get('prompt_tokens')}")
print("Q:",q)
print("A:", c[0]["message"]["content"] if c else d)
