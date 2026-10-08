"""Search-G1 representation-readout server with hot reload.

Serves /judge_probe -> {flip, boundary} for each rollout (prompt, response).
A background thread watches a manifest JSON; when the co-evolution controller writes
a new round (new policy checkpoint + freshly refit probes), the server reloads BOTH
the frozen policy-encoder snapshot and the two readout heads, then atomically swaps
them in.  Reward calls return detached scalars, so no gradient enters the evaluator.

manifest JSON:
  {"round": int, "encoder": <hf_ckpt_dir>, "commit": <pkl>, "boundary": <pkl>}
"""
from __future__ import annotations
import os
for _k in list(os.environ.keys()):
    if "proxy" in _k.lower():
        del os.environ[_k]

import argparse, json, re, threading, time
from typing import List
import hashlib
import gc
import numpy as np
import torch
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE = torch.bfloat16
app = FastAPI()
LOCK = threading.Lock()
STATE = {"round": -1, "encoder": None, "model": None, "tok": None,
         "commit": None, "boundary": None}
MANIFEST = None
OUTER_BARRIER = False


def _load(encoder, commit_pkl, boundary_pkl):
    tok = AutoTokenizer.from_pretrained(encoder, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        encoder, torch_dtype=DTYPE, device_map="auto",
        low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
    commit = joblib.load(commit_pkl)
    boundary = joblib.load(boundary_pkl)
    return tok, model, commit, boundary


def _release_encoder_locked():
    model=STATE.pop('model',None)
    STATE['model']=None
    del model
    gc.collect()
    if torch.cuda.is_initialized(): torch.cuda.empty_cache()


def _apply(manifest_path, force=False):
    if not os.path.exists(manifest_path): return
    with open(manifest_path,'rb') as f: raw=f.read()
    m=json.loads(raw)
    with LOCK:
        if not force and m.get('round',-1)==STATE['round']: return
        reload_model=force or m['encoder']!=STATE['encoder'] or STATE['model'] is None
        # All inference holds LOCK. No request can retain an old GPU encoder
        # while a replacement is constructed; barrier mode drains then unloads.
        if reload_model:
            _release_encoder_locked()
            tok,model,commit,boundary=_load(m['encoder'],m['commit'],m['boundary'])
        else:
            tok,model=STATE['tok'],STATE['model']
            commit,boundary=joblib.load(m['commit']),joblib.load(m['boundary'])
        STATE.update(round=m['round'],encoder=m['encoder'],tok=tok,model=model,
                     commit=commit,boundary=boundary,manifest_sha256=hashlib.sha256(raw).hexdigest(),paused=False)


@app.post('/barrier_pause')
def barrier_pause():
    if not OUTER_BARRIER: raise HTTPException(403,'outer barrier not enabled')
    with LOCK:
        STATE['paused']=True
        _release_encoder_locked()
        return {'status':'PAUSED_ENCODER_RELEASED','round':STATE['round']}


@app.post('/barrier_resume')
def barrier_resume():
    if not OUTER_BARRIER: raise HTTPException(403,'outer barrier not enabled')
    _apply(MANIFEST,force=True)
    with LOCK:
        if STATE['model'] is None or STATE.get('paused'): raise HTTPException(503,'encoder not ready')
        return {'status':'READY','round':STATE['round'],'encoder':STATE['encoder'],'manifest_sha256':STATE['manifest_sha256']}

def _watcher():
    while True:
        try:
            _apply(MANIFEST)
        except Exception as e:
            print(f"[probe] watcher error: {e}", flush=True)
        time.sleep(15)


class Item(BaseModel):
    row_id: str
    prompt: str
    response: str = ""


class ProbeRequest(BaseModel):
    items: List[Item]
    identity: dict


@torch.no_grad()
def _last_hidden(texts, layer, model, tok, micro=16, max_len=4096):
    base = model.model
    cap = {}
    h = base.layers[layer - 1].register_forward_hook(
        lambda mod, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    out = []
    try:
        for s in range(0, len(texts), micro):
            enc = tok(texts[s:s + micro], return_tensors="pt", add_special_tokens=False,
                      padding=True, truncation=True, max_length=max_len).to(base.device)
            base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                 use_cache=False)
            out.append(cap["h"][:, -1, :].float().cpu().numpy())
    finally:
        h.remove()
    return np.concatenate(out) if out else np.zeros((0, 1))


def _commit_ctx(prompt, response, position):
    """Build the prefix corresponding to the representation used for calibration."""
    if position == "Xc":
        start = response.find("<answer>")
        if start >= 0:
            return prompt + response[:start + len("<answer>")]
        return prompt + response
    if position == "Xa":
        end = response.rfind("</answer>")
        return prompt + (response[:end] if end >= 0 else response)
    raise ValueError(f"unsupported commit position: {position!r}")


@app.post("/judge_probe")
def judge_probe(req: ProbeRequest):
    if not req.items:
        return {"results": []}
    with LOCK:
        identity={'manifest_sha256':STATE.get('manifest_sha256'),'encoder':STATE['encoder'],'round':STATE['round']}
        if STATE.get('paused') or STATE['model'] is None: raise HTTPException(503,'encoder paused/unready')
        if req.identity!=identity: raise HTTPException(409,'frozen encoder identity mismatch')
        model,tok=STATE['model'],STATE['tok']
        commit,boundary=STATE['commit'],STATE['boundary']
        position=commit.get('position','Xa')
        cc=[_commit_ctx(it.prompt,it.response,position) for it in req.items]
        bc=[it.prompt for it in req.items]
        Hc=_last_hidden(cc,commit['layer'],model,tok)
        Hb=_last_hidden(bc,boundary['layer'],model,tok)
        flip=commit['clf'].predict_proba(Hc)[:,1]
        bnd=boundary['clf'].predict_proba(Hb)[:,1]
        return {'identity':identity,'results':[{'row_id':it.row_id,'flip':float(flip[i]),'boundary':float(bnd[i])} for i,it in enumerate(req.items)]}


@app.get("/health")
def health():
    return {"ok": STATE["model"] is not None, "round": STATE["round"],
            "encoder": STATE["encoder"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--outer-barrier",action="store_true")
    args = ap.parse_args()
    MANIFEST = args.manifest
    OUTER_BARRIER = args.outer_barrier
    print(f"[probe] manifest={MANIFEST} waiting for first round...", flush=True)
    for _ in range(600):
        if os.path.exists(MANIFEST):
            break
        time.sleep(2)
    _apply(MANIFEST, force=True)
    if not OUTER_BARRIER: threading.Thread(target=_watcher, daemon=True).start()
    print(f"[probe] serving on 127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
