"""Local-only BEV event editor. Run inside the data pod and port-forward it.

Drafts are editable JSON; exports are immutable paired frozen recipes.
Neither the editor nor its geometry checks claim gRPC qualification.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import pickle
import re
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlparse
import uuid
import zipfile
import numpy as np
from shapely.geometry import Polygon, LineString, box
from shapely.ops import unary_union

from navsafe.benchmark.editing.assets.registry import AssetRegistry
from navsafe.benchmark.editing.recipe.schema import Recipe, HostSpec, Frames, EgoSpec, ActorRecipe, AssetRef, load_recipe
from navsafe.benchmark.editing.recipe.freeze import freeze_recipe
from navsafe.benchmark.editing.recipe.replay import edits_from_recipe
from navsafe.traffic.geometry import body
from navsafe.traffic.authored_path import AuthoredMotion, smooth_path

EVENTS=("cut_in","lead_brake","vru_cross")
ASSETS={"car":("hb_car_1","Car"),"pedestrian":("hb_pedestrian_1","Pedestrian"),
        "dog":("hb_animal_3","Dog"),"cow":("hb_animal_cow","Cow"),"horse":("hb_animal_4","Horse")}
STATIC=Path(__file__).parent/"static"

def plain(value):
    if isinstance(value,np.ndarray): return np.round(value,4).tolist()
    if isinstance(value,(np.integer,np.floating)): return value.item()
    raise TypeError(type(value).__name__)

def dumps(obj): return json.dumps(obj,ensure_ascii=False,default=plain,allow_nan=False)

class Editor:
    def __init__(self,inventory,output,recipe_dir):
        self.inventory=Path(inventory)
        rows=json.loads(self.inventory.read_text())
        if len(rows)!=28 or len({r["token"] for r in rows})!=28: raise ValueError("Exactly the 28 fixed scenarios are required")
        self.rows={r["token"]:r for r in rows}; self.output=Path(output); self.recipe_dir=Path(recipe_dir)
        (self.output/"drafts").mkdir(parents=True,exist_ok=True)
        (self.output/"exports").mkdir(exist_ok=True)
        self.lock=threading.RLock()
        self.registry=AssetRegistry.load()
        from navsafe.benchmark import config as cfg
        # Freeze against the deployed asset bytes, not another library copy.
        if cfg.ASSET_BANK:
            for key,entry in list(self.registry.entries.items()):
                if entry.ply:
                    self.registry.entries[key]=replace(entry,ply=str(cfg.ASSET_BANK/Path(entry.ply).name))

    def token(self,token):
        if token not in self.rows: raise ValueError("Unknown scenario")
        return token

    def original(self,token):
        row=self.rows[self.token(token)]
        paths=list(self.recipe_dir.glob("*."+token+".yaml"))
        if not paths and row.get("scenario_meta",{}).get("has_inserted_actors"):
            raise ValueError("This scenario has no original recipe, so its actors cannot be omitted")
        return load_recipe(paths[0]) if paths else None

    @lru_cache(maxsize=6)
    def host(self,token):
        row=self.rows[self.token(token)]; cache=self.inventory.parent/(token+".pkl")
        if cache.exists():
            with cache.open("rb") as f: sd=pickle.load(f)
        else:
            from navsafe.benchmark.editing.host import load_host_scenario
            sd,_=load_host_scenario(row["data_root"],scene_id=row["scene_id"])
        return sd

    def catalog(self):
        result=[]
        for name,(key,label) in ASSETS.items():
            try:
                asset=self.registry.resolve(key)
                gait=self.registry.resolve_gait_bank(key)
                result.append(dict(id=name,label=label,key=key,dims=asset.entry.dims,
                                   available=True,animated=bool(gait)))
            except Exception as e:
                result.append(dict(id=name,label=label,key=key,available=False,reason=str(e)))
        return result

    def index(self):
        rows=[]
        for token,row in self.rows.items():
            status={}
            for event in EVENTS:
                path=self.output/"drafts"/(token+"_"+event+".json")
                status[event]=bool(json.loads(path.read_text()).get("actors")) if path.exists() else False
            rows.append(dict(token=token,leaf=row["leaf"],saved=status))
        return dict(scenes=rows,assets=self.catalog(),output=str(self.output))

    @lru_cache(maxsize=6)
    def scene(self,token):
        sd=self.host(token); ego_id=str(sd["metadata"]["sdc_id"]); ego=sd["tracks"][ego_id]["state"]
        pos=np.asarray(ego["position"]); lower=pos[:,:2].min(axis=0)-90; upper=pos[:,:2].max(axis=0)+90
        features=[]
        for key,f in sd.get("map_features",{}).items():
            points=f.get("polygon")
            polygon=points is not None
            if points is None: points=f.get("polyline")
            if points is None or len(points)<2: continue
            points=np.asarray(points)[:,:2]
            if not np.isfinite(points).all() or np.any(points.max(axis=0)<lower) or np.any(points.min(axis=0)>upper): continue
            shape=Polygon(points).buffer(0) if polygon and len(points)>=3 else LineString(points)
            if shape.is_empty: continue
            if shape.geom_type=="Polygon": coords=np.asarray(shape.simplify(.15).exterior.coords)
            elif shape.geom_type=="LineString": coords=np.asarray(shape.simplify(.15).coords)
            else: continue
            features.append(dict(id=str(key),type=str(f.get("type","")),polygon=polygon,points=np.round(coords,3).tolist()))
        T=len(pos); original=self.original(token); removed=set()
        if original:
            removed={a.source_track_id for a in original.actors.values() if a.source_track_id and a.op in ("remove","replace","relocate")}
        tracks=[]
        for key,tr in sd["tracks"].items():
            if str(key) in removed: continue
            st=tr["state"]; p=np.asarray(st["position"])
            if not len(p) or not np.isfinite(p).all(): continue
            if str(key)!=ego_id and (np.any(p[:,:2].max(axis=0)<lower) or np.any(p[:,:2].min(axis=0)>upper)): continue
            def series(name,default):
                a=np.asarray(st.get(name,default))
                return [float(a if a.ndim==0 else a[min(i,len(a)-1)]) for i in range(T)]
            samples=[]
            h=series("heading",0); lengths=series("length",4.5); widths=series("width",1.8); valid=series("valid",1)
            for i in range(T):
                p0=p[min(i,len(p)-1)]; samples.append([*map(float,p0[:2]),h[i],lengths[i],widths[i],bool(valid[i])])
            tracks.append(dict(id=str(key),type=str(tr.get("type","VEHICLE")),ego=str(key)==ego_id,
                               samples=samples,original=False))
        warnings=[]
        # Existing recipe actors: roll their actual drivers against logged ego
        # for a BEV preview. This is a logged-ego preview, not a closed-loop run.
        if original and original.actors:
            from navsafe.traffic import idm_driver,social_force,timed_braking,timed_cut_in,authored_path
            from navsafe.traffic.navsafe import DRIVERS
            from navsafe.traffic.stage_free import WorldView
            drivers={}
            for name,a in original.actors.items():
                if a.op=="remove": continue
                try:
                    cls=DRIVERS[a.policy.get("kind","static")]
                    drivers[name]=cls(name,a.spawn,policy=a.policy,**a.policy.get("params",{}))
                except Exception as e:
                    warnings.append(f"Original actor {name} shows its start point only: {e}")
                    drivers[name]=None
            collected={name:[] for name in drivers}
            previous={}
            for i in range(T):
                others=[]
                for tr in tracks:
                    x,y,h,l,w,v=tr["samples"][i]
                    if v and not tr["ego"]: others.append(dict(id=tr["id"],position=[x,y,0],heading=h,length=l,width=w))
                others.extend(dict(id=k,**v) for k,v in previous.items())
                ep=dict(position=pos[i],heading=float(np.asarray(ego["heading"])[i]),
                        velocity=np.asarray(ego["velocity"])[i],length=4.5,width=1.8)
                world=WorldView(t=i,dt=.1,ego=ep,agents=others,ego_id=ego_id)
                for name,driver in drivers.items():
                    a=original.actors[name]
                    try:
                        if driver:
                            if i: driver.step(world,.1)
                            pose=driver.pose()
                        else: pose=a.spawn
                    except Exception as e:
                        if not any(name in w for w in warnings): warnings.append(f"Original actor {name}: BEV preview stopped updating: {e}")
                        pose=previous.get(name,a.spawn)
                    previous[name]=pose
                    collected[name].append([float(pose["position"][0]),float(pose["position"][1]),float(pose.get("heading",0)),float(pose.get("length",4.5)),float(pose.get("width",1.8)),True])
            for name,samples in collected.items():
                tracks.append(dict(id="original_"+name,type=original.actors[name].track_type,ego=False,original=True,samples=samples))
        return dict(token=token,leaf=self.rows[token]["leaf"],T=T,dt=.1,
                    handoff=original.ego.replay_frames*.1 if original else .8,
                    bounds=[*lower,*upper],features=features,tracks=tracks,ego_id=ego_id,
                    warnings=warnings,original_preview=bool(original and original.actors),
                    frame="ego_frame0",units="m")

    def document(self,doc):
        token=self.token(doc.get("token")); event=doc.get("event")
        if event not in EVENTS: raise ValueError("Unknown event type")
        actors=doc.get("actors",[])
        if not isinstance(actors,list) or len(actors)>12: raise ValueError("At most 12 actors per event")
        ids=set()
        for actor in actors:
            key=actor.get("id","")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",key) or key in ids: raise ValueError("Invalid or duplicate actor ID")
            ids.add(key)
            if actor.get("asset") not in ASSETS: raise ValueError("Unknown asset")
            if (event=="vru_cross") == (actor["asset"]=="car"): raise ValueError("Event and actor type do not match")
            smooth_path(actor.get("points",[]))
            for key,low,high,default in (("onset",0,19,3),("speed",.1,35,5),("deceleration",.1,10,3),("hold",0,10,1),("recovery",.1,10,2)):
                v=float(actor.get(key,default))
                if not math.isfinite(v) or not low<=v<=high: raise ValueError(f"{key} out of range")
            source=actor.get("source")
            if source:
                sd=self.host(token)
                if source==str(sd["metadata"]["sdc_id"]) or source not in sd["tracks"] or str(sd["tracks"][source].get("type"))!="VEHICLE" or actor["asset"]!="car":
                    raise ValueError("Only background vehicles of the scenario can be taken over")
                if any(source==a.get("source") for a in actors if a is not actor): raise ValueError("The same vehicle cannot be taken over twice")
        return dict(version=1,token=token,event=event,actors=deepcopy(actors),updated_at=time.time())

    def policy(self,actor,event,enabled=True):
        path=smooth_path(actor["points"])
        return dict(kind="authored_path",event_kind=event,path_polyline=path.tolist(),
                    onset_s=float(actor.get("onset",3)),speed=float(actor.get("speed",5)),
                    deceleration=float(actor.get("deceleration",3)),speed_reduction=float(actor.get("speed",5)),
                    hold_s=float(actor.get("hold",1)),recovery_acceleration=float(actor.get("recovery",2)),enabled=enabled)

    def preview(self,doc):
        doc=self.document(doc); T=self.scene(doc["token"])["T"]; actors=[]
        for actor in doc["actors"]:
            motion=AuthoredMotion(self.policy(actor,doc["event"])); entry=self.registry.get(ASSETS[actor["asset"]][0])
            samples=[]
            for i in range(T):
                state=motion.sample(i*.1)
                samples.append([*map(float,state["position"][:2]),state["heading"],*entry.dims[:2],True,state["speed"],state["exhausted"]])
            base=AuthoredMotion(self.policy(actor,doc["event"],False))
            baseline=[[float(base.sample(i*.1)["position"][0]),float(base.sample(i*.1)["position"][1])] for i in range(T)]
            actors.append(dict(id=actor["id"],samples=samples,path=motion.path[:,:2].tolist(),baseline=baseline))
        return dict(actors=actors)

    @lru_cache(maxsize=6)
    def road(self,token):
        scene=self.scene(token)
        return unary_union([Polygon(f["points"]).buffer(0) for f in scene["features"] if f["polygon"] and "LANE" in f["type"]])

    def analyze(self,doc):
        doc=self.document(doc); scene=self.scene(doc["token"]); preview=self.preview(doc); road=self.road(doc["token"]); road_buffer=road.buffer(.1)
        issues=[]; removed={a.get("source") for a in doc["actors"] if a.get("source")}
        byid={a["id"]:a for a in doc["actors"]}
        def pose(s): return dict(position=s[:2],heading=s[2],length=s[3],width=s[4])
        for tr in preview["actors"]:
            actor=byid[tr["id"]]; hits={}; crossed=False
            for i in range(0,scene["T"],2):
                state=tr["samples"][i]; polygon=body(pose(state))
                crossed |= bool(road.intersects(polygon))
                if actor["asset"]=="car" and state[7]: hits.setdefault("short",i*.1)
                if actor["asset"]=="car" and not road_buffer.covers(polygon): hits.setdefault("road",i*.1)
                for bg in scene["tracks"]:
                    if bg["id"] in removed: continue
                    other=bg["samples"][i]
                    if other[5] and np.linalg.norm(np.array(other[:2])-state[:2])<15 and polygon.intersects(body(pose(other))):
                        hits.setdefault("ego" if bg["ego"] else "background",i*.1)
                for other in preview["actors"]:
                    if other["id"]!=tr["id"] and polygon.intersects(body(pose(other["samples"][i]))): hits.setdefault("authored",i*.1)
            labels={"short":"The vehicle path is too short; extend its tail","road":"The vehicle leaves the map lanes","ego":"Crosses the logged ego path (needs closed-loop validation)","background":"Overlaps an actor of the original scenario","authored":"New actors overlap each other"}
            for kind,t in hits.items(): issues.append(dict(actor=tr["id"],kind=kind,time=round(t,2),severity="info" if kind=="ego" else "warning",message=labels[kind]))
            if actor["asset"]!="car" and not crossed: issues.append(dict(actor=tr["id"],kind="crossing",severity="warning",message="The VRU path does not cross a map lane yet"))
        return dict(issues=issues,scope="BEV geometry check; not a closed-loop or render validation",original_warnings=scene["warnings"],checked_frames=len(range(0,scene["T"],2)))

    def save(self,doc):
        doc=self.document(doc); path=self.output/"drafts"/(doc["token"]+"_"+doc["event"]+".json")
        with self.lock:
            tmp=path.with_suffix(".tmp"); tmp.write_text(dumps(doc)); tmp.replace(path)
        return dict(saved=True,updated_at=doc["updated_at"],path=str(path))

    def export(self,doc):
        doc=self.document(doc)
        if not doc["actors"]: raise ValueError("Place at least one actor first")
        report=self.analyze(doc); token=doc["token"]; row=self.rows[token]; original=self.original(token); scene=self.scene(token)
        ident=time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
        out=self.output/"exports"/ident; out.mkdir()
        files=[]
        for enabled,name in ((True,"event"),(False,"baseline")):
            recipe=deepcopy(original) if original else Recipe(recipe_id="",leaf=row["leaf"],
                host=HostSpec(scene=row["scene_id"],world_version="installed_reconstruction",temporal_extent_s=(scene["T"]-1)*.1),
                frames=Frames(T=scene["T"],dt_s=.1,after_frame=8),ego=EgoSpec(replay_frames=8,cam_height="navsim",z_to_ground=0))
            recipe.recipe_id=f"editor/{token}/{doc['event']}/{ident}/{name}"; recipe.provenance="constructed"
            recipe.selection=dict(recipe.selection,editor_event=doc["event"],editor_variant=name,qualification="UNVALIDATED_EDITOR_DRAFT")
            for actor in doc["actors"]:
                key=ASSETS[actor["asset"]][0]; asset=self.registry.resolve(key); policy=self.policy(actor,doc["event"],enabled)
                motion=AuthoredMotion(policy); state=motion.sample(0)
                gait=self.registry.resolve_gait_bank(key)
                spec=ActorRecipe(name="editor_"+actor["id"],op="replace" if actor.get("source") else "insert",
                    source_track_id=actor.get("source") or "",asset=AssetRef(**asset.to_recipe_asset()),
                    track_type=asset.entry.track_type,semantic_class="vehicle" if actor["asset"]=="car" else ("pedestrian" if actor["asset"]=="pedestrian" else "animal"),
                    nurec_asset_id=str(asset.ply),spawn=dict(position=state["position"].tolist(),heading=state["heading"],velocity=state["velocity"].tolist(),length=asset.entry.dims[0],width=asset.entry.dims[1],height=asset.entry.dims[2]),policy=policy)
                if gait: spec.nurec_pose_bank=str(gait[0]); spec.pose_bank_sha256=gait[1]
                recipe.actors[spec.name]=spec
            path=out/(name+".yaml"); freeze_recipe(recipe,path)
            # Validate against the actual replay/asset integrity contract.
            edits_from_recipe(load_recipe(path))
            files.append(dict(name=path.name,path=str(path)))
        (out/"draft.json").write_text(dumps(doc)); (out/"checks.json").write_text(dumps(report))
        command=(f"python navsafe/cli/eval_entry.py --scenario-source py123d --py123d-data-root {row['data_root']} --py123d-scene-index 0 --render-backend nurec_grpc --model-type pdm_closed --checkpoint none --recipe {out/'event.yaml'} --traffic-mode navsafe --execution-mode controller --controller lqr --replan-rate 5 --enable-vis --output-dir YOUR_NEW_OUTPUT_DIR")
        (out/"README.txt").write_text("Unvalidated editor recipe. The BEV and the runtime share AuthoredMotion; the original recipe actors and handoff are kept.\nConfigure the gRPC renderer, handoff, assets and environment as in docs/navsafe_eval.md, then run:\n"+command+"\nDo not treat the geometry check as an evaluation result.\n")
        archive=out.with_suffix(".zip")
        with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
            for f in out.iterdir(): z.write(f,f.name)
        self.save(doc)
        return dict(id=ident,files=files,path=str(out),download="/api/download/"+ident,checks=report)


def handler(editor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args): pass
        def send(self,obj,status=200):
            data=gzip.compress(dumps(obj).encode()); self.send_response(status)
            self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Encoding","gzip")
            self.send_header("Content-Length",str(len(data))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            try:
                path=urlparse(self.path).path
                if path=="/api/scenes": return self.send(editor.index())
                if path.startswith("/api/scene/"): return self.send(editor.scene(path.rsplit("/",1)[1]))
                if path.startswith("/api/draft/"):
                    _,_,_,token,event=path.split("/"); editor.token(token)
                    if event not in EVENTS: raise ValueError("Unknown event")
                    p=editor.output/"drafts"/(token+"_"+event+".json")
                    return self.send(json.loads(p.read_text()) if p.exists() else dict(version=1,token=token,event=event,actors=[],updated_at=0))
                if path.startswith("/api/download/"):
                    ident=path.rsplit("/",1)[1]
                    if not re.fullmatch(r"\d{8}_\d{6}_[a-f0-9]{8}",ident): raise ValueError("invalid export")
                    f=editor.output/"exports"/(ident+".zip"); mime="application/zip"
                else:
                    name={"/":"index.html","/app.js":"app.js","/style.css":"style.css"}.get(path)
                    if not name: return self.send(dict(error="Not found"),404)
                    f=STATIC/name; mime={".html":"text/html; charset=utf-8",".js":"text/javascript",".css":"text/css"}[f.suffix]
                data=f.read_bytes(); self.send_response(200); self.send_header("Content-Type",mime); self.send_header("Content-Length",str(len(data)))
                if mime=="application/zip": self.send_header("Content-Disposition",f'attachment; filename="{f.name}"')
                self.end_headers(); self.wfile.write(data)
            except Exception as e: self.send(dict(error=str(e)),400)
        def do_POST(self):
            try:
                origin=self.headers.get("Origin")
                if origin and origin!="http://"+self.headers.get("Host",""): raise ValueError("Cross-origin write rejected")
                if "application/json" not in self.headers.get("Content-Type",""): raise ValueError("JSON required")
                n=int(self.headers.get("Content-Length","0"))
                if not 0<n<2_000_000: raise ValueError("Request too large")
                doc=json.loads(self.rfile.read(n))
                action={"/api/preview":editor.preview,"/api/analyze":editor.analyze,"/api/save":editor.save,"/api/export":editor.export}.get(self.path)
                if action is None: raise ValueError("unknown action")
                self.send(action(doc))
            except Exception as e: self.send(dict(error=str(e)),400)
    return Handler

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--inventory",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--recipe-dir",required=True); ap.add_argument("--port",type=int,default=8765)
    args=ap.parse_args(); editor=Editor(args.inventory,args.output,args.recipe_dir)
    server=ThreadingHTTPServer(("127.0.0.1",args.port),handler(editor))
    print(f"NavSafe BEV editor ready at http://127.0.0.1:{args.port}",flush=True); server.serve_forever()

if __name__=="__main__": main()
