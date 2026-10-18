import os, json, sys, time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
from multiprocessing import Pool
from kresling_sim import kresling_2026_supp_config as cfg
from kresling_sim.config import WORKSPACE_X_MM
from kresling_sim.inverse import LeastSquaresSolver
from kresling_sim.montecarlo import run_monte_carlo, DEFAULT_NOISE
from kresling_sim.sensors import sensor_grid

CONFIGS=[(5,16),(5,18),(5,20),(6,16),(6,18),(6,20),(7,14),(7,16),(7,18),(8,16)]
zs=np.linspace(cfg.Z_MIN_MM,cfg.Z_MAX_MM,4); qs=np.linspace(0,WORKSPACE_X_MM/2,3)
POS=np.array([[x,y,z] for z in zs for x in qs for y in qs])

def run(c):
    side,pitch=c; s=sensor_grid(side,side,pitch,pitch); rng=np.random.default_rng(42)
    rm=[];cr=[];cv=[]
    for p in POS:
        r=run_monte_carlo(p,s,noise=DEFAULT_NOISE,n_trials=20,warm_start_sigma=3.0,rng=rng,
                          magnet_kwargs=cfg.MAGNET_KWARGS,solver=LeastSquaresSolver(n_starts=5))
        rm.append(r.rmse_mm);cr.append(r.crb_rmse_mm);cv.append(r.converged.mean())
    rm,cr,cv=map(np.array,(rm,cr,cv))
    return dict(side=side,pitch=pitch,n=side*side,span=(side-1)*pitch,mc_mean=float(rm.mean()),mc_max=float(rm.max()),
                crb_mean=float(cr.mean()),crb_max=float(cr.max()),conv_mean=float(cv.mean()),conv_min=float(cv.min()),
                ratio=float((rm/cr).mean()))
if __name__=="__main__":
    t=time.time()
    with Pool(min(len(CONFIGS),os.cpu_count())) as p: out=p.map(run,CONFIGS)
    json.dump(out,open(sys.argv[1],"w"),indent=1)
    print("done",time.time()-t,"s")
