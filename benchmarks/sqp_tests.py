import os
import sys
import shutil
import platform
import numpy as np
import pandas as pd
import platform
import pyemu

bin_path = os.path.join("test_bin")
if "linux" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"linux")
elif "darwin" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"mac")
else:
    bin_path = os.path.join(bin_path,"win")

bin_path = os.path.abspath("test_bin")
os.environ["PATH"] += os.pathsep + bin_path


bin_path = os.path.join("..","..","..","bin")
exe = ""
if "windows" in platform.platform().lower():
    exe = ".exe"
exe_path = os.path.join(bin_path, "pestpp-sqp" + exe)


noptmax = 4
num_reals = 20
port = 4021


def basic_sqp_test():
    model_d = "mf6_freyberg"
    local=True
    if "linux" in platform.platform().lower() and "10par" in model_d:
        #print("travis_prep")
        #prep_for_travis(model_d)
        local=False
    
    t_d = os.path.join(model_d,"template")
    m_d = os.path.join(model_d,"master_sqp_fd")
    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    pst = pyemu.Pst(os.path.join(t_d,"freyberg6_run_opt.pst"))
    pst.control_data.noptmax = 0
    pst.write(os.path.join(t_d,"freyberg6_run_sqp.pst"))
    pyemu.os_utils.run("{0} freyberg6_run_sqp.pst".format(exe_path.replace("-ies","-sqp")),cwd=t_d)

    assert os.path.exists(os.path.join(t_d,"freyberg6_run_sqp.base.par"))
    assert os.path.exists(os.path.join(t_d,"freyberg6_run_sqp.base.rei"))

    pst.pestpp_options["sqp_num_reals"] = 10
    pst.pestpp_options["opt_risk"] = 0.95
    pst.pestpp_options["sqp_ensemble_gradient"] = False

    pst.control_data.noptmax = -1
    pst.write(os.path.join(t_d,"freyberg6_run_sqp.pst"))
    pyemu.os_utils.start_workers(t_d, exe_path.replace("-ies","-sqp"), "freyberg6_run_sqp.pst", 
                                 num_workers=15, master_dir=m_d,worker_root=model_d,
                                 port=port)

    assert os.path.exists(os.path.join(m_d,"freyberg6_run_sqp.0.jcb"))
    jco = pyemu.Jco.from_binary(os.path.join(m_d,"freyberg6_run_sqp.0.jcb"))
    print(jco.shape)
   

    m_d = os.path.join(model_d,"master_sqp_en")
    pst.pestpp_options["sqp_ensemble_gradient"] = True
    pst.control_data.noptmax = -1
    pst.write(os.path.join(t_d,"freyberg6_run_sqp.pst"))
    pyemu.os_utils.start_workers(t_d, exe_path.replace("-ies","-sqp"), "freyberg6_run_sqp.pst", 
                                 num_workers=15, master_dir=m_d,worker_root=model_d,
                                 port=port)

    assert os.path.exists(os.path.join(m_d,"freyberg6_run_sqp.0.par.csv"))
    df = pd.read_csv(os.path.join(m_d,"freyberg6_run_sqp.0.par.csv"),index_col=0)
    assert df.shape == (pst.pestpp_options["sqp_num_reals"],pst.npar),str(df.shape)
    assert os.path.exists(os.path.join(m_d,"freyberg6_run_sqp.0.obs.csv"))
    df = pd.read_csv(os.path.join(m_d,"freyberg6_run_sqp.0.obs.csv"),index_col=0)
    assert df.shape == (pst.pestpp_options["sqp_num_reals"],pst.nobs),str(df.shape)


def sqp_ensemble_jco_test():
    model_d = "mf6_freyberg"
    local=True
    if "linux" in platform.platform().lower() and "10par" in model_d:
        #print("travis_prep")
        #prep_for_travis(model_d)
        local=False
    
    t_d = os.path.join(model_d,"template")
    m_d = os.path.join(model_d,"master_sqp_jco")
    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    pst = pyemu.Pst(os.path.join(t_d,"freyberg6_run_opt.pst"))
    
    par = pst.parameter_data
    par.loc[par.pargp=="welflux","parval1"] = (par.loc[par.pargp=="welflux","parubnd"] + par.loc[par.pargp=="welflux","parlbnd"]) / 2.0

    pst.pestpp_options["sqp_num_reals"] = 30
    pst.pestpp_options["opt_risk"] = 0.5
    pst.pestpp_options["sqp_ensemble_gradient"] = True

    pst.control_data.noptmax = 1
    pst.write(os.path.join(t_d,"freyberg6_run_sqp.pst"))
    pyemu.os_utils.start_workers(t_d, exe_path.replace("-ies","-sqp"), "freyberg6_run_sqp.pst", 
                                 num_workers=15, master_dir=m_d,worker_root=model_d,
                                 port=port)
    jco_file = os.path.join(m_d,"freyberg6_run_sqp.1.jcb")
    assert os.path.exists(jco_file),jco_file
    jco = pyemu.Jco.from_binary(jco_file)
    print(jco.shape)
    assert jco.shape[1] == par.loc[par.pargp=="welflux",:].shape[0],"{0} : {1}".format(jco.shape[1],par.loc[par.pargp=="welflux",:].shape[0])
    obs = pst.observation_data
    nconst = obs.loc[obs.obgnme.str.startswith("less"),:].shape[0]
    assert jco.shape[0] == nconst + pst.nprior,"{0} : {1}".format(jco.shape[0], nconst + pst.nprior)
   
def start_workers():
    model_d = "mf6_freyberg"
    t_d = os.path.join(model_d,"template")
    pyemu.os_utils.start_workers(t_d, exe_path.replace("-ies","-sqp"), "freyberg6_run_sqp.pst", 
                                 num_workers=15,worker_root=model_d,
                                 port=port)

if __name__ == "__main__":
    
    shutil.copy2(os.path.join("..","exe","windows","x64","Debug","pestpp-sqp.exe"),os.path.join("..","bin","pestpp-sqp.exe"))
    #basic_sqp_test()
    sqp_ensemble_jco_test()
    #start_workers()