import MeasurementScheme as MS
import argparse
import nlopt
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sf

hbar=1.05*10**-34 # Plank constant
w0=12.56*10**14 # Frequency of 1.55 um radiation

tau1 = 1-(10**-2) # Spectral Filer's sideband transmission
etaAPD = 0.1  # Quantum efficiency of APD 
etaSCPD = 0.5 # Quantum efficiency of superconducting single photon detector
gammaAPD = 50 # Dark Count Rate for APD single photon detector
gammaSCPD = 1 # Dark Count Rate for superconducting single photon detector
rho = 10**(-2.5) # Spectral Filer's carrier transmission
T=10**-8 # time Period 
dt = 3.3*10**-9 # gating time window




def lossDb(DB):
    return 10**(DB/10)



def output_key_rate_Original(p, grad):
    mu0 = p[0] 
    beta = p[1]
    Original = MS.OriginalScheme()
    sk = Original.secretKey(etaD=etaAPD,gammaDet=gammaAPD,T=T,dt=dt,eta=eta1,mu0=mu0,rho=rho,beta=beta)
    print(f"secret key is {sk}  at dt={dt}, mu={mu0},beta={beta}")
    return sk


def output_key_rate_Interface(p, grad):
    mu0 = p[0] 
    beta = p[1]
    Interface = MS.InterfaceScheme()
    sk = Interface.secretKey(etaD=etaAPD,gammaDet=gammaAPD,T=T,dt=dt,eta=eta1,tau=tau1,mu0=mu0,rho=rho,beta=beta)
    print(f"secret key is {sk}  at dt={dt}, mu={mu0},beta={beta}")
    return sk


def Pin(mu0,T):
        return mu0*(hbar*w0)/(T)




def optimizeKR(protocol, dstart,dstop,folderName,step=5):
    
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    
    darr = np.arange(dstart,dstop+1,step=1)
    kr = np.zeros(np.size(darr))
    parameters = np.zeros((np.size(darr),2))
    
            
    i=0
    for distance in darr:
        #  Amplitude  range
        mu_min = 0.01;
        mu_max = 150;
        # Modulation depth range
        beta_min = 0.01;
        beta_max = 1.199;
        lossCoeff = 0.2
        dBLoss = lossCoeff*distance
        global eta1
        eta1 = 10**(-lossCoeff*distance/10)
        
        switcher = {0: output_key_rate_Interface,1: output_key_rate_Original}
        p=switcher.get(protocol, 0)   

        mu_0 =  5
        beta_0 = 0.5
        
        opt = nlopt.opt(nlopt.LN_COBYLA, 2)
        opt.set_max_objective(p)
        initialVector = [mu_0, beta_0]
        optmizeVectorMin = [ mu_min , beta_min]
        optmizeVectorMax =  [ mu_max , beta_max]


        opt.set_lower_bounds(optmizeVectorMin)
        opt.set_upper_bounds(optmizeVectorMax)
        opt.set_ftol_rel(0.005)
        opt.set_xtol_rel(0.005)

        opt.max_eval = 2000
        x = opt.optimize(initialVector)
        maxf = opt.last_optimum_value()
        kr[i] = maxf
        parameters[i,0] =x[0]
        parameters[i,1] =x[1]
        #parameters = np.append(parameters,[[x[0],x[1]]],axis=0)
        i+=1
    
       
          
    print("Saving test")
    np.savetxt(f'{folderName}/distance-array.out', darr, delimiter=',')
    np.savetxt(f'{folderName}/key-rate-array.out', kr, delimiter=',')
    np.savetxt(f'{folderName}/parameters.out', parameters, delimiter=',')
    
    
    
    return [darr, kr*100, parameters]

    
    




def main(args):
    folderName = args.folderName
    protocol = args.scheme
    dstart = args.dStart
    dstop = args.dStop
    
    optimizeKR(protocol=protocol, dstart=dstart,dstop=dstop,folderName=folderName,step=5,)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-scheme', type=int, default=0, help='Point to Point with double state detection=0, Point to Point Original=1')
    parser.add_argument('-dStart', type=float, default=50, help='Start distance for optimization in km')
    parser.add_argument('-dStop', type=float, default=200, help='Stop distance for optimization in km')
    parser.add_argument('-folderName', default="Optimization-Data", help='Folder for saving data')
    args = parser.parse_args()
    main(args)
