import numpy as np
from scipy.stats import entropy
import scipy.special as sf

imi=np.array(1.j);


    
class OriginalScheme:
    def __init__(self):
        pass
    
    def chi(self, mu0, eta, beta):
        x=(1-np.exp(-mu0*(1-eta)*(1-sf.jv(0, 2*beta))) )/2
        return entropy([x,1-x],base=2)
    
            
    def secretKey(self,etaD,gammaDet,T,dt,eta,mu0,rho,beta):
        E=self.Pdet(etaD,gammaDet,T,dt,eta,mu0,rho,beta,phiA=0,phiB=np.pi)/2
        SD=self.Pdet(etaD,gammaDet,T,dt,eta,mu0,rho,beta,phiA=0,phiB=0)/2
        PD=E+SD # Probability of photodetection
        K=PD*(1-self.QBEREntropy(etaD,gammaDet,T,dt,eta,mu0,rho,beta)-self.chi(mu0, eta, beta))/(2*T)        
        return K/1000
            

    def Pdet(self,etaD,gammaDet,T,dt,eta,mu0,rho,beta,phiA,phiB):
        return (etaD*self.Nph(eta,mu0,rho,beta,phiA,phiB)/T + gammaDet)*dt
            

    def Fk(self,k,phiA,phiB,beta):
        # k is the sideband order
        res=0
        for n in np.arange(-20,20):
            res+=sf.jv(n, beta)*sf.jv(k-n, beta)*(np.exp(imi*n*phiA+imi*(k-n)*phiB )+ np.exp(imi*n*phiB+imi*(k-n)*phiA ))
        return res/2

    def Nph(self,eta,mu0,rho,beta,phiA,phiB):
        # Number of photons 
        nar=np.arange(-20,20)
        nar=np.delete(nar,np.argwhere(nar==0))
        res = 0 
        for n in nar:
            res+=np.absolute(self.Fk(n,phiA,phiB,beta))**2
        res+=rho*np.absolute(self.Fk(0,phiA,phiB,beta))**2
        return eta*mu0*res

    def QBER(self,etaD,gammaDet,T,dt,eta,mu0,rho,beta):
        E=self.Pdet(etaD,gammaDet,T,dt,eta,mu0,rho,beta,phiA=0,phiB=np.pi)/2
        SD=self.Pdet(etaD,gammaDet,T,dt,eta,mu0,rho,beta,phiA=0,phiB=0)/2
        return E/(E+SD)

    def QBEREntropy(self,etaD,gammaDet,T,dt,eta,mu0,rho,beta):
        x =self.QBER(etaD,gammaDet,T,dt,eta,mu0,rho,beta)
        return entropy([x,1-x],base=2)

    
    
class InterfaceScheme:
    def __init__(self):
        pass
    
    def chi(self,mu0, eta, beta):
        x=(1-np.exp(-mu0*(1-eta)*(1-sf.jv(0, 2*beta))) )/2
        return entropy([x,1-x],base=2)

    def Fk(self,k,phiA,phiB,beta):
        # k is the order
        res=0
        for n in np.arange(-20,20):
            res+=sf.jv(n, beta)*sf.jv(k-n, beta)*(np.exp(imi*n*phiA+imi*(k-n)*phiB )+ np.exp(imi*n*phiB+imi*(k-n)*phiA ))
        return res/2            
            
    
    def Fk1(self,k,phiA,phiB,beta,rho,tau,mu0):
        # k is the order
        return 2**(-1/2)*(mu0*tau)**(1/2)*(sf.jv(k, beta)*np.exp(imi*k*phiA) + (-rho**(1/2)+tau**(1/2))*sf.jv(0, beta)*sf.jv(k, beta)*np.exp(imi*k*phiB) + (rho)**(1/2)*self.Fk(k,phiA,phiB,beta))

    def F01(self,phiA,phiB,beta,rho,tau,mu0):
        # k is the order
        return 2**(-1/2)*(mu0*rho)**(1/2)*(sf.jv(0, beta)+ (-rho**(1/2)+tau**(1/2))*sf.jv(0, beta)**2 + rho**(1/2)*self.Fk(0,phiA,phiB,beta))

    def Fk2(self,k,phiA,phiB,beta,rho,tau,mu0):
        # k is the order
        return 2**(-1/2)*(mu0*tau)**(1/2)*(sf.jv(k, beta)*np.exp(imi*k*phiA) - (-rho**(1/2)+tau**(1/2))*sf.jv(0, beta)*sf.jv(k, beta)*np.exp(imi*k*phiB) - (rho)**(1/2)*self.Fk(k,phiA,phiB,beta))

    def F02(self,phiA,phiB,beta,rho,tau,mu0):
        # k is the order
        return 2**(-1/2)*(mu0*rho)**(1/2)*(- sf.jv(0, beta)+ (-rho**(1/2)+tau**(1/2))*sf.jv(0, beta)**2 + rho**(1/2)*self.Fk(0,phiA,phiB,beta))


    def Nph1(self,eta,mu0,rho,tau,beta,phiA,phiB):
        # mu0 = |alpha0|**2
        nar=np.arange(-20,20)
        nar=np.delete(nar,np.argwhere(nar==0))
        res = 0 
        for k in nar:
            res+=np.absolute(self.Fk1(k,phiA,phiB,beta,rho,tau,mu0))**2
        res+=np.absolute(self.F01(phiA,phiB,beta,rho,tau,mu0))**2
        return eta*res

    def Nph2(self,eta,mu0,rho,tau,beta,phiA,phiB):
        # mu0 = |alpha0|**2
        nar=np.arange(-20,20)
        nar=np.delete(nar,np.argwhere(nar==0))
        res = 0 
        for k in nar:
            res+=np.absolute(self.Fk2(k,phiA,phiB,beta,rho,tau,mu0))**2
        res+=np.absolute(self.F02(phiA,phiB,beta,rho,tau,mu0))**2
        return eta*res

    def Pdet1(self,etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA,phiB):
        # Probability of first detector click
        return (etaD * self.Nph1(eta=eta,mu0=mu0,rho=rho,tau=tau,beta=beta,phiA=phiA,phiB=phiB)/T + gammaDet)*dt

    def Pdet2(self,etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA,phiB):
        # Probability of second detector click
        return (etaD * self.Nph2(eta,mu0,rho,tau,beta,phiA,phiB)/T + gammaDet)*dt
    
    def secretKey(self,etaD,gammaDet,T,dt,eta,tau,mu0,rho,beta):
        # secret key generation rate
        a=self.Pdet2(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=0)*(1 - self.Pdet1(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=0))
        b=self.Pdet1(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=np.pi)*(1 - self.Pdet2(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=np.pi))
        E=a+b
        SD=self.Pdet1(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=0)*(1 - self.Pdet2(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=0))+self.Pdet2(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=np.pi)*(1 - self.Pdet1(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=np.pi))
        PD=E+SD # Probability of photodetection
        K=PD*(1-self.QBEREntropy(etaD,gammaDet,T,dt,eta,tau,mu0,rho,beta)-self.chi(mu0, eta, beta))/(2*T)
        return K/1000
    

    def QBEREntropy(self,etaD,gammaDet,T,dt,eta,tau,mu0,rho,beta):
        x=self.QBER(etaD,gammaDet,T,dt,eta,tau,mu0,rho,beta)
        return entropy([x,1-x],base=2)
    

    

    def QBER(self,etaD,gammaDet,T,dt,eta,tau,mu0,rho,beta):
        a=self.Pdet2(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=0)*(1 - self.Pdet1(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=0))
        b=self.Pdet1(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=np.pi)*(1 - self.Pdet2(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=np.pi))
        E=a+b
        SD=self.Pdet1(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=0)*(1 - self.Pdet2(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=0))+self.Pdet2(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=np.pi)*(1 - self.Pdet1(etaD,gammaDet,T,dt,eta,mu0,rho,tau,beta,phiA=0,phiB=np.pi))
        return E/(E+SD)
    