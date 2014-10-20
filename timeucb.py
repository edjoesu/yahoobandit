import numpy as np
import pystan

class timeUCBpolicy:
    def __init__(self, timegroup):
        self.arms = {}
        self.neval = 1
        
        self.startTime = None
        self.currentBin = 0
        
        self.maxArmNumber = -1
        self.timegroup = timegroup
        self.timestepWeights = [1.]
        
        self.stanModel = self.compileModel()
        
        self.y = []
        self.a = []
        self.t = []
        
    def select(self, armChoices, timestamp):
        if self.neval == 1:
            self.startTime = int(timestamp)     
            
        for arm in armChoices:
            if arm not in self.arms:
                self.maxArmNumber = self.maxArmNumber + 1
                self.arms[arm]=armstats(self.maxArmNumber, self.currentBin)
                
                
        return max(self.arms, key=lambda aa: self.arms[aa].index)
    
    def timestampToBin(self,timestamp):
        return int((int(timestamp) - self.startTime)/self.timegroup)
    
    def newTimeBin(self, timebin):
        if timebin > 1:
            self.fit_model() 
                    
        for arm in self.arms:
            self.arms[arm].computeIndex(self.timestepWeights, self.neval)       
                    
        for arm in self.arms:
            while len(self.arms[arm].timebin_ntrials) < (timebin + 1):
                self.arms[arm].timebin_ntrials = self.arms[arm].timebin_ntrials + [0]
                self.arms[arm].timebin_nsuccess = self.arms[arm].timebin_nsuccess + [0]
                            
            while len(self.timestepWeights) < (timebin + 1):
                self.timestepWeights = self.timestepWeights + [self.timestepWeights[-1]]
                
        self.currentBin = timebin 
    
    def update(self, testedArm, testedReward, timestamp): 
        
        timebin = self.timestampToBin(timestamp)
        
        if timebin > self.currentBin: 
            self.newTimeBin(timebin)        

        self.y.append(testedReward)
        self.t.append(timebin + 1) #stan arrays are one-referenced 
        self.a.append(self.arms[testedArm].armnumber + 1) #stan arrays are one-referenced
        
        self.neval = self.neval + 1 
        
        self.arms[testedArm].ntrials = self.arms[testedArm].ntrials + 1
        self.arms[testedArm].timebin_ntrials[-1] = self.arms[testedArm].timebin_ntrials[-1] + 1 
        
        self.arms[testedArm].nsuccess = self.arms[testedArm].nsuccess + testedReward
        self.arms[testedArm].timebin_nsuccess[-1] = self.arms[testedArm].timebin_nsuccess[-1] + testedReward
   
        self.arms[testedArm].computeIndex(self.timestepWeights,self.neval)     
    def compileModel(self):
        model_string= r"""
        data { 
        int<lower=0> i; 
        int<lower=0> j; 
        int<lower=0> k;
        int<lower=0,upper=1> y[i]; 
        int<lower=1,upper=j> t[i]; 
        int<lower=1,upper=k> a[i]; 
        }
        parameters {
        real theta[j-1];
        real eta[k];
        real<lower=0> S;
        }       
        model {
        vector[j] theta_temp;
        
        S ~ gamma(1,0.5);
        

        
        for (kk in 1:k) {
        eta[kk] ~ normal(-3,10);
        }
               
        theta_temp[1] <- 0.0;
        for (ll in 1:(j-1)) theta_temp[ll+1] <- theta[ll];
        
        for (mm in 2:j) {
        theta_temp[mm] ~ normal(theta_temp[mm-1],S);
        }        
        
        for (nn in 1:i) {
        y[nn] ~ bernoulli(inv_logit(eta[a[nn]]+j*theta_temp[t[nn]]));
        }
        }
        """   
        
        return pystan.StanModel(model_code = model_string)
        
    def fit_model(self):
        ucb_data = {'i':len(self.y),
                    'j':self.currentBin + 1,
                    'k':self.maxArmNumber + 1, 
                    'y':self.y,
                    't':self.t,
                    'a':self.a}        
        
        fit = self.stanModel.sampling(data=ucb_data, iter=1000, chains=4)     
        
        stan_samples = fit.extract(permuted=True) 
        
        #Stan arrays are 1-referenced so there is an evil off-by-one
        theta_hat = np.mean(stan_samples['theta'],0)
        try: 
            tmp = theta_hat[0]
        except:
            theta_hat = [theta_hat]
        
        self.timestepWeights = [1.] + [np.exp(-1 * ll) for ll in theta_hat]
        
        return True
                         
        
class armstats:
    def __init__(self, armnumber = 0, timegroup = 0):
        self.ntrials = 0 
        self.nsuccess = 0 
        self.index = 1.   
        self.armnumber = armnumber
        self.timebin_ntrials = [0] * (timegroup + 1)
        self.timebin_nsuccess = [0] * (timegroup + 1)
    
    def computeIndex(self, timestepWeights, neval):
        phat_arm = np.dot(self.timebin_nsuccess, timestepWeights) / self.ntrials 
        n_arm = self.ntrials + self.nsuccess        
        ucb_arm = phat_arm + np.sqrt(2 * np.log(neval)/n_arm)     
        
        self.index = ucb_arm