import numpy as np
import pystan

class timeUCBpolicy:
    def __init__(self, timegroup):
        self.arms = {}
        self.neval = 1
        self.startTime = None
        self.maxArmNumber = 0
        self.timegroup = timegroup
        self.timestepWeights = [1]
        
        self.y = []
        self.a = []
        self.t = []
        
    def select(self, armChoices, timestamp):
        if self.neval == 1:
            self.startTime = int(timestamp)     
            
        for arm in armChoices:
            if arm not in self.arms:
                self.arms[arm]=armstats(self.maxArmNumber)
                self.maxArmNumber = self.maxArmNumber + 1
                
        return max(self.arms, key=lambda aa: self.arms[aa].index)
    
    def timestampToBin(self,timestamp):
        return 1+int((int(timestamp) - self.startTime)/self.timegroup)
    
    def update(self, testedArm, testedReward, timestamp): 

        self.y.append(testedReward)
        self.t.append(self.timestampToBin(timestamp))
        self.a.append(self.arms[testedArm].armnumber)
        
        self.neval = self.neval + 1 
        
        self.arms[testedArm].ntrials = self.arms[testedArm].ntrials + 1
        self.arms[testedArm].timebin_ntrials[-1] = self.arms[testedArm].timebin_ntrials[-1] + 1 
        
        self.arms[testedArm].nsuccess = self.arms[testedArm].nsuccess + testedReward
        self.arms[testedArm].timebin_nsuccess[-1] = self.arms[testedArm].timebin_nsuccess[-1] + testedReward
   
        phat_arm = np.dot(self.arms[testedArm].timebin_nsuccess, self.timestepWeights) / self.arms[testedArm].ntrials
        #phat_arm = float(self.arms[testedArm].nsuccess) / self.arms[testedArm].ntrials
        n_arm = self.arms[testedArm].ntrials + self.arms[testedArm].nsuccess        
        ucb_arm = phat_arm + np.sqrt(2 * np.log(self.neval)/n_arm)
        
        self.arms[testedArm].index = ucb_arm
        
        if self.neval % self.timegroup == 0: 
            pp_means = self.fit_model() 
            
            self.arms[testedArm].timebin_ntrials = self.arms[testedArm].timebin_ntrials + [0]
            self.arms[testedArm].timebin_nsuccess = self.arms[testedArm].timebin_nsuccess + [0]
            
            for arm in self.arms:
                phat_arm = np.dot(self.arms[arm].timebin_success, self.timestepWeights) / self.arms[arm].ntrials
            
    def fit_model(self):
        model_string= r"""
        data { 
        int<lower=0> i; 
        int<lower=0> j; 
        int<lower=0> k;
        int<lower=0,upper=1> y[i]; 
        int<lower=1,upper=j> t[i]; 
        int<lower=0,upper=k> a[i]; 
        }
        parameters {
        simplex theta[j];
        real eta[k];
        real<lower=0> S;
        }
        model {
        S ~ gamma(1,0.5);
        
        for (mm in 2:j) {
        theta[mm] ~ normal(theta[mm-1],S);
        }
        
        for (kk in 1:k) {
        eta[kk] ~ normal(-3,10);
        }
        
        for (nn in 1:i) {
        y[nn] ~ bernoulli(inv_logit(eta[a[nn]]+j*theta[t[nn]]));
        }
        }
        """    
                    
        ucb_data = {'i':len(self.y),
                    'j':max(self.t),
                    'k':self.maxArmNumber, 
                    'y':self.y,
                    't':self.t,
                    'a':self.a}
        
        fit = pystan.stan(model_code=model_string, data=ucb_data,
                          iter=1000, chains=4)     
        
        stan_samples = fit.extract(permuted=True) 
        
        #Stan arrays are 1-referenced so there is an evil off-by-one
        pp_mean = np.mean(stan_samples['theta'])           
        return pp_mean                    
        
class armstats:
    def __init__(self, armnumber = 0):
        self.ntrials = 0 
        self.nsuccess = 0 
        self.index = 1.   
        self.armnumber = armnumber
        self.timebin_ntrials = [0]
        self.timebin_nsuccess = [0]
        