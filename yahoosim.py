import numpy as np
import pystan as stan

class YahooBanditSimulator:
    def __init__(self, policy, logfilename, writeinterval = 1000):
        self.policy = policy
        self.logfilename = logfilename 
        self.writeinterval = writeinterval 
        
        self.ntrials = 0
        self.nsuccess = 0
    
    def writeLog(self, writelist):
        line = '|'.join([str(el) for el in writelist]) + '\n'
        if self.logfilename:
            with open(self.logfilename, 'a') as fo:
                fo.write(line)   
    
    def simulateFile(self, infilename):
        with open(infilename,'r') as fh:
            for line in fh:
                linePieces = [barGroup.split() for barGroup in line.split('|')]
                
                timestamp = linePieces[0][0]
                armChoices = [piece[0] for piece in linePieces[2:]]
                
                testedArm = linePieces[0][1]
                testedReward = int(linePieces[0][2])               
                
                selectedArm = self.policy.select(armChoices, timestamp)
                
                if selectedArm == testedArm:
                    self.policy.update(testedArm, testedReward, timestamp)
                    
                    self.ntrials = self.ntrials + 1 
                    self.nsuccess = self.nsuccess + testedReward
                    
                    if self.ntrials % self.writeinterval == 0:
                        self.writeLog([timestamp, self.ntrials, self.nsuccess])
                

class UCBPolicy:
    def __init__(self):
        self.arms = {}
        self.neval = 1
    def select(self, armChoices, timestamp):
        for arm in armChoices:
            if arm not in self.arms:
                self.arms[arm]=armstats()
                
        return max(self.arms, key=lambda aa: self.arms[aa].index)
    def update(self, testedArm, testedReward, timestamp):
        self.neval = self.neval + 1 
        self.arms[testedArm].ntrials = self.arms[testedArm].ntrials + 1
        self.arms[testedArm].nsuccess = self.arms[testedArm].nsuccess + testedReward
   
        phat_arm = float(self.arms[testedArm].nsuccess) / self.arms[testedArm].ntrials
        n_arm = self.arms[testedArm].ntrials + self.arms[testedArm].nsuccess        
        ucb_arm = phat_arm + np.sqrt(2 * np.log(self.neval)/n_arm)
        
        self.arms[testedArm].index = ucb_arm

class armstats:
    def __init__(self):
        self.ntrials = 0 
        self.nsuccess = 0 
        self.index = 1. 