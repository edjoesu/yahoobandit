import yahoosim 
import timeucb 

mypolicy = timeucb.timeUCBpolicy(100)
mysim = yahoosim.YahooBanditSimulator(mypolicy, 'log_ucbextract', writeinterval=100) 

mysim.simulateFile('yextract')
