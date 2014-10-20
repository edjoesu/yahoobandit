import yahoosim 
import timeucb 

mypolicy = timeucb.timeUCBpolicy(10000)
mysim = yahoosim.YahooBanditSimulator(mypolicy, 'log_ucbextract', writeinterval=100) 

mysim.simulateFile('yextract_big')
