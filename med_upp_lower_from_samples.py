import numpy as np
import matplotlib.pyplot as plt 

def model(amp,avg,sigma,t):
	arg = -(t-avg)**2/(2.*sigma**2)
	val = amp*np.exp(arg)
	return val

data_file = 'samples.dat'
data = np.genfromtxt(data_file)

burn_in = 1000

amp = data[burn_in:,2]
avg = data[burn_in:,3]
sigma = data[burn_in:,4]


no_samples = len(amp)
time = np.linspace(0,10,1000)

model_median = np.zeros(len(time))
model_upper = np.zeros(len(time))
model_lower = np.zeros(len(time))
actual = np.zeros(len(time))
stand_dev = np.zeros(len(time))

for i in range(0,len(time)):
	actual[i] = model(1.0,4.0,2.0,time[i])

# the first entry is the number of rows it seems (for np.zeros that is)
models = np.zeros((len(time),no_samples))


for i in range(0,no_samples):
	for j in range(0,len(time)):
		models[j,i] = model(amp[i],avg[i],sigma[i],time[j])


	
for i in range(0,len(time)):
	model_median[i] = np.median(models[i,:])
	#stand_dev = np.std(models[i,:])
	model_upper[i] = np.max(models[i,:])
	model_lower[i] = np.min(models[i,:])
	
no_sigma = 3.0
#model_upper = model_median+no_sigma*stand_dev
#model_lower = model_median-no_sigma*stand_dev
	
		
plt.figure(1)
plt.plot(time, model_median,'k-.',label='median',linewidth=2)
plt.fill_between(time,model_lower,model_upper,alpha=0.3,facecolor='red')
#plt.fill_between(time,model_upper,model_lower,alpha=0.3,facecolor='red')
plt.plot(time,actual,label='actual')
plt.legend()
plt.show()









