'''
Created on Sep 13, 2013
Simulate Band Pass filters,  High pass or low pass
@author: pq
'''
import numpy as np 
import matplotlib.pyplot as plt

def LowPass_1st(signal,timeline,freq):
    dt=timeline[1]-timeline[0]
    RL=freq*2*np.pi*dt
    out=np.zeros(len(timeline))
    out[0]=signal[0]
    for i in range(1,len(timeline)):
        out[i]=out[i-1]+RL*(signal[i-1]-out[i-1])
    return out

def LowPassFilter(signal,timeline,order=1,freq=0.6e3):
    rlt=signal
    for i in range(order):
        rlt=LowPass_1st(rlt,timeline,freq)
    return rlt

def HighPass_1st(signal,timeline,freq):
    ### 1st order high pass
    dt=timeline[1]-timeline[0]
    RL=freq*2*np.pi*dt
    out=np.zeros(len(timeline))
    cache=np.zeros(len(timeline))
    cache[0]=signal[0]
    for i in range(1,len(timeline)):
        out[i]=signal[i]-cache[i-1]
        cache[i]=cache[i-1]+RL*out[i]
    return out

def HighPassFilter(signal,timeline,order=2,freq=0.4e3):
    rlt=signal
    for i in range(order):
        rlt=HighPass_1st(rlt,timeline,freq)
    return rlt    
    
def VoltagetoCurrent(voltage,timeline):
    return LowPassProcess(voltage,timeline,1,4.3e3)

def LowPassProcess(signal,timeline,order,freq=1.9e3,shotno=None,plot=False):
    if plot:
        plt.figure()
        plt.suptitle('low pass filter, shot:%d' % shotno)
    lowpass=np.zeros(signal.shape)
    if(len(signal.shape)==1):
        lowpass=LowPassFilter(signal,timeline,order,freq)
        if plot:
            plt.plot(1e3*timeline,1e4*signal,'r')
            plt.plot(1e3*timeline,1e4*lowpass,'b')
            plt.xlim([2,6.5])
    else:
        num=signal.shape[0]
        for i in range(num):
            lowpass[i]=LowPassFilter(signal[i],timeline,order,freq)
            if plot:
                plt.subplot(num,1,i+1)
                plt.plot(1e3*timeline,1e4*signal[i],'r')
                plt.plot(1e3*timeline,1e4*lowpass[i],'b')
                plt.xlim([2,6.5])
                if i<num/2:
                    plt.ylim([0,9])
                else:
                    plt.ylim([0,1])
    if plot:
#         plt.ylim([0,7])
        pass
    return lowpass

def HighPassProcess(signal,timeline,order,freq=0.4e3,shotno=None,plot=False):
    if plot:
        plt.figure()
        plt.suptitle('Highpass filter. Original: red, Highpass blue')
    highpass=np.zeros(signal.shape)
    if(len(signal.shape)==1):
        highpass=HighPassFilter(signal,timeline,order,freq)
        if plot:
            plt.plot(1e3*timeline,1e4*signal,'r')
            plt.plot(1e3*timeline,1e4*highpass,'b')
    else:
        for i in range(len(signal)):
            highpass[i]=HighPassFilter(signal[i],timeline,order,freq)
            if plot and i<4:
                plt.subplot(4,1,i+1)
                plt.plot(timeline*1e3,1e4*signal[i],'r')
                plt.plot(timeline*1e3,1e4*highpass[i],'b')
                plt.plot(timeline*1e3,0*timeline,'k')
#                 if i<2:
#                     plt.ylim([-6,6])
#                 else:
#                     plt.ylim([-0.8,0.8])
#                 plt.ylim([-10,10])
#                 plt.xlim([2,6.5])
    return highpass

def LowPassFilter_Compensate_1st(signal,timeline,freq):
    dt=timeline[1]-timeline[0]
    RL=freq*2*np.pi*dt
    out=np.zeros(len(timeline))
    out[0]=0#signal[0]
    cache=0
    limit=4
    for i in range(1,len(timeline)):
        out[i]=signal[i-1]+(signal[i]-signal[i-1])/RL
#         out[i]+=cache
#         cache=0
#         if (out[i]-out[i-1]>limit):
#             cache=out[i]-out[i-1]-limit
#             out[i]=out[i-1]+limit
#         if (out[i]-out[i-1]<-limit):
#             cache=out[i]-out[i-1]+limit
#             out[i]=out[i-1]-limit
#         cache=cache*(1-RL)
    return out

def HighPassFilter_compensate_1st(signal,timeline,freq):
    dt=timeline[1]-timeline[0]
    RL=3e3*2*np.pi*dt
    out=np.zeros_like(signal)
    sumI=0
    for i in range(1,len(timeline)):
        sumI+=signal[i]
        out[i]=signal[i]+RL*sumI
    return out

def LowPassFit1Pair_mtx(Input,output,timeline,freq,gain):
    ''' try to fit the 1st order low pass filter parameter
    generate the  A delta_x=b  matrix and b term
    '''
    I=LowPass_1st(Input,timeline,freq)
    alpha=2*np.pi*freq
    I_freq=I/alpha-timeline*I+LowPass_1st(Input*timeline,timeline,freq)
    I_freq*=2*np.pi
    deltaOut=output-I*gain
    A=np.zeros([2,len(timeline)])
    A[0]=I
    A[1]=gain*I_freq
    mtx=np.dot(A,np.transpose(A))
    b=np.dot(A,deltaOut)
    return (mtx,b,deltaOut)    
    
    
def LowPassFit1Pair(Input,output,timeline,freq,gain,round=10):
    for i in range(round):
        (mtx,b,deltaOut)=LowPassFit1Pair_mtx(Input,output,timeline,freq,gain)
        (deltagain,deltafreq)=np.dot(np.linalg.inv(mtx),b)
        freq+=deltafreq*0.5
        gain+=deltagain*0.5
        print(freq,gain)
    simu=LowPass_1st(Input,timeline,freq)*gain
    plt.figure()
    plt.plot(timeline,Input,'b')
    plt.plot(timeline,output,'r')
    plt.plot(timeline,simu,'k')
    return(freq,gain)
    
def LowPassFit2Pairs_mtx(Input,output,timeline,freq,gain):
    ''' try to fit the 1st order low pass filter parameter
    generate the  A delta_x=b  matrix and b term
    using two parallel low pass filter
    '''
    freq1=freq[0]
    freq2=freq[1]
    gain1=gain[0]
    gain2=gain[1]
    I1=LowPass_1st(Input,timeline,freq1)
    alpha1=2*np.pi*freq1
    I1_freq=I1/alpha1-timeline*I1+LowPass_1st(Input*timeline,timeline,freq1)
    I1_freq*=2*np.pi
    #####
    I2=Input
#     I2=LowPass_1st(Input,timeline,freq2)
#     alpha2=2*np.pi*freq2
#     I2_freq=I2/alpha2-timeline*I2+LowPass_1st(Input*timeline,timeline,freq2)
#     I2_freq*=2*np.pi
    ######
    deltaOut=output-I1*gain1-I2*gain2
    A=np.zeros([3,len(timeline)])
    A[0]=I1
    A[1]=gain1*I1_freq
    A[2]=I2
#     A[3]=gain2*I2_freq
    mtx=np.dot(A,np.transpose(A))
    b=np.dot(A,deltaOut)
    return (mtx,b,deltaOut)  

def LowPassFit2Pairs(Input,output,timeline,freq,gain,round=10):
    ''' A pole plus a direct transfer'''
    step=0.5
    for i in range(round):
        (mtx,b,deltaOut)=LowPassFit2Pairs_mtx(Input,output,timeline,freq,gain)
        (Dgain1,Dfreq1,Dgain2)=np.dot(np.linalg.inv(mtx),b)
        freq[0]+=Dfreq1*step
#         freq[1]+=Dfreq2*step
        gain[0]+=Dgain1*step
        gain[1]+=Dgain2*step
        print(freq,gain)
    simu=LowPass_1st(Input,timeline,freq[0])*gain[0]  \
            +Input*gain[1]
#             +LowPass_1st(Input,timeline,freq[1])*gain[1]
    plt.figure()
    plt.plot(timeline,Input,'b')
    plt.plot(timeline,output,'r')
    plt.plot(timeline,simu,'k')
    return(freq,gain)

def CCtoBr(input,timeline,freq=[2.254e3,80e3],gain=[0.77,0.29]):
    ''' simulate the Control current to Br field. using the 100us step fn data
    Use two poles'''
#     freq=[2.254e3,80e3];gain=[0.77,0.29]
    out=LowPassProcess(input,timeline, 1,freq[0])*gain[0]
#     out+=LowPassProcess(input,timeline,1,freq[1])*gain[1]
    out+=input*gain[1]
    return out

def BrtoCC(Br,timeline,freq=[2.075e3,37.572e3],gain=[0.76,0.29]):
    '''Try to revert the CC to Br process. revert the parallel two lowpass filter
    Process a single channel'''
    dt=timeline[1]-timeline[0]
    alpha1=freq[0]*2*np.pi*dt
    alpha2=freq[1]*2*np.pi*dt
    g1=gain[0]
    g2=gain[1]
    rlt=np.zeros(Br.shape)
    I1=0
    I2=0
    for i in range(len(Br)):
        rlt[i]=(Br[i]+g1*(alpha1-1)*I1+g2*(alpha2-1)*I2)/(alpha1*g1+alpha2*g2)
        I1+=alpha1*(rlt[i]-I1)
        I2+=alpha2*(rlt[i]-I2)
    return rlt

def BrtoCC_beta(Br,timeline,freqnew=80e3,freq=[2.075e3,37.572e3],gain=[0.76,0.29]):
    ''' shift the crit freq to  freqnew  single channel'''
    dt=timeline[1]-timeline[0]
#     alpha1=freq[0]*2*np.pi
#     alpha2=freq[1]*2*np.pi
#     alpha3=freqnew*2*np.pi
#     g1=gain[0];g2=gain[1]
#     rlt=np.zeros(Br.shape)
#     sum=0
#     for i in range(len(timeline)):
#         rlt[i]=alpha3*(g1/alpha1+g2/alpha2)*Br[i]+sum
#         sum+=alpha3*dt*((g1+g2)*Br[i]-rlt[i])
    freq=[2.254e3,80e3];gain=[0.77,0.29]
    g1=gain[0]
    g2=gain[1]
    alpha1=freq[0]*2*np.pi
    rlt=np.zeros(Br.shape)
    sum=0
    for i in range(len(timeline)):
        rlt[i]=(Br[i]+sum)/g2
        sum+=alpha1*dt*(Br[i]-(g1+g2)*rlt[i])
    return rlt
    
def CCtoVoltage(Current,timeline,freqold=4.3e3,freqnew=20e3):
    return lowPassFilter_Compensate_1st_beta(Current,timeline,freqold,freqnew)    
    
def lowPassFilter_Compensate_1st_beta(signal,timeline,freqold,freqnew):
    ''' shift the crit freq from freqOld to  freqnew
    single channel'''
    dt=timeline[1]-timeline[0]
    alpha1=freqold*2*np.pi
    alpha2=freqnew*2*np.pi
    rlt=np.zeros(signal.shape)
    sum=0
    for i in range(len(timeline)):
        rlt[i]=alpha2/alpha1*signal[i]+sum
        sum+=alpha2*dt*(signal[i]-rlt[i])
    return rlt
    


if __name__=="__main__":
    times=np.arange(1e-3,5e-3,2e-6)
    freq=6e3
    Input=np.sin(times*2*np.pi*freq)
    output=LowPass_1st(Input,times,3e3)
    output+=LowPass_1st(Input,times,7e3)*0.8
#    plt.plot(times,input,times,output)
    LowPassFit2Pairs(Input,output,times,[4e3,6e3],[0.8,0.6],20)
    








    plt.show()
    
        
