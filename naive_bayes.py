import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
import math
from scipy.stats import norm
import matplotlib.patches as mpatches

def naive_bayes(tsize0,tsize1):
    mean1 = [1,0]
    mean2 = [0,1]

    Sigma1 = [[1,0.75],[0.75,1] ]
    Sigma2 = [[1,0.75],[0.75,1] ]

    size_train0=tsize0
    size_train1=tsize1

    colmap = {1: 'r', 2: 'g', 3: 'b',4:'y',5:'pink',6:'brown'}

    x1,y1 = np.random.multivariate_normal(mean1, Sigma1, size_train0).T #zero training
    x2,y2 = np.random.multivariate_normal(mean2, Sigma2, size_train1).T #one  training
    x3,y3 = np.random.multivariate_normal(mean1, Sigma1, 100).T #testing set1
    x4,y4 = np.random.multivariate_normal(mean2, Sigma2, 100).T #testing set2

    meanx0 = np.mean(x1)
    meany0 = np.mean(y1)
    stdx0 = np.std(x1)
    stdy0 = np.std(y1)

    meanx1 = np.mean(x2)
    meany1 = np.mean(y2)
    stdx1 = np.std(x2)
    stdy1 = np.std(y2)

    d={'x':x1,'y':y1,'label':np.zeros(len(x1),dtype=int)}
    dtrain=pd.DataFrame(data=d)

    d={'x':x2,'y':y2,'label':np.ones(len(x2),dtype=int)}
    dtrain=dtrain.append(pd.DataFrame(data=d))

    d={'x':x3,'y':y3,'actual':np.zeros(len(x3),dtype=int)}                  
    dtest=pd.DataFrame(data=d)

    d={'x':x4,'y':y4,'actual':np.ones(len(x4),dtype=int)}                    
    dtest=dtest.append(pd.DataFrame(data=d))

    prob0 = size_train0/(size_train0+size_train1)
    prob1 = size_train1/(size_train0+size_train1)


    label0 = norm.pdf(dtest['x'],loc=meanx0,scale=stdx0)*norm.pdf(dtest['y'],loc=meany0,scale=stdy0)
    label1 = norm.pdf(dtest['x'],loc=meanx1,scale=stdx1)*norm.pdf(dtest['y'],loc=meany1,scale=stdy1)

    post0=label0*prob0
    post1=label1*prob1

    dtest['post0'] = post0
    dtest['post1']=post1
    dtest['pred']=dtest[["post0", "post1"]].max(axis=1)
    dtest.loc[dtest.pred > post1, 'predicted'] = 0 
    dtest.loc[dtest.pred > post0, 'predicted'] = 1
    dtest.loc[dtest.pred > post1, 'color'] = 'r' 
    dtest.loc[dtest.pred > post0, 'color'] = 'b'
    dtest.loc[np.logical_and(dtest.predicted==0, dtest.actual==0),'true_negative']=1
    dtest.loc[np.logical_and(dtest.predicted==0, dtest.actual==1),'true_negative']=0  #extra
    dtest.loc[np.logical_and(dtest.predicted==1, dtest.actual==1),'true_negative']=0 #extra
    dtest.loc[np.logical_and(dtest.predicted==1, dtest.actual==0),'true_negative']=0 #extra
    
    dtest.loc[np.logical_and(dtest.predicted==1, dtest.actual==1),'true_positive']=1
    dtest.loc[np.logical_and(dtest.predicted==1, dtest.actual==0),'true_positive']=0  #extra
    dtest.loc[np.logical_and(dtest.predicted==0, dtest.actual==0),'true_positive']=0  #extra
    dtest.loc[np.logical_and(dtest.predicted==0, dtest.actual==1),'true_positive']=0 #extra
    dtest.fillna(0)
    
    dtest.loc[np.logical_and(dtest.predicted==0, dtest.actual==1),'false_positive']=1
    dtest.loc[np.logical_and(dtest.predicted==0, dtest.actual==0),'false_positive']=0       #extra
    dtest.loc[np.logical_and(dtest.predicted==1, dtest.actual==0),'false_positive']=0       #extra
    dtest.loc[np.logical_and(dtest.predicted==1, dtest.actual==1),'false_positive']=0       #extra
    dtest.fillna(0)
    
    dtest.loc[np.logical_and(dtest.predicted==1, dtest.actual==0),'false_negative']=1
    dtest.loc[np.logical_and(dtest.predicted==0, dtest.actual==1),'false_negative']=0       #extra
    dtest.loc[np.logical_and(dtest.predicted==1, dtest.actual==1),'false_negative']=0       #extra
    dtest.loc[np.logical_and(dtest.predicted==0, dtest.actual==0),'false_negative']=0       #extra
    dtest.fillna(0)
    
    tn=dtest.true_negative.sum(axis = 0, skipna = True)
    tp=dtest.true_positive.sum(axis = 0, skipna = True)
    fp=dtest.false_positive.sum(axis = 0, skipna = True)
    fn=dtest.false_negative.sum(axis = 0, skipna = True)
    
    dtest['tp_cum'] = dtest.true_positive.cumsum(axis = 0, skipna = True)
    dtest['tn_cum'] = dtest.true_negative.cumsum(axis = 0, skipna = True)
    dtest['fn_cum'] = dtest.false_negative.cumsum(axis = 0, skipna = True)
    dtest['fp_cum'] = dtest.false_positive.cumsum(axis = 0, skipna = True)
    dtest.fillna(0)

    dtest['tpr']=dtest['tp_cum']/(dtest['tp_cum']+dtest['fn_cum'])
    dtest.fillna(0)
    
    dtest['fpr']=dtest['fp_cum']/(dtest['fp_cum']+dtest['tn_cum'])
    dtest.fillna(0)
    
    accuracy=(tp+tn)/(tp+fp+fn+tn)
    error_rate=1-accuracy
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)
    print("accuracy is {},error rate is {},recall is {} and precision is {} True positive is {} true negative is {} False positive is {} False negative is {}".format(accuracy,error_rate,recall,precision,tp,tn,fp,fn))                                                            #uncomment this for confusion matrix
    fig = plt.figure(figsize=(10, 10))                                                                    # uncomment this for scatter plot
    plt.scatter(dtest['x'], dtest['y'], color=dtest['color'], alpha=0.5, edgecolor='k',marker="x")
    plt.show()
    #find_roc(dtest['fpr'],dtest['tpr'])                                # uncomment this for  roc 
    return accuracy

def find_roc(A,B):
    x = np.array(A)
    y = np.array(B)
    maxx=np.nanmax(x)
    maxy=np.nanmax(y)
    x=np.append(x,1)
    y=np.append(y,1)
    plt.plot(x,y,color='blue')
    plt.xlabel('FPR', fontsize=16)
    plt.ylabel('TPT', fontsize=16)
    area_curve(maxx,maxy)
    plt.show()
def area_curve(maxx,maxy):
    topcurve = math.sqrt((maxx - 1)**2 + (maxy - 1)**2)
    trap=((topcurve+1)*maxy)/2         # ((base1+base2)*height)/2
    print("Area under the curve is",trap)

def main():
    # accuracy1=naive_bayes(10,10)
    # accuracy2=naive_bayes(20,20)
    # accuracy3=naive_bayes(50,50)    
    # accuracy4=naive_bayes(100,100)    
    #accuracy5=naive_bayes(300,300)    
    accuracy6=naive_bayes(500,500)
    #accuracy7=naive_bayes(300,700)                                                     #uncomment this to get part 1-4
    # plotter=[accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6]             # uncomment this to get accuracy plot for different training sizes
    # plt.plot(plotter, color='blue')
    # plt.ylabel('Probability')
    # plt.show()
if __name__ == "__main__": main()
