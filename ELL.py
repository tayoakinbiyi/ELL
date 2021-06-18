import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import pandas as pd
from multiprocessing import shared_memory, cpu_count
from socket import gethostname
import numpy as np
import pdb
import time

import sys
import time
import os

import mmap
import subprocess

from concurrent.futures import ProcessPoolExecutor, wait
from statsmodels.stats.moment_helpers import cov2corr
from scipy.stats import norm, beta, betabinom
from scipy.special import gammaln,betaln,gamma
import rpy2.robjects.numpy2ri as n2ri
n2ri.activate()
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects as ro

def compute(parms):     
    print('began compute',flush=True)

    vZ=np.loadtxt('input/vZ',delimiter=',')
    numVzEigenValsToKeep=parms['numVzEigenValsToKeep']
    numHermite=parms['numHermite']
    minEta=parms['minEta']
    maxEta=parms['maxEta']
    numLam=parms['numLam']

    L=eigen(vZ,numVzEigenValsToKeep)
    D=L.shape[1]
    corr=cov2corr(L@L.T)

    offDiag=corr[np.triu_indices(corr.shape[1],1)]   
    hermiteDiag=[0 if (n%2==1) else np.mean((np.abs(offDiag)**(n+2))/gamma(n+3)) for n in range(numHermite)]

    minLamPerD,maxLamPerD,lam=minMaxLamPerD(hermiteDiag,D,parms)
    minDPerBin,maxDPerBin,minBinPerD,maxBinPerD=minMaxDPerBin(minLamPerD,maxLamPerD,lam)

    precomputeDF,startPrecomputeRowPerD,endPrecomputeRowPerD=makePrecompute(hermiteDiag,lam,minBinPerD,maxBinPerD,minDPerBin,maxDPerBin,
        D,parms)

    precomputeDF['eta'][startPrecomputeRowPerD]=2
    precomputeDF['eta'][endPrecomputeRowPerD]=1
    precomputeDF['lam'][endPrecomputeRowPerD]=1
    precomputeDF=precomputeDF[precomputeDF['eta']>0]
    precomputeDF['eta'][precomputeDF['eta']==2]=0

    etaPrev=precomputeDF['eta'][0:-1]
    etaNext=precomputeDF['eta'][1:]
    loc=np.append(np.where(etaPrev!=etaNext)[0],np.array([len(precomputeDF)-1]))
    precomputeDF=precomputeDF[loc]

    np.savetxt('intermediate/precompute',precomputeDF,delimiter=',')
    np.savetxt('intermediate/L',L,delimiter=',')
    
    print('finished compute',flush=True)
    
    return()
    
def monteCarlo(parms):
    print('began monteCarlo',flush=True)
    
    if not os.path.exists('intermediate/precompute'):
        sys.exit('Run precompute first')
    
    mcReps=parms['mcReps']
    folder=parms['folder']
    numCores=parms['numCores']

    precomputeDF=np.loadtxt('intermediate/precompute',delimiter=',',dtype=[('lam','float64'),('eta','float64')])
    b_precompute,precomputeNm=bufCreate(precomputeDF.shape[0],dtype=[('lam','float64'),('eta','float64')])
    b_precompute[0][:]=precomputeDF

    startPrecomputeRowPerD=np.where(precomputeDF['eta']==0)[0].astype(int)
    endPrecomputeRowPerD=np.where(precomputeDF['eta']==1)[0].astype(int)

    L=np.loadtxt('intermediate/L',delimiter=',')
    b_L,LNm=bufCreate(L.shape)
    b_L[0][:]=L
    
    D=L.shape[1]
    xargs=[]

    with ProcessPoolExecutor(numCores) as executor:
        for core in range(numCores):
            xargs+=[(LNm,int(np.ceil(mcReps/numCores)),startPrecomputeRowPerD,endPrecomputeRowPerD,precomputeNm,D,parms,core)]

        ff=executor.map(monteCarloHelp,xargs)
    
    bufClose(precomputeNm)
    bufClose(LNm)
    
    print('finished monteCarlo',flush=True)

    return()

def monteCarloCombine(parms):
    print('began monteCarloCombine',flush=True)
    
    folder=parms['folder']

    ell=[]
    files=os.listdir('intermediate/ellRef')
    for file in files:  
        ell+=[np.loadtxt('intermediate/ellRef/'+file,delimiter=',').reshape(-1,1)]
    ell=np.sort(np.concatenate(ell,axis=0))
    np.savetxt('intermediate/ellMC',ell,delimiter=',')

    naive=[]
    files=os.listdir('intermediate/naiveRef')
    for file in files:  
        naive+=[np.loadtxt('intermediate/naiveRef/'+file,delimiter=',').reshape(-1,1)]
    naive=np.sort(np.concatenate(naive,axis=0))
    np.savetxt('intermediate/naiveMC',naive,delimiter=',')

    print('finished monteCarloCombined',flush=True)
    
    return()
    
def score(parms):
    print('began score',flush=True)

    if not (os.path.exists('intermediate/ellMC') and os.path.exists('intermediate/naiveMC')):
        sys.exit('Run monteCarlo First')
        
    globalPvalThresh=parms['globalPvalThresh']
    numCores=parms['numCores']

    Z=np.loadtxt('input/Z',delimiter=',')
    
    precomputeDF=np.loadtxt('intermediate/precompute',delimiter=',',dtype=[('lam','float64'),('eta','float64')])
    b_precompute,precomputeNm=bufCreate(precomputeDF.shape[0],dtype=[('lam','float64'),('eta','float64')])
    b_precompute[0][:]=precomputeDF

    startPrecomputeRowPerD=np.where(precomputeDF['eta']==0)[0].astype(int)
    endPrecomputeRowPerD=np.where(precomputeDF['eta']==1)[0].astype(int)
    
    b_ell,ellNm=bufCreate(Z.shape[0])
    b_naive,naiveNm=bufCreate(Z.shape[0])

    snpLabels=np.loadtxt('input/snpLabels',delimiter=',',dtype='object')

    D=Z.shape[1]
    calD=int(np.ceil(parms['delta']*D))

    xargs=[]
    with ProcessPoolExecutor(numCores) as executor:
        for core in range(numCores):
            rows=np.arange(core*int(np.ceil(Z.shape[0]/numCores)),min(Z.shape[0],(core+1)*int(np.ceil(Z.shape[0]/numCores))))
            xargs+=[(Z[rows],rows,startPrecomputeRowPerD,endPrecomputeRowPerD,precomputeNm,D,parms,ellNm,naiveNm)]
                 
        ff=executor.map(scoreHelp,xargs)

    bufClose(precomputeNm)
    ellStats=bufClose(ellNm)
    naiveStats=bufClose(naiveNm)
    
    ellMC=np.sort(np.loadtxt('intermediate/ellMC',delimiter=','))
    naiveMC=np.sort(np.loadtxt('intermediate/naiveMC',delimiter=','))

    sortOrd=np.argsort(ellStats)
    ellPvals=ellStats.copy()
    ellPvals[sortOrd]=(np.searchsorted(ellMC,ellStats[sortOrd],side='left')+1)/(len(ellMC)+1)

    sortOrd=np.argsort(naiveStats)
    naivePvals=naiveStats.copy()
    naivePvals[sortOrd]=(np.searchsorted(naiveMC,naiveStats[sortOrd],side='left')+1)/(len(naiveMC)+1)

    scoreDF=pd.DataFrame({'snpLabels':snpLabels,'ellStats':ellStats,'ellPvals':ellPvals,'naiveStats':naiveStats,'naivePvals':naivePvals})
    scoreDF.to_csv('output/score',index=False)
    
    print('finished score, numCandidates {}'.format(np.sum(scoreDF['ellPvals']<globalPvalThresh)),flush=True)
    
    return()

def associatedTraits(parms):
    print('began associatedTraits',flush=True)
    
    if not os.path.exists('output/score'):
        sys.exit('Run score First')

    FDR=parms['FDR']
    globalPvalThresh=parms['globalPvalThresh']
    mM=(parms['numCandidates']/parms['totalSnps'])

    traitLabels=np.loadtxt('input/traitLabels',delimiter=',',dtype='object')
    snpLabels=np.loadtxt('input/snpLabels',delimiter=',',dtype='object')

    D=len(traitLabels)
    calD=int(np.ceil(parms['delta']*D))

    score=pd.read_csv('output/score',index_col=None,header=0,dtype={'snpLabels':'object','ellStats':float,
        'ellPvals':float,'naiveStats':float,'naivePvals':float})[['snpLabels','ellStats','ellPvals','naiveStats','naivePvals']]    
    pi=2*norm.sf(np.loadtxt('input/Z',delimiter=','))

    associatedTraits=pd.DataFrame([[snpLabels[ind],trait] 
        for (traitLabels,snpLabels,pi,mM,FDR,ellPvals,globalPvalThresh) in 
                [(traitLabels,snpLabels,pi,mM,FDR,score['ellPvals'].values.flatten(),globalPvalThresh)]
            for ind,row in enumerate(pi) 
                for trait in traitLabels[np.argsort(row)[:np.max(np.where(np.sort(row)<FDR*mM*np.arange(1,len(row)+1)/len(row))[0],
                                                                 initial=0)+1]]
                    if ellPvals[ind]<globalPvalThresh],columns=['snpLabels','traitLabels'])
    associatedTraits.to_csv('output/associatedTraits',index=False)

    score=score.merge(associatedTraits.groupby('snpLabels')['traitLabels'].count().reset_index().rename(
        columns={'traitLabels':'numAssociatedTraits'}),on='snpLabels',how='left').fillna(0).astype({'numAssociatedTraits':int})

    score.to_csv('output/score',index=False)
    
    print('finished associatedTraits',flush=True)
    
    return()                
        
def eqtlDistance(parms):
    print('began eqtlDistance',flush=True)
    
    if not os.path.exists('output/associatedTraits'):
        sys.exit('Run associatedTraits First')

    numCores=parms['numCores']
    
    associatedTraits=pd.read_csv('output/associatedTraits',index_col=None,header=0,dtype={'snpLabels':'object',
        'traitLabels':'object'})[['snpLabels','traitLabels']]
    score=pd.read_csv('output/score',index_col=None,header=0,dtype={'snpLabels':'object','ellStats':float,
        'ellPvals':float,'naiveStats':float,'naivePvals':float,'numAssociatedTraits':int})[['snpLabels','ellStats',
        'ellPvals','naiveStats','naivePvals','numAssociatedTraits']]    
    score=score[score['numAssociatedTraits']>0]
                                                                                              
    numEQTL=len(score)

    b_distance,distanceNm=bufCreate([numEQTL,numEQTL])

    eqtlPairs=np.concatenate([x.reshape(-1,1) for x in np.triu_indices(numEQTL,1)],axis=1)
    numEqtlPairs=eqtlPairs.shape[0]
    
    snpLocs=np.cumsum(np.append(np.array([0]),score['numAssociatedTraits'])).astype(int)

    xargs=[]
    with ProcessPoolExecutor(numCores) as executor:
        for core in range(numCores):
            eqtlPairRange=np.arange(core*int(np.ceil(numEqtlPairs/numCores)),min(numEqtlPairs,(core+1)*int(np.ceil(
                numEqtlPairs/numCores))))
            if len(eqtlPairRange)==0:
                continue

            xargs+=[(associatedTraits,distanceNm,snpLocs,eqtlPairs[eqtlPairRange])]

        ff=executor.map(distanceHelp,xargs)

    distanceDF=bufClose(distanceNm)

    clusterString='''
    cluster <- function(distDF) {
        ans=hclust(as.dist(distDF),method='average')                    
        return(ans)
    }    
    '''
    f_clust=SignatureTranslatedAnonymousPackage(clusterString,'cluster')    
    ans=f_clust.cluster(distanceDF)
    merge=np.array(ans.rx2('merge')) 
    height=np.array(ans.rx2('height')) 

    np.savetxt('intermediate/merge',merge,delimiter=',')
    np.savetxt('intermediate/height',height,delimiter=',')
    np.savetxt('intermediate/eqtlDistance',distanceDF,delimiter=',')
    
    print('finished eqtlDistance',flush=True)
    
    return()

def cluster(parms):   
    print('began cluster',flush=True)
    
    if not os.path.exists('intermediate/eqtlDistance'):
        sys.exit('Run eqtlDistance First')

    merge=np.loadtxt('intermediate/merge',delimiter=',')
    height=np.loadtxt('intermediate/height',delimiter=',')
    distanceDF=np.loadtxt('intermediate/eqtlDistance',delimiter=',')
    associatedTraits=pd.read_csv('output/associatedTraits',index_col=None,header=0,dtype={'snpLabels':'object',
        'traitLabels':'object'})[['snpLabels','traitLabels']]

    score=pd.read_csv('output/score',index_col=None,header=0)[['snpLabels','ellStats','ellPvals','naiveStats',
        'naivePvals','numAssociatedTraits']]
    loc=(score['numAssociatedTraits']>0)

    clusterDistance=parms['clusterDistance']
    
    labels=np.array([])

    cutString='''
    f_cut <- function(merge,height,labels,clusterDistance) {
        ans=cutree(list(merge=merge,height=height,labels=labels),h=clusterDistance)                    
        return(ans)
    }    
    '''
    f_cut=SignatureTranslatedAnonymousPackage(cutString,'f_cut')    

    cuts=np.array(f_cut.f_cut(merge,height,labels,clusterDistance))
    
    clusterOverlapMatrix=np.zeros(distanceDF.shape)
    clusterOverlapMatrix[np.triu_indices(distanceDF.shape[0],1)]=np.array([[(1 if i==j else 0) for i in ls] 
        for ls in [cuts] for j in ls])[np.triu_indices(distanceDF.shape[0],1)]
    clusterOverlapMatrix[np.tril_indices(distanceDF.shape[0],0)]=(1-distanceDF)[np.tril_indices(distanceDF.shape[0],0)]

    score.insert(1,'cluster',0)
    score.loc[loc,'cluster']=cuts
    score.to_csv('output/score',index=False)
    
    np.savetxt('output/clusterOverlapMatrix',clusterOverlapMatrix,delimiter=',')

    print('finished cluster',flush=True)
    
    return()

def plotEllPvals(parms):
    print('began plotEllPvals',flush=True)
    
    if not os.path.exists('output/clusterOverlapMatrix'):
        sys.exit('Run cluster First')

    globalPvalThresh=parms['globalPvalThresh']
    
    scoreDF=pd.read_csv('output/score',index_col=None,header=0)

    try:
        ID=[float(x) for x in scoreDF['snpLabels']]
    except:
        print('Non float snp labels so using snp index',flush=True)
        ID=np.arange(scoreDF.shape[0])
    
    fig, axs = plt.subplots(1,1,dpi=400)  
    fig.set_figwidth(3,forward=True)
    fig.set_figheight(3,forward=True)     
    axs.scatter(ID,-np.log10(scoreDF['naivePvals']),c='blue',marker='.',s=.7,linewidths = 0,label='Naive-Pvalue')
    axs.scatter(ID,-np.log10(scoreDF['ellPvals']),c='red',marker='.',s=.7,linewidths = 0,label='ELL-Pvalue')
    axs.set_xlabel('Chromosomal Location (Mb)',fontsize=5)
    axs.set_ylabel('-log10 PValue',fontsize=5)
    axs.axhline(y=-np.log10(globalPvalThresh),ls='--',c='black',linewidth=1,label='Genome Wide Threshold')
    axs.title.set_position([0.5,0.98])
    axs.tick_params(labelsize=5)                    
    axs.set_title('Manhattan Plot of Trans eQTL P-values for eQTLs on Chromosome',fontsize=4)
    axs.legend(fontsize=3)
    fig.savefig('output/plotEllPvals.png')    
    plt.close('all')
    
    print('finished plotEllPvals',flush=True)
    
    return()

def plotEllStats(parms):
    print('began plotEllStats',flush=True)
    
    if not os.path.exists('output/clusterOverlapMatrix'):
        sys.exit('Run cluster First')

    globalPvalThresh=parms['globalPvalThresh']
    ellMC=np.loadtxt('intermediate/ellMC')
    
    scoreDF=pd.read_csv('output/score',index_col=None,header=0)

    try:
        ID=[float(x) for x in scoreDF['snpLabels']]
    except:
        print('Non float snp labels so using snp index',flush=True)
        ID=np.arange(scoreDF.shape[0])
    
    fig, axs = plt.subplots(1,1,dpi=400)  
    fig.set_figwidth(3,forward=True)
    fig.set_figheight(3,forward=True)     
    axs.scatter(ID,-np.log10(scoreDF['ellStats']),c='red',marker='.',s=.7,linewidths = 0,label='ELL-Statistics')
    axs2=axs.twinx()
    axs2.scatter(ID,-np.log10(scoreDF['naiveStats']),c='blue',marker='.',s=.7,linewidths = 0,label='Naive-Statistics')
    axs2.set_ylabel('-log10 Naive Statistics',fontsize=5)
    axs.set_ylabel('-log10 Ell Statistics',fontsize=5)
    axs.set_xlabel('Chromosomal Location (Mb)',fontsize=5)
    axs.axhline(y=-np.log10(ellMC[int(len(ellMC)*globalPvalThresh)]),ls='--',c='black',linewidth=1)
    axs.title.set_position([0.5,0.98])
    axs.tick_params(labelsize=5)                    
    axs2.tick_params(labelsize=5)                    
    axs.set_title('Manhattan Plot of Trans eQTL P-values for eQTLs on Chromosome',fontsize=4)
    axs.legend(fontsize=3)
    fig.savefig('output/plotEllStats.png')    
    plt.close('all')
    
    print('finished plotEllStats',flush=True)
    
    return()

def plotOverlap(parms):
    print('began plotOverlap',flush=True)
    
    if not os.path.exists('output/clusterOverlapMatrix'):
        sys.exit('Run cluster First')

    hspace=0.14
    wspace=0.11

    clusterOverlapMatrix=np.loadtxt('output/clusterOverlapMatrix',delimiter=',')
    scoreDF=pd.read_csv('output/score',index_col=None,header=0)

    ID=scoreDF['snpLabels'].values.flatten().tolist()
    
    scoreDF=scoreDF[scoreDF['numAssociatedTraits']>0]

    try:
        ID=np.array([float(x) for x in ID])
    except:
        print('Non float snp labels so using snp index',flush=True)
        ID=np.arange(len(ID))
    
    tick=np.unique(np.round(np.linspace(0,scoreDF.shape[0]-1,10))).astype(int)                    
    tickLabels=np.round(ID[tick].astype(float),3)    

    fig, axs = plt.subplots(1,1,dpi=400)  
    fig.set_figwidth(3,forward=True)
    fig.set_figheight(3,forward=True)         
    im=axs.imshow(clusterOverlapMatrix,vmin=0,vmax=1,cmap='Blues',origin='lower',aspect='auto')
    axs2=axs.twinx()
    axs2.scatter(np.arange(scoreDF.shape[0]),scoreDF['numAssociatedTraits'],color='black',s=.3,linewidths=0)
    axs2.set_ylim([0,int(scoreDF['numAssociatedTraits'].max())])
    axs2.tick_params(axis='both',labelsize=5,pad=0)
    axs.set_xticks(tick)
    axs.set_yticks(tick)
    axs.set_xticklabels(tickLabels)
    axs.set_yticklabels(tickLabels)
    axs.tick_params(axis='both',labelsize=5,pad=0)  

    fig.subplots_adjust(right=0.82,hspace=0)
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar_ax.tick_params(axis='y',labelsize=4,pad=0,length=1)

    fig.colorbar(im, cax=cbar_ax)

    fig.text(0.5, 0.05,'Trans-eQTL Chromosomal Location (Mb)', ha='center',fontsize=5) # x
    fig.text(0.01, 0.5,'Trans-eQTL Chromosomal Location (Mb)', va='center',rotation='vertical',fontsize=5) # y
    fig.text(0.89, 0.3,'[#] Associated Traits Per Snp', va='bottom',rotation=270,fontsize=5)
    fig.text(0.975, 0.4,'Overlap Score', va='bottom',rotation=270,fontsize=5)
    fig.suptitle('{} eQTLs, {} clusters'.format(scoreDF.shape[0],scoreDF['cluster'].drop_duplicates().shape[0]),
        fontsize=8,va='bottom',y=0.9)
    fig.savefig('output/plotOverlap.png')

    plt.close('all')
    
    print('finished plotOverlap',flush=True)
    
    return()

##################################################################################################################3
######################################### compute #################################################################
##################################################################################################################33

def minMaxLamPerD(hermiteDiag,D,parms): 
    calD=int(np.ceil(parms['delta']*D))
    minEta=parms['minEta']
    maxEta=parms['maxEta']
    numLam=parms['numLam']
    numCores=parms['numCores']
    maxIters=parms['maxIters']

    start=ordPvalInvCDF(hermiteDiag,0,D,minEta,minEta,0,(1/D),maxIters)[1]
    end=ordPvalInvCDF(hermiteDiag,calD-1,D,maxEta,maxEta*1e-3,(1/D),1,maxIters)[0]

    lam=np.unique(geomBins(numLam,start,end))
    lam=lam[lam>0]

    b_minLamPerD,minNm=bufCreate([calD])
    b_maxLamPerD,maxNm=bufCreate([calD])

    xargs=[]
    with ProcessPoolExecutor(numCores) as executor:
        for core in range(numCores):
            dRange=np.arange(core*int(np.ceil(calD/numCores)),min(calD,(core+1)*int(np.ceil(calD/numCores))))
            if len(dRange)==0:
                continue

            xargs+=[(dRange,D,minNm,maxNm,minEta,maxEta,maxIters,hermiteDiag,start,end,core)]

        ff=executor.map(minMaxLamPerDHelp,xargs)

    return(np.sort(bufClose(minNm)),np.sort(bufClose(maxNm)),lam)

def minMaxLamPerDHelp(args):
    j=0
    dRange=args[j];j+=1
    D=args[j];j+=1
    minNm=args[j];j+=1
    maxNm=args[j];j+=1
    minEta=args[j];j+=1
    maxEta=args[j];j+=1
    maxIters=args[j];j+=1
    hermiteDiag=args[j];j+=1
    low=args[j];j+=1
    high=args[j];j+=1
    core=args[j];j+=1
    
    b_minLamPerD=bufLoad(minNm)
    b_maxLamPerD=bufLoad(maxNm)

    for dInd in range(len(dRange)):
        lam=ordPvalInvCDF(hermiteDiag,dRange[dInd],D,minEta,minEta,low,high,maxIters)
        b_minLamPerD[0][dRange[dInd]]=lam[0]
        low=lam[0]

        lam=ordPvalInvCDF(hermiteDiag,dRange[-dInd-1],D,maxEta,maxEta*1e-3,low,high,maxIters)
        b_maxLamPerD[0][dRange[-dInd-1]]=lam[1]
        high=lam[1]
        
    b_minLamPerD[1].close()
    b_maxLamPerD[1].close()
    
    return()

def minMaxDPerBin(minLamPerD,maxLamPerD,lam):
    numLam=len(lam)

    minBinPerD=np.clip(np.searchsorted(lam,minLamPerD),0,numLam-1)
    minBinPerD[0]=0

    maxBinPerD=np.clip(np.searchsorted(lam,maxLamPerD)-1,0,numLam-1)

    numLam=np.max(maxBinPerD)+1
    lam=lam[0:numLam]  

    minDPerBin=minYPerXFromMaxXPerY(maxXPerY=maxBinPerD,minX=0)
    maxDPerBin=maxYPerXFromMinXPerY(minXPerY=minBinPerD,maxX=numLam-1)

    return(minDPerBin,maxDPerBin,minBinPerD,maxBinPerD)

def makePrecompute(hermiteDiag,lam,minBinPerD,maxBinPerD,minDPerBin,maxDPerBin,D,parms):
    numCores=parms['numCores']
    calD=int(np.ceil(parms['delta']*D))
    
    precomputeLen=(maxBinPerD-minBinPerD+1)

    b_precompute,precomputeNm=bufCreate([np.sum(precomputeLen)],dtype=[('lam','float64'),('eta','float64')])

    startPrecomputeRowPerD=np.cumsum([0]+precomputeLen[:-1].tolist())
    endPrecomputeRowPerD=np.cumsum(precomputeLen)-1

    precomputeLen=np.sum(precomputeLen)
    maxBin=len(lam)-1

    xargs=[]
    with ProcessPoolExecutor(numCores) as executor:
        for core in range(numCores):
            binRange=np.arange(core*int(np.ceil(maxBin/numCores)),min(maxBin,(core+1)*int(np.ceil(maxBin/numCores))))
            if len(binRange)==0:
                continue

            xargs+=[(binRange,D,minDPerBin,maxDPerBin,startPrecomputeRowPerD,precomputeNm,minBinPerD,lam,hermiteDiag,core)]

        ff=executor.map(makePrecomputeHelp,xargs)

    for d in range(calD):
        b_precompute[0][startPrecomputeRowPerD[d]:endPrecomputeRowPerD[d]+1]['eta']=np.sort(
            b_precompute[0][startPrecomputeRowPerD[d]:endPrecomputeRowPerD[d]+1]['eta'])

    return(bufClose(precomputeNm),startPrecomputeRowPerD,endPrecomputeRowPerD)

def makePrecomputeHelp(args):
    j=0;
    binRange=args[j];j+=1
    D=args[j];j+=1
    minDPerBin=args[j];j+=1
    maxDPerBin=args[j];j+=1
    startPrecomputeRowPerD=args[j];j+=1
    precomputeNm=args[j];j+=1
    minBinPerD=args[j];j+=1
    lam=args[j];j+=1
    hermiteDiag=args[j];j+=1
    core=args[j];j+=1
    
    b_precompute=bufLoad(precomputeNm)
    
    for Bin in binRange: 
        dList=np.arange(minDPerBin[Bin],maxDPerBin[Bin]+1).astype(int)
        fval=ordPvalCDF(D,lam[Bin],minDPerBin[Bin],maxDPerBin[Bin],hermiteDiag)
        loc=startPrecomputeRowPerD[minDPerBin[Bin]:maxDPerBin[Bin]+1]+(Bin-minBinPerD[minDPerBin[Bin]:maxDPerBin[Bin]+1])
        b_precompute[0]['lam'][loc]=lam[Bin]
        b_precompute[0]['eta'][loc]=fval
    
    b_precompute[1].close()
    
    return()

def ordPvalInvCDF(hermiteDiag,d,D,eta,eps,low,high,maxIters):
    lam=[low,high]
    F=[ordPvalCDF(D,lam[0],d,d,hermiteDiag)[0],ordPvalCDF(D,lam[1],d,d,hermiteDiag)[0]]
    badRange=True
    while badRange:
        if F[0]>eta:
            lam[0]/=2
            F[0]=ordPvalCDF(D,lam[0],d,d,hermiteDiag)[0]
        elif F[1]<eta:
            lam[1]=np.clip(lam[1]*2,0,1)
            F[1]=ordPvalCDF(D,lam[1],d,d,hermiteDiag)[0]
        else:
            badRange=False
    
    count=0
    
    while (F[1]-F[0])>eps:           
        count+=1
        newLam=(lam[1]+lam[0])/2
        newF=ordPvalCDF(D,newLam,d,d,hermiteDiag)[0]
        
        if newF<eta:
            lam[0]=newLam
            F[0]=newF
        else:
            lam[1]=newLam
            F[1]=newF
            
        #print('count {} newLam {} newF {} lam0 {} lam1 {} F0 {} F1 {}'.format(count,newLam,newF,lam[0],lam[1],F[0],F[1]),flush=True)
            
        if count>maxIters:
            print('convergence d {}, eta {}, iteration limit reached {}'.format(d,eta,maxIters),flush=True)
            break

    return(lam)

def ordPvalCDF(D,lam,minK,maxK,hermiteDiag):
    if lam==0:
        return(np.array([0]*(maxK-minK+1)))
    if lam==1:
        return(np.array([1]*(maxK-minK+1)))
    
    if np.sum(hermiteDiag)==0:
        kvec=np.arange(minK,maxK+1)+1
        ans=beta.cdf(lam,kvec,D+1-kvec)
    else:
        z=-norm.ppf(lam/2)
        lnBeta=np.log(4)+2*norm.logpdf(z)+np.log(np.polynomial.hermite_e.hermeval2d(z,z,np.diag(hermiteDiag)))-np.log(lam)-np.log(1-lam)
        gamma=np.exp(lnBeta-np.log(1-np.exp(lnBeta)))
        one=gammaln(D+1)-gammaln(np.arange(minK+1,D+1)+1)-gammaln(D-np.arange(minK+1,D+1)+1)
        two=np.sum(np.log(lam+gamma*np.arange(0,minK))) + np.cumsum(np.log(lam+gamma*np.arange(minK,D)))
        three=np.append(np.cumsum(np.log(1-lam+gamma*np.arange(0,D-minK-1)))[::-1],np.array([0]))
        four=np.sum(np.log(1+gamma*np.arange(0,D)))
        lnVec=one+two+three-four
        ans=np.sum(np.exp(lnVec[(maxK-minK+1):])) + np.cumsum(np.exp(lnVec[(maxK-minK)::-1]))[::-1]

    return(ans)

def minYPerXFromMaxXPerY(maxXPerY,minX):
    maxX=np.max(maxXPerY)
    uniqueMaxXPerY=np.sort(np.unique(maxXPerY))
    minYPer_uniqueMaxXPerY=np.searchsorted(maxXPerY,uniqueMaxXPerY,side='left')
    minYPerX=minYPer_uniqueMaxXPerY[np.searchsorted(uniqueMaxXPerY,np.arange(minX,maxX+1),side='left')].astype(int)
    
    return(minYPerX)

def maxYPerXFromMinXPerY(minXPerY,maxX):
    minX=np.min(minXPerY)
    uniqueMinXPerY=np.sort(np.unique(minXPerY))    
    maxYPer_uniqueMinXPerY=np.searchsorted(minXPerY,uniqueMinXPerY,side='right')-1    
    maxYPerX=maxYPer_uniqueMinXPerY[np.searchsorted(uniqueMinXPerY,np.arange(minX,maxX+1),side='right')-1].astype(int)
    
    return(maxYPerX)

def geomBins(numLam,minVal,maxVal):
    zeta=np.power(minVal/maxVal,1/numLam)
    bins=np.append(np.array([maxVal]),maxVal*np.power(zeta,np.arange(1,numLam+1)))[::-1]
    return(bins)

def eigen(vZ,numValsToKeep):        
    eigenString='''
    eigenDF <- function(covDF,numValsToKeep) {
        ans=eigen(covDF,symmetric=T)

        evals=ans$values
        cutOff=sort(evals,decreasing=T)[numValsToKeep]
        evals[evals<cutOff]=0

        newCovDF=ans$vectors%*%diag(evals)%*%t(ans$vectors)

        covDiag=diag(newCovDF)
        covOffDiag=newCovDF[lower.tri(newCovDF, diag = F)]

        covOldDiag=diag(covDF)
        covOldOffDiag=covDF[lower.tri(covDF, diag = F)]

        P=diag(1/sqrt(covDiag))

        L=P%*%ans$vectors%*%diag(sqrt(evals))
        diff=abs(c((L%*%t(L)-cov2cor(covDF))[lower.tri(newCovDF, diag = F)]))

        return(L)
    }    
    '''
    f_eigen=SignatureTranslatedAnonymousPackage(eigenString,'eigenDF')   
    L=np.array(f_eigen.eigenDF(vZ,ro.IntVector([numValsToKeep])))

    return(L)

##################################################################################################################3
######################################### monteCarlo #################################################################
##################################################################################################################33

def monteCarloHelp(args):
    j=0;
    LNm=args[j];j+=1
    mcReps=args[j];j+=1
    startPrecomputeRowPerD=args[j];j+=1
    endPrecomputeRowPerD=args[j];j+=1
    precomputeNm=args[j];j+=1
    D=args[j];j+=1
    parms=args[j];j+=1
    core=args[j];j+=1

    b_L=bufLoad(LNm)   

    ellNm='intermediate/ellRef/{}-{}-{}'.format(gethostname(),core,time.time())
    naiveNm='intermediate/naiveRef/{}-{}-{}'.format(gethostname(),core,time.time())

    Z=norm.rvs(size=[mcReps,b_L[0].shape[1]])@b_L[0].T

    ellStats,naiveStats=getStatistics((Z,startPrecomputeRowPerD,endPrecomputeRowPerD,precomputeNm,D,parms))

    np.savetxt(ellNm,np.sort(ellStats))
    np.savetxt(naiveNm,np.sort(naiveStats))
    print(core)
    b_L[1].close()
    
    return()

##################################################################################################################3
######################################### score #################################################################
##################################################################################################################33

def scoreHelp(args):
    j=0;
    Z=args[j];j+=1
    rows=args[j];j+=1
    startPrecomputeRowPerD=args[j];j+=1
    endPrecomputeRowPerD=args[j];j+=1
    precomputeNm=args[j];j+=1
    D=args[j];j+=1
    parms=args[j];j+=1
    ellNm=args[j];j+=1
    naiveNm=args[j];j+=1
    
    b_ell=bufLoad(ellNm)
    b_naive=bufLoad(naiveNm)
    
    ellStat,naiveStat=getStatistics((Z,startPrecomputeRowPerD,endPrecomputeRowPerD,precomputeNm,D,parms))
    
    b_ell[0][rows]=ellStat
    b_naive[0][rows]=naiveStat
    
    b_ell[1].close()
    b_naive[1].close()
    
    return()

def getStatistics(args):
    j=0;
    Z=args[j];j+=1
    startPrecomputeRowPerD=args[j];j+=1
    endPrecomputeRowPerD=args[j];j+=1
    precomputeNm=args[j];j+=1
    D=args[j];j+=1
    parms=args[j];j+=1

    b_precompute=bufLoad(precomputeNm)
    
    minEta=parms['minEta']
    maxEta=parms['maxEta']
    numCores=parms['numCores']
    calD=int(np.ceil(parms['delta']*D))

    Reps,_=Z.shape

    pi=2*norm.sf(-np.sort(-np.abs(Z))[:,0:calD])
    ellStat=np.zeros(pi.shape)
    for d in range(calD):
        t_precompute=b_precompute[0][startPrecomputeRowPerD[d]:endPrecomputeRowPerD[d]+1]
        order=np.argsort(pi[:,d])
        loc=np.searchsorted(t_precompute['lam'],pi[order,d],side='left').astype(int)
        ellStat[order,d]=t_precompute['eta'][loc]

    ellStat=np.clip(np.min(ellStat,axis=1),minEta,maxEta)
    naiveStat=np.min(pi,axis=1)*D
    
    b_precompute[1].close()

    return(ellStat,naiveStat)

##################################################################################################################3
######################################### overlap #################################################################
##################################################################################################################33

def distanceHelp(args):
    j=0;
    associatedTraits=args[j];j+=1
    distanceNm=args[j];j+=1
    snpLocs=args[j];j+=1
    eqtlPairs=args[j];j+=1
    
    b_distance=bufLoad(distanceNm)

    for i1,i2 in eqtlPairs:                
        nm1=set(associatedTraits['traitLabels'].iloc[snpLocs[i1]:snpLocs[i1+1]])
        nm2=set(associatedTraits['traitLabels'].iloc[snpLocs[i2]:snpLocs[i2+1]])
        val=len(nm1&nm2)/min(len(nm1),len(nm2))
        b_distance[0][i1,i2]=1-val
        b_distance[0][i2,i1]=1-val

    b_distance[1].close()
    
    return()
    
##################################################################################################################3
######################################### shared memory #################################################################
##################################################################################################################33

def bufCreate(shape,dtype='float64'):
    name=str(np.random.uniform())+gethostname()+str(os.getpid())
    buf = shared_memory.SharedMemory(create=True, name=name,size=np.prod(shape)*np.dtype(dtype).itemsize)

    arr=np.ndarray(shape, dtype=dtype, buffer=buf.buf)
    arr[:]=0

    return((arr,buf),(name,shape,dtype))

def bufLoad(nm):
    name,shape,dtype=nm
    buf = shared_memory.SharedMemory(create=False, name=name)
    arr=np.ndarray(shape, dtype=dtype, buffer=buf.buf)

    return((arr,buf))

def bufClose(nm):
    name,shape,dtype=nm

    buf=shared_memory.SharedMemory(create=False, name=name)
    arr=np.ndarray(shape, dtype=dtype, buffer=buf.buf).copy()

    buf.close()
    buf.unlink()
    
    return(arr)

##################################################################################################################3
######################################### main #################################################################
##################################################################################################################33

def main(parms):
    parms['numCores']=int(float(parms['numCores']))
    parms['delta']=float(parms['delta'])
    parms['minEta']=float(parms['minEta'])
    parms['maxEta']=float(parms['maxEta'])
    parms['maxIters']=int(float(parms['maxIters']))
    parms['numVzEigenValsToKeep']=int(float(parms['numVzEigenValsToKeep']))
    parms['numHermite']=int(float(parms['numHermite']))
    parms['numLam']=int(float(parms['numLam']))
    parms['mcReps']=int(float(parms['mcReps']))
    parms['globalPvalThresh']=float(parms['globalPvalThresh'])
    parms['totalSnps']=int(float(parms['totalSnps']))
    parms['FDR']=float(parms['FDR'])
    parms['clusterDistance']=float(parms['clusterDistance'])
    
    parms['numCores']=cpu_count() if parms['numCores']==0 else parms['numCores']

    os.chdir(parms['folder'])
    subprocess.call(['mkdir','-p','intermediate','intermediate/ellRef','intermediate/naiveRef','output'])
    
    Z = open('input/Z', "r")
    line = Z.readline().split(',')
    Z.close()

    parms['numCandidates']=len(line)

    if parms['compute']=='True':
        compute(parms)
    
    if parms['monteCarlo']=='True':
        monteCarlo(parms)
    
    if parms['monteCarloCombine']=='True':
        monteCarloCombine(parms)
    
    if parms['score']=='True':
        score(parms)
    
    if parms['associatedTraits']=='True':
        associatedTraits(parms)
        
    if parms['cluster']=='True':
        eqtlDistance(parms)
        cluster(parms)
        
    if parms['plotEllPvals']=='True':
        plotEllPvals(parms)
        
    if parms['plotEllStats']=='True':
        plotEllStats(parms)
        
    if parms['plotOverlap']=='True':
        plotOverlap(parms)
        
    return()


configFile = open(sys.argv[1], "r")
myList = configFile.readlines()
configFile.close()
config={x[0]:x[1] for x in [x.replace('\n','').replace(' ','').replace('\t','').split(',') for x in myList]}

main(config)