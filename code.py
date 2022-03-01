# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np, scipy.stats as st
import statistics
import math
import scipy.stats as stats
import pylab
import seaborn as sns

#Pre-processing Stage

#taking data
players_data=pd.read_csv("players.csv")
players_data=players_data.dropna()#removes missing/misleading players
players_data=players_data.loc[players_data['yob'].isin(list(range(1925,1993)))] #13 to 80 years old

#taking data
data=pd.read_csv("ratings_2015.csv")
data=data.groupby('fide_id')['rating_standard'].mean()
data=data.reset_index()
data=data.dropna() #removes missing/misleading values

#intersection of data and players 
fide_ids1=players_data['fide_id'].values
fide_ids2=data['fide_id'].values
intersect_ids=list(set.intersection(set(fide_ids1), set(fide_ids2)))

players_data=players_data.loc[players_data["fide_id"].isin(intersect_ids)].reset_index().drop(columns=['index'])
data=data.loc[data['fide_id'].isin(intersect_ids)].reset_index().drop(columns=['index'])
players_data['score']=data['rating_standard'] #add ratings to players_data dataframe

#Analyze
def find_means_vars(datas):
    data=list()
    data.append(datas.loc[datas["yob"].isin(list(range(1982,1993)))])#23-13
    data[0]=data[0].groupby("gender")["score"].mean().reset_index()
    
    data.append(datas.loc[datas["yob"].isin(list(range(1972,1982)))])#33-24
    data[1]=data[1].groupby("gender")["score"].mean().reset_index()
    
    data.append(datas.loc[datas["yob"].isin(list(range(1962,1972)))])#43-34
    data[2]=data[2].groupby("gender")["score"].mean().reset_index()
    
    data.append(datas.loc[datas["yob"].isin(list(range(1952,1962)))])#53-44
    data[3]=data[3].groupby("gender")["score"].mean().reset_index()
    
    data.append(datas.loc[datas["yob"].isin(list(range(1925,1952)))])#54-80
    data[4]=data[4].groupby("gender")["score"].mean().reset_index()
    return data

def mean_score(datas):
    for i in range(1,5):
        datas[0] =datas[0].append(datas[i])
    return datas[0].groupby("gender")["score"].mean().reset_index()

def vars_score(datas):
    datas1=list()
    datas1.append(datas[0])
    for i in range(1,5):
        datas1[0] =datas1[0].append(datas[i])
    return datas1[0].groupby("gender")["score"].var().reset_index()


male_Mean=[]
female_Mean=[]
men_means=[[],[],[]]
men_vars=[[],[],[]]
women_means=[[],[],[]]
women_vars=[[],[],[]]
#FM
FM_datas=players_data.loc[players_data['title'].isin(['FM'])]
FM_means=find_means_vars(FM_datas)
FM_mean=mean_score(FM_means)

male_Mean.append(FM_mean['score'][1])
female_Mean.append(FM_mean['score'][0])

#means
men_means[0]=list(FM_means[0].loc[FM_means[0]["gender"].isin(['M'])]['score'])
women_means[0]=list(FM_means[0].loc[FM_means[0]["gender"].isin(['F'])]['score']) 
#vars
FM_vars=vars_score(FM_means)

#IM
IM_datas=players_data.loc[players_data['title'].isin(['IM'])]
IM_means=find_means_vars(IM_datas)
IM_mean=mean_score(IM_means)
male_Mean.append(IM_mean['score'][1])
female_Mean.append(IM_mean['score'][0])

#means
men_means[1]=list(IM_means[0].loc[IM_means[0]["gender"].isin(['M'])]['score'])
women_means[1]=list(IM_means[0].loc[IM_means[0]["gender"].isin(['F'])]['score']) 
women_means[1].append(female_Mean[1])  #for filling

#vars
IM_vars=vars_score(IM_means)

#GM
GM_datas=players_data.loc[players_data['title'].isin(['GM'])]
GM_means=find_means_vars(GM_datas)
GM_mean=mean_score(GM_means)
male_Mean.append(GM_mean['score'][1])
female_Mean.append(GM_mean['score'][0])

#means
men_means[2]=list(GM_means[0].loc[GM_means[0]["gender"].isin(['M'])]['score'])
women_means[2]=list(GM_means[0].loc[GM_means[0]["gender"].isin(['F'])]['score'])

#vars
GM_vars=vars_score(GM_means)

#For Avarage rating graph
def compare():
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(3)
    bar_width = 0.2
    opacity = 0.8
    
    plt.bar(index, male_Mean, bar_width,
    alpha=opacity,
    color='b',
    label='male')
    
    plt.bar(index + bar_width, female_Mean, bar_width,
    alpha=opacity,
    color='r',
    label='female')
    
    plt.xlabel('Chess Title')
    plt.ylabel('Avarage Chess Score')
    plt.title('Avarage Chess Score of Genders and Titles')
    plt.xticks(index + bar_width, ('FM', 'IM', 'GM'))
    plt.legend()
    
    plt.tight_layout()
    plt.show()

#Avarage rating graph
compare()

#T-Interval
t_Value=2.776
stds=[ [ [],[] ],[ [],[] ],[ [],[] ] ] 
errors=[ [ [],[] ],[ [],[] ],[ [],[] ] ] 
t_intervals=[ [ [],[] ],[ [],[] ],[ [],[] ] ] 
for i in range(3):
    stds[i][0]=statistics.stdev(men_means[i])
    stds[i][1]=statistics.stdev(women_means[i])
    errors[i][0]=t_Value*stds[i][0]/ math.sqrt(5)
    errors[i][1]=t_Value*stds[i][1]/ math.sqrt(5)
    t_intervals[i][0]=st.t.interval(0.95,4,loc=male_Mean[i],scale=st.sem(men_means[i]))
    t_intervals[i][1]=st.t.interval(0.95,4,loc=female_Mean[i],scale=st.sem(women_means[i]))

    
    
    
#normality tests
#uantile plot
stats.probplot(data['rating_standard'].values,dist="norm",plot=pylab) 
pylab.show()

#histogram
ax=sns.distplot(data['rating_standard'].values)
    
#correalation scores
#FM
FM_Male_data=list(FM_means[0].loc[FM_means[0]["gender"].isin(['M'])].reset_index()['score'].values)
FM_Female_data=list(FM_means[0].loc[FM_means[0]["gender"].isin(['F'])].reset_index()['score'].values)
FM_corrr=stats.pearsonr(FM_Male_data,FM_Female_data)


#IM
IM_Male_data=list(IM_means[0].loc[IM_means[0]["gender"].isin(['M'])].reset_index()['score'].values)
IM_Female_data=list(IM_means[0].loc[IM_means[0]["gender"].isin(['F'])].reset_index()['score'].values)

IM_Female_data.append(IM_mean['score'][0])
IM_corr=stats.pearsonr(IM_Male_data,IM_Female_data)  
    
#GM
GM_Male_data=list(GM_means[0].loc[GM_means[0]["gender"].isin(['M'])].reset_index()['score'].values)
GM_Female_data=list(GM_means[0].loc[GM_means[0]["gender"].isin(['F'])].reset_index()['score'].values)
GM_corr=stats.pearsonr(GM_Male_data,GM_Female_data)
    