#!/usr/bin/env python
# coding: utf-8

# Maor's coding trip

# 

# In[ ]:


print("Hi man")


# In[ ]:


a = 5.6
b= "yaaah"
print(type(a))
get_ipython().run_line_magic('whos', '')


# In[ ]:


a,b,c,d,f = 1,2,2.5,"Hi",4
get_ipython().run_line_magic('whos', '')
del a


# In[ ]:


sumbandc= b+c
get_ipython().run_line_magic('whos', '')
powbA
ndc = b**c


# In[ ]:


c= True
b= False
6==7
get_ipython().run_line_magic('whos', '')
b or c


# In[ ]:


isinstance("HI",str)
isinstance(3+2j,(int,float,complex))
pow(5,2,7)
t = input("pick anything :")
type(t)
int(t)


# In[ ]:


a = int(t)
type(a)
p = float(input("enter a real number :"))


# In[ ]:


type(p)
get_ipython().run_line_magic('whos', '')
del b,c


# In[ ]:


a = input("a :")
b = input("b :")
if a>b:
    print(a)
    print("yaaah")
print("nop")        
9>10    


# In[ ]:


a = float(input("a :"))
b = float(input("b :"))
if a>b:
    print(a)
    print("yaaah")
elif a==b:
    print("samsam")
else:    
    print("nop")     


# In[ ]:


"""
sfrdhtfyikrjdf
"""
n = int(input("pick :"))
i=1
while i<n:
    print(i**2)
    print("this is iteration number", i)
    i += 1
print("done")


# In[ ]:


n = int(input("pick :"))
while True:
    if n%17 == 0:
        print("break")
        break
    else: 
        n += 1
        print("I'm in")
print("done")
n = int(input("pick :"))
while True:
    if n%17 != 0:
        print("contin.")
        n +=1
        continue
    else: 
        print("I'm in")
        break
print("done")
6%17    


# In[ ]:


a = range(10)
print(a)
L = []
for i in range(0,10,2):
    print(i)
    L.append(i**2)
print(L)
L.remove(0)


# In[ ]:


L[0] =1
L


# In[ ]:


L = [3,5,3,6,0,2,-2,5,-10]
#for i in range(len(L)):
  #  m = L[i]
 #   c = i
 #   for j in range(i,len(L) ):
       # if L[j]<m:
         #   m = L[j]
 #   L[i]=m
 #   i +=1
print(L)
def YAH(a,b):
    c = a**b%b
    return(c)
c = print(YAH(2,3))
print(YAH(2,3))
def H():
    a = "suck"
    b = "my"
    c = "dick"
    return a,b,c
c,d,f = H()
print(c,d,f)


# In[ ]:


def f(*s):
    sum = 0
    for i in range(len(s)):
        sum += s[i] 
    return(sum)
print(f(5,3,4,7,8))        


# In[ ]:


D = {"color": "red", "number": 44}
for i in D:
    print(i,D[i])
def p(**c):
    for i in c:
        print("variable name is :",i, "baiable value is :", c[i])
p(a = "fuucj", b = 6, c = "yakitoru")


# MY DIRECTORY

# In[ ]:


import sys
sys.path.append('C:/Users/maorb/PYfunctions')
import fun as myfuns
myfuns.countUniqueWords(("boom","boom"),"boom")
from fun import countUniqueWords
countUniqueWords(("boom","boom"),"boom")


# In[ ]:


pric= 12
print("th pric is" + str(pric) )


# In[ ]:


print("""fuck thic ssdgjlkdzg
        sa dfhgfjh
        sad=
""")


# In[ ]:


L = "Hi my mood is good my dude"
L[::-1]
print("HI I'm \"Maor\" boyy")


# In[ ]:


L = [a**2 for a in range(10)]
print(L)
S = {a**2 for a in range(2,20,2)}
print(S)
print(set(sorted(S)))



# In[ ]:


def getGradesOfStudents():
    D = {}
    while True:
        studentId = input("Student ID is: ")
        studentGrades = input("enter marks seperated by coma :")
        moreStudents = input('enter "no" if you want to quit: ')
        if studentId in D:
            print(studentId, "is already in")
        else:
            D[studentId] = studentGrades.split(",")
        if moreStudents.lower() == "no":
            return D
def averageMarks(D):
    avgMark = {}
    for i in D:
        L = D[i]
        s = 0
        for marks in L:
            s += int(marks)
            avgMark[i] = s/len(L)
    return avgMark
dat = getGradesOfStudents()
print(dat)
    
        
    


# In[ ]:


print(dat)
avg = averageMarks(dat)
for i in avg:
    print("stud", i, "got marks", avg[i])


# Numpy

# In[ ]:


import numpy as np
a = np.array([[1,2,3],[3,5,6], [2,3,4]], dtype="i")
b = np.array((1,2,3,4,6,7), dtype="f")
c = np.array([[[1,2,3],[2,0,0]],[[0,2,0],[0,0,2]],[[-1,-2,-3],[-2,0,0]],[[0,-2,0],[0,0,-2]]])
print(a[1,2])
print(a)
print(c)
a.dtype
a.ndim
print(c.ndim)


# In[ ]:


print(c.shape)
print(c.shape[0])
print(c.shape[1])
print(c.size)
print(c)


# In[ ]:


import numpy as np
g = np.arange(100)
gPo = np.arange(20,100,5)
r = np.random.permutation(np.arange(20,100,5))

print(r.reshape(4,4))

type(r)


# In[ ]:


list(r).ndim


# In[ ]:


import matplotlib.pyplot as plt
#plt.hist(u,bins= 100)
plt.hist(n,bins= 100)


# In[ ]:


l = np.random.rand(2,2,2,2)
print(l)
l.ndim
s = np.arange(100).reshape(4,5,5)
s.shape
z = np.zeros(100)
np.ones(100)


# In[ ]:


z[0] = 6
z1 = z.copy()
print(z1)
z1[0] = 1
z
z1
list(range(10)[::2])
t = np.arange(12).reshape(4,3)
idl = np.argwhere(t==5)[0][0]
print(t)
print(idl)
t[1,:]
t[:,1]
t.T
t.sort(axis=0)
t.sort(axis=1)


# In[ ]:


import numpy.linalg as la
la.inv(np.array([1,2,3,4]).reshape(2,2))


# In[ ]:


A = np.arange(100)
b = A[[2,3,4]]
b[0] = 5
b
A # didnt change
B = A[A<40]
BC = A[(A<41) & (A>30)] #(), &, |, ~ for arrays, 'and', 'or', 'not' for single objects
BC


# In[ ]:


A = np.round(10*np.random.rand(2,3))
B = np.random.randn(2,2)
c = np.hstack((B,A))
c.sort()
print(c)
#cd = np.vstack((A,B)) #error
A =np.random.permutation(np.arange(10))
A.sort() #np.sort(A)
A = A[::-1]


# In[ ]:


B = np.random.randn(1000)
get_ipython().run_line_magic('timeit', 'sum(B)')
get_ipython().run_line_magic('timeit', 'np.sum(B) #B.sum')


# Pandas
# 

# In[ ]:


import pandas as pd
print(pd.__version__)
A = pd.Series([2,3,4,6], index=['a','b','c','d'])
A.values
A.index[0]
A['a']
A['a':'d']
grads = {'A':'A+', 'B':'B+', 'C':'C+', 'D':'B+'}
grad = pd.Series(grads)
marks = {'A':95, 'B':85, 'C':66, 'D':87}
mark = pd.Series(marks)
grad.values
grad[0:2]
D = pd.DataFrame({'Marks':mark, 'Grades': grad})
D.T
print(D.values)
D.values[2,1]
D.columns
D['scalMarks'] = D['Marks']/90
del D['scalMarks']
D


# In[ ]:


del D['Marks']

D


# In[ ]:


D


# In[ ]:


A = pd.DataFrame([{'a':1,'b':2}, {'b':5,'c':7}])
A
A.fillna(0)
A.dropna


# In[ ]:


A = pd.Series(['a','v','c'], index= [11,2,3])
A[2]
A.index[0]
print(A[1:3])
A.loc[11:2]
A.iloc[1:3]
D
D.iloc[::-1,:]


# Data analysis

# In[ ]:


get_ipython().system('conda install scikit-learn')

from sklearn.impute import SimpleImputer
df = pd.read_csv('C:/Users/maorb/CSVs/YAH.csv')
df
df.head(10)
df.drop(['Consent Confirmation','Unnamed: 41'], axis=1, inplace= True)
df
#df.rename(columns= {'Gender': 'Matysuckmydick','Age':'FUmaty'}, inplace=True)
#pd.to_datetime #change date format to panda friendly
df.describe()
df.info()
#df= df.fillna('Not filled')
#df1 = df.groupby('FUmaty')[['Empathy Sum', 'Sum conflict 1-3', 'Hope-Inducing/Neutral Articles']].sum().r
#df2 = df.groupby(['FUmaty','Sum Conflict 1+3'])[['Empathy Sum', 'Sum conflict 1-3', 'Hope-Inducing/Neutral Articles']].sum()
df3 = df[df['Sum Conflict 1+3']>9]
dfs = df.groupby(['High/Low Empathy Scale', 'Hope-Inducing/Neutral Articles'])['Sum conflict 1-3'].apply(lambda x: x).reset_index
print(dfs)


# In[ ]:


#!conda install scikit-learn

#from sklearn.impute import SimpleImputer
df = pd.read_csv('C:/Users/maorb/CSVs/YAH.csv')
df
df.head(10)
df.drop(['Consent Confirmation','Unnamed: 41'], axis=1, inplace= True)
df
#df.rename(columns= {'Gender': 'Matysuckmydick','Age':'FUmaty'}, inplace=True)
#pd.to_datetime #change date format to panda friendly
df.describe()
df.info()
#df= df.fillna('Not filled')
#df1 = df.groupby('FUmaty')[['Empathy Sum', 'Sum conflict 1-3', 'Hope-Inducing/Neutral Articles']].sum().r
#df2 = df.groupby(['FUmaty','Sum Conflict 1+3'])[['Empathy Sum', 'Sum conflict 1-3', 'Hope-Inducing/Neutral Articles']].sum()
df3 = df[df['Sum Conflict 1+3']>9]
dfs = df.groupby(['High/Low Empathy Scale', 'Hope-Inducing/Neutral Articles'])['Sum conflict 1-3'].apply(lambda x: x).reset_index
print(dfs)


# Anova for my Experiment

# In[ ]:


### print(df2)
#dfs = df.groupby(['Sum conflict 1-3'])['Sum conflict 1-3','High/Low Empathy Scale', 'Hope-Inducing/Neutral Articles'].apply(lambda x:x)
#dfs = df.groupby("race").groups
#df =df.sort_values(by=['Sum conflict 1-3','High/Low Empathy Scale', 'Hope-Inducing/Neutral Articles'])
dp = pd.DataFrame({"sumc":sumc,"HL":HL,"hopn":hopn})
sumc = df.iloc[:,37]
HL = df.iloc[:,32]
hopn = df.iloc[:,39]
model = ols('sumc ~ HL * hopn',                  # Model formula
            data = dp).fit()
                
anova_result = sm.stats.anova_lm(model, typ=2)
print (anova_result)
#FU = dfs[dfs['High/Low Empathy Scale'] == "LOW"]
#FU['Sum conflict 1-3']
#dfs.iloc[:2,1]


# In[ ]:


dfs.head()
df.columns.get_loc('Sum conflict 1-3')
#df.head()


# In[ ]:


#df.info()

type(HL)


# In[ ]:


### print(df2)
dfs = df.groupby(['Sum conflict 1-3'],sort =False)[['Sum conflict 1-3','High/Low Empathy Scale', 'Hope-Inducing/Neutral Articles']].apply(lambda x:x)
#dfs = df.groupby(['Sum conflict 1-3'], as_index=False)['Sum conflict 1-3','High/Low Empathy Scale', 'Hope-Inducing/Neutral Articles']
sumc = dfs.iloc[:,0]
HL = dfs.iloc[:,1]
hopn = dfs.iloc[:,2]

model = ols('sumc ~ HL * hopn',                 # Model formula
            data = dfs).fit()
                
anova_result = sm.stats.anova_lm(model, typ=2)
print(anova_result)
FU = dfs[dfs['High/Low Empathy Scale'] == "LOW"]
FU['Sum conflict 1-3']
dfs.iloc[:2,1]
dfs
dfs.head()
#df.head()


# Plotting

# In[ ]:


import matplotlib.pyplot as plt
a = np.linspace(0,10,1000)
b = np.sin(a)
plt.plot(a,b)
plt.scatter(a[:30],b[:30])


# In[ ]:


plt.scatter(a[::10],b[::10], color = 'purple')


# In[ ]:


plt.plot(a,b,color='b')
plt.plot(a,np.cos(a),color='purple')


# Analysis

# In[ ]:


imputer = SimpleImputer(strategy='constant') #solving missing values
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
uniqueFUM = df['FUmaty'].unique() #finding different values
len(uniqueFUM)


# instructor's coding

# In[ ]:


for idx in range(0,len(countries)):    
    C = df3[df3['Country']==countries[idx]].reset_index()        
    plt.scatter(np.arange(0,len(C)),C['Confirmed'],color='blue',label='Confirmed')
    plt.scatter(np.arange(0,len(C)),C['Recovered'],color='green',label='Recovered')
    plt.scatter(np.arange(0,len(C)),C['Deaths'],color='red',label='Deaths')
    plt.title(countries[idx])
    plt.xlabel('Days since the first suspect')
    plt.ylabel('Number of cases')
    plt.legend()
    plt.show()    
df4 = df3.groupby(['Date'])[['Date','Confirmed','Deaths','Recovered']].sum().reset_index()
C = df4
plt.scatter(np.arange(0,len(C)),C['Confirmed'],color='blue',label='Confirmed')
plt.scatter(np.arange(0,len(C)),C['Recovered'],color='green',label='Recovered')
plt.scatter(np.arange(0,len(C)),C['Deaths'],color='red',label='Deaths')
plt.title('World')
plt.xlabel('Days since the first suspect')
plt.ylabel('Number of cases')
plt.legend()
plt.show()


# In[ ]:


b = np.sort(np.random.random(10))
fig, a = plt.subplots(figsize = (12,6))
a.plot(b,b**2,linestyle= 'solid')
a.plot(b,b**3,linestyle= 'solid')
a.plot(b,b**4,linestyle= 'solid')
a.plot(b,b**5,linestyle= 'solid')


# Statistics

# In[ ]:


import scipy.stats as stats
import math
from scipy.stats import chi2_contingency
df = pd.read_csv(('C:/Users/maorb/CSVs/titanic.csv'))
df
df1 = df.groupby('Pclass')[['Pclass','Survived']].apply(lambda x: x)
dsum = df.groupby('Pclass')[['Pclass','Survived']].sum()
df1


# In[ ]:


print(dsum)
dsum.iloc[:,1]
f1 =sum(dsum.iloc[:,1])
f2 =sum(dsum.iloc[:,0])
M = dsum.iloc[:,1]/dsum.iloc[:,0]
M
print(dsum['Pclass'].sum())
print(dsum['Survived'].sum())
popMean = dsum['Survived'].sum()/dsum['Pclass'].sum()
print(popMean)
dsum['Mean'].values[[0,2]]
(0.629630+0.236413+0.080788)/3


# In[ ]:


dsum['Mean']= M
dsum
contingency_table = pd.crosstab(df['Pclass'],df['Survived'])
print(contingency_table)
class3 = df[df['Pclass'] == 3].shape[0]
class3
dSumN = df.groupby('Pclass')[['Pclass','Survived']].sum()
dSumN
yo = chi2, p, dof, expected = chi2_contingency(contingency_table)
yo
#df['mean'] = dsum[:0]/dsum[:1]


# In[ ]:


df3 = df.groupby('Pclass').apply(lambda x: x).reset_index()
df3


# T.test one pop.

# In[ ]:


p1 = stats.poisson.rvs(loc = 18, mu = 35, size = 150000)
p2 = stats.poisson.rvs(loc = 18, mu = 10, size = 100000)
pAll = np.concatenate((p1,p2))
mAP1 = stats.poisson.rvs(loc = 18, mu = 30, size = 30)
mAP2 = stats.poisson.rvs(loc = 18, mu = 10, size = 30)
mALL = np.concatenate((mAP1,mAP2))
stats.ttest_1samp(a =mALL, popmean = pAll.mean())
stats.t.ppf(q = 0.025, df = 59) # T statistc
stats.t.cdf(x= -2.9582, df = 59)*2 #Pvalue'multiply by 2 tail'
sigma = mALL.std()/math.sqrt(60)
stats.t.interval(0.95,            #confi. lvl, DF, Sam. M,SD estimate
                df =59,
                loc = mALL.mean(),
                scale= sigma)


# T.test two pop. comparison

# In[ ]:


np.random.seed(12)
wa1 = stats.poisson.rvs(loc = 18, mu = 33, size= 30)
wa2 = stats.poisson.rvs(loc = 18, mu = 13, size= 20)
wAll = np.hstack((wa1,wa2))
wAll.mean()
stats.ttest_ind(a = mALL, b = wAll, equal_var=False) 


# Paired T.test

# In[ ]:


np.random.seed(11)
b = stats.norm.rvs(scale = 30, loc = 250, size=100)
a = b + stats.norm.rvs(scale = 5, loc = -1.25, size=100)
w_df = pd.DataFrame({'weight_before':b,'weight_after':a,'wc':a-b})
w_df.describe()
stats.ttest_rel(a = b, b = a)


# Anova

# In[ ]:


np.random.seed(12)

races =   ["asian","black","hispanic","other","white"]

# Generate random data
voter_race = np.random.choice(a= races,
                              p = [0.05, 0.15 ,0.25, 0.05, 0.5],
                              size=1000)

voter_age = stats.poisson.rvs(loc=18,
                              mu=30,
                              size=1000)
# Group age data by race
voter_frame = pd.DataFrame({"race":voter_race,"age":voter_age})

groups = voter_frame.groupby("race").groups

# Etract individual groups
asian = voter_age[groups["asian"]]
black = voter_age[groups["black"]]
hispanic = voter_age[groups["hispanic"]]
other = voter_age[groups["other"]]
white = voter_age[groups["white"]]
# Perform the ANOVA
stats.f_oneway(asian, black, hispanic, other, white)


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('age ~ race',                 # Model formula
            data = voter_frame).fit()
                
anova_result = sm.stats.anova_lm(model, typ=2)
print (anova_result)


# In[ ]:


np.random.seed(12)

# Generate random data
voter_race = np.random.choice(a= races,
                              p = [0.05, 0.15 ,0.25, 0.05, 0.5],
                              size=1000)

# Use a different distribution for white ages
white_ages = stats.poisson.rvs(loc=18, 
                              mu=32,
                              size=1000)

voter_age = stats.poisson.rvs(loc=18,
                              mu=30,
                              size=1000)

voter_age = np.where(voter_race=="white", white_ages, voter_age)

# Group age data by race
voter_frame = pd.DataFrame({"race":voter_race,"age":voter_age})
groups = voter_frame.groupby("race").groups    

# Extract individual groups
asian = voter_age[groups["asian"]]
black = voter_age[groups["black"]]
hispanic = voter_age[groups["hispanic"]]
other = voter_age[groups["other"]]
white = voter_age[groups["white"]]

# Perform the ANOVA
stats.f_oneway(asian, black, hispanic, other, white)


# In[ ]:


voter_frame.groupby("race").groups['asian']


# In[ ]:


model = ols('age ~ race',                 # Model formula
            data = voter_frame).fit()
                
anova_result = sm.stats.anova_lm(model, typ=2)
print (anova_result)


# In[ ]:


"""post hoc after anova to determine which of them is 
   significantly different, multiple T
"""
race_pairs = []
for race1 in range(4):
    for race2  in range(race1+1,5):
        race_pairs.append((races[race1], races[race2]))

# Conduct t-test on each pair
for race1, race2 in race_pairs: 
    print(race1, race2)
    print(stats.ttest_ind(voter_age[groups[race1]], 
                          voter_age[groups[race2]]))


# In[ ]:


import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
"""post hoc after anova to determine which of them is 
   significantly different, Tukey
"""
tukey = pairwise_tukeyhsd(endog=voter_age,     # Data
                          groups=voter_race,   # Groups
                          alpha=0.05)          # Significance level

tukey.plot_simultaneous()    # Plot group confidence intervals
plt.vlines(x=49.57,ymin=-0.5,ymax=4.5, color="red")

tukey.summary()              # See test summary


# Chi Square

# In[ ]:


national = pd.DataFrame(["white"]*100000 + ["hispanic"]*60000 +\
                        ["black"]*50000 + ["asian"]*15000 + ["other"]*35000)
           

minnesota = pd.DataFrame(["white"]*600 + ["hispanic"]*300 + \
                         ["black"]*250 +["asian"]*75 + ["other"]*150)

national_table = pd.crosstab(index=national[0], columns="count")
minnesota_table = pd.crosstab(index=minnesota[0], columns="count")

print( "National")
print(national_table)
print(" ")
print( "Minnesota")
print(minnesota_table)


# In[ ]:


observed = minnesota_table

national_ratios = national_table/len(national)  # Get population ratios

expected = national_ratios * len(minnesota)   # Get expected counts

chi_squared_stat = (((observed-expected)**2)/expected).sum() #actually for each row

print(chi_squared_stat)


# In[ ]:


national_ratios
expected - observed # each row


# In[ ]:





# In[ ]:


crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 4)   # Df = number of variable categories - 1

print("Critical value")
print(crit)

p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=4)
print("P value")
print(p_value)


# In[ ]:


stats.chisquare(f_obs= observed,   # Array of observed counts
                f_exp= expected)   # Array of expected counts


# In[ ]:


""" chi of independence to find if variables are actually
independent!!
"""
np.random.seed(10)

# Sample data randomly at fixed probabilities
voter_race = np.random.choice(a= ["asian","black","hispanic","other","white"],
                              p = [0.05, 0.15 ,0.25, 0.05, 0.5],
                              size=1000)

# Sample data randomly at fixed probabilities
voter_party = np.random.choice(a= ["democrat","independent","republican"],
                              p = [0.4, 0.2, 0.4],
                              size=1000)

voters = pd.DataFrame({"race":voter_race, 
                       "party":voter_party})

voter_tab = pd.crosstab(voters.race, voters.party, margins = True)

voter_tab.columns = ["democrat","independent","republican","row_totals"]

voter_tab.index = ["asian","black","hispanic","other","white","col_totals"]

observed = voter_tab.iloc[0:5,0:3]   # Get table without totals for later use
voter_tab


# In[ ]:


voters


# In[ ]:


#expected =  np.outer(voter_tab.iloc[:,3][0:5],
                   #  voter_tab.loc["col_totals"][0:3]) / 1000
expected =  np.outer(voter_tab["row_totals"][0:5], #outer layer of a 2 vectors
                     voter_tab.loc["col_totals"][0:3]) / 1000
expected = pd.DataFrame(expected)

expected.columns = ["democrat","independent","republican"]
expected.index = ["asian","black","hispanic","other","white"]

expected


# In[ ]:


expected
#np.outer(voter_tab.loc["col_totals"][0:3], #first the number of rows
                    # voter_tab["row_totals"][0:5]) / 1000


# In[ ]:


hi_squared_stat = (((observed-expected)**2)/expected).sum().sum()
# twice sum cuz 2d table
print(chi_squared_stat)


# In[ ]:


crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 8)   # (5-1)*(3-1)

print("Critical value")
print(crit)

p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=8)
print("P value")
print(p_value)


# In[ ]:


stats.chi2_contingency(observed= observed) #independent


# Linear Regression

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
matplotlib.style.use('ggplot')


# In[ ]:


mtcars = pd.read_csv("C:/Users/maorb/CSVs/mtcars.csv")

mtcars.plot(kind="scatter",
           x="wt",
           y="mpg",
           figsize=(9,9),
           color="black");


# In[ ]:


from sklearn import linear_model
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
regression_model = linear_model.LinearRegression()

# Train the model using the mtcars data
lm1 = regression_model.fit(X = pd.DataFrame(mtcars["wt"]), 
                     y = mtcars["mpg"])
model = sm.OLS(mtcars["mpg"], pd.DataFrame(mtcars["wt"])).fit()
# Check trained model y-intercept
print(regression_model.intercept_)

# Check trained model coefficients
print(regression_model.coef_)
model.summary()


# In[ ]:





# In[ ]:


regression_model.score(X = pd.DataFrame(mtcars["wt"]),#good R sq
                     y = mtcars["mpg"])


# In[ ]:


trainPrediction = regression_model.predict(X = pd.DataFrame(mtcars["wt"]))
                     
residuals = mtcars["mpg"]-trainPrediction
residuals.describe()


# In[ ]:


SSresiduals = (residuals**2).sum()
SStotal = ((mtcars["mpg"]-mtcars["mpg"].mean())**2).sum()
#R sq
1 - SSresiduals/SStotal


# In[ ]:


mtcars.plot(kind="scatter",
           x="wt",
           y="mpg",
           figsize=(9,9),
           color="black",
           xlim = (0,7))

# Plot regression line
plt.plot(mtcars["wt"],      # Explanitory variable
         trainPrediction,  # Predicted values
         color="blue");


# In[ ]:


"""mtcars_subset = mtcars[["mpg","wt"]]
mtcars_subset
super_car = {"mpg":50,"wt":10}
mtcars_subset.loc[len(mtcars_subset)] = super_car
mtcars_subset.rename(index={32: 'sup'}, inplace=True) #Renaming
mtcars_subset"""


# In[ ]:


mtcars_subset = mtcars[["mpg","wt"]]

super_car = pd.DataFrame({"mpg":50,"wt":10}, index=["super"])

new_cars = pd.concat([mtcars_subset, super_car])

# Initialize model
regression_model = linear_model.LinearRegression()

# Train the model using the new_cars data
regression_model.fit(X = pd.DataFrame(new_cars["wt"]), 
                     y = new_cars["mpg"])

train_prediction2 = regression_model.predict(X = pd.DataFrame(new_cars["wt"]))

# Plot the new model
new_cars.plot(kind="scatter",
           x="wt",
           y="mpg",
           figsize=(9,9),
           color="black", xlim=(1,11), ylim=(10,52))
              
# Plot regression line
plt.plot(new_cars["wt"],     # Explanatory variable
         train_prediction2,  # Predicted values
         color="blue");


# In[ ]:


plt.figure(figsize=(9,9))

stats.probplot(residuals, dist="norm", plot=plt);


# In[ ]:


plt.hist(new_cars["mpg"])


# In[ ]:


def rmse(predicted, targets):
    """
    Computes root mean squared error of two numpy ndarrays
    
    Args:
        predicted: an ndarray of predictions
        targets: an ndarray of target values
    
    Returns:
        The root mean squared error as a float
    """
    return (np.sqrt(np.mean((targets-predicted)**2)))

rmse(trainPrediction, mtcars["mpg"])


# In[ ]:


new_cars


# In[ ]:


from sklearn.metrics import mean_squared_error

RMSE = mean_squared_error(trainPrediction, mtcars["mpg"])**0.5

RMSE


# In[ ]:


poly_model = linear_model.LinearRegression()

# Make a DataFrame of predictor variables
predictors = pd.DataFrame([mtcars["wt"],           # Include weight
                           mtcars["wt"]**2]).T     # Include weight squared

# Train the model using the new_cars data
poly_model.fit(X = predictors, 
               y = mtcars["mpg"])

# Check trained model y-intercept
print("Model intercept")
print(poly_model.intercept_)
# Check trained model coefficients (scaling factor given to "wt")
print("Model Coefficients")
print(poly_model.coef_)

# Check R-squared
print("Model Accuracy:")
print(poly_model.score(X = predictors, 
                 y = mtcars["mpg"]))


# In[ ]:


## Plot the curve from 1.5 to 5.5
poly_line_range = np.arange(1.5, 5.5, 0.1)

# Get first and second order predictors from range
poly_predictors = pd.DataFrame([poly_line_range,
                               poly_line_range**2]).T

# Get corresponding y values from the model
y_values = poly_model.predict(X = poly_predictors)

mtcars.plot(kind="scatter",
           x="wt",
           y="mpg",
           figsize=(9,9),
           color="black",
           xlim = (0,7))

# Plot curve line
plt.plot(poly_line_range,   # X-axis range
         y_values,          # Predicted values
         color="blue");


# In[ ]:


preds = poly_model.predict(X=predictors)

rmse(preds , mtcars["mpg"])


# In[ ]:


poly_model = linear_model.LinearRegression()

# Make a DataFrame of predictor variables
predictors = pd.DataFrame([mtcars["wt"],           
                           mtcars["wt"]**2,
                           mtcars["wt"]**3,
                           mtcars["wt"]**4,
                           mtcars["wt"]**5,
                           mtcars["wt"]**6,
                           mtcars["wt"]**7,
                           mtcars["wt"]**8,
                           mtcars["wt"]**9,
                           mtcars["wt"]**10]).T     

# Train the model using the new_cars data
poly_model.fit(X = predictors, 
               y = mtcars["mpg"])

# Check trained model y-intercept
print("Model intercept")
print(poly_model.intercept_)

# Check trained model coefficients (scaling factor given to "wt")
print("Model Coefficients")
print(poly_model.coef_)

# Check R-squared
poly_model.score(X = predictors, 
                 y = mtcars["mpg"])


# In[ ]:


p_range = np.arange(1.5, 5.45, 0.01)

poly_predictors = pd.DataFrame([p_range, p_range**2, p_range**3,
                              p_range**4, p_range**5, p_range**6, p_range**7, 
                              p_range**8, p_range**9, p_range**10]).T  

# Get corresponding y values from the model
y_values = poly_model.predict(X = poly_predictors)

mtcars.plot(kind="scatter",
           x="wt",
           y="mpg",
           figsize=(9,9),
           color="black",
           xlim = (0,7))

# Plot curve line
plt.plot(p_range,   # X-axis range
         y_values,          # Predicted values
         color="blue");


# In[ ]:


multi_reg_model = linear_model.LinearRegression()

# Train the model using the mtcars data
multi_reg_model.fit(X = mtcars.loc[:,["wt","hp"]], 
                     y = mtcars["mpg"])

# Check trained model y-intercept
print(multi_reg_model.intercept_)

# Check trained model coefficients (scaling factor given to "wt")
print(multi_reg_model.coef_)

# Check R-squared
a =multi_reg_model.score(X = mtcars.loc[:,["wt","hp"]], 
                      y = mtcars["mpg"])


# In[ ]:


mtcars.plot(kind="scatter",
           x="hp",
           y="mpg",
           figsize=(9,9),
           color="black");


# In[ ]:


multi_reg_model = linear_model.LinearRegression()

# Include squared terms
poly_predictors = pd.DataFrame([mtcars["wt"],
                                mtcars["hp"],
                                mtcars["wt"]**2,
                                mtcars["hp"]**2]).T

# Train the model using the mtcars data
multi_reg_model.fit(X = poly_predictors, 
                    y = mtcars["mpg"])

# Check R-squared
print("R-Squared")
print( multi_reg_model.score(X = poly_predictors , 
                      y = mtcars["mpg"]) )

# Check RMSE
print("RMSE")
print(rmse(multi_reg_model.predict(poly_predictors),mtcars["mpg"]))


# In[ ]:


#Formula = 1-(1-R**2)*(n-1)/(n-p-1)
print('Adjusted R sq')
#print(1-(1-0.89)()... a bit less than Rsq but still 0.87


# Logistic Regression

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
plt.figure(figsize=(9,9))

def sigmoid(t):                          # Define the sigmoid function
    return (1/(1 + np.e**(-t)))    

plot_range = np.arange(-6, 6, 0.1)       

y_values = sigmoid(plot_range)

# Plot curve
plt.plot(plot_range,   # X-axis range
         y_values,          # Predicted values
         color="red");


# In[ ]:


titanic_train = pd.read_csv("C:/Users/maorb/CSVs/train.csv")    # Read the data

char_cabin = titanic_train["Cabin"].astype(str)     # Convert cabin to str

new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter

titanic_train["Cabin"] = pd.Categorical(new_Cabin)  # Save the new cabin var

# Impute median Age for NA Age values
new_age_var = np.where(titanic_train["Age"].isnull(), # Logical check
                       28,                       # Value if check is true
                       titanic_train["Age"])     # Value if check is false

titanic_train["Age"] = new_age_var 

new_fare_var = np.where(titanic_train["Fare"].isnull(), # Logical check
                       50,                         # Value if check is true
                       titanic_train["Fare"])     # Value if check is false

titanic_train["Fare"] = new_fare_var 


# In[ ]:


from sklearn import linear_model
from sklearn import preprocessing
# Initialize label encoder
label_encoder = preprocessing.LabelEncoder()

# Convert Sex variable to numeric
encoded_sex = label_encoder.fit_transform(titanic_train["Sex"])

# Initialize logistic regression model
log_model = linear_model.LogisticRegression(solver = 'lbfgs')

# Train the model
log_model.fit(X = pd.DataFrame(encoded_sex), 
              y = titanic_train["Survived"])

# Check trained model intercept
print(log_model.intercept_)

# Check trained model coefficients
print(log_model.coef_)


# In[ ]:


preds = log_model.predict_proba(X= pd.DataFrame(encoded_sex))
preds = pd.DataFrame(preds)
preds.columns = ["Death_prob", "Survival_prob"]

# Generate table of predictions vs Sex
pd.crosstab(titanic_train["Sex"], preds.loc[:, "Survival_prob"])


# In[ ]:


encoded_class = label_encoder.fit_transform(titanic_train["Pclass"])
encoded_cabin = label_encoder.fit_transform(titanic_train["Cabin"])

train_features = pd.DataFrame([encoded_class,
                              encoded_cabin,
                              encoded_sex,
                              titanic_train["Age"]]).T

# Initialize logistic regression model
log_model = linear_model.LogisticRegression(solver = 'lbfgs')

# Train the model
log_model.fit(X = train_features ,
              y = titanic_train["Survived"])

# Check trained model intercept
print(log_model.intercept_)

# Check trained model coefficients
print(log_model.coef_)


# In[ ]:


preds = log_model.predict(X= train_features)

# Generate table of predictions vs actual
pd.crosstab(preds,titanic_train["Survived"])


# In[ ]:


log_model.score(X = train_features ,
                y = titanic_train["Survived"])


# In[ ]:


from sklearn import metrics

# View confusion matrix
metrics.confusion_matrix(y_true=titanic_train["Survived"],  # True labels
                         y_pred=preds) # Predicted labels
# View summary of common classification metrics
print(metrics.classification_report(y_true=titanic_train["Survived"],
                                    y_pred=preds) )


# In[ ]:


# Read and prepare test data
titanic_test = pd.read_csv("../input/test.csv")    # Read the data

char_cabin = titanic_test["Cabin"].astype(str)     # Convert cabin to str

new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter

titanic_test["Cabin"] = pd.Categorical(new_Cabin)  # Save the new cabin var

# Impute median Age for NA Age values
new_age_var = np.where(titanic_test["Age"].isnull(), # Logical check
                       28,                       # Value if check is true
                       titanic_test["Age"])      # Value if check is false

titanic_test["Age"] = new_age_var 


# In[ ]:


# Convert test variables to match model features
encoded_sex = label_encoder.fit_transform(titanic_test["Sex"])
encoded_class = label_encoder.fit_transform(titanic_test["Pclass"])
encoded_cabin = label_encoder.fit_transform(titanic_test["Cabin"])

test_features = pd.DataFrame([encoded_class,
                              encoded_cabin,
                              encoded_sex,
                              titanic_test["Age"]]).T
# Make test set predictions
test_preds = log_model.predict(X=test_features)

# Create a submission for Kaggle
submission = pd.DataFrame({"PassengerId":titanic_test["PassengerId"],
                           "Survived":test_preds})

# Save submission to CSV
submission.to_csv("tutorial_logreg_submission.csv", 
                  index=False)       # Do not save index values


# Decision Trees

# In[ ]:


import pandas as pd
import numpy as np
import sklearn as sk
# Load and prepare Titanic data
titanic_train = pd.read_csv("C:/Users/maorb/CSVs/train.csv")    # Read the data

# Impute median Age for NA Age values
new_age_var = np.where(titanic_train["Age"].isnull(), # Logical check
                       28,                       # Value if check is true
                       titanic_train["Age"])     # Value if check is false

titanic_train["Age"] = new_age_var 
from sklearn import tree #The tree itself
from sklearn import preprocessing #converting to numerical categorial variables
# Initialize label encoder
label_encoder = preprocessing.LabelEncoder()

# Convert Sex variable to numeric
encoded_sex = label_encoder.fit_transform(titanic_train["Sex"])

# Initialize model
tree_model = tree.DecisionTreeClassifier()

# Train the model
tree_model.fit(X = pd.DataFrame(encoded_sex), 
               y = titanic_train["Survived"])

import graphviz

# Save tree as dot file
dot_data = tree.export_graphviz(tree_model, out_file=None) 
graph = graphviz.Source(dot_data)  
graph 
preds = tree_model.predict_proba(X = pd.DataFrame(encoded_sex))

pd.crosstab(preds[:,0], titanic_train["Sex"])
# Make data frame of predictors
predictors = pd.DataFrame([encoded_sex, titanic_train["Pclass"]]).T

# Train the model
tree_model.fit(X = predictors, 
               y = titanic_train["Survived"])
# Save tree as dot file
dot_data = tree.export_graphviz(tree_model, out_file=None) 
graph = graphviz.Source(dot_data)  
graph 
# Get survival probability
preds = tree_model.predict_proba(X = predictors)
female = preds[titanic_train['Sex']=='female'][0][1]
male = preds[titanic_train['Sex']=='male'][0][1]

# Create a table of predictions by sex and class
pd.crosstab(preds[:,0], columns = [titanic_train["Pclass"], 
                                   titanic_train["Sex"]])
predictors = pd.DataFrame([encoded_sex,
                           titanic_train["Pclass"],
                           titanic_train["Age"],
                           titanic_train["Fare"]]).T

# Initialize model with maximum tree depth set to 8
tree_model = tree.DecisionTreeClassifier(max_depth = 8)

lm1 =tree_model.fit(X = predictors, 
               y = titanic_train["Survived"])
# Save tree as dot file
dot_data = tree.export_graphviz(tree_model, out_file=None) 
graph = graphviz.Source(dot_data)  
graph 
tree_model.score(X = predictors, 
                 y = titanic_train["Survived"])
# Read and prepare test data
titanic_test = pd.read_csv("C:/Users/maorb/CSVs/train.csv")    # Read the data

# Impute median Age for NA Age values
new_age_var = np.where(titanic_test["Age"].isnull(), # Logical check
                       28,                       # Value if check is true
                       titanic_test["Age"])      # Value if check is false

new_fare_var = np.where(titanic_test["Fare"].isnull(), # Logical check
                       50,                       # Value if check is true
                       titanic_test["Fare"])      # Value if check is false

titanic_test["Age"] = new_age_var 
titanic_test["Fare"] = new_fare_var


# In[ ]:


pd.crosstab(preds[:,0], titanic_train["Sex"])


# In[ ]:


print(female)


# F.test or Anova prediction --> Too damn advanced right now

# In[ ]:


from sklearn import datasets
df=pd.read_csv('C:/Users/maorb/CSVs/winequality-red.csv')
df.head()
df2 = df.sort_values(by ='quality')
df.groupby('quality').mean()
df.boxplot(column='fixed acidity', by ='quality',grid=False)
df.boxplot(column='volatile acidity', by ='quality',grid=False )
df


# In[ ]:


y = df.iloc[:,11] #pick quality
x = df.iloc[:,0:11] #pick all but quality(11)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2,
                                                    random_state=42)
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=4)
selector.fit(x_train, y_train)
selector.scores_
cols = selector.get_support(indices=True)
cols
x_train_s = x_train.iloc[:,cols]
x_test_s = x_test.iloc[:,cols]
from sklearn.tree import DecisionTreeClassifier
#Initalize the classifier
clf = DecisionTreeClassifier()
#Fitting the training data
clf.fit(x_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(x_test)
# Model Accuracy, how often is the classifier correct?
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Fitting the training data
clf.fit(x_train_s, y_train)
#Predict the response for test dataset
y_pred = clf.predict(x_test_s)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy
print('Accuracy = ', knn.score(x_test_s, y_test))


# In[ ]:


get_ipython().system('pip install scikit-learn-intelex')


# XGBoost

# In[ ]:


get_ipython().system('pip install --upgrade pandas')


# In[ ]:


import xgboost as xgb


# In[ ]:


import pandas as pd 
import numpy as np
import xgboost as xgb
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import ConfusionMatrixDisplay


# In[ ]:


df = pd.read_csv('C:/Users/maorb/CSVs/Telco.csv')
df['ChurnValue'] = (df['Churn']=='Yes') *1
df.drop('customerID',axis =1, inplace=True)
df.head()


# In[ ]:


df['Dependents'].unique()
df['MultipleLines'].replace(' ','_',regex=True, inplace=True)
df['InternetService'].replace(' ','_',regex=True, inplace=True)
df['Contract'].replace(' ','_',regex=True, inplace=True)
df['PaymentMethod'].replace(' ','_',regex=True, inplace=True)
df.columns = df.columns.str.replace(' ', '_') #in the headlines as well


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


b = df['TotalCharges'].unique()
a = df['MonthlyCharges'].unique() #float
a.dtype
b.dtype #OBJECT even though it seems like it contains numbers!!


# In[ ]:


b = pd.to_numeric(b) #ValueError: Unable to parse string " "


# In[ ]:


len(df.loc[df['TotalCharges'] == ' ']) #count number of Trues


# In[ ]:


df.loc[df['TotalCharges'] == ' ', 'TotalCharges'] = 0 #set the TC in rows with blanks in TC to 0


# In[ ]:


df.loc[df['tenure'] == 0]


# In[ ]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges']) # now that 0s turn to numeric


# In[ ]:


df.dtypes #Success


# In[ ]:


df.replace(' ', '_', regex=True, inplace=True)
df.head() #we did it to print nice trees xgb doesnt care


# In[ ]:


X = df.drop('ChurnValue',axis =1).copy()
y = df['ChurnValue'].copy()
y


# In[ ]:


X.dtypes


# In[ ]:


x = pd.get_dummies(X,columns = ['PaymentMethod',
                                'StreamingMovies','gender',
                                'Partner','Dependents',
                                'PhoneService','MultipleLines',
                                'InternetService','OnlineSecurity',
                                'OnlineBackup','DeviceProtection',
                                'TechSupport','StreamingTV',
                                'Contract','PaperlessBilling',])                                
x = x*1
x.dtypes


# In[ ]:





# In[ ]:





# In[ ]:


y.unique()


# In[ ]:


sum(y)/len(y)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=42,
                                                stratify=y)


# In[ ]:


sum(y_train)/len(y_train) #stratify keeps the proportion in the trained data like the obsewrved


# In[ ]:


sum(y_test)/len(y_test)


# In[ ]:


x['Churn'] = x['Churn'].astype('category')


# In[ ]:


x


# In[ ]:


conda install -c conda-forge xgboost


# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def data_handling(data: dict) -> tuple:
    # Split dataset into features and target
    # data is features
    """
    >>> data_handling(({'data':'[5.1, 3.5, 1.4, 0.2]','target':([0])}))
    ('[5.1, 3.5, 1.4, 0.2]', [0])
    >>> data_handling(
    ...     {'data': '[4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2]', 'target': ([0, 0])}
    ... )
    ('[4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2]', [0, 0])
    """
    return (data["data"], data["target"])


def xgboost(features: np.ndarray, target: np.ndarray) -> XGBClassifier:
    """
    # THIS TEST IS BROKEN!! >>> xgboost(np.array([[5.1, 3.6, 1.4, 0.2]]), np.array([0]))
    XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                  colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                  early_stopping_rounds=None, enable_categorical=False,
                  eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                  importance_type=None, interaction_constraints='',
                  learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
                  max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
                  missing=nan, monotone_constraints='()', n_estimators=100,
                  n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=0,
                  reg_alpha=0, reg_lambda=1, ...)
    """
    classifier = XGBClassifier()
    classifier.fit(features, target)
    return classifier


def main() -> None:
    """
    >>> main()

    Url for the algorithm:
    https://xgboost.readthedocs.io/en/stable/
    Iris type dataset is used to demonstrate algorithm.
    """

    # Load Iris dataset
    iris = load_iris()
    features, targets = data_handling(x)
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.25
    )

    #names = iris["target_names"]

    # Create an XGBoost Classifier from the training data
    #xgboost_classifier = xgboost(x_train, y_train)

    # Display the confusion matrix of the classifier with both training and test sets
    #ConfusionMatrixDisplay.from_estimator(
     #   xgboost_classifier,
      #  x_test,
      #  y_test,
       # display_labels=names,
        #cmap="Blues",
        #normalize="true",
    #)
   # plt.title("Normalized Confusion Matrix - IRIS Dataset")
    #plt.show()


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
    main()
lm_xgb = xgb.XGBClassifier(objective='binary:logistic', missing = None,
                          seed = 42,enable_categorical=True)
lm_xgb.fit(X_train,y_train,
          eval_set=[(X_test, y_test)],
          verbose=True)
lm_xgb.set_params(
    early_stopping_rounds= 1, 
    eval_metric='auc') 


# In[ ]:


lm_xgb = xgb.XGBClassifier(objective='binary:logistic', missing=None,
                          seed=42, enable_categorical=True)
lm_xgb.set_params(early_stopping_rounds=10, eval_metric='logloss')
lm_xgb.fit(X_train, y_train,
           eval_set=[(X_test, y_test)],
           verbose=True)


# In[ ]:


lm_xgb = xgb.XGBClassifier(objective='binary:logistic', missing=None, seed=42, enable_categorical=True)

# Fit the model with early stopping and evaluation metric
lm_xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=True,
            early_stopping_rounds=10,  # Set early stopping rounds here
            eval_metric='auc'  # Set the evaluation metric here
)


# Image proccessing first attempt

# In[ ]:


get_ipython().system('pip install opencv-contrib-python')


# In[ ]:


get_ipython().system('pip install pytesseract')


# In[ ]:


get_ipython().system('pip install pdf2image')


# # Photo convertion to CSV

# In[ ]:


import cv2
import pytesseract
from pdf2image import convert_from_path
import pandas as pd

# Path to your PDF file
pdf_path = 'C:/Users/maorb/Desktop/Work/Evaluation Form Sustainable Food Production and Consumption.pdf'
# Convert PDF pages to images
images = convert_from_path(pdf_path)

# Initialize empty list to store extracted data
extracted_data = []
from pdf2image import convert_from_path,convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError,PDFPageCountError,PDFSyntaxError


images = convert_from_bytes(open(pdf_path).read())

for i, image in enumerate(images):
    fname = "image" + str(i) + ".png"
    image.save(fname, "PNG")

for i, image in enumerate(images):
    # Convert PIL image to OpenCV format
    images[i].save('image_name'+ str(i) +'.jpg', 'JPEG')
    #img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Preprocess image if needed (e.g., resize, grayscale, filters)
    # Example: img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Use Pytesseract to extract text from image
    extracted_text = pytesseract.image_to_string(img_cv)
    
    # Parse and process extracted text to get specific data
    # Example: extracted_data.append({'Page': i+1, 'Data': extracted_text})
    extracted_data.append(extracted_text)

# Save extracted data to CSV
df = pd.DataFrame(extracted_data, columns=['Extracted Data'])
df.to_csv('extracted_data.csv', index=False)


# # PDF TO CSV

# In[ ]:


import os
import tabula

# Set Java path
java_path = 'C:/Program Files (x86)/Common Files/Oracle/Java/javapath'  # Replace with the path you found
os.environ['PATH'] += os.pathsep + java_path

# Specify PDF and CSV file paths
pdf_name = r'C:\Users\maorb\Classes\Seminar\Final pitches - Groups.pdf'
output_folder = 'C:/Users/maorb/CSVs/'
tables = tabula.read_pdf(pdf_name, pages='all')

# Iterate over each table and save as CSV
for i, df in enumerate(tables):
    csv_file_path = os.path.join(output_folder, f'Yes{i+1}.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved: {csv_file_path}")


# In[ ]:


import os
import tabula

# Set Java path
java_path = 'C:/Program Files (x86)/Common Files/Oracle/Java/javapath'  # Replace with the path you found
os.environ['PATH'] += os.pathsep + java_path

# Specify PDF and CSV file paths
pdf_name = 'C:/Users/maorb/Desktop/Work/Evaluation Form Sustainable Food Production and Consumption.pdf'
output_folder = 'C:/Users/maorb/CSVs/'
tables = tabula.read_pdf(pdf_name, pages='all')
tables


# In[ ]:


import os
import tabula

# Set Java path
java_path = 'C:/Program Files (x86)/Common Files/Oracle/Java/javapath'  # Replace with the path you found
os.environ['PATH'] += os.pathsep + java_path

# Specify PDF file path
pdf_name = 'C:/Users/maorb/Desktop/Work/Evaluation Form Sustainable Food Production and Consumption.pdf'

# Read PDF and extract tables
tables = tabula.read_pdf(pdf_name, pages='all', multiple_tables=True)

# Specify the output folder
output_folder = 'C:/Users/maorb/CSVs/'

# Iterate over each table and save as CSV
for i, df in enumerate(tables):
    csv_file_path = os.path.join(output_folder, f'Seminar_{i+1}.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved: {csv_file_path}")


# EXCEL ROWS EDITIng

# In[ ]:


import pandas as pd

# Load both sheets into pandas dataframes
df1 = pd.read_excel("C:/Users/maorb/CSVs/Biztec+2023+(Time+1)_February+18,+2024_17.05 (1).xlsx", sheet_name='Time1')
df2 = pd.read_excel("C:/Users/maorb/CSVs/Biztec+2023+(Time+1)_February+18,+2024_17.05 (1).xlsx", sheet_name='Time2')

# Merge the dataframes based on the values in columns C and D
merged_df = pd.merge(df1, df2, how='left', left_on='Q3', right_on='first_name')

# Filter out rows where column D is not null (meaning it was matched)
matched_rows = merged_df[merged_df['S'].notnull()]

# Write the matched rows to a new Excel file in Sheet1 starting from column F
matched_rows.to_excel('matched_data.xlsx', sheet_name='Time1', startcol=5, index=False)

print("Matching and writing to Excel completed successfully.")


# In[ ]:


import pandas as pd

# Load both sheets into pandas dataframes
df1 = pd.read_excel("C:/Users/maorb/CSVs/Biztec+2023+(Time+1)_February+18,+2024_17.05 (1).xlsx", sheet_name='Time1')
df2 = pd.read_excel("C:/Users/maorb/CSVs/Biztec+2023+(Time+1)_February+18,+2024_17.05 (1).xlsx", sheet_name='Time2')

# Merge the dataframes based on the values in the first name columns
merged_df = pd.merge(df1, df2, how='inner', left_on='שם פרטי', right_on='first_name')

# Print the merged dataframe
print(merged_df)


# In[ ]:


import pandas as pd

# Load both sheets into pandas dataframes
df1 = pd.read_excel("C:/Users/maorb/CSVs/Biztec+2023+(Time+1)_February+18,+2024_17.05 (1).xlsx", sheet_name='Time1')
df2 = pd.read_excel("C:/Users/maorb/CSVs/Biztec+2023+(Time+1)_February+18,+2024_17.05 (1).xlsx", sheet_name='Time2')

# Rename the column in df1 to use English characters
df1.rename(columns={'Unnamed: 18': 'first_name'}, inplace=True)
# Merge the dataframes based on the values in the first name columns
merged_df = pd.merge(df1, df2, how='inner',left_on='first_name', right_on='first_name')
merged_df.to_csv("C:/Users/maorb/CSVs/merged_data.csv", index=False, encoding='utf-8-sig')



# In[ ]:


import pandas as pd

# Load both sheets into pandas dataframes
df3 = pd.read_excel("C:/Users/maorb/CSVs/Inspire Time 1.xlsx", sheet_name='Time1')
df4 = pd.read_excel("C:/Users/maorb/CSVs/Inspire Time 1.xlsx", sheet_name='Time2')
df3.drop(0, inplace=True)
# Rename the column in df1 to use English characters
df3.rename(columns={'Unnamed: 18': 'first_name'}, inplace=True)
df3.rename(columns={'Unnamed: 19': 'last_name'}, inplace=True)

# Merge the dataframes based on the values in the first name columns, using a left join
merged_df = pd.merge(df3, df4, how='left', on='first_name')

# Keep the 'first_name' column from df2 in the merged dataframe


# Print the merged dataframe
print(merged_df)

# Save the merged dataframe to a new CSV file with UTF-8 encoding
merged_df.to_csv("C:/Users/maorb/CSVs/merged_data40.csv", index=False, encoding='utf-8-sig')




# VIDEO EDITING

# In[ ]:


import cv2
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Drawing specifications for landmarks and connections
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))
connection_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, color=(255, 0,0))

# Video file to process
input_video_path = 'Liron-Head_move.mp4'
output_video_path = 'liron_head_move_mesh.mp4'

# OpenCV video capture and writer
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 output
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    # Draw the face mesh on the frame
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=connection_spec
            )
            # Draw contours
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=connection_spec
            )
            # Draw irises
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=connection_spec
            )
    
    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()


# PHOTO EDITING

# In[ ]:


get_ipython().system('pip install Audio-classifier')


# In[ ]:


import mediapipe as mp


# In[ ]:


get_ipython().system('conda activate root')


# In[ ]:


import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Drawing specifications for landmarks and connections
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))
connection_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, color=(255, 0,0))

# Folder containing input images
input_folder = r"C:\Users\maorb\Desktop\Work\sad_2\photo_14.jpeg"
# Folder to save processed images
output_folder = r"C:\Users\maorb\Desktop\Work\sad_2\new.jpeg"

# Iterate over subdirectories in the input folder
for root, dirs, files in os.walk(input_folder):
    for subdir in dirs:
        sub_folder_path = os.path.join(root, subdir)
        output_sub_folder = os.path.join(output_folder, subdir)
        
        # Create output subfolder if it doesn't exist
        if not os.path.exists(output_sub_folder):
            os.makedirs(output_sub_folder)
        
        # Process each image in the subfolder
        for filename in os.listdir(sub_folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust file extensions as needed
                input_image_path = os.path.join(sub_folder_path, filename)
                output_image_path = os.path.join(output_sub_folder, filename)
                
                # Read the input image
                frame = cv2.imread(input_image_path)
                
                # Convert the image to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the image to detect face landmarks
                results = face_mesh.process(frame_rgb)
                
                # Draw an ellipse around the face
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Iterate over each landmark in the face
                        for lm in face_landmarks.landmark:
                            # Get the x and y coordinates of the landmark
                            x = int(lm.x * frame.shape[1])
                            y = int(lm.y * frame.shape[0])
                            
                            # Draw a circle around each landmark
                            cv2.circle(frame, (x, y), 1, (255, 0, 0), 1)
                
                # Save the output image
                cv2.imwrite(output_image_path, frame)

# Release resources
cv2.destroyAllWindows()


# In[ ]:


import os

# Directory containing the files to rename
directory = 'C:/Users/maorb/Desktop/Work/disgusted_4'

# Loop through each file in the directory
for k, filename in enumerate(os.listdir(directory)):
    # Split the filename and its extension
    name, extension = os.path.splitext(filename)
    
    # Construct the full old and new file paths
    old_path = os.path.join(directory, filename)
    
    # Rename the file using a new name (e.g., using an index)
    new_filename = f'photo_{k}{extension}'
    new_path = os.path.join(directory, new_filename)
    
    # Rename the file
    os.rename(old_path, new_path)


# In[ ]:


import os

# Directory containing the files to rename
directory = 'C:/Users/maorb/Desktop/Work/happy_1'

# Loop through each file in the directory
for k, filename in enumerate(os.listdir(directory)):
    # Split the filename and its extension
    name, extension = os.path.splitext(filename)
    
    # Construct the full old and new file paths
    old_path = os.path.join(directory, filename)
    
    # Rename the file using a new name (e.g., using an index)
    new_filename = f'photo_{k}{extension}'
    new_path = os.path.join(directory, new_filename)
    
    # Rename the file
    os.rename(old_path, new_path)


# In[ ]:


import os

# Directory containing the files to rename
directory = 'C:/Users/maorb/Desktop/Work/scared_6'

# Loop through each file in the directory
for k, filename in enumerate(os.listdir(directory)):
    # Split the filename and its extension
    name, extension = os.path.splitext(filename)
    
    # Construct the full old and new file paths
    old_path = os.path.join(directory, filename)
    
    # Rename the file using a new name (e.g., using an index)
    new_filename = f'photo_{k}{extension}'
    new_path = os.path.join(directory, new_filename)
    
    # Rename the file
    os.rename(old_path, new_path)


# In[ ]:


import os

# Directory containing the files to rename
directory = 'C:/Users/maorb/Desktop/Work/surprised_3'

# Loop through each file in the directory
for k, filename in enumerate(os.listdir(directory)):
    # Split the filename and its extension
    name, extension = os.path.splitext(filename)
    
    # Construct the full old and new file paths
    old_path = os.path.join(directory, filename)
    
    # Rename the file using a new name (e.g., using an index)
    new_filename = f'photo_{k}{extension}'
    new_path = os.path.join(directory, new_filename)
    
    # Rename the file
    os.rename(old_path, new_path)


# In[ ]:


import os

# Directory containing the files to rename
directory = 'C:/Users/maorb/Desktop/Work/sad_2'

# Loop through each file in the directory
for k, filename in enumerate(os.listdir(directory)):
    # Split the filename and its extension
    name, extension = os.path.splitext(filename)
    
    # Construct the full old and new file paths
    old_path = os.path.join(directory, filename)
    
    # Rename the file using a new name (e.g., using an index)
    new_filename = f'photo_{k}{extension}'
    new_path = os.path.join(directory, new_filename)
    
    # Rename the file
    os.rename(old_path, new_path)


# In[ ]:


import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Folder containing input images
input_folder = 'C:/Users/maorb/Desktop/Work'
# Folder to save processed images
output_folder = 'C:/Users/maorb/Desktop/Work/Processed7'

# Iterate over subdirectories in the input folder
for root, dirs, files in os.walk(input_folder):
    for subdir in dirs:
        sub_folder_path = os.path.join(root, subdir)
        output_sub_folder = os.path.join(output_folder, subdir)
        
        # Create output subfolder if it doesn't exist
        if not os.path.exists(output_sub_folder):
            os.makedirs(output_sub_folder)
        
        # Process each image in the subfolder
        for filename in os.listdir(sub_folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust file extensions as needed
                input_image_path = os.path.join(sub_folder_path, filename)
                output_image_path = os.path.join(output_sub_folder, filename)
                
                # Read the input image
                frame = cv2.imread(input_image_path)
                
                # Convert the image to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the image to detect face landmarks
                results = face_mesh.process(frame_rgb)
                
                # Extract landmarks if available
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Extract landmark points
                        landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]
                        
                        # Get bounding box around landmarks
                        x, y, w, h = cv2.boundingRect(np.array(landmarks))
                        
                        # Define bounding box around face and some hair area
                        x -= 20  # Add some margin
                        y -= 20
                        w += 40
                        h += 40
                        
                        # Ensure bounding box is within image boundaries
                        x = max(0, x)
                        y = max(0, y)
                        w = min(frame.shape[1] - x, w)
                        h = min(frame.shape[0] - y, h)
                        
                        # Crop the image to the bounding box
                        cropped_image = frame[y:y+h, x:x+w]
                        
                        # Save the cropped image
                        cv2.imwrite(output_image_path, cropped_image)

# Release resources
cv2.destroyAllWindows()


# In[ ]:


import os
import cv2
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

# Drawing specifications for landmarks and connections
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))
connection_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, color=(255, 0,0))

# Draw landmarks on the image
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, connection_spec, drawing_spec)

# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\new.jpeg"

# Save the output image
cv2.imwrite(output_image_path, frame)

# Release resources
face_mesh.close()


# ONly Lips

# In[ ]:


import os
import cv2
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

# Drawing specifications for landmarks and connections
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))
connection_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, color=(255, 0,0))

# Draw landmarks on the image
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Draw all landmarks in green
        mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, connection_spec)
        
        # Draw landmarks around lips in red
        mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_LIPS, connection_spec, mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0)))
        
# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\new1.jpeg"

# Save the output image
cv2.imwrite(output_image_path, frame)

# Release resources
face_mesh.close()


# FACE_CONNECTIONS = frozenset([
#     # Lips.
#     (61, 146),
#     (146, 91),
#     (91, 181),
#     (181, 84),
#     (84, 17),
#     (17, 314),
#     (314, 405),
#     (405, 321),
#     (321, 375),
#     (375, 291),
#     (61, 185),
#     (185, 40),
#     (40, 39),
#     (39, 37),
#     (37, 0),
#     (0, 267),
#     (267, 269),
#     (269, 270),
#     (270, 409),
#     (409, 291),
#     (78, 95),
#     (95, 88),
#     (88, 178),
#     (178, 87),
#     (87, 14),
#     (14, 317),
#     (317, 402),
#     (402, 318),
#     (318, 324),
#     (324, 308),
#     (78, 191),
#     (191, 80),
#     (80, 81),
#     (81, 82),
#     (82, 13),
#     (13, 312),
#     (312, 311),
#     (311, 310),
#     (310, 415),
#     (415, 308),
#     # Left eye.
#     (263, 249),
#     (249, 390),
#     (390, 373),
#     (373, 374),
#     (374, 380),
#     (380, 381),
#     (381, 382),
#     (382, 362),
#     (263, 466),
#     (466, 388),
#     (388, 387),
#     (387, 386),
#     (386, 385),
#     (385, 384),
#     (384, 398),
#     (398, 362),
#     # Left eyebrow.
#     (276, 283),
#     (283, 282),
#     (282, 295),
#     (295, 285),
#     (300, 293),
#     (293, 334),
#     (334, 296),
#     (296, 336),
#     # Right eye.
#     (33, 7),
#     (7, 163),
#     (163, 144),
#     (144, 145),
#     (145, 153),
#     (153, 154),
#     (154, 155),
#     (155, 133),
#     (33, 246),
#     (246, 161),
#     (161, 160),
#     (160, 159),
#     (159, 158),
#     (158, 157),
#     (157, 173),
#     (173, 133),
#     # Right eyebrow.
#     (46, 53),
#     (53, 52),
#     (52, 65),
#     (65, 55),
#     (70, 63),
#     (63, 105),
#     (105, 66),
#     (66, 107),
#     # Face oval.
#     (10, 338),
#     (338, 297),
#     (297, 332),
#     (332, 284),
#     (284, 251),
#     (251, 389),
#     (389, 356),
#     (356, 454),
#     (454, 323),
#     (323, 361),
#     (361, 288),
#     (288, 397),
#     (397, 365),
#     (365, 379),
#     (379, 378),
#     (378, 400),
#     (400, 377),
#     (377, 152),
#     (152, 148),
#     (148, 176),
#     (176, 149),
#     (149, 150),
#     (150, 136),
#     (136, 172),
#     (172, 58),
#     (58, 132),
#     (132, 93),
#     (93, 234),
#     (234, 127),
#     (127, 162),
#     (162, 21),
#     (21, 54),
#     (54, 103),
#     (103, 67),
#     (67, 109),
#     (109, 10)
# ])

# In[ ]:


import os
import cv2
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

lip_contours = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)]


lip_coors = [x[0] for x in lip_contours]
print(lip_coors)

# List to store landmarks coordinates around lips
lips_landmarks = []

# Extract landmarks around lips
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            # Check if the landmark is around the lips
            if i in lip_coors:
                lips_landmarks.append((lm.x, lm.y, lm.z))

# Print the coordinates of landmarks around lips
for i, landmark in enumerate(lips_landmarks, start=1):
    print(f"Landmark {i}: {landmark}")

# Release resources
face_mesh.close()


# Trying to paint the landmarks in different colors 

# In[ ]:


# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"

# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\maor.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# List of landmark contours for the lips

lips_landmarks = [(int(x[0]*frame.shape[0]),int(x[1]*frame.shape[1])) for x in lips_landmarks]
print(lips_landmarks)

# Draw purple dots along the lips contours
for point in lips_landmarks:
    cv2.circle(frame, point, 1, (128, 0, 128), -1)  # Purple color

# Save the modified image
cv2.imwrite(output_image_path, frame)


# In[ ]:


import cv2
import matplotlib.pyplot as plt

input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\photo_19.png"

# Read the input image
im = cv2.imread(input_image_path)
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cnt = contours[4]  # Just an example, choose the contour you want to draw
new_im = cv2.drawContours(im.copy(), [cnt], 0, (0, 255, 0), 3)

# Display the image with the contour
plt.imshow(cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB))
plt.show()


# In[ ]:


import numpy as np
import cv2 as cv
 
img = cv.imread(r"C:\Users\maorb\Desktop\Work\sad_2\photo_13.JPG", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret,thresh = cv.threshold(img,127,255,0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
 
cnt = contours[:]
#M = cv.moments(cnt)
#print( M )
cv.drawContours(img, cnt, -1, (0, 255, 0), 3)  # Draw all contours
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img,[box],0,(0,0,255),2)
# Display the image with the contour
plt.imshow(cv.cvtColor(color_img, cv.COLOR_BGR2RGB))
plt.show()


# Contours

# In[ ]:


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Read the image
img = cv.imread(r"C:\Users\maorb\Desktop\Work\sad_2\photo_13.JPG", cv.IMREAD_GRAYSCALE)

# Check if the image was read successfully
assert img is not None, "File could not be read, check with os.path.exists()"

# Threshold the image
ret, thresh = cv.threshold(img, 127, 255, 0)

# Find contours
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
color_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.drawContours(color_img, contours, -1, (0, 255, 0), 3)  # Draw all contours

# Find minimum area rectangle
cnt = contours[0]  # Take one contour for demonstration
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.intp(box)
cv.drawContours(color_img, [box], 0, (0, 0, 255), 2)  # Draw the rotated rectangle

# Display the image with the contour
plt.imshow(cv.cvtColor(color_img, cv.COLOR_BGR2RGB))
plt.show()


# Using histograms?

# In[ ]:


import numpy as np
import cv2 as cv
 
roi = cv.imread(r"C:\Users\maorb\Desktop\Work\sad_2\sad_l.jpeg")
assert roi is not None, "file could not be read, check with os.path.exists()"
hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
 
target = roi
assert target is not None, "file could not be read, check with os.path.exists()"
hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
 
# calculating object histogram
roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
 
# normalize histogram and apply backprojection
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
 
# Now convolute with circular disc
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(dst,-1,disc,dst)
 
# threshold and binary AND
ret,thresh = cv.threshold(dst,50,255,0)
thresh = cv.merge((thresh,thresh,thresh))
res = cv.bitwise_and(target,thresh)
 
res = np.vstack((target,thresh,res))
cv.imwrite(r"C:\Users\maorb\Desktop\Work\sad_2\MEss.JPG",res)


# In[ ]:


cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
area = cv.contourArea(cnt)
area
perimeter = cv.arcLength(cnt,True)
perimeter
epsilon = 0.1*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)


# In[ ]:


import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

# List of landmark contours for the lips
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]

# Extract lips landmarks coordinates
lips_landmarks = []
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            if i in [idx for pair in lip_contours for idx in pair]:
                lips_landmarks.append((lm.x * frame.shape[1], lm.y * frame.shape[0]))

# Calculate the centroid of the lips
centroid_x = sum(point[0] for point in lips_landmarks) / len(lips_landmarks)
centroid_y = sum(point[1] for point in lips_landmarks) / len(lips_landmarks)

# Scale factor
scale_factor = 1.2  # Adjust the scale factor to increase/reduce the stretching

# Move the lips landmarks outward from the centroid
lips_landmarks_scaled = [
    ((point[0] - centroid_x) * scale_factor + centroid_x, 
     (point[1] - centroid_y) * scale_factor + centroid_y-4)
    for point in lips_landmarks
]

# Prepare for drawing
pts = np.array(lips_landmarks_scaled, np.int32)
pts = pts.reshape((-1,1,2))

# Fill the lips area with a color (e.g., red)
#cv2.fillPoly(frame, [pts], (0, 0, 255))

# Optionally, draw the lips landmarks on the image
for point in lips_landmarks_scaled:
    cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color

# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\maor_realigned2.1.jpeg"

# Save the modified image
cv2.imwrite(output_image_path, frame)

# Release resources
face_mesh.close()


# Box only around lips

# In[ ]:


import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

# List of landmark contours for the lips
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]

# Extract lips landmarks coordinates
lips_landmarks = []
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            if i in [idx for pair in lip_contours for idx in pair]:
                lips_landmarks.append((lm.x * frame.shape[1], lm.y * frame.shape[0]))

# Calculate the centroid of the lips
centroid_x = sum(point[0] for point in lips_landmarks) / len(lips_landmarks)
centroid_y = sum(point[1] for point in lips_landmarks) / len(lips_landmarks)

# Scale factor
scale_factor = 1.2  # Adjust the scale factor to increase/reduce the stretching

# Move the lips landmarks outward from the centroid
lips_landmarks_scaled = [
    ((point[0] - centroid_x) * scale_factor + centroid_x, 
     (point[1] - centroid_y) * scale_factor + centroid_y - 5)
    for point in lips_landmarks
]

# Prepare for drawing
pts = np.array(lips_landmarks_scaled, np.int32)
pts = pts.reshape((-1,1,2))

# Draw a bounding box around the lips area
min_x = int(min(point[0] for point in lips_landmarks_scaled))
max_x = int(max(point[0] for point in lips_landmarks_scaled))
min_y = int(min(point[1] for point in lips_landmarks_scaled))
max_y = int(max(point[1] for point in lips_landmarks_scaled))
cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue color

# Optionally, draw the lips landmarks on the image
for point in lips_landmarks_scaled:
    cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color

# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\maor_realigned2.1_with_box.jpeg"

# Save the modified image
cv2.imwrite(output_image_path, frame)

# Release resources
face_mesh.close()


# Box around lips to the edges

# In[ ]:


import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\happy_1\sad_l.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# Get the width of the image
image_width = frame.shape[1]

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

# Function to process landmarks and draw bounding box
def process_landmarks_and_draw_bbox(landmarks, frame, vert_factor, scale_factor):
    # Calculate centroid
    centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
    centroid_y = sum(point[1] for point in landmarks) / len(landmarks)

    # Scale factor
    # Move the landmarks outward from the centroid
    landmarks_scaled = [
        ((point[0] - centroid_x) * scale_factor + centroid_x, 
         (point[1] - centroid_y) * scale_factor + centroid_y - vert_factor)
        for point in landmarks
    ]

    # Prepare for drawing
    pts = np.array(landmarks_scaled, np.int32)
    pts = pts.reshape((-1,1,2))

    # Draw a bounding box around the landmarks area
    min_x = int(min(point[0] for point in landmarks_scaled))
    max_x = int(max(point[0] for point in landmarks_scaled))
    min_y = int(min(point[1] for point in landmarks_scaled))
    max_y = int(max(point[1] for point in landmarks_scaled))
    if min_x > 0:
        min_x = 0
    if max_x < image_width:
        max_x = image_width
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue color

    # Optionally, draw the landmarks on the image
    for point in landmarks_scaled:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color

# List of landmark contours for each feature
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]
left_eye_contour = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), 
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), 
    (398, 362)
    
]
left_eyebrow_contour = [
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)
]
right_eye_contour = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), 
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)]
right_eyebrow_contour = [
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
]

# Extract landmarks coordinates for each feature
landmarks_dict = {}
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            for contour_name, contour in zip(["lips", "left_eye",'left_eyebrow', "right_eye",'left_eyebrow'],
                                            [lip_contours, left_eye_contour, left_eyebrow_contour, right_eye_contour, right_eyebrow_contour]):
                if i in [idx for pair in contour for idx in pair]:
                    if contour_name not in landmarks_dict:
                        landmarks_dict[contour_name] = []
                    landmarks_dict[contour_name].append((lm.x * frame.shape[1], lm.y * frame.shape[0]))


vert_dict = {'lips': 5, 'left_eye': 5, 'left_eyebrow': 0, "right_eye": 5, "right_eyebrow":0}
scale_dict = {'lips': 1.2, 'left_eye': 2.6, 'left_eyebrow': 1.14, "right_eye": 2.6, "right_eyebrow":1.14}
# Process each set of landmarks and draw bounding box separately
for contour_name, landmarks in landmarks_dict.items():
    process_landmarks_and_draw_bbox(landmarks, frame,vert_dict[contour_name],scale_dict[contour_name])

# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\happy_1\combine.jpeg"

# Save the modified image
cv2.imwrite(output_image_path, frame)

# Release resources
face_mesh.close()


# In[ ]:


import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\happy_1\photo_0.JPG"

# Read the input image
frame = cv2.imread(input_image_path)

# Get the width of the image
image_width = frame.shape[1]

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

# Function to process landmarks and draw bounding box
def process_landmarks_and_draw_bbox(landmarks, frame, vert_factor, scale_factor):
    # Calculate centroid
    centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
    centroid_y = sum(point[1] for point in landmarks) / len(landmarks)

    # Scale factor
    # Move the landmarks outward from the centroid
    landmarks_scaled = [
        ((point[0] - centroid_x) * scale_factor + centroid_x, 
         (point[1] - centroid_y) * scale_factor + centroid_y - vert_factor)
        for point in landmarks
    ]

    # Prepare for drawing
    pts = np.array(landmarks_scaled, np.int32)
    pts = pts.reshape((-1,1,2))

    # Draw a bounding box around the landmarks area
    min_x = int(min(point[0] for point in landmarks_scaled))
    max_x = int(max(point[0] for point in landmarks_scaled))
    min_y = int(min(point[1] for point in landmarks_scaled))
    max_y = int(max(point[1] for point in landmarks_scaled))
    if min_x > 0:
        min_x = 0
    if max_x < image_width:
        max_x = image_width
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue color

    # Optionally, draw the landmarks on the image
    for point in landmarks_scaled:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color

# List of landmark contours for each feature
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]
left_eye_contour = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), 
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), 
    (398, 362)
    
]
left_eyebrow_contour = [
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)
]
right_eye_contour = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), 
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)]
right_eyebrow_contour = [
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
]

# Extract landmarks coordinates for each feature
landmarks_dict = {}
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            for contour_name, contour in zip(["lips", "left_eye",'left_eyebrow', "right_eye",'left_eyebrow'],
                                            [lip_contours, left_eye_contour, left_eyebrow_contour, right_eye_contour, right_eyebrow_contour]):
                if i in [idx for pair in contour for idx in pair]:
                    if contour_name not in landmarks_dict:
                        landmarks_dict[contour_name] = []
                    landmarks_dict[contour_name].append((lm.x * frame.shape[1], lm.y * frame.shape[0]))


vert_dict = {'lips': 5, 'left_eye': 5, 'left_eyebrow': 0, "right_eye": 5, "right_eyebrow":0}
scale_dict = {'lips': 1.2, 'left_eye': 2.6, 'left_eyebrow': 1.14, "right_eye": 2.6, "right_eyebrow":1.14}
# Process each set of landmarks and draw bounding box separately
for contour_name, landmarks in landmarks_dict.items():
    process_landmarks_and_draw_bbox(landmarks, frame,vert_dict[contour_name],scale_dict[contour_name])

# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\happy_1\new.JPG"

# Save the modified image
cv2.imwrite(output_image_path, frame)

# Release resources
face_mesh.close()


# ## final code

# In[ ]:


import cv2
import numpy as np
import mediapipe as mp

"""
Boxes continue on a straight line right to the photo edges.

In the function process_landmarks_and_draw_bbox I have vert_factor, scale_factor for customizing 
and aligning the face mesh.
"""
# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\NEWest.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# Get the width of the image
image_width = frame.shape[1]

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

def bounding_box(landmarks):
    global min_y, max_y,max_y_region1,y_coords_max,y_coords_min,min_y_region2
    x_coords_min = [landmark[0] for landmark in landmarks]
    y_coords_min = [landmark[1] for landmark in landmarks]
    x_coords_max = [landmark[2] for landmark in landmarks]
    y_coords_max = [landmark[3] for landmark in landmarks]
    min_x = int(min(x_coords_min))
    min_y = int(min(y_coords_min))
    max_x = int(max(x_coords_max))
    max_y = int(max(y_coords_max))
    max_y_region1 = int(max([y for y in y_coords_max if y < max_y and y>= int(min(y_coords_max))]))
    min_y_region2 = int(max(y_coords_min))
    return [[min_x, min_y, max_x, max_y_region1], [min_x, min_y_region2, max_x, max_y]]

# Function to process landmarks and draw bounding box
def process_landmarks_and_draw_bbox(landmarks, frame, vert_factor, scale_factor):
    global min_x, min_y, max_x, max_y
    # Calculate centroid
    centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
    centroid_y = sum(point[1] for point in landmarks) / len(landmarks)

    # Scale factor
    # Move the landmarks outward from the centroid
    landmarks_scaled = [
        ((point[0] - centroid_x) * scale_factor + centroid_x, 
         (point[1] - centroid_y) * scale_factor + centroid_y - vert_factor)
        for point in landmarks
    ]

    # Prepare for drawing
    pts = np.array(landmarks_scaled, np.int32)
    pts = pts.reshape((-1,1,2))

    # Draw a bounding box around the landmarks area
    min_x = int(min(point[0] for point in landmarks_scaled))
    max_x = int(max(point[0] for point in landmarks_scaled))
    min_y = int(min(point[1] for point in landmarks_scaled))
    max_y = int(max(point[1] for point in landmarks_scaled))
    if min_x > 0:
        min_x = 0
    if max_x < image_width:
        max_x = image_width
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue color

    # Optionally, draw the landmarks on the image
    for point in landmarks_scaled:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color

# List of landmark contours for each feature
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]
left_eye_contour = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), 
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), 
    (398, 362)
    
]
left_eyebrow_contour = [
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)
]
right_eye_contour = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), 
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)]
right_eyebrow_contour = [
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
]

# Extract landmarks coordinates for each feature
landmarks_dict = {}
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            for contour_name, contour in zip(["lips", "left_eye",'left_eyebrow', "right_eye",'left_eyebrow'],
                                            [lip_contours, left_eye_contour, left_eyebrow_contour, right_eye_contour, right_eyebrow_contour]):
                if i in [idx for pair in contour for idx in pair]:
                    if contour_name not in landmarks_dict:
                        landmarks_dict[contour_name] = []
                    landmarks_dict[contour_name].append((lm.x * frame.shape[1], lm.y * frame.shape[0]))


vert_dict = {'lips': 5, 'left_eye': 5, 'left_eyebrow': 0, "right_eye": 5, "right_eyebrow":0}
scale_dict = {'lips': 1.2, 'left_eye': 2.2, 'left_eyebrow': 1.14, "right_eye": 2.2, "right_eyebrow":1.14}
# Process each set of landmarks and draw bounding box separately
for contour_name, landmarks in landmarks_dict.items():
    process_landmarks_and_draw_bbox(landmarks, frame,vert_dict[contour_name],scale_dict[contour_name])
box_coordinates = []
for contour_name, landmarks in landmarks_dict.items():
    process_landmarks_and_draw_bbox(landmarks, frame, vert_dict[contour_name], scale_dict[contour_name])
    box_coordinates.append((min_x, min_y, max_x, max_y))

           
               
        # Extract bounding box and append the bounding box area to sections list
A = bounding_box(box_coordinates)
A[0][0],A[1][0] = 0,0
section_coords = [A[0], A[1]]
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\Yes.JPG"

# Save the modified image
cv2.imwrite(output_image_path, frame)




# In[ ]:


# Create the white spacer
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\NEWest.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)
spacer_height = section_coords[1][1] - section_coords[0][3]  # Correct height calculation
spacer_width = frame.shape[1]  # Use full width of the image
spacer = np.ones((spacer_height, spacer_width, 3), dtype=np.uint8) * 255  # White spacer

# Crop the sections between the white space
cropped_sections = []
for idx, coords in enumerate(section_coords):
    section = frame[coords[1]:coords[3], coords[0]:coords[2]]
    cropped_sections.append(section)
    cv2.imwrite(fr"C:\Users\maorb\Desktop\Work\sad_2\cropped_sections3_{idx}.jpg", section)  # Saving each cropped section

# Concatenate sections and spacer to create the modified image
top_section = frame[section_coords[0][1]:section_coords[0][3], :]
bottom_section = frame[section_coords[1][1]:section_coords[1][3], :]
modified_image = np.vstack([top_section, spacer, bottom_section])

# Save the modified image with the white spacer
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\FinalP.jpeg"  # Example name, replace with actual path
cv2.imwrite(output_image_path, modified_image)




# In[ ]:


import cv2

# Read the image
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"  # Modify this path

# Read the input image
frame = cv2.imread(input_image_path)
# Define the region of interest (ROI) for cropping
x1, y1, x2, y2 = 0, 46, 714, 228

# Crop the image
crop_img = frame[240:514, x1:x2]

# Display the cropped image
cv2.imshow("Cropped Image", crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


get_ipython().system('conda activate base')


# In[ ]:


import cv2
import numpy as np

# Load the image
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"
frame = cv2.imread(input_image_path)
if frame is None:
    raise ValueError("Image not found")

# Define the coordinates for the sections between the white space
section_coords = [A[0], A[1]]

spacer_height = section_coords[1][1] - section_coords[0][3]  # Correct height calculation
spacer_width = frame.shape[1]  # Use full width of the image
spacer = np.ones((spacer_height, spacer_width, 3), dtype=np.uint8) * 255  # White spacer

# Crop the sections between the white space
cropped_sections = []
for idx, coords in enumerate(section_coords):
    section = frame[coords[1]:coords[3], coords[0]:coords[2]]
    cropped_sections.append(section)
    cv2.imwrite(f"cropped_sectionss_{idx}.jpg", section)  # Saving each cropped section

# Concatenate sections and spacer to create the modified image
top_section = frame[section_coords[0][1]:section_coords[0][3], :]
bottom_section = frame[section_coords[1][1]:section_coords[1][3], :]
modified_image = np.vstack([top_section, spacer, bottom_section])
# Save the modified image with the white spacer
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\modifiedYes.jpeg"
cv2.imwrite(output_image_path, modified_image)


# In[ ]:


import cv2
import numpy as np

# Load the image
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"
frame = cv2.imread(input_image_path)
if frame is None:
    raise ValueError("Image not found")

# Define the coordinates for the sections between the white space
section_coords = [[0, 46, 714, 235], [0, 514, 714, 677]]

# Create the white spacer
spacer_coords = (235, 0)  # Coordinates for the white spacer
spacer_height = 677 - 228  # Height of the spacer
spacer_width = 714  # Width of the spacer
spacer = np.ones((spacer_height, spacer_width, 3), dtype=np.uint8) * 255  # White spacer

# Insert the white spacer at the specified coordinates
modified_image = np.hstack((frame[section_coords[0][1]:section_coords[0][3], section_coords[0][0]:section_coords[0][2]],
                            spacer,
                            frame[section_coords[1][1]:section_coords[1][3], section_coords[1][0]:section_coords[1][2]]))

# Crop the sections between the white space
cropped_sections = []
for coords in section_coords:
    section = frame[coords[1]:coords[3], coords[0]:coords[2]]
    cropped_sections.append(section)

# Save each cropped section
for i, section in enumerate(cropped_sections):
    output_path = f"C:\\Users\\maorb\\Desktop\\Work\\sad_2\\cropped_section_{i}.jpeg"
    cv2.imwrite(output_path, section)

# Save the modified image with the white spacer
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\modified_image_with_spacer.jpeg"
cv2.imwrite(output_image_path, modified_image)


# In[ ]:


import cv2
import numpy as np

# Load the image
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"  # Example name, replace with actual path
frame = cv2.imread(input_image_path)
if frame is None:
    raise ValueError("Image not found")

# Define the coordinates for the sections between the white space
section_coords = [[0, 46, 714, 235], [0, 514, 714, 677]]

# Create the white spacer
spacer_height = section_coords[1][1] - section_coords[0][3]  # Correct height calculation
spacer_width = frame.shape[1]  # Use full width of the image
spacer = np.ones((spacer_height, spacer_width, 3), dtype=np.uint8) * 255  # White spacer

# Crop the sections between the white space
cropped_sections = []
for idx, coords in enumerate(section_coords):
    section = frame[coords[1]:coords[3], coords[0]:coords[2]]
    cropped_sections.append(section)
    cv2.imwrite(fr"C:\Users\maorb\Desktop\Work\sad_2\cropped_sectionss_{idx}.jpg", section)  # Saving each cropped section

# Concatenate sections and spacer to create the modified image
top_section = frame[section_coords[0][1]:section_coords[0][3], :]
bottom_section = frame[section_coords[1][1]:section_coords[1][3], :]
modified_image = np.vstack([top_section, spacer, bottom_section])

# Save the modified image with the white spacer
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\Final.jpg"  # Example name, replace with actual path
cv2.imwrite(output_image_path, modified_image)


# In[ ]:


A


# In[ ]:


import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)
if frame is None:
    raise ValueError("Image not found")

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

# Create mask for regions to keep
mask = np.zeros(frame.shape[:2], dtype=np.uint8)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Extract landmarks for regions of interest
        lips = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark[61:68]]
        left_eye = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark[133:144]]
        right_eye = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark[362:373]]

        # Create a list of regions - skipping drawing code, focus on the vertices
        regions = [lips, left_eye, right_eye]

        # Fill the regions in the mask and frame
        for region in regions:
            cv2.fillPoly(mask, [np.array(region, dtype=np.int32)], (255))

# Apply the mask to the frame
masked_image = cv2.bitwise_and(frame, frame, mask=mask)

# Invert mask
inverse_mask = cv2.bitwise_not(mask)

# Apply white color on the inverse_mask area of the original frame
frame[inverse_mask == 255] = (255, 255, 255)

# Save the modified image
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\new_combined.jpeg"
cv2.imwrite(output_image_path, frame)

# Optionally save regions as new images
for index, region in enumerate(regions):
    region_image = cv2.polylines(frame.copy(), [np.array(region, dtype=np.int32)], isClosed=True, color=(255,0,0), thickness=1)
    x, y, w, h = cv2.boundingRect(np.array(region, dtype=np.int32))
    roi = region_image[y:y+h, x:x+w]
    cv2.imwrite(f"region_{index}.jpg", roi)

# Clear face_mesh resources
face_mesh.close()


# In[ ]:


regions


# In[ ]:


import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"  # Modify this path

# Read the input image
frame = cv2.imread(input_image_path)
if frame is None:
    raise ValueError("Image not found")

# Get the width of the image
image_width = frame.shape[1]

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

# Define the bounding box function
def bounding_box(landmarks):
    x_coords_min = [landmark[0] for landmark in landmarks]
    y_coords_min = [landmark[1] for landmark in landmarks]
    x_coords_max = [landmark[2] for landmark in landmarks]
    y_coords_max = [landmark[3] for landmark in landmarks]
    min_x = int(min(x_coords_min))
    min_y = int(min(y_coords_min))
    max_x = int(max(x_coords_max))
    max_y = int(max(y_coords_max))
    max_y_region1 = int(max([y for y in y_coords_max if y < max_y and y >= int(min(y_coords_max))]))
    min_y_region2 = int(max(y_coords_min))
    return [[min_x, min_y, max_x, max_y_region1], [min_x, min_y_region2, max_x, max_y]]

# Function to process landmarks and draw bounding box
def process_landmarks_and_draw_bbox(landmarks, frame, vert_factor, scale_factor):
    global min_x,max_x, min_y, max_y
    # Calculate centroid
    centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
    centroid_y = sum(point[1] for point in landmarks) / len(landmarks)

    # Scale factor
    # Move the landmarks outward from the centroid
    landmarks_scaled = [
        ((point[0] - centroid_x) * scale_factor + centroid_x, 
         (point[1] - centroid_y) * scale_factor + centroid_y - vert_factor)
        for point in landmarks
    ]

    # Prepare for drawing
    pts = np.array(landmarks_scaled, np.int32)
    pts = pts.reshape((-1,1,2))

    # Draw a bounding box around the landmarks area
    min_x = int(min(point[0] for point in landmarks_scaled))
    max_x = int(max(point[0] for point in landmarks_scaled))
    min_y = int(min(point[1] for point in landmarks_scaled))
    max_y = int(max(point[1] for point in landmarks_scaled))
    if min_x > 0:
        min_x = 0
    if max_x < image_width:
        max_x = image_width
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue color

    # Optionally, draw the landmarks on the image
    for point in landmarks_scaled:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color

# Define the vertical factor and scale factor dictionaries
vert_dict = {'lips': 5, 'left_eye': 5, 'left_eyebrow': 0, "right_eye": 5, "right_eyebrow": 0}
scale_dict = {'lips': 1.2, 'left_eye': 2.2, 'left_eyebrow': 1.14, "right_eye": 2.2, "right_eyebrow": 1.14}

# Initialize sections list
sections = []

# If face landmarks are detected, handle bounding boxes
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Extract landmarks coordinates for each feature
        landmarks_dict = {}
        for i, lm in enumerate(face_landmarks.landmark):
            for contour_name, contour in zip(["lips", "left_eye", "left_eyebrow", "right_eye", "right_eyebrow"],
                                             [lip_contours, left_eye_contour, left_eyebrow_contour, right_eye_contour, right_eyebrow_contour]):
                if i in [idx for pair in contour for idx in pair]:
                    if contour_name not in landmarks_dict:
                        landmarks_dict[contour_name] = []
                    landmarks_dict[contour_name].append((lm.x * frame.shape[1], lm.y * frame.shape[0]))

        # Process each set of landmarks and draw bounding box separately
        for contour_name, landmarks in landmarks_dict.items():
            process_landmarks_and_draw_bbox(landmarks, frame, vert_dict[contour_name], scale_dict[contour_name])
            sections.extend([frame[min_y:max_y, min_x:max_x] for min_x, min_y, max_x, max_y in bounding_box(landmarks)])

# If there are multiple sections, combine them with a white spacer
if len(sections) > 1:
    # Find the dimensions to create a spacer
    max_height = max(section.shape[0] for section in sections)
    spacer = np.ones((max_height, 10, 3), dtype=np.uint8) * 255  # White spacer

    # Combine sections with a spacer
    final_image = np.hstack([sections[0], spacer] + sections[1:]) # Using only the first and the last section here
else:
    final_image = sections[0] if sections else frame

# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\new_combined.jpeg"  # Modify this path

# Save the modified image
cv2.imwrite(output_image_path, final_image)

# Release resources 
face_mesh.close()


# Boxes for each feature

# In[ ]:


import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

# Function to process landmarks and draw bounding box
def process_landmarks_and_draw_bbox(landmarks, frame,vert_factor, scale_factor):
    # Calculate centroid
    centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
    centroid_y = sum(point[1] for point in landmarks) / len(landmarks)

    # Scale factor  # Adjust the scale factor to increase/reduce the stretching

    # Move the landmarks outward from the centroid
    landmarks_scaled = [
        ((point[0] - centroid_x) * scale_factor + centroid_x, 
         (point[1] - centroid_y) * scale_factor + centroid_y - vert_factor)
        for point in landmarks
    ]

    # Prepare for drawing
    pts = np.array(landmarks_scaled, np.int32)
    pts = pts.reshape((-1,1,2))

    # Draw a bounding box around the landmarks area
    min_x = int(min(point[0] for point in landmarks_scaled))
    max_x = int(max(point[0] for point in landmarks_scaled))
    min_y = int(min(point[1] for point in landmarks_scaled))
    max_y = int(max(point[1] for point in landmarks_scaled))
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue color

    # Optionally, draw the landmarks on the image
    for point in landmarks_scaled:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color

# List of landmark contours for each feature
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]
left_eye_contour = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), 
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), 
    (398, 362)
    
]
left_eyebrow_contour = [
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)
]
right_eye_contour = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), 
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)]
right_eyebrow_contour = [
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
]

# Extract landmarks coordinates for each feature
landmarks_dict = {}
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            for contour_name, contour in zip(["lips", "left_eye",'left_eyebrow', "right_eye",'left_eyebrow'],
                                            [lip_contours, left_eye_contour, left_eyebrow_contour, right_eye_contour, right_eyebrow_contour]):
                if i in [idx for pair in contour for idx in pair]:
                    if contour_name not in landmarks_dict:
                        landmarks_dict[contour_name] = []
                    landmarks_dict[contour_name].append((lm.x * frame.shape[1], lm.y * frame.shape[0]))


vert_dict = {'lips': 5, 'left_eye': 5, 'left_eyebrow': 0, "right_eye": 5, "right_eyebrow":0}
scale_dict = {'lips': 1.2, 'left_eye': 2.2, 'left_eyebrow': 1.14, "right_eye": 2.2, "right_eyebrow":1.14}
# Process each set of landmarks and draw bounding box separately
for contour_name, landmarks in landmarks_dict.items():
    process_landmarks_and_draw_bbox(landmarks, frame,vert_dict[contour_name],scale_dict[contour_name])

# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\maorMultipleBoxes.jpeg"

# Save the modified image
cv2.imwrite(output_image_path, frame)

# Release resources
face_mesh.close()


# With adaptive boxes

# In[ ]:


import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

# Function to process landmarks and draw bounding box
def process_landmarks_and_draw_bbox(landmarks, frame,vert_factor, scale_factor):
    # Calculate centroid
    centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
    centroid_y = sum(point[1] for point in landmarks) / len(landmarks)

    # Scale factor  # Adjust the scale factor to increase/reduce the stretching

    # Move the landmarks outward from the centroid
    landmarks_scaled = [
        ((point[0] - centroid_x) * scale_factor + centroid_x, 
         (point[1] - centroid_y) * scale_factor + centroid_y - vert_factor)
        for point in landmarks
    ]

    # Prepare for drawing
    pts = np.array(landmarks_scaled, np.int32)
    pts = pts.reshape((-1,1,2))

    # Draw a bounding box around the landmarks area
    min_x = int(min(point[0] for point in landmarks_scaled))
    max_x = int(max(point[0] for point in landmarks_scaled))
    min_y = int(min(point[1] for point in landmarks_scaled))
    max_y = int(max(point[1] for point in landmarks_scaled))
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)  # Blue color
    # Optionally, draw the landmarks on the image
    for point in landmarks_scaled:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color

# List of landmark contours for each feature
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]
left_eye_contour = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), 
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362)
]
left_eyebrow_contour = [
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)
]
right_eye_contour = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), 
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)
]
right_eyebrow_contour = [
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
]

# Extract landmarks coordinates for each feature
landmarks_dict = {}
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            for contour_name, contour in zip(["lips", "left_eye", "left_eyebrow", "right_eye", "right_eyebrow"],
                                            [lip_contours, left_eye_contour, left_eyebrow_contour, right_eye_contour, right_eyebrow_contour]):
                if i in [idx for pair in contour for idx in pair]:
                    if contour_name not in landmarks_dict:
                        landmarks_dict[contour_name] = []
                    landmarks_dict[contour_name].append((lm.x * frame.shape[1], lm.y * frame.shape[0]))


vert_dict = {'lips': 5, 'left_eye': 5, 'left_eyebrow': 0, "right_eye": 5, "right_eyebrow":0}
scale_dict = {'lips': 1.2, 'left_eye': 2, 'left_eyebrow': 0, "right_eye": 2, "right_eyebrow":0}
# Process each set of landmarks and draw bounding box separately
for contour_name, landmarks in landmarks_dict.items():
    process_landmarks_and_draw_bbox(landmarks, frame,vert_dict[contour_name],scale_dict[contour_name])

# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\maorMultipleBoxes.jpeg"

# Save the modified image
cv2.imwrite(output_image_path, frame)

# Release resources
face_mesh.close()


# In[ ]:


import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

# Function to process landmarks and draw bounding box
def process_landmarks_and_draw_bbox(landmarks, frame, vert_factor, scale_factor):
    # Calculate centroid
    centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
    centroid_y = sum(point[1] for point in landmarks) / len(landmarks)

    # Scale factor
    # Move the landmarks outward from the centroid
    landmarks_scaled = [
        ((point[0] - centroid_x) * scale_factor + centroid_x, 
         (point[1] - centroid_y) * scale_factor + centroid_y - vert_factor)
        for point in landmarks
    ]

    # Prepare for drawing
    pts = np.array(landmarks_scaled, np.int32)
    pts = pts.reshape((-1,1,2))

    # Draw a bounding box around the landmarks area
    min_x = int(min(point[0] for point in landmarks_scaled))
    max_x = int(max(point[0] for point in landmarks_scaled))
    min_y = int(min(point[1] for point in landmarks_scaled))
    max_y = int(max(point[1] for point in landmarks_scaled))
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue color

    # Optionally, draw the landmarks on the image
    for point in landmarks_scaled:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color

    return min_y, max_y

# List of landmark contours for each feature
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]
left_eye_contour = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), 
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), 
    (398, 362)
]
left_eyebrow_contour = [
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)
]
right_eye_contour = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), 
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)]
right_eyebrow_contour = [
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
]

# Extract landmarks coordinates for each feature
landmarks_dict = {}
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            for contour_name, contour in zip(["lips", "left_eye", "left_eyebrow", "right_eye", "right_eyebrow"],
                                            [lip_contours, left_eye_contour, left_eyebrow_contour, right_eye_contour, right_eyebrow_contour]):
                if i in [idx for pair in contour for idx in pair]:
                    if contour_name not in landmarks_dict:
                        landmarks_dict[contour_name] = []
                    landmarks_dict[contour_name].append((lm.x * frame.shape[1], lm.y * frame.shape[0]))

# Process each set of landmarks and draw bounding box separately
bbox_y_coords = {}
for contour_name, landmarks in landmarks_dict.items():
    min_y, max_y = process_landmarks_and_draw_bbox(landmarks, frame, 0, 1)
    bbox_y_coords[contour_name] = (min_y, max_y)

# Calculate the height of the upper and lower sections
upper_section_height = min(bbox_y_coords['left_eye'][0], bbox_y_coords['left_eyebrow'][0],
                            bbox_y_coords['right_eye'][0], bbox_y_coords['right_eyebrow'][0],
                            bbox_y_coords['lips'][0])
lower_section_height = max(bbox_y_coords['left_eye'][1], bbox_y_coords['left_eyebrow'][1],
                            bbox_y_coords['right_eye'][1], bbox_y_coords['right_eyebrow'][1],
                            bbox_y_coords['lips'][1])

# Create two separate images for each section
upper_section_image = frame[:upper_section_height, :]
lower_section_image = frame[lower_section_height:, :]

# Output image paths
output_image_path_upper = r"C:\Users\maorb\Desktop\Work\sad_2\maorUpperSection.jpeg"
output_image_path_lower = r"C:\Users\maorb\Desktop\Work\sad_2\maorLowerSection.jpeg"

# Save the upper and lower section images
cv2.imwrite(output_image_path_upper, upper_section_image)
cv2.imwrite(output_image_path_lower, lower_section_image)

# Release resources
face_mesh.close()


# In[ ]:


lips_landmarks + [(1,1)]


# In[ ]:


import os
import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

# List of landmark contours for the lips
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]

# Extract lips landmarks coordinates
lips_landmarks = []
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            if i in [idx for pair in lip_contours for idx in pair]:
                lips_landmarks.append((int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])))

# Define the lips area polygon
pts = np.array(lips_landmarks, np.int32)
pts = pts.reshape((-1,1,2))

# Expand the lips area polygon by dilating it
dilated_pts = cv2.convexHull(pts)
expand_factor = 10  # Adjust this value as needed
dilated_pts = cv2.convexHull(pts, False)
dilated_pts = cv2.dilate(dilated_pts, None, iterations=expand_factor)

# Fill the expanded lips area with a color (e.g., red)
cv2.fillPoly(frame, [dilated_pts], (0, 0, 255))

# Optionally, draw the lips landmarks on the image
for point in lips_landmarks:
    cv2.circle(frame, point, 3, (0, 255, 0), -1)  # Green color

# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\maor_with_expanded_lips_marked3.jpeg"

# Save the modified image
cv2.imwrite(output_image_path, frame)

# Release resources
face_mesh.close()


# In[ ]:


# Define the lips area polygon
pts = np.array(lips_landmarks, np.int32)
pts = pts.reshape((-1,1,2))

# Find the convex hull of the lips area
dilated_pts = cv2.convexHull(pts, False)

# Define the expansion factor
expand_factor = 10  # Adjust this value as needed

# Dilate the lips area polygon to expand the marking area
dilated_pts = cv2.dilate(dilated_pts, None, iterations=expand_factor)


# Fill the expanded lips area with a color (e.g., red)
cv2.fillPoly(frame, [dilated_pts], (0, 0, 255))

# Optionally, draw the lips landmarks on the image
for point in lips_landmarks:
    cv2.circle(frame, point, 3, (0, 255, 0), -1)  # Green color

# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\maor_with_expanded_lips_marked5.jpeg"

# Save the modified image
cv2.imwrite(output_image_path, frame)


# In[ ]:


lips_landmarks


# In[ ]:


import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Drawing specifications for landmarks and connections
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))
connection_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, color=(255, 0,0))

# Folder containing input images
input_folder = r'C:\Users\maorb\Desktop\Work\sad_2\photo_14.jpeg'
# Folder to save processed images
output_folder = r'C:\Users\maorb\Desktop\Work\sad_2\newjpg.jpg'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust file extensions as needed
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, filename)
        
        # Read the input image
        frame = cv2.imread(input_image_path)
        
        # Convert the image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect face landmarks
        results = face_mesh.process(frame_rgb)
        
        # Draw an ellipse around the face
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get bounding box coordinates of the face
                bbox = cv2.boundingRect(cv2.convexHull(cv2.UMat(cv2.boxPoints(cv2.minAreaRect(cv2.UMat(np.array([[lm.x, lm.y] for lm in face_landmarks])))).reshape(-1, 2).astype(np.int32))))
                # Draw the ellipse
                cv2.ellipse(frame, (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)), (int(bbox[2] / 2), int(bbox[3] / 2)), 0, 0, 360, (0, 255, 0), 2)
        
        # Save the output image
        cv2.imwrite(output_image_path, frame)

# Release resources
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:


import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Folder containing input images
input_folder = 'C:/Users/maorb/Desktop/Work'
# Folder to save processed images
output_folder = 'C:/Users/maorb/Desktop/Work/Processed7'

# Iterate over subdirectories in the input folder
for root, dirs, files in os.walk(input_folder):
    for subdir in dirs:
        sub_folder_path = os.path.join(root, subdir)
        output_sub_folder = os.path.join(output_folder, subdir)
        
        # Create output subfolder if it doesn't exist
        if not os.path.exists(output_sub_folder):
            os.makedirs(output_sub_folder)
        
        # Process each image in the subfolder
        for filename in os.listdir(sub_folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust file extensions as needed
                input_image_path = os.path.join(sub_folder_path, filename)
                output_image_path = os.path.join(output_sub_folder, filename)
                
                # Read the input image
                frame = cv2.imread(input_image_path)
                
                # Convert the image to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the image to detect face landmarks
                results = face_mesh.process(frame_rgb)
                
                # Extract landmarks if available
                if results.multi_face_landmarks:
                    # Extract only the first face landmarks
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Extract landmark points
                    landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]
                    
                    # Get bounding box around landmarks
                    x, y, w, h = cv2.boundingRect(np.array(landmarks))
                    
                    # Define bounding box around face and some hair area
                    x -= 20  # Add some margin
                    y -= 20
                    w += 40
                    h += 40
                    
                    # Ensure bounding box is within image boundaries
                    x = max(0, x)
                    y = max(0, y)
                    w = min(frame.shape[1] - x, w)
                    h = min(frame.shape[0] - y, h)
                    
                    # Crop the image to the bounding box
                    cropped_image = frame[y:y+h, x:x+w]
                    
                    # Save the cropped image
                    cv2.imwrite(output_image_path, cropped_image)
                else:
                    # If face detection fails, use an approximate cropping method
                    # Define bounding box around the center of the image
                    x = frame.shape[1] // 4
                    y = frame.shape[0] // 4
                    w = frame.shape[1] // 2
                    h = frame.shape[0] // 2
                    
                    # Crop the image to the bounding box
                    cropped_image = frame[y:y+h, x:x+w]
                    
                    # Save the cropped image
                    cv2.imwrite(output_image_path, cropped_image)

# Release resources
cv2.destroyAllWindows()


# Adjust photo based on defined limits

# In[ ]:


import cv2

def adaptive_equalize_histogram(image_path, clip_limit=3, grid_size=(8, 8)):
    image = cv2.imread(image_path)
    
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply adaptive histogram equalization to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    clahe_l = clahe.apply(l_channel)
    
    # Merge the equalized L channel with the original A and B channels
    equalized_lab_image = cv2.merge((clahe_l, a_channel, b_channel))
    
    # Convert the equalized LAB image back to BGR color space
    equalized_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2BGR)
    
    return equalized_image

image_path = r'C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG'
equalized_image = adaptive_equalize_histogram(image_path)
cv2.imwrite(r'C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\eq12.JPG', equalized_image)


# Adjust photo based on brightness_factor, contrast_factor

# In[ ]:


from PIL import Image, ImageEnhance

def adjust_brightness_contrast(image_path, brightness_factor, contrast_factor):
    image = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    return image

image_path = r'C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG'
adjusted_image = adjust_brightness_contrast(image_path, 0.3, 1)
adjusted_image.save(r'C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\eq12.JPG')


# In[ ]:


from PIL import Image, ImageEnhance

def match_brightness_contrast(reference_image_path, target_image_path):
    reference_image = Image.open(reference_image_path)
    target_image = Image.open(target_image_path)
    
    ref_brightness = ImageEnhance.Brightness(reference_image).enhance(1.0).convert('RGB')
    tar_brightness = ImageEnhance.Brightness(target_image).enhance(1.0).convert('RGB')
    
    ref_contrast = ImageEnhance.Contrast(reference_image).enhance(1.0).convert('RGB')
    tar_contrast = ImageEnhance.Contrast(target_image).enhance(1.0).convert('RGB')

    target_image_brightness = Image.blend(tar_brightness, ref_brightness, 0.5)
    target_image_contrast = Image.blend(tar_contrast, ref_contrast, 0.5)
    
    return target_image_brightness, target_image_contrast

reference_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19HAHR.JPG"
target_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG"
adjusted_brightness, adjusted_contrast = match_brightness_contrast(reference_image_path, target_image_path)

adjusted_brightness.save(r'C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\eqbr12.JPG')
adjusted_contrast.save(r'C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\eqco12.JPG')


# In[ ]:


from PIL import Image, ImageEnhance

def adjust_brightness(image_path, brightness_factor):
    image = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(image)
    adjusted_image = enhancer.enhance(brightness_factor)
    return adjusted_image

def adjust_contrast(image_path, contrast_factor):
    image = Image.open(image_path)
    enhancer = ImageEnhance.Contrast(image)
    adjusted_image = enhancer.enhance(contrast_factor)
    return adjusted_image

def combine_brightness_contrast(reference_image_path, target_image_path, brightness_factor, contrast_factor):
    reference_image = Image.open(reference_image_path)
    
    # Adjust brightness and contrast of the target image
    target_brightness = adjust_brightness(target_image_path, brightness_factor)
    target_contrast = adjust_contrast(target_image_path, contrast_factor)
    
    # Blend the adjusted target images with the reference image
    adjusted_brightness = Image.blend(target_brightness, reference_image, 0.5)
    adjusted_contrast = Image.blend(target_contrast, reference_image, 0.5)
    
    return adjusted_brightness, adjusted_contrast

reference_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19HAHR.JPG"
target_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG"
brightness_factor = 1.2  # Adjust as needed
contrast_factor = 1.5  # Adjust as needed

adjusted_brightness, adjusted_contrast = combine_brightness_contrast(reference_image_path, target_image_path, brightness_factor, contrast_factor)

adjusted_brightness.save(r'C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\adjusted_brightness.JPG')
adjusted_contrast.save(r'C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\adjusted_contrast.JPG')


# In[ ]:


from PIL import Image, ImageEnhance

def adjust_brightness_contrast_with_reference(reference_image_path, target_image_path):
    # Open the reference and target images
    reference_image = Image.open(reference_image_path).convert('L')  # Convert to grayscale for easier comparison
    target_image = Image.open(target_image_path).convert('L')

    # Calculate the mean brightness and contrast of each image
    reference_brightness = reference_image.getextrema()[0]
    target_brightness = target_image.getextrema()[0]
    reference_contrast = reference_image.getextrema()[1] - reference_brightness
    target_contrast = target_image.getextrema()[1] - target_brightness

    # Calculate the adjustment factors
    brightness_factor = reference_brightness / target_brightness
    contrast_factor = reference_contrast / target_contrast

    # Open the target image
    target_image = Image.open(target_image_path)

    # Adjust brightness and contrast of the target image
    brightness_enhancer = ImageEnhance.Brightness(target_image)
    adjusted_brightness = brightness_enhancer.enhance(brightness_factor)

    contrast_enhancer = ImageEnhance.Contrast(adjusted_brightness)
    adjusted_image = contrast_enhancer.enhance(contrast_factor)

    return adjusted_image

reference_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19HAHR.JPG"
target_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG"

adjusted_image = adjust_brightness_contrast_with_reference(reference_image_path, target_image_path)
adjusted_image.save(r'C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\adjusted_brightness_contrast.JPG')


# In[ ]:


import cv2
import numpy as np

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    return faces

# Load the first and second photos
photo1 = cv2.imread(r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19HAHR.JPG")
photo2 = cv2.imread(r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG")

# Convert photo1 to HSV color space
photo1_hsv = cv2.cvtColor(photo1, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for skin color in HSV
lower_skin = np.array([10, 40, 20], dtype=np.uint8)
upper_skin = np.array([25, 255, 255], dtype=np.uint8)

# Create the mask for skin color in photo1
mask = cv2.inRange(photo1_hsv, lower_skin, upper_skin)

# Apply face detection and use the detected faces as the mask
faces = detect_faces(photo1)
mask_faces = np.zeros_like(mask)
for (x, y, w, h) in faces:
    mask_faces[y:y+h, x:x+w] = 255

# Apply the mask to the second photo
masked_photo2 = cv2.bitwise_and(photo2, photo2, mask=mask_faces)

# Blend the skin color regions from photo1 with photo2
result = cv2.addWeighted(photo1, 0.5, masked_photo2, 0.5, 0)

# Display or save the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


mask


# In[ ]:


import cv2
import numpy as np

def detect_face(image):
    # Detect faces using a face detection library
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    
    # Assuming a single face for simplicity; handle multiple faces as needed
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return (x, y, w, h)
    else:
        return None

def create_mask_for_face(image, face_coords):
    # Create a mask for the detected face
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if face_coords:
        x, y, w, h = face_coords
        mask[y:y+h, x:x+w] = 255
    return mask

def apply_histogram_matching(src_image, ref_image, mask=None):
    # Apply histogram matching separately to each channel
    matched_image = np.zeros_like(src_image)
    for channel in range(src_image.shape[2]):
        if mask is not None:
            # Apply histogram matching only to the masked region
            matched_channel = cv2.equalizeHist(src_image[:,:,channel], mask=mask)
        else:
            # Apply histogram matching to the entire image
            matched_channel = cv2.equalizeHist(src_image[:,:,channel])
        matched_image[:,:,channel] = matched_channel
    return matched_image

# Paths to the reference and target images
reference_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19HAHR.JPG"
target_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG"

# Load images
reference_image = cv2.imread(reference_image_path)
target_image = cv2.imread(target_image_path)

# Detect face in the target image
face_coords = detect_face(target_image)

# Create mask for the detected face
face_mask = create_mask_for_face(target_image, face_coords)
background_mask = np.bitwise_not(face_mask)
# Apply histogram matching with masking
adjusted_image = apply_histogram_matching(target_image, reference_image, background_mask)

# Path to save the adjusted image
adjusted_image_path = r'C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\adjusted_brightness_contrast.JPG'

# Save the adjusted image
cv2.imwrite(adjusted_image_path, adjusted_image)
# Load images, convert to RGB, detect faces, create masks
# You'd follow similar loading and conversion steps as in the original question
# face_coords = detect_face(target_image)
# face_mask = create_mask_for_face(target_image, face_coords)
# background_mask = np.bitwise_not(face_mask)

# Adjust the codes above into your workflow to match histograms separately for face and background
# This demonstrates the concept; practically, you would need intensive testing and refinement


# In[ ]:


import cv2
import numpy as np

# Function to detect faces in an image
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    return faces

# Function to compute the histogram of the skin color in a region of interest (ROI)
def compute_skin_histogram(image, roi):
    global hist,h,s,mask
    # Convert ROI to HSV color space
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define range for skin color in HSV
    lower_skin = np.array([10, 60, 20], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    
    # Create a mask for skin color
    mask = cv2.inRange(hsv,(10, 60, 20), (25, 255, 255) )
    
    # Compute histogram of skin color
    hist = np.zeros((180, 256), dtype=np.uint8)
    for i in range(roi_hsv.shape[0]):
        for j in range(roi_hsv.shape[1]):
            h = roi_hsv[i, j, 0]
            s = roi_hsv[i, j, 1]
            if lower_skin[0] <= h <= upper_skin[0] and lower_skin[1] <= s <= upper_skin[1]:
                hist[h, s] += 1
    
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    
    return hist


# Function to match the skin color of faces in two images
def match_skin_color(target_image, reference_image):
    global hist_t, hist_r, roi_t, roi_r
    # Detect faces in both images
    target_faces = detect_faces(target_image)
    reference_faces = detect_faces(reference_image)
    
    # Iterate over detected faces in the target image
    for (x_t, y_t, w_t, h_t) in target_faces:
        # Extract ROI (face region) from the target image
        roi_t = target_image[y_t:y_t+h_t, x_t:x_t+w_t]
        
        # Compute histogram of skin color in the target ROI
        hist_t = compute_skin_histogram(target_image, roi_t)
        
        # Iterate over detected faces in the reference image
        for (x_r, y_r, w_r, h_r) in reference_faces:
            # Extract ROI (face region) from the reference image
            roi_r = reference_image[y_r:y_r+h_r, x_r:x_r+w_r]
            
            # Compute histogram of skin color in the reference ROI
            hist_r = compute_skin_histogram(reference_image, roi_r)
            
            # Apply histogram matching to match skin color
            matched_skin = cv2.calcBackProject([roi_r], [0, 1], hist_r, [0, 180, 0, 256], 1)
            
            # Check size and type of matched_skin array
            if matched_skin.shape[:2] == roi_t.shape[:2]:
                matched_skin = matched_skin.astype(np.uint8)
                matched_skin = cv2.bitwise_and(roi_t, roi_t, mask=matched_skin)
                
                # Replace the skin color in the target ROI with matched skin color
                target_image[y_t:y_t+h_t, x_t:x_t+w_t] = matched_skin
    
    return target_image

# Load target and reference images
target_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG"
reference_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19HAHR.JPG"

target_image = cv2.imread(target_image_path)
reference_image = cv2.imread(reference_image_path)

# Match skin color of faces in the target image to the reference image
matched_image = match_skin_color(target_image, reference_image)
adjusted_image_path = r'C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\adjusted_brightness_contrast.JPG'

# Save the adjusted image
cv2.imwrite(adjusted_image_path, matched_image)
# Display the matched image
#cv2.imshow('Matched Image', matched_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    return faces
# Load the first and second photos
photo2 = cv2.imread(r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG")
photo1 = cv2.imread(r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19HAHR.JPG")
faces_photo1 = detect_faces(photo1)
faces_photo2 = detect_faces(photo2)
def get_skin_mask(roi):
    # Define lower and upper bounds for skin color in HSV
    lower_skin = np.array([10, 60, 20], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    
    # Convert ROI to HSV color space
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Create mask for skin color
    mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)
    return mask

# Blend the skin color regions from photo1 with photo2
for (x, y, w, h) in faces_photo2:
    # Extract the face region from the second photo
    face_roi_photo2 = photo2[y:y+h, x:x+w]
    
    # Get skin color mask for the face region in the second photo
    skin_mask = get_skin_mask(face_roi_photo2)
    
    # Apply skin color mask to the face region in the first photo
    face_roi_photo1 = photo1[y:y+h, x:x+w]
    masked_face_photo1 = cv2.bitwise_and(face_roi_photo1, face_roi_photo1, mask=skin_mask)
    
    # Blend the skin color regions from the first photo with the corresponding regions in the second photo
    photo2[y:y+h, x:x+w] = cv2.addWeighted(face_roi_photo2, 0.5, masked_face_photo1, 0.5, 0)

# Display or save the result
cv2.imshow('Result', photo2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np



# Function to detect faces in an image
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    return faces

# Load the target image
target_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19HAHR.JPG"
target_image = cv2.imread(target_image_path)



# Detect faces in the darkened image
faces = detect_faces(target_image)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(darkened_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the result
cv2.imshow("Darkened Image with Face Detection", darkened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np

# Function to detect faces in an image
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    return faces

# Function to compute the histogram of the skin color in a region of interest (ROI)
def compute_skin_histogram(image, mask):
    # Convert image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Compute histogram of skin color using the provided mask
    hist = cv2.calcHist([image_hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    
    return hist

# Function to match the skin color of faces in two images
def match_skin_color(target_image, reference_image):
    global hist_t, hist_r, roi_t, roi_r,target_faces

    # Detect faces in both images
    target_faces = detect_faces(target_image)
    reference_faces = detect_faces(reference_image)
    
    # Iterate over detected faces in the target image
    for (x_t, y_t, w_t, h_t) in target_faces:
        # Extract ROI (face region) from the target image
        roi_t = target_image[y_t:y_t+h_t, x_t:x_t+w_t]
        
        # Compute histogram of skin color in the target ROI
        hist_t = compute_skin_histogram(target_image, roi_t)
        
        # Iterate over detected faces in the reference image
        for (x_r, y_r, w_r, h_r) in reference_faces:
            # Extract ROI (face region) from the reference image
            roi_r = reference_image[y_r:y_r+h_r, x_r:x_r+w_r]
            
            # Compute histogram of skin color in the reference ROI
            hist_r = compute_skin_histogram(reference_image, roi_r)
            
            # Apply histogram matching to match skin color
            matched_skin = cv2.calcBackProject([roi_r], [0, 1], hist_r, [0, 180, 0, 256], 1)
            
            # Check size and type of matched_skin array
            if matched_skin.shape[:2] == roi_t.shape[:2]:
                matched_skin = matched_skin.astype(np.uint8)
                matched_skin = cv2.bitwise_and(roi_t, roi_t, mask=matched_skin)
                
                # Replace the skin color in the target ROI with matched skin color
                target_image[y_t:y_t+h_t, x_t:x_t+w_t] = matched_skin
    
    return target_image

# Load target and reference images
target_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG"
reference_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19HAHR.JPG"

target_image = cv2.imread(target_image_path)
reference_image = cv2.imread(reference_image_path)

# Match skin color of faces in the target image to the reference image
matched_image = match_skin_color(target_image, reference_image)

# Display the matched image
cv2.imshow('Matched Image', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread(r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG").astype(np.float32)/255
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)


# In[ ]:


gamma = 5
gamma_img = np.power(img,gamma)
plt.imshow(gamma_img)


# In[ ]:


img = cv2.imread(r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv,(10, 60, 20), (25, 255, 255) )
cv2.imshow("orange", mask);cv2.waitKey();cv2.destroyAllWindows()


# In[ ]:


import cv2

def equalize_histogram_color(image_path):
    image = cv2.imread(image_path)
   
    
    # Split the image into its BGR channels
    b, g, r = cv2.split(image)

    # Equalize the histogram of each channel separately
    b_equalized = cv2.equalizeHist(b)
    g_equalized = cv2.equalizeHist(g)
    r_equalized = cv2.equalizeHist(r)

    # Merge the equalized channels back into an RGB image
    equalized_image = cv2.merge((b_equalized, g_equalized, r_equalized))
    
    return equalized_image

image_path =  r'C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG'
equalized_image = equalize_histogram_color(image_path)
cv2.imwrite(r'C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\eq12.JPG', equalized_image)


# Create Excel for Psychopy

# # Convert PPT slides to jpg and save

# In[ ]:


import os
import win32com.client

# Path to the PowerPoint file
pptx_path = r'C:\Users\maorb\Desktop\Experiment Builder\Emotions.pptx'

# Output folder for saving the JPG images
output_folder = r'C:\Users\maorb\Desktop\Experiment Builder\stimuli'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open PowerPoint application
Application = win32com.client.Dispatch("PowerPoint.Application")
Presentation = Application.Presentations.Open(pptx_path)

# Iterate through each slide and save them as JPG images
for i, slide in enumerate(Presentation.Slides):
    image_path = os.path.join(output_folder, f'slide_{i + 1}.jpg')  # Output file path
    slide.Export(image_path, "JPG")  # Export slide as JPG image

# Close PowerPoint application
Application.Quit()
Presentation = None
Application = None


# # Creating Csv of photos and emotions path based on key words

# In[ ]:


#Liron's Photos
import os
import random
import pandas as pd

# Path to the directory containing the images
image_path = r'C:\Users\maorb\Desktop\Experiment Builder\stimuli\Liron Photos'

# List all directories inside the image path
photo_folders = [folder for folder in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, folder)) and folder != 'Emotions']

# Path to the Emotions folder
emotion_path = r'C:\Users\maorb\Desktop\Experiment Builder\stimuli\Emotions'

# List of emotions
emotions = ['Angry', 'Disgusted', 'Happy', 'Sad', 'Surprised', 'Scared']  # Add more if needed

# Initialize DataFrame
data = {'Emotion': [], 'Photo': [], 'Congruity': [], 'Correct Answer': []}
congruity = [1] * 33 + [0] * 33
random.shuffle(congruity)
# Randomly select images and assign congruity
for i in range(64):
    congruity1 = congruity[i]
    # For Congruity 0, select a random photo and emotion that do not match
    if congruity1 == 0:
        # Randomly select a photo and an emotion with non-matching names
        photo_folder = random.choice(photo_folders)
        photo = random.choice(os.listdir(os.path.join(image_path, photo_folder)))
        photo_path = os.path.join(image_path, photo_folder, photo)
        correct_emotion = photo_folder.split('_')[0].capitalize()
        non_matching_emotions = [e for e in emotions if e.lower() not in photo.lower()]
        emotion = random.choice(non_matching_emotions)
        emotion_file = f"{emotion}.jpg"
        emotion_file_path = os.path.join(emotion_path, emotion_file)
    
    # For Congruity 1, select a random photo and match it with the correct emotion
    else:
        photo_folder = random.choice(photo_folders)
        photo = random.choice(os.listdir(os.path.join(image_path, photo_folder)))
        photo_path = os.path.join(image_path, photo_folder, photo)
        
        correct_emotion = photo_folder.split('_')[0].capitalize()
        emotion_file = f"{correct_emotion}.jpg"
        emotion_file_path = os.path.join(emotion_path, emotion_file)
    
    # Add data to DataFrame
    data['Emotion'].append(emotion_file_path)
    data['Photo'].append(photo_path)
    data['Congruity'].append(congruity1)
    data['Correct Answer'].append(correct_emotion)

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
csv_path = os.path.join(image_path, 'CSV_exp1.csv')
df.to_csv(csv_path, index=False)


# In[ ]:


#KDEF Photos:
import os
import random
import pandas as pd

# Path to the directory containing the images
image_path = r'C:\Users\maorb\Desktop\Experiment Builder\stimuli\KDEF Photos'

# List all directories inside the image path
photo_folders = [folder for folder in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, folder)) and folder != 'Emotions']

# Path to the Emotions folder
emotion_path = r'C:\Users\maorb\Desktop\Experiment Builder\stimuli\Emotions'

# List of emotions
emotions = ['Angry', 'Disgusted', 'Happy', 'Sad', 'Surprised', 'Scared']  # Add more if needed

# Initialize DataFrame
data = {'Emotion': [], 'Photo': [], 'Congruity': [], 'Correct_Answer': []}
congruity = [1] * 105 + [0] * 105
random.shuffle(congruity)
# Randomly select images and assign congruity
for i in range(210):
    congruity1 = congruity[i]
    # For Congruity 0, select a random photo and emotion that do not match
    if congruity1 == 0:
        # Randomly select a photo and an emotion with non-matching names
        photo_folder = random.choice(photo_folders)
        photo = random.choice(os.listdir(os.path.join(image_path, photo_folder)))
        photo_path = os.path.join(image_path, photo_folder, photo)
        correct_emotion = photo_folder.split('_')[0].capitalize()
        non_matching_emotions = [e for e in emotions if e.lower() not in correct_emotion.lower()]
        emotion = random.choice(non_matching_emotions)
        emotion_file = f"{emotion}.jpg"
        emotion_file_path = os.path.join(emotion_path, emotion_file)
    
    # For Congruity 1, select a random photo and match it with the correct emotion
    else:
        photo_folder = random.choice(photo_folders)
        photo = random.choice(os.listdir(os.path.join(image_path, photo_folder)))
        photo_path = os.path.join(image_path, photo_folder, photo)
        
        correct_emotion = photo_folder.split('_')[0].capitalize()
        emotion_file = f"{correct_emotion}.jpg"
        emotion_file_path = os.path.join(emotion_path, emotion_file)
    
    # Add data to DataFrame
    data['Emotion'].append(emotion_file_path)
    data['Photo'].append(photo_path)
    data['Congruity'].append(congruity1)
    data['Correct_Answer'].append(correct_emotion)

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
csv_path = os.path.join(image_path, 'CSV_KDEF5.csv')
df.to_csv(csv_path, index=False)


# In[ ]:


import os
import random
import pandas as pd

# Path to the directory containing the images
image_path = r'C:\Users\maorb\Desktop\Experiment Builder\stimuli\KDEF Photos'

# List all directories inside the image path
photo_folders = [folder for folder in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, folder)) and folder != 'Emotions']

# Path to the Emotions folder
emotion_path = r'C:\Users\maorb\Desktop\Experiment Builder\stimuli\Emotions'

# List of emotions
emotions = ['Angry', 'Disgusted', 'Happy', 'Sad', 'Surprised', 'Scared']  # Add more if needed

# Initialize DataFrame
data = {'Emotion': [], 'Photo': [], 'Congruity': [], 'Correct_Answer': []}
congruity = [1] * 105 + [0] * 105
random.shuffle(congruity)
# Initialize list to keep track of selected photos
selected_photos = []
# Randomly select images and assign congruity
for i in range(210):
    congruity1 = congruity[i]
    # For Congruity 0, select a random photo and emotion that do not match
    if congruity1 == 0:
        # Randomly select a photo folder
        photo_folder = random.choice(photo_folders)
        # Get the list of photos in the selected folder
        available_photos = [photo for photo in os.listdir(os.path.join(image_path, photo_folder)) if photo not in selected_photos]
        # Select a random photo from the available choices
        photo = random.choice(available_photos)
        photo_path = os.path.join(image_path, photo_folder, photo)
        correct_emotion = photo_folder.split('_')[0].capitalize()
        non_matching_emotions = [e for e in emotions if e.lower() not in correct_emotion.lower()]
        emotion = random.choice(non_matching_emotions)
        emotion_file = f"{emotion}.jpg"
        emotion_file_path = os.path.join(emotion_path, emotion_file)
    
    # For Congruity 1, select a random photo and match it with the correct emotion
    else:
        # Randomly select a photo folder
        photo_folder = random.choice(photo_folders)
        # Get the list of photos in the selected folder
        available_photos = [photo for photo in os.listdir(os.path.join(image_path, photo_folder)) if photo not in selected_photos]
        # Select a random photo from the available choices
        photo = random.choice(available_photos)
        photo_path = os.path.join(image_path, photo_folder, photo)
        
        correct_emotion = photo_folder.split('_')[0].capitalize()
        emotion_file = f"{correct_emotion}.jpg"
        emotion_file_path = os.path.join(emotion_path, emotion_file)
    
    # Add selected photo to the list
    selected_photos.append(photo)
    
    # Add data to DataFrame
    data['Emotion'].append(emotion_file_path)
    data['Photo'].append(photo_path)
    data['Congruity'].append(congruity1)
    data['Correct_Answer'].append(correct_emotion)

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
csv_path = os.path.join(image_path, 'CSV_KDEFfi.csv')
df.to_csv(csv_path, index=False)


# # Organize photos in set folder based on letter

# In[ ]:


import os
import shutil

# Path to the kdef directory
kdef_path = r'C:\Users\maorb\Desktop\Experiment Builder\KDEF'

# Path to the stimuli directory where emotion folders will be created
stimuli_path = r'C:\Users\maorb\Desktop\Experiment Builder\stimuli'

# Mapping of emotion codes to emotion folder names
emotion_mapping = {
    'AF': 'Scared',
    'AN': 'Angry',
    'HA': 'Happy',
    'SU': 'Surprised',
    'SA': 'Sad',
    'NE': 'Nervous',
    'DI': 'Disgusted'
}

# Function to extract emotion code from photo name
# Function to extract emotion code from photo name
def extract_emotion_code(photo_name):
    if photo_name.endswith('S.JPG'):
        return photo_name[-7:-5]
    return None


# Iterate through folders in the kdef directory
for folder_name in os.listdir(kdef_path):
    folder_path = os.path.join(kdef_path, folder_name)
    
    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Iterate through files in the folder
        for photo_name in os.listdir(folder_path):
            photo_path = os.path.join(folder_path, photo_name)
            
            # Extract emotion code from photo name
            emotion_code = extract_emotion_code(photo_name)
            print(emotion_code)
            # Debug print
            
            # If an emotion code is found and it exists in the mapping
            if emotion_code in emotion_mapping:
                # Determine the emotion folder
                emotion_folder = emotion_mapping[emotion_code]
                emotion_folder_path = os.path.join(stimuli_path, emotion_folder)
                
                # Debug print
                print(f'Emotion Folder: {emotion_folder}, Emotion Folder Path: {emotion_folder_path}')
                
                # Create the emotion folder if it doesn't exist
                if not os.path.exists(emotion_folder_path):
                    os.makedirs(emotion_folder_path)
                
                # Move the photo to the emotion folder
                shutil.move(photo_path, os.path.join(emotion_folder_path, photo_name))


# In[ ]:


import os 
stimuli_path = r'C:\Users\maorb\Desktop\Experiment Builder\stimuli'

relative_path = os.path.join('.', 'stimuli')
relative_path


# # Creating a csv from extracted specific details in excel

# In[ ]:


csv_path = 'C:/Users/maorb/CSVs/Inspire.xls'
time1_df = pd.read_excel(csv_path, sheet_name='InspireTime1+2')
time2_df = pd.read_excel(csv_path, sheet_name='Time2')
semifinal_df = pd.read_excel(csv_path, sheet_name='My_tab')

# Extract required columns
team_names = time1_df.iloc[3:]['Unnamed: 98']
venture_names = time2_df.iloc[2:]['Q41']
semi_teams = semifinal_df.iloc[:,0]
semi_grade = semifinal_df.iloc[:,1]
teamNames = list(set(team_names) | set(venture_names))
data = {'Team name': teamNames, 'Answered_1': ['Yes' if name in list(team_names) else 'No' for name in teamNames],
        'Answered_2': ['Yes' if name in list(venture_names) else 'No' for name in teamNames]}

semi_rank = []
for name in teamNames:
    if name in list(semi_teams): 
        rank = semi_teams[semi_teams == name].index
        grade = semi_grade.iloc[rank[0]]
    else:
        grade = 'No'
    semi_rank.append(grade)
data['Semi_rank'] = semi_rank

# Create DataFrame
result_df = pd.DataFrame(data)

# Save DataFrame to CSV
result_csv_path = 'C:/Users/maorb/CSVs/BiztecTabl.csv'
result_df.to_csv(result_csv_path, index=False)

print("CSV file with the desired columns created successfully.")


# In[ ]:


import pandas as pd

csv_path = 'C:/Users/maorb/CSVs/Inspire.xls'
time1_df = pd.read_excel(csv_path, sheet_name='InspireTime1+2')
time2_df = pd.read_excel(csv_path, sheet_name='Time2')
semifinal_df = pd.read_excel(csv_path, sheet_name='My_tab')

# Extract required columns
team_names = time1_df.iloc[3:]['Unnamed: 98']
venture_names = time2_df.iloc[2:]['Q41']
semi_teams = semifinal_df.iloc[:,0]
semi_grade = semifinal_df.iloc[:,1]
teamNames = list(set(team_names) | set(venture_names))
data = {'Team name': teamNames, 
        'Answered_time1_count': [sum(team_names == name) for name in teamNames],
        'Answered_time2_count': [sum(venture_names == name) for name in teamNames]}

semi_rank = []
for name in teamNames:
    if name in list(semi_teams): 
        rank = semi_teams[semi_teams == name].index
        grade = semi_grade.iloc[rank[0]]
    else:
        grade = 'No'
    semi_rank.append(grade)
data['Mark'] = semi_rank

# Create DataFrame
result_df = pd.DataFrame(data)

# Save DataFrame to CSV
result_csv_path = 'C:/Users/maorb/CSVs/insTab1.csv'
result_df.to_csv(result_csv_path, index=False)

print("CSV file with the desired columns created successfully.")


# In[ ]:


import pandas as pd

# Read the Excel file into a DataFrame
semifinal_df = pd.read_excel(csv_path, sheet_name='Marks')

# Get unique names from the 4th column (assuming the names are in the 4th column)
uni = semifinal_df.iloc[:, 3].unique()

# Group by unique names and their marks (assuming marks are in the 'Unnamed: 4' column)
new_marks = semifinal_df.groupby(semifinal_df.iloc[:, 3])['Unnamed: 4'].unique()

# Convert the resulting Series to DataFrame
new_table = new_marks.reset_index()

# Print the new DataFrame
pd.DataFrame(new_table)


# In[ ]:


semifinal_df


# In[ ]:


time2_df.iloc[2:]['Q41']


# In[ ]:


import pandas as pd

# Read the Biztec CSV file
csv_path = 'C:/Users/maorb/CSVs/Biztec+2023+(Time+1)_February+18,+2024_17.05 (1).xlsx'
final_df = pd.read_excel(csv_path, sheet_name='Final Table')
# Count 'Yes' responses for Time1
sum1 = final_df[final_df['Answered_1'] == 'Yes']['Answered_1'].count()

# Count 'Yes' responses for Time2
sum2 = final_df[final_df['Answered_2'] == 'Yes']['Answered_2'].count()

# Group by team name and calculate sum of 'Answered_1', 'Answered_2', and 'Semi_rank' columns
summary_df = final_df.groupby('Team name').agg({'Answered_1': 'sum', 'Answered_2': 'sum', 'Semi_rank': 'sum'})

# Rename the columns
summary_df = summary_df.rename(columns={'Answered_1': 'Time1', 'Answered_2': 'Time2'})
#df_biz = pd.DataFrame({'Team_name': })
# Save the summary DataFrame to CSV
summary_csv_path = 'C:/Users/maorb/CSVs/FinalSummary.csv'
summary_df.to_csv(summary_csv_path)

print("Summary CSV file created successfully.")


# In[ ]:


import pandas as pd

# Read the Biztec CSV file
csv_path = 'C:/Users/maorb/CSVs/Biztec+2023+(Time+1)_February+18,+2024_17.05 (1).xlsx'
final_df = pd.read_excel(csv_path, sheet_name='Final Table')

# Lowercase the 'Team name' column
final_df['Team name'] = final_df['Team name'].str.lower()

# Group by lowercased 'Team name' and sum the counts of 'Yes' for Time1 and Time2
summary_df = final_df.groupby('Team name').agg({'Answered_1': lambda x: (x == 'Yes').sum(),
                                                'Answered_2': lambda x: (x == 'Yes').sum(),
                                                'Semi_rank': 'first'})

# Reset index to make 'Team name' a column again
summary_df.reset_index(inplace=True)

# Save the summary DataFrame to CSV
summary_csv_path = 'C:/Users/maorb/CSVs/Summary_Table.csv'
summary_df.to_csv(summary_csv_path, index=False)

print("Summary table created successfully.")
summary_df


# In[ ]:


summary_df


# In[ ]:


import pandas as pd

# Read the Biztec CSV file
csv_path = 'C:/Users/maorb/CSVs/Biztec+2023+(Time+1)_February+18,+2024_17.05 (1).xlsx'
final_df = pd.read_excel(csv_path, sheet_name='Final Table')

# Remove leading and trailing whitespace from 'Team name' column
final_df['Team name'] = final_df['Team name'].str.strip()

# Create a new column to store the original team names
final_df['Original Team Name'] = final_df['Team name']

# Lowercase the 'Team name' column
final_df['Team name'] = final_df['Team name'].str.lower()

# Group by lowercased 'Team name' and sum the counts of 'Yes' for Time1 and Time2
summary_df = final_df.groupby('Team name').agg({'Answered_1': lambda x: (x == 'Yes').sum(),
                                                'Answered_2': lambda x: (x == 'Yes').sum(),
                                                'Semi_rank': 'first'})

# Reset index to make 'Team name' a column again
summary_df.reset_index(inplace=True)

# Replace 'Team name' with the original names
summary_df['Team name'] = final_df.groupby('Team name')['Original Team Name'].first().values

# Drop the 'Original Team Name' column

# Save the summary DataFrame to CSV
summary_csv_path = 'C:/Users/maorb/CSVs/Summary_Table.csv'
summary_df.to_csv(summary_csv_path, index=False)

print("Summary table created successfully.")


# In[ ]:


import pandas as pd

team_names = [
    'SafeButton', 'Nobook', 'Inhalify', 'HeritageCube', 'Walnut', 'DropMaster',
    'C-AIR', 'C-AIR', 'Ming', 'DropMaster', 'VISTA (WORKING TITLE)', 'C-AIR',
    'DropMaster', 'C-Air', 'NOA', 'Ecoegg pot', 'Ticketrust', 'Heritage Cube',
    'RefineRobotics', 'GOS', 'אבן אור', 'Fall Prevention', 'DOCUSHIELD', 'CommU',
    'CowVolution', 'impro', 'StoryTellE', 'BUYZI', 'HeritageCube', 'HighEye',
    'Ecoegg pots', 'SOONER', 'Fitech', 'C-Air', 'HighEye', 'Ticketrust',
    'HeritageCube', 'MING', 'CommU', 'Atom Construction', 'Red-eye', 'Bina',
    'EYEOP', 'Eye-Tech', 'CommU', 'EYEHOPE', 'HTI', 'Vacure', 'Inhalify',
    'Medilink', 'EyeOP', 'sooner', 'Sooner', 'Impro', 'Gyg', 'Medilink',
    'CowVolution', 'petBud', 'SOONer', 'Vacure', 'impro', 'FitTech', 'PetBud',
    'PetBud', 'eye-tech', 'commu', 'Perbud', 'Nomos'
]

# Create a DataFrame from the list of team names
df = pd.DataFrame({'Team Name': team_names})

# Convert Team Name to lowercase
df['Team Name'] = df['Team Name'].str.lower()

# Count the number of instances for each name
counts = df['Team Name'].value_counts()
counts.sum()


# # Tkinter App builder

# In[ ]:


get_ipython().system('pip install tkinter')


# In[ ]:


import tkinter as tk
from tkinter import messagebox
import random

# Function to display a random number between 1 and 100
def display_random_number():
    random_number = random.randint(1, 100)
    messagebox.showinfo("Random Number", f"The random number is: {random_number}")

# Function to display a message box with a custom message
def display_custom_message():
    message = "Hello, welcome to my simple Python application!"
    messagebox.showinfo("Custom Message", message)

# Create the main application window
root = tk.Tk()
root.title("Simple Python Application")

# Create buttons for displaying a random number and a custom message
random_number_button = tk.Button(root, text="Display Random Number", command=display_random_number)
random_number_button.pack(pady=10)

custom_message_button = tk.Button(root, text="Display Custom Message", command=display_custom_message)
custom_message_button.pack(pady=10)

# Run the application
root.mainloop()


# In[ ]:


def generate_d_splits(nums, d):
    n = len(nums)
    all_splits = []

    # Helper function to generate all possible d-splits recursively
    def generate_splits(start, current_split):
        if len(current_split) == d - 1:  # We have d - 1 partitions
            all_splits.append(current_split + [nums[start:]])
            return

        for i in range(start + 1, n):
            generate_splits(i, current_split + [nums[start:i]])

    for i in range(1, n - (d - 1) + 1):  # Adjusted the range
        generate_splits(i, [nums[:i]])

    return all_splits

# Example usage:
nums = [1, 2, 3, 4]
d = 2
splits = generate_d_splits(nums, d)
for split in splits:
    print(split)


# In[ ]:


import numpy as np
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
matrix[0][1]


# In[ ]:


import pandas as pd
def min_price_of_d_splits(nums, d):
    global prefix_sum
    n = len(nums)
    # Initialize dp table
    dp = [[float('inf')] * (d + 1) for _ in range(n + 1)]
    # Initialize dp[i][1] as the sum of the first i elements
    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1]
        dp[i][1] = prefix_sum[i]

    # Compute dp table
    for j in range(2, d + 1):
        for i in range(1, n + 1):
            for k in range(1, i):
                dp[i][j] = min(dp[i][j], max(dp[k][j - 1], prefix_sum[i] - prefix_sum[k]))

    return pd.DataFrame(dp), print(dp[n][d])


# In[ ]:


min_price_of_d_splits(range(10),5)


# ## Photo editing app

# In[ ]:


def main():
    # Load image
    image_path = input("Enter the path to the image: ")
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Image not found or unable to load.")
        return
main()


# In[ ]:


import cv2

def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

def apply_edge_detection(image):
    return cv2.Canny(image, 100, 200)

def improve_resolution(image):
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def main():
    # Load image
    image_path = input("Enter the path to the image: ")
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print("Error: Image not found or unable to load.")
        return
    
    current_image = original_image.copy()
    
    while True:
        # Display options
        print("\nChoose an option:")
        print("1. Grayscale")
        print("2. Blur")
        print("3. Edge Detection")
        print("4. Improve Resolution")
        print("5. Save")
        print("6. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            current_image = apply_grayscale(current_image)
        elif choice == '2':
            current_image = apply_blur(current_image)
        elif choice == '3':
            current_image = apply_edge_detection(current_image)
        elif choice == '4':
            current_image = improve_resolution(current_image)
            print("Resolution improved successfully.")
        elif choice == '5':
            output_path = input("Enter the path to save the edited image: ")
            cv2.imwrite(output_path, current_image)
            print("Image saved successfully.")
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()


# # POMODORO TIMER

# In[ ]:


get_ipython().system('pip install winsound')


# In[ ]:


import time
from playsound import playsound

def play_sound(filename):
    """
    Function to play a sound file.

    Args:
    - filename (str): Path to the sound file.

    Returns:
    None
    """
    playsound(filename)

def pomodoro_timer(pomodoro_duration=25, break_duration=5, num_cycles=4):
    """
    Function to run an interactive Pomodoro timer with audio notifications.

    Args:
    - pomodoro_duration (int): Duration of each Pomodoro session in minutes (default is 25 minutes).
    - break_duration (int): Duration of each short break in minutes (default is 5 minutes).
    - num_cycles (int): Number of Pomodoro cycles before a long break (default is 4 cycles).

    Returns:
    None
    """

    # Convert durations to seconds
    pomodoro_seconds = pomodoro_duration * 60
    break_seconds = break_duration * 60

    # Run Pomodoro cycles
    for cycle in range(1, num_cycles + 1):
        print(f"Pomodoro Cycle {cycle} started.")
        play_sound("start_sound.mp3")  # Play start sound
        
        # Pomodoro session
        print(f"Pomodoro session started for {pomodoro_duration} minutes.")
        play_sound("work_sound.mp3")  # Play work sound
        time.sleep(pomodoro_seconds)
        print("Pomodoro session ended. Time for a break!")
        play_sound("end_sound.mp3")  # Play end sound
        
        # Break session
        if cycle < num_cycles:
            print(f"Short break started for {break_duration} minutes.")
            play_sound("start_sound.mp3")  # Play start sound
            time.sleep(break_seconds)
            print("Short break ended. Let's start the next Pomodoro!")
            play_sound("end_sound.mp3")  # Play end sound
        else:
            print("Congratulations! You've completed all Pomodoro cycles. Time for a long break!")
            play_sound("finish_sound.mp3")  # Play finish sound
            # You can optionally include a longer break session here

    print("All Pomodoro cycles completed. Great job!")

# Example usage
pomodoro_timer()


# In[ ]:


import time
import winsound
import tkinter as tk

def pomodoro_timer(work_duration=25, break_duration=5, cycles=4):
    root = tk.Tk()
    root.title("Pomodoro Timer")

    label = tk.Label(root, font=("Arial", 20))
    label.pack()

    for cycle in range(cycles):
        label.config(text=f"Cycle {cycle + 1}\nWork session started.")
        root.update()

        # Work session
        for remaining in range(work_duration, 0, -1):
            mins, secs = divmod(remaining, 60)
            time_str = f"{mins:02d}:{secs:02d}"
            label.config(text=time_str)
            root.update()
            time.sleep(1)
        winsound.Beep(440, 1000)  # Play a system sound to signal the end of the work session

        # Break session
        label.config(text="Break time!")
        root.update()
        for remaining in range(break_duration, 0, -1):
            mins, secs = divmod(remaining, 60)
            time_str = f"{mins:02d}:{secs:02d}"
            label.config(text=time_str)
            root.update()
            time.sleep(1)
        winsound.Beep(440, 1000)  # Play a system sound to signal the end of the break

    label.config(text="Pomodoro session ended!")
    root.update()
    winsound.Beep(440, 1000)  # Play a system sound to indicate the end of all cycles
    root.mainloop()

# Example usage
pomodoro_timer()


# In[ ]:


import tkinter as tk
from tkinter import messagebox
import winsound
import threading
import time

class PomodoroApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pomodoro Timer")

        self.work_minutes = 25
        self.rest_minutes = 5
        self.is_working = False
        self.is_resting = False

        self.label = tk.Label(root, text="", font=("Arial", 24))
        self.label.pack(pady=20)

        self.start_button = tk.Button(root, text="Start", command=self.start_timer)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_timer, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.set_button = tk.Button(root, text="Set Time", command=self.set_time)
        self.set_button.pack(pady=10)

    def start_timer(self):
        self.is_working = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.root.after(1000, self.update_timer)

    def stop_timer(self):
        self.is_working = False
        self.is_resting = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def update_timer(self):
        if self.is_working:
            if self.work_minutes > 0:
                self.work_minutes -= 1
                self.label.config(text=f"Work Time: {self.work_minutes:02d}:{00}")
                self.root.after(1000, self.update_timer)
            else:
                winsound.Beep(440, 1000)
                messagebox.showinfo("Work Time Ended", "Take a rest now!")
                self.work_minutes = 25
                self.is_working = False
                self.is_resting = True
                self.update_timer()
        elif self.is_resting:
            if self.rest_minutes > 0:
                self.rest_minutes -= 1
                self.label.config(text=f"Rest Time: {self.rest_minutes:02d}:{00}")
                self.root.after(1000, self.update_timer)
            else:
                winsound.Beep(440, 1000)
                messagebox.showinfo("Rest Time Ended", "Back to work now!")
                self.rest_minutes = 5
                self.is_resting = False
                self.is_working = True
                self.update_timer()

    def set_time(self):
        def set_time_window():
            set_time_root = tk.Toplevel()
            set_time_root.title("Set Time")

            work_label = tk.Label(set_time_root, text="Work Time (minutes):")
            work_label.pack()
            work_entry = tk.Entry(set_time_root)
            work_entry.insert(0, str(self.work_minutes))
            work_entry.pack()

            rest_label = tk.Label(set_time_root, text="Rest Time (minutes):")
            rest_label.pack()
            rest_entry = tk.Entry(set_time_root)
            rest_entry.insert(0, str(self.rest_minutes))
            rest_entry.pack()

            def save_time():
                nonlocal work_minutes, rest_minutes
                work_minutes = int(work_entry.get()) * 60  # Convert minutes to seconds
                rest_minutes = int(rest_entry.get()) * 60  # Convert minutes to seconds
                set_time_root.destroy()

            save_button = tk.Button(set_time_root, text="Save", command=save_time)
            save_button.pack()

            set_time_root.grab_set()
            set_time_root.wait_window()

            return work_minutes, rest_minutes

        work_minutes, rest_minutes = set_time_window()
        self.work_minutes = work_minutes
        self.rest_minutes = rest_minutes

def main():
    root = tk.Tk()
    app = PomodoroApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


# In[4]:


A = (40-80/3)**2+(5-80/3)**2 + (35-80/3)**2 + 10**2 *4 +  2 *5**2 
A/7

