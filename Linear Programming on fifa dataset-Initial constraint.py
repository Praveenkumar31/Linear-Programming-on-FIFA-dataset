
# coding: utf-8

# In[1]:


#importing packages
from pulp import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
get_ipython().run_line_magic('matplotlib', 'inline')

#reading data file
data = pd.read_csv("LP.csv")
#data.head
#list(data.columns.values)


# In[2]:


prob = pulp.LpProblem('dream_attack', pulp.LpMaximize)

#creating decision variables - with bound between 0 and 1 to buy a player
decision_variables = []
for rownum, row in data.iterrows():
    variable = str('x' + str(rownum))
    variable = pulp.LpVariable(str(variable), lowBound = 0, upBound = 1, cat= 'Integer')
    decision_variables.append(variable)

print ("Total number of decision_variables: " + str(len(decision_variables)))
print(decision_variables)


# In[3]:


#optimization function
max_potential = ""
for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            formula = row['Potential']*player
            max_potential += formula

prob += max_potential
print ("Optimization function: " + str(max_potential))                


# In[4]:


#creating constrainsts

#budget constraint
max_budget = 650000000
used_budget = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            #print(player)
            formula = player*row['Vvalue']
            used_budget += formula

#print(used_budget)
prob += (used_budget <= max_budget)


# In[5]:


#Age constraint
max_age = 270
player_age = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            #print(player)
            formula = player*row['Age']
            player_age += formula

#print(used_budget)
prob += (player_age <= max_age)


# In[6]:


#goalkeeper_constraint

#setting max goalkeeper players
avail_gk = 1
total_gk = ""
for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'GK':
                formula = 1*player
                total_gk += formula
prob += (total_gk == avail_gk)
#print(total_def)

#goalkeeper_diving
max_div  = 1
exp_div = 85
opt_div = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'GK':
                if row['GK diving'] >= 85:
                    formula = player*row['GK diving']
                    opt_div += formula

prob += (opt_div >= exp_div)

#goalkeeper_handling
max_han = 1
exp_han = 85
opt_han = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'GK':
                if row['GK handling'] >=85:
                    formula = player*row['GK handling']
                    opt_han += formula

prob += (opt_han >= exp_han)

#goalkeeper_kicking
max_kic  = 1
exp_kic = 85
opt_kic = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'GK':
                if row['GK kicking'] >=85:
                    formula = player*row['GK kicking']
                    opt_kic += formula

prob += (opt_kic >= exp_kic)

#goalkeeper_positioning 
max_pos = 1
exp_pos = 85
opt_pos = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'GK':
                if row['GK positioning'] >=85:
                    formula = player*row['GK positioning']
                    opt_pos += formula

prob += (opt_pos >= exp_pos)

#goalkeeper_reflexes
max_ref = 1
exp_ref = 85
opt_ref = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'GK':
                if row['GK reflexes'] >=85:
                    formula = player*row['GK reflexes']
                    opt_ref += formula

prob += (opt_ref >= exp_ref)


# In[7]:


#defender_constraint

#setting max defence players
avail_def = 4
total_def = ""
for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'Defence':
                formula = 1*player
                total_def += formula
prob += (total_def == avail_def)
#print(total_def)

#defender_interception
max_int  = 4
exp_int = 320
opt_int = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'Defence':
                if row['Interceptions'] >=80:
                    formula = player*row['Interceptions']
                    opt_int += formula

prob += (opt_int >= exp_int)

#defender_marking
max_mar = 4
exp_mar = 320
opt_mar = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'Defence':
                if row['Marking'] >=80:
                    formula = player*row['Marking']
                    opt_mar += formula

prob += (opt_mar >= exp_mar)

#defender_sliding_tackle
max_sli = 2
exp_sli = 150
opt_sli = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'Defence':
                if row['Sliding tackle'] >=75:
                    formula = player*row['Sliding tackle']
                    opt_sli += formula

prob += (opt_sli >= exp_sli)

#defender_standing_tackle
max_sta = 4
exp_sta = 320
opt_sta = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'Defence':
                if row['Standing tackle'] >=80:
                    formula = player*row['Standing tackle']
                    opt_sta += formula

#print(opt_diving)
prob += (opt_sta >= exp_sta)


# In[8]:


#attack constraints

#setting max attack players
avail_att = 3
total_att = ""
for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'Attack':
                formula = 1*player
                total_att += formula
prob += (total_att == avail_att)
        
#attack_finish
max_fin = 1
exp_fin = 85
opt_fin = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'Attack':
                if row['Finishing'] >=85:
                    formula = player*row['Finishing']
                    opt_fin += formula
                
prob += (opt_fin >= exp_fin)

#attack_cross
max_cro = 2
exp_cro = 150
opt_cro = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'Attack':
                if row['Crossing'] >=75:
                    formula = player*row['Crossing']
                    opt_cro += formula
                
prob += (opt_cro >= exp_cro)


# In[9]:


#midfielders constraints

#setting maximum mid fielders
avail_mid = 3
total_mid = ""
for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Player_type'] == 'Mid':
                formula = 1*player
                total_mid += formula
prob += (total_mid == avail_mid)


# In[10]:


#team_acceleration
max_acc = 3
exp_acc = 240
opt_acc = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Acceleration'] >= 80:
                formula = player*row['Acceleration']
                opt_acc += formula
                
prob += (opt_acc >= exp_acc)

#team_ball control
max_bal = 4
exp_bal = 280
opt_bal = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Ball control'] >= 70:
                formula = player*row['Ball control']
                opt_bal += formula
                
prob += (opt_bal >= exp_bal)

#team_dribble
max_dri = 2
exp_dri = 140
opt_dri = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Dribbling'] >= 70:
                formula = player*row['Dribbling']
                opt_dri += formula
                
prob += (opt_dri >= exp_dri)

#team_free kick
max_fre = 1
exp_fre = 80
opt_fre = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Free kick accuracy'] >= 80:
                formula = player*row['Free kick accuracy']
                opt_fre += formula
                
prob += (opt_fre >= exp_fre)

#team_jumping
max_jum = 3
exp_jum = 225
opt_jum = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Jumping'] >= 75:
                formula = player*row['Jumping']
                opt_jum += formula
                
prob += (opt_jum >= exp_jum)

#team_penalty
max_pen = 5 
exp_pen = 375
opt_pen = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Penalties'] >=75:
                formula = player*row['Penalties']
                opt_pen += formula
                
prob += (opt_pen >= exp_pen)


#team_short pass
max_spa = 5
exp_spa = 400
opt_spa = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Short passing'] >= 80:
                formula = player*row['Short passing']
                opt_spa += formula
                
prob += (opt_spa >= exp_spa)

#team_long pass
max_lop = 4
exp_lop = 320
opt_lop = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Long passing'] >= 80:
                formula = player*row['Long passing']
                opt_lop += formula
                
prob += (opt_lop >= exp_lop)

#team_reaction
max_rea = 3
exp_rea = 240
opt_rea = ""

for rownum, row in data.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['Reactions'] >= 80:
                formula = player*row['Reactions']
                opt_rea += formula
                
prob += (opt_rea >= exp_rea)

print(prob)


# In[11]:


#running optimization
optimization_result = prob.solve()
assert optimization_result == pulp.LpStatusOptimal
print("Status:", LpStatus[prob.status])
# print("Optimal Solution to the problem: ", value(prob.objective))
#print ("Individual decision_variables: ")
#for v in prob.variables():
#    print(v.name, "=", v.varValue)


# In[12]:


#reorder results
variable_name = []
variable_value = []

for v in prob.variables():
    variable_name.append(v.name)
    variable_value.append(v.varValue)

df = pd.DataFrame({'variable': variable_name, 'value': variable_value})
for rownum, row in df.iterrows():
    value = re.findall(r'(\d+)', row['variable'])
    df.loc[rownum, 'variable'] = int(value[0])

df = df.sort_values(by='variable')

#append results
for rownum, row in data.iterrows():
    for results_rownum, results_row in df.iterrows():
        if rownum == results_row['variable']:
            data.loc[rownum, 'decision'] = results_row['value']


# In[13]:


#optimal solution

data[data['decision'] == 1]

#df1 = pd.DataFrame(data = data[data['decision'] == 1])
#df1.to_csv("C:/Users/91979/Desktop/output.csv")


# In[14]:


print("\t Sensitivity Analysis \t Constraint \t Shadow Price \t Slack")
aa=[]
for name, c in prob.constraints.items():
    print(name, ":", c, "\t", c.pi, "\t", c.slack, "\n")


# In[15]:


prob.writeLP("dream_attack.lp")

