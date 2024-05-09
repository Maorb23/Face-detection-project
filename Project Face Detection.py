#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os as os
import pandas as pd
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
##All the different participants
Elon_df = pd.read_csv("C:/Users/maorb/CSVs/Elon_ExpFinal.csv")
Meler_df = pd.read_csv("C:/Users/maorb/CSVs/Meler_ExpFinal.csv")
Gal_df = pd.read_csv("C:/Users/maorb/CSVs/Gal_ExpFinal.csv")
Yael_df = pd.read_csv("C:/Users/maorb/CSVs/Yael_ExpFinal.csv")
Maayan_df = pd.read_csv("C:/Users/maorb/CSVs/Maayan_ExpFinal.csv")


# In[172]:


##Getting rid of training trials
Elon_df.drop(index=Elon_df.index[0:10], inplace=True)
Meler_df.drop(index=Meler_df.index[0:10], inplace=True)
Gal_df.drop(index=Gal_df.index[0:10], inplace=True)
Yael_df.drop(index=Yael_df.index[0:10], inplace=True)
Maayan_df.drop(index=Maayan_df.index[0:10], inplace=True)
combined_df = pd.concat([Gal_df, Elon_df, Meler_df, Yael_df,Maayan_df], ignore_index=True)


# In[79]:


a = Elon_df['isUserRight'].mean()
b = Meler_df['isUserRight'].mean()
c = Gal_df['isUserRight'].mean()
d = Yael_df['isUserRight'].mean()
e = Maayan_df['isUserRight'].mean()


# In[80]:


# Calculate the mean for each DataFrame
a = Elon_df['isUserRight'].mean()
b = Meler_df['isUserRight'].mean()
c = Gal_df['isUserRight'].mean()
d = Yael_df['isUserRight'].mean()
e = Maayan_df['isUserRight'].mean()

# Calculate the weighted average
weighted_avg = ((Elon_df.shape[0] * a) + (Meler_df.shape[0] * b) + (Gal_df.shape[0] * c) + (Yael_df.shape[0] * d) + (Maayan_df.shape[0] * e)) / 564

print(weighted_avg)


# In[173]:


import os

def extract_emotion_name(file_path):
    # Split the file path by the backslash ('\') characters
    path_parts = file_path.split('\\')
    # Find the index of 'Emotions' in the path
    emotions_index = path_parts.index('Emotions')
    # Extract the emotion name, which is the element after 'Emotions'
    emotion_name_with_extension = path_parts[emotions_index + 1]
    # Remove the file extension to get the emotion name
    emotion_name = os.path.splitext(emotion_name_with_extension)[0]
    return emotion_name

# Apply the function to extract the emotion name and create a new column
Elon_df['Emotion_name'] = Elon_df['Emotion'].apply(extract_emotion_name)
Gal_df['Emotion_name'] = Gal_df['Emotion'].apply(extract_emotion_name)
Yael_df['Emotion_name'] = Yael_df['Emotion'].apply(extract_emotion_name)
Meler_df['Emotion_name'] = Meler_df['Emotion'].apply(extract_emotion_name)
Maayan_df['Emotion_name'] = Maayan_df['Emotion'].apply(extract_emotion_name)



# In[174]:


combined_df = pd.concat([Gal_df, Elon_df, Meler_df, Yael_df,Maayan_df], ignore_index=True)
combined_df['Emotion_name']


# # Anova for each group, combined in the end

# ## Elon

# In[84]:


import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Read the CSV file into a DataFrame

# Drop the first 10 rows

# Display the first row

# Convert to factor variables
Elon_df['Correct_Answer'] = pd.Categorical(Elon_df['Correct_Answer'])
Elon_df['Response'] = pd.Categorical(Elon_df['Response'])
Elon_df['Emotion_name'] = pd.Categorical(Elon_df['Emotion_name'])


# Fit the ANOVA model
model_aov1 = smf.ols("isUserRight ~ Emotion_name + C(Response)*C(Correct_Answer)", Elon_df).fit()


# Print the summary of the ANOVA model
print(model_aov1.summary())

# Get the means table
anov_table = sm.stats.anova_lm(model_aov1, typ=2)
print(anov_table)


# In[85]:


# Group the DataFrame by 'Correct_Answer' and calculate the mean for each group
means = Elon_df.groupby('Correct_Answer',observed = True)['isUserRight'].mean()

# Get the counts for each group
counts = Elon_df['Correct_Answer'].value_counts()

# Combine means and counts into a DataFrame
result_df = pd.DataFrame({'Mean': means, 'Count': counts})

# Print the result DataFrame
print(result_df)


# ## Yael

# In[86]:


Yael_df['Correct_Answer'] = pd.Categorical(Yael_df['Correct_Answer'])
Yael_df['Response'] = pd.Categorical(Yael_df['Response'])
Yael_df['Emotion_name'] = pd.Categorical(Yael_df['Emotion_name'])


# Fit the ANOVA model
model_aov2 = smf.ols("isUserRight ~ Emotion_name + C(Response)*C(Correct_Answer)", Yael_df).fit()


# Print the summary of the ANOVA model
print(model_aov2.summary())

# Get the means table
anov_table = sm.stats.anova_lm(model_aov2, typ=2)
print(anov_table)


# In[87]:


means = Yael_df.groupby('Correct_Answer',observed = True)['isUserRight'].mean()

# Get the counts for each group
counts = Yael_df['Correct_Answer'].value_counts()

# Combine means and counts into a DataFrame
result_df = pd.DataFrame({'Mean': means, 'Count': counts})

# Print the result DataFrame
print(result_df)


# ## Meler

# In[88]:


Meler_df['Correct_Answer'] = pd.Categorical(Meler_df['Correct_Answer'])
Meler_df['Response'] = pd.Categorical(Meler_df['Response'])
Meler_df['Emotion_name'] = pd.Categorical(Meler_df['Emotion_name'])


# Fit the ANOVA model
model_aov3 = smf.ols("isUserRight ~ Emotion_name + C(Response)*C(Correct_Answer)", Meler_df).fit()


# Print the summary of the ANOVA model
print(model_aov3.summary())

# Get the means table
anov_table = sm.stats.anova_lm(model_aov3, typ=2)
print(anov_table)


# In[89]:


means = Meler_df.groupby('Correct_Answer',observed = True)['isUserRight'].mean()

# Get the counts for each group
counts = Meler_df['Correct_Answer'].value_counts()

# Combine means and counts into a DataFrame
result_df = pd.DataFrame({'Mean': means, 'Count': counts})

# Print the result DataFrame
print(result_df)


# ## Gal_df

# In[90]:


Gal_df['Correct_Answer'] = pd.Categorical(Gal_df['Correct_Answer'])
Gal_df['Response'] = pd.Categorical(Gal_df['Response'])
Gal_df['Emotion_name'] = pd.Categorical(Gal_df['Emotion_name'])

# Fit the ANOVA model
model_aov4 = smf.ols("isUserRight ~ Emotion_name + Response*Correct_Answer", Gal_df).fit()


# Print the summary of the ANOVA model
print(model_aov4.summary())

# Get the means table
anov_table = sm.stats.anova_lm(model_aov4, typ=2)
print(anov_table)


# In[91]:


means = Gal_df.groupby('Correct_Answer',observed = True)['isUserRight'].mean()

# Get the counts for each group
counts = Gal_df['Correct_Answer'].value_counts()

# Combine means and counts into a DataFrame
result_df = pd.DataFrame({'Mean': means, 'Count': counts})

# Print the result DataFrame
print(result_df)


# ## Maayan

# In[92]:


Maayan_df['Correct_Answer'] = pd.Categorical(Maayan_df['Correct_Answer'])
Maayan_df['Response'] = pd.Categorical(Maayan_df['Response'])
Maayan_df['Emotion_name'] = pd.Categorical(Maayan_df['Emotion_name'])

# Fit the ANOVA model
model_aov6 = smf.ols("isUserRight ~ Emotion_name + C(Response)*C(Correct_Answer)", Yael_df).fit()


# Print the summary of the ANOVA model
print(model_aov6.summary())

# Get the means table
anov_table = sm.stats.anova_lm(model_aov6, typ=2)
print(anov_table)


# In[93]:


means = Maayan_df.groupby('Correct_Answer',observed = True)['isUserRight'].mean()

# Get the counts for each group
counts = Maayan_df['Correct_Answer'].value_counts()

# Combine means and counts into a DataFrame
result_df = pd.DataFrame({'Mean': means, 'Count': counts})

# Print the result DataFrame
print(result_df)


# ## Combined

# In[94]:


combined_df = combined_df.rename(columns = {'trials_response.rt':'Response_time'})


# In[123]:


combined_df['Correct_Answer'] = pd.Categorical(combined_df['Correct_Answer'])
combined_df['Response'] = pd.Categorical(combined_df['Response'])
combined_df['Emotion_name'] = pd.Categorical(combined_df['Emotion_name'])
# Fit the ANOVA model
model_aov5 = smf.ols("isUserRight ~ Emotion_name + Response*Correct_Answer", combined_df).fit()


# Print the summary of the ANOVA model
print(model_aov5.summary())

# Get the means table
anov_table = sm.stats.anova_lm(model_aov5, typ=2)
print(anov_table)


# In[161]:


model_aov5 = smf.ols("isUserRight ~ Emotion_name + Response*Response_time", combined_df).fit()


# Print the summary of the ANOVA model
print(model_aov5.summary())

# Get the means table
anov_table = sm.stats.anova_lm(model_aov5, typ=2)
print(anov_table)


# In[170]:


model_aov5 = smf.ols("isUserRight ~ Emotion_name + Response_time*Correct_Answer", combined_df).fit()


# Print the summary of the ANOVA model
print(model_aov5.summary())

# Get the means table
anov_table = sm.stats.anova_lm(model_aov5, typ=2)
print(anov_table)


# In[ ]:


model_aov5 = smf.ols("isUserRight ~ Emotion_name + Response_time*Correct_Answer", combined_df).fit()


# In[140]:


means = combined_df.groupby('Correct_Answer',observed = True)['isUserRight'].mean()
means1 = combined_df.groupby('Correct_Answer',observed = True)['Response_time'].mean()
means2 = combined_df.groupby('Emotion_name',observed = True)['Response_time'].mean()

# Get the counts for each group
counts = combined_df['Correct_Answer'].value_counts()

# Combine means and counts into a DataFrame
result_correct = pd.DataFrame({'Mean': means, 'Count': counts})
result_cor_response = pd.DataFrame({'Mean': means1, 'Count': counts})
result_Emotion_response = pd.DataFrame({'Mean': means2, 'Count': counts})

# Print the result DataFrame
print(colored('UserIsRightRatio', 'black', 'on_cyan', attrs=["blink"]))
print(result_correct)
print("\n")  # Empty line
print(colored('CorrectPhoto_ResponseTime', 'black', 'on_cyan', attrs=["blink"]))
print(result_cor_response)
print("\n")  # Empty line
print(colored('EmotionName_ResponseTime', 'black', 'on_cyan', attrs=["blink"]))
print(result_Emotion_response)


# ## Visualization

# In[154]:


import seaborn as sns
import matplotlib.pyplot as plt

# Define the y-axis limits
y_min = 0.4
y_max = 0.88

# Create a DataFrame from the means
mean_df = pd.DataFrame({'Correct_Answer': means.index, 'Mean': means.values})

# Plot the histogram
sns.set_theme(style='darkgrid')
sns.barplot(data=mean_df, x='Correct_Answer', y='Mean', palette='bright')

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

plt.title('Right Answers Ratio')


# Set y-axis limits
plt.ylim(y_min, y_max)



# In[155]:


mean_df1 = pd.DataFrame({'Correct_Answer': means1.index, 'Mean': means1.values})
sns.set_theme(style='darkgrid')
sns.barplot(data=mean_df1, x='Correct_Answer', y='Mean', palette='bright')
plt.xticks(rotation=45)
y_min = 0.4
y_max = 0.95


# Set y-axis limits
plt.ylim(y_min, y_max)
plt.title('Mean Response Time')


# ## Logistic Regression

# In[168]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas

res1 = smf.logit(formula='isUserRight ~ Response*Response_time', data=combined_df).fit()

res1.summary()
                 


# In[ ]:


import rpy2


# In[1]:


mod = pymer4('isUserRight ~ Response + (1|Response_time)', data=combined_df)

