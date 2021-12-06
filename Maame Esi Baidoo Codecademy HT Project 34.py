# Import libraries
import numpy as np
import pandas as pd
import codecademylib3

# Import data
dogs = pd.read_csv('dog_data.csv')
#Inspecting the dataset
print(dogs.head())
#Finding out if whippets are significantly more or less likely than other dogs to be a rescue
whippet_rescue = dogs.is_rescue[dogs.breed == "whippet"]
num_whippet_rescues = np.sum(whippet_rescue == 1)
print("There are " + str(num_whippet_rescues) + " whippets that are rescues.")
print("There are " + str(len(whippet_rescue)) + " whippets in this data.")
#Testing whether the percentage of rescue whippets in this data is significantly different from the population value of 8%
from scipy.stats import binom_test
pval_whippet = binom_test(x = num_whippet_rescues, n = len(whippet_rescue), p = 0.08)
#Using a significance threshold of 0.05:
if pval_whippet < 0.05:
  print("The percentage of whippets that are rescues is significantly different from 8%")
else:
  print("The percentage of whippets that are rescues is not significantly different from 8%")
#Finding out if there is a significant difference in the average weights of three dog breeds
wt_whippets = dogs.weight[dogs.breed == "whippet"]
wt_terriers = dogs.weight[dogs.breed == "terrier"]
wt_pitbulls = dogs.weight[dogs.breed == "pitbull"]
#Running a hypothesis test to determine whether the average weights are significantly different or not
from scipy.stats import f_oneway
fstat, pval_weight = f_oneway(wt_whippets, wt_terriers, wt_pitbulls)
#Using a significance threshold of 0.05:
if pval_weight < 0.05:
  print("At least, two of the three dog breeds have average weights that are significantly different.")
else:
  print("The average weights of the three dog breeds are not significantly different from each other.")
#Running another hypothesis test to determine which of the three breeds weigh different amounts on average
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# Subset to just whippets, terriers, and pitbulls
dogs_wtp = dogs[dogs.breed.isin(['whippet', 'terrier', 'pitbull'])]
weight_tukey = pairwise_tukeyhsd(dogs_wtp.weight, dogs_wtp.breed)
print(weight_tukey)
#Looks like the average weights between pitbulls and terriers, and also between terriers and whippets are significantly different from each other.

#Finding out if poodles and shihtzus come in different colors
# Subset to just poodles and shihtzus
dogs_ps = dogs[dogs.breed.isin(['poodle', 'shihtzu'])]
#Creating a contingency table since breed and color are both categorical variables
dog_cont = pd.crosstab(dogs_ps.color, dogs_ps.breed)
print(dog_cont)
#Running a hypothesis test to find out if there is an association between breed and color
from scipy.stats import chi2_contingency
chi2, pval_color, dof, expected = chi2_contingency(dog_cont)
#Using a significance threshold of 0.05:
if pval_color < 0.05:
  print("There is no association between breed (poodle vs. shihtzu) and color.")
else:
  print("There is an association between breed (poodle vs. shihtzu) and color.")

#Finding if there is an association between greyhounds and pitull breeds and color
dogs_gp = dogs[dogs.breed.isin(['greyhound', 'pitbull'])]
#Creating a contingency table since breed and color are both categorical variables
dog_cont2 = pd.crosstab(dogs_gp.color, dogs_gp.breed)
print(dog_cont2)
#Running a hypothesis test to find out if there is an association between those breeds and color
chi2, pval_color2, dof, expected = chi2_contingency(dog_cont2)
#Using a significance threshold of 0.05:
if pval_color2 < 0.05:
  print("There is no association between breed (greyhound vs. pitbull) and color.")
else:
  print("There is an association between breed (greyhound vs. pitbull) and color.")