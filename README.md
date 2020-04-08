
# Statistical Power - Lab

## Introduction


In this lesson, you'll practice doing a power-analysis during experimental design. As you've seen, power analysis allows you to determine the sample size required to detect an effect of a given size with a given degree of confidence. In other words, it allows you to determine the probability of detecting an effect of a given size with a given level of confidence, under-sample size constraints.

The following four factors have an intimate relationship:

* Sample size
* Effect size
* Significance level = P (Type I error) = probability of finding an effect that is not there
* **Power = 1 - P (Type II error)** = probability of finding an effect that is there

Given any three of these, we can easily determine the fourth.

## Objectives

In this lab you will: 

- Describe the impact of sample size and effect size on power 
- Perform power calculation using SciPy and Python 
- Demonstrate the combined effect of sample size and effect size on statistical power using simulations

## Let's get started!
  
To start, let's import the necessary libraries required for this simulation: 


```python
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels,stats.power import TTestIndPower, TTestPower
sns.set_style('darkgrid')
```

## Scenario

A researcher wants to study how daily protein supplementation in the elderly population will affect baseline liver fat. The study budget will allow enrollment of 24 patients. Half will be randomized to a placebo group and half to the protein supplement treatment group and the trial will be carried out over one month. It is desired to see whether the mean change in percentage of liver fat from baseline to the end of the study differs between the two groups in the study. 

With this, the researcher writes the null hypothesis: 

    There is no difference between experimental and control group mean change in percentage of liver fat 
    
$$\mu_{1} = \mu_{2}$$
  
And the alternative Hypothesis:

    There is a difference between experimental and control group mean change in percentage of liver fat 

$$\mu_{1} \neq \mu_{2}$$
    
  

The researcher needs to know what power  will be obtained under the sample size restrictions to identify a change in mean percent liver fat of 0.17. Based on past results, a common standard deviation of 0.21 will be used for each treatment group in the power analysis. 

To determine the practicality of this experimental design, you'll run a power analysis simulation: 


```python
# Number of patients in each group
sample_size = 12

# Control group
control_mean = 0
control_sd = 0.21

# Experimental group
experimental_mean = 0.17
experimental_sd = 0.21

# Set the number of simulations for our test = 1000
n_sim = 1000
```

You can now start running simulations to run an independent t-test with above data and store the calculated p-value in our `p` array. Perform following tasks: 

* Initialize a numpy array and fill it with `NaN` values for storing the results (p_value) of the independent t-test  
* For a defined number of simulations (i.e., 1000), do the following:

    * Generate a random normal variable with control mean and sd
    * Generate a random normal variable with experimental mean and sd
    * Run and independent t-test using control and experimental data
    * Store the p value for each test

* Calculate the total number and overall proportion of simulations where the null hypothesis is rejected



```python
# For reproducibility 
np.random.seed(10)

# Initialize array to store results
p = (np.empty(n_sim))
p.fill(np.nan)

#  Run a for loop for range of values in n_sim
for n in range(n_sim):
    control_sample = np.random.normal(control_mean, control_sd, sample_size)
    exp_sample = np.random.normal(experimental_mean, experimental_sd, sample_size)
    t_test = stats.ttest_ind(control_sample, exp_sample)
    p[n] = t_test[1]
    
# number of null hypothesis rejections
num_null_rejects = np.sum(p < 0.05)
power = num_null_rejects / float(n_sim) #num_null_rejects / num_sim is times we had 

# 0.495
power
```




    0.495



These results indicate that using 12 participants in each group and with given statistics, the statistical power of the experiment is 49%. This can be interpreted as follows:

> **If a large effect (0.17 or greater) is truly present between control and experimental groups, then the null hypothesis (i.e. no difference with alpha 0.05) would be rejected 49% of the time. **

## Sample size requirements for a given effect size

Often in behavioral research 0.8 is accepted as a sufficient level of power.  

Clearly, this is not the case for the experiment as currently designed. Determine the required sample size in order to identify a difference of 0.17 or greater between the group means with an 80% power.


```python
# Required power
target = 0.8
```


```python
from statsmodels.stats.power import TTestIndPower
power = TTestIndPower()
```


```python
# Determine the sample size
sample_size = power.solve_power(effect_size=0.17/0.21, alpha=.05, power=target)
```


```python
sample_size
```




    24.951708908275144




```python
# Minimum sample size to start the simulations 
sample_size = 12
null_rejected = 0
n_sim = 10000
```

As above, perform the following

* Initialize an empty array for storing results
* initialize a list for storing sample size x power summary
* While current power is less than the target power
    * Generate distributions for control and experimental groups using given statistics (as before)
    * Run a t-test and store results
    * Calculate current power 
    * Output current sample size and power calculated for inspection
    * Store results: Sample size, power
    * increase the sample size by 1 and repeat


```python
np.random.seed(10)

p = (np.empty(n_sim))
p.fill(np.nan)

power_sample = []

# Keep iterating as shown above until desired power is obtained  
while null_rejected < target:
    data = np.empty([n_sim, sample_size, 2])
    data[:,:,0] = np.random.normal(control_mean, control_sd, size=[n_sim, sample_size])
    data[:,:,1] = np.random.normal(experimental_mean, experimental_sd, size=[n_sim,sample_size])
    
    t_test = stats.ttest_ind(data[:,:,0], data[:,:,1], axis=1)
    p_vals = t_test[1]
    null_rejected = np.sum(p_vals < 0.05) / n_sim
    print('Number of Samples:', sample_size,', Calculated Power =', null_rejected)
    power_sample.append([sample_size, null_rejected])
    sample_size += 1
```

    Number of Samples: 12 , Calculated Power = 0.4754
    Number of Samples: 13 , Calculated Power = 0.5066
    Number of Samples: 14 , Calculated Power = 0.5423
    Number of Samples: 15 , Calculated Power = 0.5767
    Number of Samples: 16 , Calculated Power = 0.6038
    Number of Samples: 17 , Calculated Power = 0.6297
    Number of Samples: 18 , Calculated Power = 0.658
    Number of Samples: 19 , Calculated Power = 0.6783
    Number of Samples: 20 , Calculated Power = 0.7056
    Number of Samples: 21 , Calculated Power = 0.7266
    Number of Samples: 22 , Calculated Power = 0.7481
    Number of Samples: 23 , Calculated Power = 0.7624
    Number of Samples: 24 , Calculated Power = 0.7864
    Number of Samples: 25 , Calculated Power = 0.8031


You can also plot the calculated power against sample size to visually inspect the effect of increasing sample size. 


```python
# Plot a sample size X Power line graph 
plt.figure(figsize=(10,5))
df = pd.DataFrame(power_sample)
plt.plot(df[0], df[1])
plt.title("Power with increasing sample size")
plt.xlabel('Sample Size')
plt.ylabel("Power")
plt.show();
```


![png](index_files/index_16_0.png)


This output indicates that in order to get the required power (80%) to detect a difference of 0.17, you would need a considerably higher number of patients. 

## BONUS: Investigating the relationship between Power, Sample Size, and Effect Size

You've seen how to calculate power given alpha, sample size, and effect size. To further investigate this relationship, it is interesting to plot the relationship between power and sample size for various effect sizes. 

To do this, run multiple simulations for varying parameters. Then store the parameters and plot the resulting dataset. Specifically:

1. Use a value of $\alpha$ = 0.05 for all of your simulations
2. Use the following effect sizes: [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
3. Use the sample sizes from 10 to 500
4. For each effect size sample size combination, calculate the accompanying power
5. Plot a line graph of the power vs sample size relationship. You should have 7 plots; one for each of the effect sizes listed above. All 7 plots can be on the same graph but should be labeled appropriately. Plot the power on the y-axis and sample size on the x-axis.


```python
alpha = 0.05
effect_sizes = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
sample_sizes = np.arange(10, 501, 10)
n_sim = 1000
```


```python
power_data = np.empty([len(effect_sizes), len(sample_sizes)])
```


```python
def power_explore(n_sim=1000, sample_sizes=np.arange(10, 501, 10), 
                  control_mean=0, control_sd=0.21, experimental_mean=0.17, experimental_sd=0.21):
    np.random.seed(10)

    p = (np.empty(n_sim))
    p.fill(np.nan)

    power_sample = []

    # Keep iterating as shown above until desired power is obtained  
    for sample_size in sample_sizes:
        data = np.empty([n_sim, sample_size, 2])
        data[:,:,0] = np.random.normal(control_mean, control_sd, size=[n_sim, sample_size])
        data[:,:,1] = np.random.normal(experimental_mean, experimental_sd, size=[n_sim,sample_size])

        t_test = stats.ttest_ind(data[:,:,0], data[:,:,1], axis=1)
        p_vals = t_test[1]
        null_rejected = np.sum(p_vals < 0.05) / n_sim
        #print('Number of Samples:', sample_size,', Calculated Power =', null_rejected)
        power_sample.append(null_rejected)
     
    return power_sample
```


```python
power_analysis = {}
for mean in effect_sizes:
    power_analysis[mean] = power_explore(experimental_mean = mean)
```


```python
pwr_pd = pd.DataFrame.from_dict(power_analysis)
pwr_pd
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.01</th>
      <th>0.05</th>
      <th>0.10</th>
      <th>0.15</th>
      <th>0.20</th>
      <th>0.30</th>
      <th>0.50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.043</td>
      <td>0.084</td>
      <td>0.177</td>
      <td>0.329</td>
      <td>0.498</td>
      <td>0.850</td>
      <td>0.998</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.040</td>
      <td>0.085</td>
      <td>0.299</td>
      <td>0.599</td>
      <td>0.851</td>
      <td>0.996</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.071</td>
      <td>0.156</td>
      <td>0.454</td>
      <td>0.781</td>
      <td>0.961</td>
      <td>0.999</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.061</td>
      <td>0.200</td>
      <td>0.585</td>
      <td>0.881</td>
      <td>0.984</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.054</td>
      <td>0.215</td>
      <td>0.670</td>
      <td>0.948</td>
      <td>0.999</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.055</td>
      <td>0.264</td>
      <td>0.768</td>
      <td>0.973</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.060</td>
      <td>0.281</td>
      <td>0.789</td>
      <td>0.988</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.065</td>
      <td>0.318</td>
      <td>0.853</td>
      <td>0.992</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.072</td>
      <td>0.342</td>
      <td>0.887</td>
      <td>0.996</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.062</td>
      <td>0.405</td>
      <td>0.923</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.066</td>
      <td>0.430</td>
      <td>0.936</td>
      <td>0.999</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.061</td>
      <td>0.473</td>
      <td>0.953</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.064</td>
      <td>0.466</td>
      <td>0.974</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.068</td>
      <td>0.520</td>
      <td>0.982</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.066</td>
      <td>0.501</td>
      <td>0.984</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.055</td>
      <td>0.572</td>
      <td>0.996</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.078</td>
      <td>0.591</td>
      <td>0.993</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.077</td>
      <td>0.618</td>
      <td>0.995</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.085</td>
      <td>0.657</td>
      <td>0.996</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.069</td>
      <td>0.650</td>
      <td>0.997</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.073</td>
      <td>0.693</td>
      <td>0.997</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.077</td>
      <td>0.693</td>
      <td>0.996</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.078</td>
      <td>0.727</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.073</td>
      <td>0.752</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.083</td>
      <td>0.772</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.088</td>
      <td>0.793</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.070</td>
      <td>0.789</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.083</td>
      <td>0.833</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.101</td>
      <td>0.816</td>
      <td>0.999</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.076</td>
      <td>0.834</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.077</td>
      <td>0.829</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>31</td>
      <td>0.094</td>
      <td>0.848</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>32</td>
      <td>0.092</td>
      <td>0.860</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>33</td>
      <td>0.101</td>
      <td>0.855</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>34</td>
      <td>0.099</td>
      <td>0.891</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>35</td>
      <td>0.095</td>
      <td>0.888</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>36</td>
      <td>0.079</td>
      <td>0.907</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>37</td>
      <td>0.109</td>
      <td>0.898</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>38</td>
      <td>0.098</td>
      <td>0.912</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>39</td>
      <td>0.083</td>
      <td>0.925</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0.090</td>
      <td>0.935</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>41</td>
      <td>0.109</td>
      <td>0.949</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>42</td>
      <td>0.107</td>
      <td>0.941</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>43</td>
      <td>0.096</td>
      <td>0.940</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>44</td>
      <td>0.107</td>
      <td>0.952</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>45</td>
      <td>0.091</td>
      <td>0.954</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>46</td>
      <td>0.124</td>
      <td>0.956</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>47</td>
      <td>0.110</td>
      <td>0.955</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>48</td>
      <td>0.100</td>
      <td>0.967</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>49</td>
      <td>0.120</td>
      <td>0.959</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
pwr_pd.index = sample_sizes
pwr_pd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.01</th>
      <th>0.05</th>
      <th>0.10</th>
      <th>0.15</th>
      <th>0.20</th>
      <th>0.30</th>
      <th>0.50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>10</td>
      <td>0.043</td>
      <td>0.084</td>
      <td>0.177</td>
      <td>0.329</td>
      <td>0.498</td>
      <td>0.850</td>
      <td>0.998</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.040</td>
      <td>0.085</td>
      <td>0.299</td>
      <td>0.599</td>
      <td>0.851</td>
      <td>0.996</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.071</td>
      <td>0.156</td>
      <td>0.454</td>
      <td>0.781</td>
      <td>0.961</td>
      <td>0.999</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0.061</td>
      <td>0.200</td>
      <td>0.585</td>
      <td>0.881</td>
      <td>0.984</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>50</td>
      <td>0.054</td>
      <td>0.215</td>
      <td>0.670</td>
      <td>0.948</td>
      <td>0.999</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(16,6))
pwr_pd.plot()
plt.xlabel('Sample Size')
plt.ylabel('Power')
plt.title("Power analysis with varying sample and effect sizes")
plt.show();
```


    <Figure size 1152x432 with 0 Axes>



![png](index_files/index_25_1.png)


## Summary

In this lesson, you gained further practice with "statistical power" and how it can be used to analyze experimental design. You ran a simulation to determine the sample size that would provide a given value of power (for a given alpha and effect size). Running simulations like this, as well as further investigations regarding required sample sizes for higher power thresholds or smaller effect sizes is critical in designing meaningful experiments where one can be confident in the subsequent conclusions drawn.
