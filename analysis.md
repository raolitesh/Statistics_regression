### <center>Liteshwar Rao</center>
# <a name="Contents">Table of Contents</a>
- [1 Introduction](#Section-1)
<p>

- [2 Data Summary](#Section-2)
    - [2.1 Imports and Reading Data](#Section-21)
    - [2.2 Data Understanding](#Section-22)
<p>

- [3 Exploratory Data Analysis (EDA)](#Section-3)
    - [3.1 Summary Statistics for Each Attribute](#Section-31)
    - [3.2 Country Count for Individual Continents](#Section-32)
    - [3.3 Development of World Population Since 1970](#Section-33)
    - [3.4 Linear Regression of the World Population](#Section-34)
        - [3.4.1 Estimated Regression Line](#Section-341)
        - [3.4.2 Point Estimate of $\sigma$](#Section-342)
        - [3.4.3 Test of Hypothesis $H_0: \beta_1 = \beta_{1,0}$ vs. $H_a: \beta_1 \ne \beta_{1,0}$](#Section-343)
        - [3.4.4 95% CI for the Slope of the Regression Line](#Section-344)
        - [3.4.5 95% Prediction Interval for $population$, when $year = 2005$](#Section-345)
        - [3.4.6 95% Prediction Band](#Section-346)
        - [3.4.7 Coefficient of Determination](#Section-347)
    - [3.5 Population Development of Each Continents](#Section-35)
    - [3.6 Population Distribution in Each Years](#Section-36)
    - [3.7 Area Distribution on Each Continents](#Section-37)
<p>

- [4 Research Questions](#Section-4)
    - [4.1 Which twenty countries are most populous in the world?](#Section-41)
    - [4.2 Which twenty countries have the highest population growth?](#Section-42)
    - [4.3 What is the population growth or decline of the 20 most populous countries?](#Section-43)
    - [4.4 Which twenty countries have the highest population density?](#Section-44)
    - [4.5 Which country belongs to both lists of the twenty countries with the highest population growth and the highest population density?](#Section-45)
    - [4.6 What was the world's population growth rate in % since 1980?](#Section-46)
    - [4.7 Is this world's population growth increasing or decreasing over the years?](#Section-47)
    - [4.8 What world's population can be expected in 2050?](#Section-48)
        - [4.8.1 95% Prediction Interval](#Section-481)
<p>
    
- [5 Conclusion](#Section-5)
<p>

- [References](#Section-6)
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
# <a name="Section-1">1 Introduction</a>
[top](#Contents)
For my project, I have chosen the data set of the world population, respectively the population of individual countries. This data set records the population of each country since 1970 until 2022 in the following distribution of years: 1970, 1980, 1990, 2000, 2010, 2015, 2020 and 2022. Furthermore, every record contains the following items: rank, country code (CCA3), country name, capital, continent, area, density, growth rate and world population percentage.
<p>
In my project, I will answer the following research questions:

- Which twenty countries are the most populous in the world?
- What is the population growth or decline of the 20 most populous countries?
- Which twenty countries have the highest population growth?
- Which twenty countries have the highest population density?
- Which country belongs to both lists of the twenty countries with the highest population growth and the highest population density?
- What was the world's population growth rate in % since 1980?
- Is this world's population growth increasing or decreasing over the years?
- What world's population can be expected in 2050?
# <a name="Section-2">2 Data Summary</a>
[top](#Contents)
#### My dataset was obtained from https://worldpopulationreview.com/
```python
# importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from scipy.stats import norm
from scipy.stats import t

pd.set_option('display.max_rows', 200)
import warnings
warnings.filterwarnings("ignore")
```
## <a name="Section-21">2.1 Imports and Reading Data</a>
