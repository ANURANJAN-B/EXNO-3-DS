## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
import pandas as pd

df=pd.read_csv("Encoding Data.csv")
df

  <img width="1018" height="570" alt="image" src="https://github.com/user-attachments/assets/e2b9714d-f3ad-4658-844d-38b79f5f3b45" />
  from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

<img width="740" height="352" alt="image" src="https://github.com/user-attachments/assets/4b96167c-4505-44ff-a615-7fc377c31aaa" />

df['bo2']=e1.fit_transform(df[["ord_2"]])
df

<img width="722" height="528" alt="image" src="https://github.com/user-attachments/assets/cae30dd4-ee16-47db-8095-c3ecbdcbf992" />

le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

<img width="644" height="569" alt="image" src="https://github.com/user-attachments/assets/391a0b6f-56b6-4cf6-9447-d9b16b5f91b2" />

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2

<img width="550" height="441" alt="image" src="https://github.com/user-attachments/assets/8506d871-0ae8-4237-8ffb-039833e54e21" />

pd.get_dummies(df2,columns=["nom_0"])

<img width="828" height="456" alt="image" src="https://github.com/user-attachments/assets/3443e47b-7dfc-446c-b7a9-b3689b53b1a5" />

pip install --upgrade category_encoders

<img width="1382" height="428" alt="image" src="https://github.com/user-attachments/assets/c64dd9c3-02b6-496d-99d3-a62ca378fb7f" />

from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df

<img width="622" height="439" alt="image" src="https://github.com/user-attachments/assets/8c9e2324-4e02-4cd5-a49a-f6d3400f47b5" />

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df

<img width="620" height="444" alt="image" src="https://github.com/user-attachments/assets/b010784c-a35f-45b7-bd88-1147ccda3d1c" />

dfb=pd.concat([df,nd],axis=1)
dfb

<img width="877" height="441" alt="image" src="https://github.com/user-attachments/assets/213f99e8-f2a8-439c-ad2a-3689e8902b73" />

from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

<img width="709" height="445" alt="image" src="https://github.com/user-attachments/assets/88c05963-63ce-41d0-9b9b-7c4359b10b9a" />

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df

<img width="986" height="498" alt="image" src="https://github.com/user-attachments/assets/06bbe7c9-3e94-4150-9222-c1c67337a45e" />

df.skew()

<img width="434" height="243" alt="image" src="https://github.com/user-attachments/assets/c3f45792-0510-4534-af2a-c76f0a33dffe" />

np.log(df["Highly Positive Skew"])

<img width="470" height="557" alt="image" src="https://github.com/user-attachments/assets/b30ccef6-c0ee-4e5e-a982-7b4dddd78ebe" />

np.reciprocal(df["Moderate Positive Skew"])

<img width="536" height="586" alt="image" src="https://github.com/user-attachments/assets/67b9432f-9dca-44ee-b440-6a3a4efcb8eb" />

np.sqrt(df["Highly Positive Skew"])

<img width="597" height="577" alt="image" src="https://github.com/user-attachments/assets/76ccc7a8-0cac-4002-9881-bb21b76522c2" />

np.square(df["Highly Positive Skew"])

<img width="651" height="567" alt="image" src="https://github.com/user-attachments/assets/b00db09e-50ae-4e7e-aca8-3ad6d9ecfa83" />

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

<img width="1276" height="517" alt="image" src="https://github.com/user-attachments/assets/39612dd3-a017-4547-bf0d-cd303242b10c" />

df.skew()

<img width="515" height="294" alt="image" src="https://github.com/user-attachments/assets/48eb653a-76f1-47b0-8003-7cc8f771b78a" />

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

<img width="615" height="354" alt="image" src="https://github.com/user-attachments/assets/6f03352f-471e-48e0-94b9-4586262eefcc" />

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

<img width="1343" height="556" alt="image" src="https://github.com/user-attachments/assets/ae92ecd8-b92b-46e4-b789-4721924108bd" />

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

<img width="910" height="561" alt="image" src="https://github.com/user-attachments/assets/2815a025-6ca7-4f0e-b34f-ce16cef49a3f" />


sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()


<img width="929" height="562" alt="image" src="https://github.com/user-attachments/assets/3514bf75-a7f4-4996-9bda-2d91453e6261" />

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

<img width="1062" height="557" alt="image" src="https://github.com/user-attachments/assets/fdc0df94-ff4f-494f-b9ee-8f172e84066a" />

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

<img width="828" height="546" alt="image" src="https://github.com/user-attachments/assets/e6a91604-cf7d-4e86-be39-e35b836e8e64" />

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()

<img width="1360" height="620" alt="image" src="https://github.com/user-attachments/assets/5c3e5e30-009a-4528-9186-cc87305e55ae" />

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

<img width="1090" height="558" alt="image" src="https://github.com/user-attachments/assets/cfa9abd2-c879-41a4-999d-2d42bef0391d" />
<img width="831" height="559" alt="image" src="https://github.com/user-attachments/assets/8303163e-5be2-4b9f-871d-f078d0aa69e3" />




# RESULT:
       # INCLUDE YOUR RESULT HERE

       
