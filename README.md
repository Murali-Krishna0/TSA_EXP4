# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 
13/05/2026
### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
df = pd.read_excel("/content/Sales_v1.xlsx")
X = df[' Sales'].dropna()
plt.rcParams['figure.figsize'] = [12, 6]
plt.plot(X)
plt.title('Original Sales Data')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.grid(True)
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=min(len(X)//4, 40), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=min(len(X)//4, 40), ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

N = 1000

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.grid(True)
plt.show()

plot_acf(ARMA_1)
plt.title('ACF of ARMA(1,1)')
plt.show()

plot_pacf(ARMA_1)
plt.title('PACF of ARMA(1,1)')
plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']

theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.grid(True)
plt.show()
plot_acf(ARMA_2)
plt.title('ACF of ARMA(2,2)')
plt.show()

plot_pacf(ARMA_2)
plt.title('PACF of ARMA(2,2)')
plt.show()
```
### OUTPUT:
<img width="1003" height="552" alt="image" src="https://github.com/user-attachments/assets/0fada30a-99f6-433e-8636-7edee41897a0" />

<img width="1197" height="298" alt="image" src="https://github.com/user-attachments/assets/545df865-98b0-420b-9ce3-36044eaf3def" />

<img width="1196" height="291" alt="image" src="https://github.com/user-attachments/assets/51b3d19f-192c-4a0e-8807-ea8fabfa7f75" />

<img width="987" height="521" alt="image" src="https://github.com/user-attachments/assets/dd780be8-3971-4685-bfaf-0c31ec75c20f" />

<img width="1007" height="537" alt="image" src="https://github.com/user-attachments/assets/c95a87d8-bfce-4b2e-805d-e98a587c9521" />

<img width="999" height="529" alt="image" src="https://github.com/user-attachments/assets/ce8d10da-b5cf-4d59-b557-a6ea4b5bf06d" />

<img width="990" height="537" alt="image" src="https://github.com/user-attachments/assets/862bc65c-668d-4f44-b51e-851d080735ce" />

<img width="1007" height="536" alt="image" src="https://github.com/user-attachments/assets/cf776520-0514-48dc-8b4d-91f7e3773705" />

<img width="1006" height="532" alt="image" src="https://github.com/user-attachments/assets/0619c582-e3a3-4170-991b-c5e33a9e02b2" />

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
