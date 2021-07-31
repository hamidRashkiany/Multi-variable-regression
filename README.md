# Multi-variable-regression
Estimate Stock_Index_Price
'''
Date:2021-07-21
Author: Hamid Rashkiany
Description: Multiple variable regression implementation. In this practice, we will work with one dataset that created by ourself. The target is 
Stock_Index_Price (This is our dependent variable). The value of target is depending on two other variables as Interest_Rate and Unemployment_Rate.
These are our features or independent variables. There is a linear relationship (correlation) between independent variable and dependent variable.
Thus we define one linear regression model to estimate the output for new data. At the end we defined manually two new data and we estimate the 
related output. Also we compare our model output with statsmodel.
Reference: https://datatofish.com/multiple-linear-regression-python/
'''
from tkinter.constants import CENTER
import pandas as pd
import numpy as np
from matplotlib.backends._backend_tk import FigureCanvasTk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import tkinter as tk

'''
Create a pandas data frame. To make a data frame in Pandas, first we need to input our data as a set of values
as below'''
y2017=[2017]*12
y2016=[2016]*12
month=[]
for i in range(12,0,-1):
    month.append(i)
month=month+month
year=y2017+y2016
data={
    "Year":year,
    "Month":month,
    "Interest_Rate":[2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
    "Unemployment_Rate":[5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
    "Stock_Index_Price":[1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]
}
dataframe=pd.DataFrame(data,columns=["Year","Month","Interest_Rate","Unemployment_Rate","Stock_Index_Price"])
#After make dataset here print it to check it that it works or not.
print(dataframe)
'''
This is a multiple variable regression. It means output value estimation depends to more than one input variable.
each row in our data set is related to one input x. For more clarity, the first row say for x1={x11 year 2017, 
x12 month 12, with x13 interest_rate 2.75, x14 unimployment_rate 5.3 the output y as stock_price y1 will be 1464}
it means here each input will have 4 features and there are the inputs for 24 months (Two years), thus:
n=4 (number of featurs), m= 24 (number of samples)
This practical example is goiong to do one multiple variable regression. Hence before any further steps, we need to check is there any linear
relationship between output and inputs or not.One way to check linear relationship is data observation. Scatter plot is one robust way to observe 
data. So here we are going to obsereve the relation between dependent variable (as Stock_Index_Price) and indepent variable (as Interest Rate).
Also we will obsereve the relation between  Stock_Index_Price and Unemployment_Rate.

'''
#Observe relation between Interest-Rate and Stock_Index_Price
plt.scatter(dataframe["Interest_Rate"],dataframe["Stock_Index_Price"],color="r")
plt.xlabel("Interest_Rate")
plt.ylabel("Stock_Index_Price")
plt.title("Observe relation between Interest-Rate and Stock_Index_Price")
plt.grid(True)
plt.show()
#Observe relation between Unemployment-Rate and Stock_Index_Price
plt.scatter(dataframe["Unemployment_Rate"],dataframe["Stock_Index_Price"],color="b")
plt.title("Observe relation between Unemployment-Rate and Stock_Index_Price")
plt.xlabel("Unemployment_Rate")
plt.ylabel("Stock_Index_Price")
plt.grid(True)
plt.show()
'''
After observation of both plots, we can see that there is a linear relationship between independen variables and dependent variable in our dataset.
In first plot (Observe relation between Interest-Rate and Stock_Index_Price), by increasing the value of Interest_Rate, the amount of Stock_Index_Price
increases as well. In second plot (Observe relation between Unemployment-Rate and Stock_Index_Price) the value of Stock_Index_Price is decaying by rising
the Unemployment_rate. Thus the multiple linear regression for this dataset can denote by:
htheta(x)=theta0+theta1*X1+theta2*X2
X1=Interest_Rate
X2=Unemployment_Rate
theta0= intercept in stright line equation
theta1=The slop of line for Interest_Rate and Stock_Index_Price
theta2= The slop of line for Unemployment_Rate and Stock_Index_Price
Now from sklearn library we can import our Linear regression and calculate above prameters.
'''
#define our X matrices
X=dataframe[["Interest_Rate","Unemployment_Rate"]]
Y=dataframe["Stock_Index_Price"]
#If you want to split out the X1, X2 and ... parameters from dataset, utilize below format:
X1=X.iloc[:,0]
X2=X.iloc[:,1]
print(X)
print(Y)
#Now call linear model from sklearn
#The ordinary least square linear regression is a sort of regression that  minimize the below cost function:
# min(||Xw-Y||^2) where Xw is our stimated value and Y is the real value of output. Thus in machine learning in least linear 
#regression, we are going to find the optimum value for w. This optimum value will minimize  ||Xw-Y||^2 as cost function. 
#As cost function has less amount, it means the gap between real value and estimated value will be less. There are many methods 
#that can apply to above equation and extract the optimum value for w such as descent gradient. Here in this parctical example
#I am not going to get deep in method. I will just apply the linear_model.LinearRegression() and extract the optimum values for 
#my parameters according to my dataset. linear_model.LinearRegression() will fit on X and Y and returns the optimum values on two
#objects that I need to decompose those parameters and put in my linear equation.
reg=linear_model.LinearRegression()
reg.fit(X,Y)
print("The optimum values of theta1 and theta2:","\n",reg.coef_)
print("The optimum value for theta0:","\n",reg.intercept_)
#After run the code the output will be:
# The optimum values of theta1 and theta2:
#  [ 345.54008701 -250.14657137]
# The optimum value for theta0:
#  1798.4039776258546
#Now let's substitute these values in my line equation:
#My model: htheta(x)=theta0+theta1*X1+theta2*X2 => h(x)=1798.4039776258546+(345.54008701)*X1+(-250.14657137)*X2
#Now assume we will have new datapints as : New_Interest_Rate = 2.75 and New_Unemployment_Rate = 5.3 and according to these new data
#we want to predict the relevant Y. There is another function in sklearn as regr.predict() that receive the new data points and return 
#the prediction value.Let's implement our model for those new data as below:
New_Interest_Rate=2.75
New_Unemployment_Rate=5.3
prediction=reg.predict([[New_Interest_Rate,New_Unemployment_Rate]])
print("prediction for new data:","\n",prediction)
'''
#Another way to check your method and it is correction is utilizing statsmodel. Statsmodel is a Python module that provide many functions
#which are use in estimation and other regression models in data science. The output of its models give us a good scale to compare our output
#with other models. You find one compelet documentary about statsmodel in : https://www.statsmodels.org/stable/gettingstarted.html
# But here I will give you one summary about this module. In this module we only use functions that provided by statsmodel, pandas and patsy.
# First of all we need to opt the model that we want to utiliz in our model. One of very well known model in statsmodel is Ordinary Least Squares 
#regression presents by OLS. The second steps is denoting our input and output matrices. In machine learning, input matrices contains the values of 
#independent values (other names in other sources predictor, regresspr) and output matrix involves the values of dependent variable (response, regressand
# in other sources). These matrices in statsmodel illustrates by endogenous variable matrix (endog) for output and exogenous variable matrix for 
# input (exog matix). At the end we need to fit our data in model and extract the parmeter from model and print them out.
#Fitting data has three steps:
#1. Use the model class to describe the model. 2. Fit the model using a class methos 3. Inspect the result to extract the parameters.
# Implementation of OLS model in our dataset:
'''
X=sm.add_constant(X) #adding a constant
model=sm.OLS(Y,X) #Describe the model
res=model.fit()
print(res.summary())
'''
#After printing the result, we can observe the values of const and Interes_Rate and Unemployment_Rate  in coef column and compare them
# with optimum values that we calculate with sklearn module for multiple variable linear regression and we can conclude that all items 
# have same values.
# Optimum values calculated by sklrean:
# The optimum values of theta1 and theta2:
#  [ 345.54008701 -250.14657137]
# The optimum value for theta0:
#  1798.4039776258546
# The results of statsmodels:
#                       coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------
# const              1798.4040    899.248      2.000      0.059     -71.685    3668.493
# Interest_Rate       345.5401    111.367      3.103      0.005     113.940     577.140
# Unemployment_Rate  -250.1466    117.950     -2.121      0.046    -495.437      -4.856

'''
###################################################################
'''
#In this section of this tutorial, I prefer to make one GUI (Graphical User Interface) for my code that user can input new data and receive the
#pridiction in output. To implement this section, I will utilize tkinter module.
#To create one GUI, the first step is defining one interface page that user can make communicate with your program. 
#  1. Create Canvas:
# The fisrt step of making one GUI in building one canvas screen.You can define one canvas as below:
# canvasName=tk.Canvas(root,width= .... , height= ....)
# The size of canvas screen can change by altering the weidth and high atributes. Consider that all bottun, label, text boxs 
#and etc will put in canvas screen. 

'''
#Create our GUI screen by Canvas widget
root=tk.Tk()
myGUIwindow=tk.Canvas(root,width=800,height=300)
myGUIwindow.pack()
#Put labek for our GUI
label1 = tk.Label(root, text='Graphical User Interface')
label1.config(font=('Arial', 20))
myGUIwindow.create_window(400, 50, window=label1)
#Calvulate intercept by sklearn and put a label to present it
interceptResult=("Intercept: ", reg.intercept_)
labelIntercept=tk.Label(root,text=interceptResult,justify=CENTER)
myGUIwindow.create_window(260,220,window=labelIntercept)
#Calculate coefficients by sklearn and make a label to present it
coefficientsResult=("Coefficients: ", reg.coef_)
labelCoefficients=tk.Label(root,text=coefficientsResult,justify=CENTER)
myGUIwindow.create_window(260,240,window=labelCoefficients)
'''
After found the optimum values and substitute in our equation, here we are going to make to entry and will ask user put his new data values
through entry to model.
'''
#New_Interest_Rate label and input box
labelNewInterestRate=tk.Label(root,text="Type the interest rate: ")
myGUIwindow.create_window(100,100,window=labelNewInterestRate)
entryNewInterestRate=tk.Entry(root)
myGUIwindow.create_window(270,100,window=entryNewInterestRate)
#New_Unemployment_Rate label and input box
labelNewUnemploymentRate=tk.Label(root,text="Type the unemployment rate: ")
myGUIwindow.create_window(120,120,window=labelNewUnemploymentRate)
entryUnemploymentRate=tk.Entry(root)
myGUIwindow.create_window(270,120,window=entryUnemploymentRate)
def values():
    global New_Interest_Rate
    New_Interest_Rate=float(entryNewInterestRate.get())

    global New_Unemployment_Rate
    New_Unemployment_Rate=float(entryUnemploymentRate.get())

    predictionResult=("Predicted Stock Index Price: ", reg.predict([[New_Interest_Rate,New_Unemployment_Rate]]))
    labelPrediction=tk.Label(root,text=predictionResult,bg="orange")
    myGUIwindow.create_window(260,280,window=labelPrediction)

btnCalculatePrediction=tk.Button(root,text="Calculate Prediction",command=values,bg="white",fg="black")
myGUIwindow.create_window(270,150,window=btnCalculatePrediction)

#Plotting our plot in our GUI
#Plot the first scatter plot
figure3=plt.figure(figsize=(5,4),dpi=100)
ax3=figure3.add_subplot(111)
ax3.scatter(dataframe["Interest_Rate"],dataframe["Stock_Index_Price"],color="r")
scatter3=FigureCanvasTk(figure3,root)
scatter3.get_tk_widget().pack(side=tk.RIGHT,fill=tk.BOTH)
ax3.legend(["Stock_Index_Price"])
ax3.set_xlabel("Interest Rate")
ax3.set_title("Interest Rate vs Stock Index Price")

#plot 2nd scatter 
figure4 = plt.Figure(figsize=(5,4), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(dataframe['Unemployment_Rate'].astype(float),dataframe['Stock_Index_Price'].astype(float), color = 'g')
scatter4 = FigureCanvasTk(figure4, root) 
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.legend(['Stock_Index_Price']) 
ax4.set_xlabel('Unemployment_Rate')
ax4.set_title('Unemployment_Rate Vs. Stock Index Price')


root.mainloop()
