import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk,Image
import tkinter.font as tkFont
import requests
import json
from datetime import datetime, timedelta

from sklearn.tree import DecisionTreeRegressor

window = tk.Tk()
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import plotly.graph_objects as go
import seaborn as sns
import plotly
import plotly.graph_objs as go
import matplotlib.dates as dates
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import seaborn as sns
import plotly
import plotly.graph_objs as go
import matplotlib.dates as dates
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)

window.title("Franfurt Stock Exchange")
window.minsize(900, 500)
def BGgraphic():
    load = Image.open(r"E:\University College Dublin\Business Analytics\Python\FinalProject\GUI\DONE\assignment\GUI\Screenshot_79.png")
    render = ImageTk.PhotoImage(load)

            # labels can be text or images
    img = Label(window, image=render)
    img.image = render
    img.place(x=730, y=0)

def clickMe():
    collectStartDate = str(startDate.get())
    collectStartDate = datetime.strptime(collectStartDate, "%Y/%m/%d")

    collectEndDate = str(endDate.get())
    collectEndDate = datetime.strptime(collectEndDate, "%Y/%m/%d")
    parameters = {"start_date": collectStartDate, "end_date": collectEndDate}

    if name.get() == 'Eon_Se'or name.get() == 'EON_X':
        Eon_Se = (requests.get("https://www.quandl.com/api/v3/datasets/FSE/EON_X.json?api_key=Jzzpuxf_rmJZyomGYZqx",
                    params=parameters))
        dataset = Eon_Se

    elif name.get() == 'Zooplus'or name.get() == 'ZO1_X':
        Zooplus = (requests.get("https://www.quandl.com/api/v3/datasets/FSE/ZO1_X.json?api_key=Jzzpuxf_rmJZyomGYZqx",
                    params=parameters))
        dataset = Zooplus
    elif name.get() == 'Wacker_Neuson_Se'or name.get() == 'WAC_X':
        Wacker_Neuson_Se = (requests.get("https://www.quandl.com/api/v3/datasets/FSE/WAC_X.json?api_key=Jzzpuxf_rmJZyomGYZqx",
                            params=parameters))
        dataset = Wacker_Neuson_Se
    elif name.get() == 'Vossloh' or name.get() == 'VOS_X':
        Vossloh = (requests.get("https://www.quandl.com/api/v3/datasets/FSE/VOS_X.json?api_key=Jzzpuxf_rmJZyomGYZqx",
                            params=parameters))
        dataset = Vossloh
    elif name.get() == 'Vtg_Aktiengesellschaft' or name.get() == 'VT9_X':
        Vtg_Aktiengesellschaft = (requests.get("https://www.quandl.com/api/v3/datasets/FSE/VT9_X.json?api_key=Jzzpuxf_rmJZyomGYZqx",
                            params=parameters))
        dataset = Vtg_Aktiengesellschaft
    elif name.get() == 'Takkt' or name.get() == 'TTK_X':
        Takkt = (requests.get("https://www.quandl.com/api/v3/datasets/FSE/TTK_X.json?api_key=Jzzpuxf_rmJZyomGYZqx",
                            params=parameters))
        dataset = Takkt
    elif name.get() == 'Tom_Tailor_Holding' or name.get() == 'TTI_X':
        Tom_Tailor_Holding = (requests.get("https://www.quandl.com/api/v3/datasets/FSE/TTI_X.json?api_key=Jzzpuxf_rmJZyomGYZqx",
                            params=parameters))
        dataset = Tom_Tailor_Holding
    elif name.get() == 'Sixt_Se_St' or name.get() == 'SIX2_X':
        Sixt_Se_St = (requests.get("https://www.quandl.com/api/v3/datasets/FSE/SIX2_X.json?api_key=Jzzpuxf_rmJZyomGYZqx",
                            params=parameters))
        dataset = Sixt_Se_St
    elif name.get() == 'Shw' or name.get() == 'SW1_X':
        Shw = (requests.get("https://www.quandl.com/api/v3/datasets/FSE/SW1_X.json?api_key=Jzzpuxf_rmJZyomGYZqx",
                            params=parameters))
        dataset = Shw
    elif name.get() == 'Puma_Se' or name.get() == 'PUM_X':
        Puma_Se = (requests.get("https://www.quandl.com/api/v3/datasets/FSE/PUM_X.json?api_key=Jzzpuxf_rmJZyomGYZqx",
                            params=parameters))
        dataset = Puma_Se
    else:
        pass

    myjson = json.dumps(dataset.json())
    myjson1 = json.loads(myjson)
    # print(myjson1)

    Json_ColumnName = myjson1['dataset']['column_names']
    Json_data = myjson1['dataset']['data']

    # 2. Convert the returned JSON object into a Python dictionary.
    mydict = []
    for values in Json_data:
        keys = Json_ColumnName
        mydict.append(dict(zip(keys, values)))

    global df
    df = pd.DataFrame(mydict, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Change', 'Traded Volume', 'Turnover',
                                       'Last Price of the day', 'Daily Traded Units', 'Daily Turnover'])
    # df.set_index('Date', inplace=True)
    df['Year'] = pd.DatetimeIndex(df['Date']).year
    df['Month'] = pd.DatetimeIndex(df['Date']).month
    df['Day'] = pd.DatetimeIndex(df['Date']).day
    # df['Month_Year'] = pd.DatetimeIndex(df['Date']).dt.to_period('M')
    # print(df.head())
    df = df.drop(['Last Price of the day', 'Daily Traded Units', 'Daily Turnover', 'Change'], axis=1)
    df = df.fillna(df.mean())

    getPrice()
    if collectCheckButtonValue.get() == 1:
        descriptiveWindow()
    getGraph()


def regression():
    global predictionValue
    predictionValue = tk.IntVar()
    linearRegression = Radiobutton(window, variable = predictionValue, value = 1, text = "Linear Regression")#, command = lambda: print(priceValue.get()))
    linearRegression.grid(column=1, row=11)
    linearRegression.font=("Courier", 10)
    multipleRegression = Radiobutton(window, variable = predictionValue, value = 2, text = "Multiple Regression")#, command = lambda: print(priceValue.get()))
    multipleRegression.grid(column=2, row=11)
    multipleRegression.font=("Courier", 10)
    decisionTree = Radiobutton(window, variable = predictionValue, value = 3, text = "Decision Tree")#, command = lambda: print(priceValue.get()))
    decisionTree.grid(column=3, row=11)
    decisionTree.font=("Courier", 10)
def getPrice():
    global getFeature
    if priceValue.get() == 1:
        getFeature = 'Open'
    elif priceValue.get() == 2:
        getFeature = 'Close'
    elif priceValue.get() == 3:
        getFeature = 'High'
    elif priceValue.get() == 4:
        getFeature = 'Low'
    else:
        pass
def getGraph():
    if graphValue.get() == 0:
        rawTimeSeries(getFeature)
    elif graphValue.get() == 1:
        linearTrendLine(getFeature)
    elif graphValue.get() == 2:
        movingAverages(collectedMovingAverage.get(),collectedMovingAverage1.get(),getFeature)
    elif graphValue.get() == 3:
        weightedMovingAverage(collectedMovingAverage1.get(),getFeature)
    elif graphValue.get() == 4:
        MACD(collectedMovingAverage.get(),getFeature)
    elif graphValue.get() == 5:
        correlationMatrix()
    elif graphValue.get() == 6:
        candelStick()
    elif graphValue.get() == 7:
        subplottingGraoh()
    elif graphValue.get() == 8:
        diplot()
    else:
        pass
    prediction()
def discriptiveStatistics(feature):
    discriptiveStats =  df[feature].describe()
    return discriptiveStats
def descriptiveWindow():
    window1 = Tk()
    window1.title("Descriptive Statitics")
    window1.geometry('250x200')
    labelNew = Label(window1, text="Descriptive Statistics for Selected Stock")
    labelNew.grid(column=2, row=2)
    buttonExit = Button(window1, text="Exit", command= lambda: window1.destroy()).grid(column=2, row=4)
    discriptiveStats = df[getFeature].describe()
    D = discriptiveStats
    D = Label(window1, text=D)
    D.grid(column=2, row=3)
def checkButtonDescriptive():
    global collectCheckButtonValue
    collectCheckButtonValue = IntVar()
    c = Checkbutton(window, text="Descriptive Statistics", variable=collectCheckButtonValue)
    c.place(x=500, y=0)
def dateStartLabel():
    global startDate
    startDatelabel = ttk.Label(window, text="Enter Start Date in YYYY/MM/DD")
    startDatelabel.grid(column=0, row=1)
    startDate = tk.StringVar()
    startDateEntered = ttk.Entry(window, width=15, textvariable=startDate)
    startDateEntered.grid(column=1, row=1)
def stockListLabel():
    global stockListLabel
    stockListLabel = ttk.Label(window, text="All the Stock Avaialable for Analysis\n1) Eon Se OR EON_X\n2) Zooplus OR Z01_X\n3) Wacker Neuson Se OR WAC_X\n4) Vossloh OR VOS_X\n5) Vtg Aktiengesellschaft OR VT9_X\n6) Takkt OR TTK_X\n7) Tom Tailor Holding OR TTI_X\n8) Sixt Se St OR SIX2_X\n9) Shw OR SW1_X\n10) Puma Se OR PUM_X")
    stockListLabel.grid(column=0, row=24)
    stockListLabel.config(font=("Courier", 10))
def dateEndLabel():
    global endDateLabel, endDate
    # endDateLabel = ttk.Label(window, text="Enter End Date")
    endDateLabel = ttk.Label(window, text="Enter End Date in YYYY/MM/DD")
    endDateLabel.grid(column=0, row=2)
    endDate = tk.StringVar()
    endDateEntered = ttk.Entry(window, width=15, textvariable=endDate)
    endDateEntered.grid(column=1, row=2)
def labelPrediction():
    global predictionLabel
    predictionLabel = ttk.Label(window, text="Select prediction by which you want to analyze")
    predictionLabel.grid(column=0, row=10)
def labelStock():
    global label,name
    label = ttk.Label(window, text="Enter Stock Symbol Or Stock Name")
    label.grid(column=0, row=0)
    name = tk.StringVar()
    nameEntered = ttk.Entry(window, width=15, textvariable=name)
    nameEntered.grid(column=1, row=0)
def labelPrediction():
    global predictlabel, predictionDays
    predictlabel = ttk.Label(window, text="Enter for how many days you want to predict from start date")
    predictlabel.grid(column=0, row=22)
    predictionDays = tk.IntVar()
    predictdays = ttk.Entry(window, width=15, textvariable=predictionDays)
    predictdays.grid(column=1, row=22)

def labelPrice():
    global priceLabel
    priceLabel = ttk.Label(window, text="Select price by which you want to analyze")
    priceLabel.grid(column=0, row=6)
def radioButtonPriceValue():
    global priceValue
    priceValue = tk.IntVar()
    open = Radiobutton(window, variable = priceValue, value = 1, text = "Open")#, command = lambda: print(priceValue.get()))
    open.grid(column=1, row=6)
    close = Radiobutton(window, variable = priceValue, value = 2, text = "Close")#, command = lambda: print(priceValue.get()))
    close.grid(column=2, row=6)
    high = Radiobutton(window, variable = priceValue, value = 3, text = "High")#, command = lambda: print(priceValue.get()))
    high.grid(column=3, row=6)
    low = Radiobutton(window, variable = priceValue, value = 4, text = "Low")#, command = lambda: print(priceValue.get()))
    low.grid(column=4, row=6)
    open.config(font=("Courier", 10))
    close.config(font=("Courier", 10))
    high.config(font=("Courier", 10))
    low.config(font=("Courier", 10))
def labelGraphs():
    global graphLabel
    graphLabel = ttk.Label(window, text="Select the graph for analysis")
    graphLabel.grid(column=0, row=7)
def ma_ma1():
    global maLabel, maLabel2, collectedMovingAverage, collectedMovingAverage1
    maLabel = ttk.Label(window, text="Enter number of days for which you want moving average")
    maLabel.grid(column=0, row=3)
    maLabel2 = ttk.Label(window, text="Enter 2nd number of days for which you want moving average")
    maLabel2.grid(column=0, row=4)
    collectedMovingAverage =  tk.IntVar()
    mA1Entered = ttk.Entry(window, width=15, textvariable=collectedMovingAverage)
    mA1Entered.grid(column=1, row=3)
    collectedMovingAverage1 = tk.IntVar()
    mA2Entered = ttk.Entry(window, width=15, textvariable=collectedMovingAverage1)
    mA2Entered.grid(column=1, row=4)
def radioButtonGraph():
    global graphValue
    graphValue = tk.IntVar()
    rawTimeSeries = Radiobutton(window, variable = graphValue, value = 0, text = "Raw Time Series")#, command = lambda: print(v.get()))
    rawTimeSeries.grid(column=1, row=7)
    linearTrendLines = Radiobutton(window, variable = graphValue, value = 1, text = "Linear trend lines")#, command = lambda: print(v.get()))
    linearTrendLines.grid(column=2, row=7)
    movingAverages = Radiobutton(window, variable = graphValue, value = 2, text = "Moving Averages")#, command = lambda: print(v.get()))
    movingAverages.grid(column=3, row=7)
    weightedMovingAverages = Radiobutton(window, variable = graphValue, value = 3, text = "Weighted Moving Averages")#, command = lambda: print(v.get()))
    weightedMovingAverages.grid(column=1, row=8)
    movingAverageConvergenceDivergence = Radiobutton(window, variable=graphValue, value=4, text="Moving Average Convergence/Divergence")#,command=lambda: print(v.get()))
    movingAverageConvergenceDivergence.grid(column=2, row=8)
    correlation = Radiobutton(window, variable=graphValue, value=5, text="Corelation Matrix")#,command=lambda: print(v.get()))
    correlation.grid(column=3, row=8)
    candelStick= Radiobutton(window, variable=graphValue, value=6, text="Candel Stick")#,command=lambda: print(v.get()))
    candelStick.grid(column=1, row=9)
    subplottingGraoh= Radiobutton(window, variable=graphValue, value=7, text="subplottingGraoh")#,command=lambda: print(v.get()))
    subplottingGraoh.grid(column=2, row=9)
    diplot= Radiobutton(window, variable=graphValue, value=8, text="DISPLOT")#,command=lambda: print(v.get()))
    diplot.grid(column=3, row=9)
    linearTrendLines.config(font=("Courier", 10))
    movingAverages.config(font=("Courier", 10))
    weightedMovingAverages.config(font=("Courier", 10))
    movingAverageConvergenceDivergence.config(font=("Courier", 10))
    correlation.config(font=("Courier", 10))
    candelStick.config(font=("Courier", 10))
    subplottingGraoh.config(font=("Courier", 10))
    diplot.config(font=("Courier", 10))
def buttonSubmit():
    global button
    button = ttk.Button(window, text="SUBMIT", command=clickMe)
    button.place(x=600, y=290)
def prediction():
    if predictionValue.get() == 1:
        LR1(getFeature)
    elif predictionValue.get() == 2:
        multipleLinearRegression()
    elif predictionValue == 3:
        decisionTree()
    else:
        pass
def resetAll():
    window.destroy()
def exit():
    global clearall
    clearall = ttk.Button(window, text='EXIT', command=resetAll)
    clearall.place(x = 700, y = 290)
def SetTodefault():
    startDate.set('')
    name.set('')
    endDate.set('')
    collectedMovingAverage.set(0)
    collectedMovingAverage1.set(0)
    priceValue.set(None)
    graphValue.set(None)
    predictionValue.set(None)
def reset():
    global resetall
    resetall = ttk.Button(window, text='RESET', command=SetTodefault)
    resetall.place(x = 500, y = 290)
def rawTimeSeries(feature):
    stockDate = df['Date']
    stockClosingPrice = df[feature]
    plt.figure(figsize=(10,4))
    ax = plt.plot(stockDate,stockClosingPrice, linestyle = 'dashed',color='#1e69a0')
    plt.xlim(right=15)  # adjust the right leaving left unchanged
    plt.xlim(left=1)  # adjust the left leaving right unchanged
    ax = plt.axes()
    # Setting the background color
    ax.set_facecolor("#212946")
    #ax.grid(False)
    plt.gca().xaxis.grid(True, color ="#252e50")
    plt.gca().yaxis.grid(True, color="#252e50")
    plt.gcf().autofmt_xdate()
    plt.title(name.get() + 'Raw Time Series', fontsize=20, fontname="Lucida Console")
    plt.xlabel('Date',fontsize=12, fontname="Lucida Console")
    plt.ylabel(feature,fontsize=12, fontname="Lucida Console")
    plt.tight_layout()
    plt.show()
#------------------------------------------Linear Trend Line------------------------------------------
def linearTrendLine(feature):

    #x = df['Day']
    x = dates.date2num(df['Date'])
    y = df[feature]
    plt.figure(figsize=(10, 4))
    plb.plot(x, y, 'o',color = "#1f74b1")
    ax = plt.axes()
    # Setting the background color
    ax.set_facecolor("#212946")
    plt.gca().xaxis.grid(True, color="#252e50")
    plt.gca().yaxis.grid(True, color="#252e50")
    # calc the trendline
    z = np.polyfit(x, y, 1)
    z = np.squeeze(z)
    p = np.poly1d(z)
    xx = np.linspace(x.min(),x.max(), 100)
    dd = dates.num2date(xx)
    plb.plot(dd, p(xx), "r--", color="#ff7f0e")
    #plb.plot(x, p(x), "r--",color = "#ff7f0e")
    plt.title(name.get() + ' Linear Trend Line', fontsize=20, fontname="Lucida Console")
    plt.xlabel('Day', fontsize=12, fontname="Lucida Console")
    plt.ylabel(feature, fontsize=12, fontname="Lucida Console")
    plt.tight_layout()
    plt.show()
#------------------------------------------Moving Averages(e.g. MA(n), with user-selectable n); ------------------------------------------
def movingAverages(firstValue,seconedValue,feature):
    n = int(firstValue)
    n2 = int(seconedValue)
    dfMA = pd.DataFrame(df[feature])
    dfMA['MA1'] = dfMA[feature].rolling(n).mean()
    dfMA['MA2'] = dfMA[feature].rolling(n2).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(dfMA[feature], label=feature)
    plt.plot(dfMA['MA1'], label='Moving Average for ' + str(n) + ' days')
    plt.plot(dfMA['MA2'], label='Moving Average for ' + str(n2) + ' days')
    leg = plt.legend(loc='best')
    for text in leg.get_texts():
        plt.setp(text, color='white')
    ax = plt.axes()
    # Setting the background color
    ax.set_facecolor("#212946")
    plt.title( ' Moving Average', fontsize=20, fontname="Lucida Console")
    # plt.xlabel('Day', fontsize=12, fontname="Lucida Console")
    plt.ylabel(feature, fontsize=12, fontname="Lucida Console")
    plt.gca().xaxis.grid(True, color="#252e50")
    plt.gca().yaxis.grid(True, color="#252e50")
    plt.tight_layout()
    plt.show()
#------------------------------------------Weighted Moving Average ------------------------------------------
def weightedMovingAverage(n,feature):
    # n = int(input("Enter number of days for which you want weighted moving average"))
    # plt.figure(16,8)
    plt.plot(df['Date'],df[feature], label = feature)
    plt.plot(df[feature].ewm(span=n).mean(), label ="WMA" + str(n) + 'days')
    # plt.plot(df['Close'].ewm(span=n2).mean(), label="WMA" + str(n2) + 'days')
    plt.xlim(right = 15)
    plt.xlim(left=1)
    plt.legend(loc = 2)
    plt.show()
#------------------------------------------Moving Average Convergence/Divergence (MACD) ------------------------------------------
def MACD(n,feature):
    df_MACD = pd.DataFrame(df[feature])
    Date = pd.DataFrame(df['Date'])
    Date = pd.to_datetime(Date['Date'], format='%Y/%m/%d')
    df_MACD['12-day EMA'] = df_MACD[feature].rolling(12).mean()
    df_MACD['26-day EMA'] = df_MACD[feature].rolling(26).mean()
    df_MACD['MACD Line'] = df_MACD['12-day EMA'] - df_MACD['26-day EMA']
    df_MACD['Signal Line EMA'] = df_MACD['MACD Line'].rolling(n).mean()
    df_MACD['MACD Histogram'] = df_MACD['MACD Line'] - df_MACD['Signal Line EMA']
    plt.figure(figsize=(20, 15))
    plt.title("Moving Average Convergence/Divergence (MACD)")
    plt.plot(Date, df_MACD['12-day EMA'], label='12 day Moving Average')
    plt.plot(Date, df_MACD['26-day EMA'], label='26 day Moving Average')
    plt.plot(Date, df_MACD['MACD Line'], label='MACD Line')
    plt.plot(Date, df_MACD['Signal Line EMA'], label='Signal Line')
    plt.plot(Date, df_MACD['MACD Histogram'], label='MACD Histogram')
    plt.legend(loc='best')
    plt.show()


#------------------------------------------Correlation Matrix ------------------------------------------
def correlationMatrix():

    import seaborn as sn
    corrMatrix = df.corr()
    print(corrMatrix)
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    # plt.boxplot(x=df["Open"])
    # plt.show()
    # sn.boxplot(x=df["High"], y=df["Low"])
    # plt.show()

def candelStick():
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],open=df['Open'],high=df['High'], low=df['Low'],close=df['Close'])])
    fig.show()

def subplottingGraoh():
    fig, axes = plt.subplots(nrows = 2, ncols=1, figsize=(15,15))

    df[['Open', 'Close']].head(20).plot(kind='bar', ax = axes[0],figsize=(16,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')  #008fd5
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black') #fb5c3f
    # axes[0].set_xlabel('Open', fontsize=12, fontname="Lucida Console")
    # axes[0].set_ylabel('Close', fontsize=12, fontname="Lucida Console")
    axes[0].legend(loc='best')

    df[['High', 'Low']].head(20).plot(kind='bar', ax = axes[1],figsize=(16,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # plt.xlabel('High', fontsize=12, fontname="Lucida Console")
    # plt.ylabel('Low', fontsize=12, fontname="Lucida Console")
    axes[1].legend(loc='best')
    plt.show()

def diplot():
    fig, axs = plt.subplots(ncols=2, nrows=2)
    sns.distplot(df['Open'],ax=axs[0,0],kde=True, color='red')#, bins=100)
    sns.distplot(df['Close'],ax=axs[0,1],kde=True, color='red')#, bins=100)
    sns.distplot(df['High'],ax=axs[1,0],kde=True, color='red')#, bins=100)
    sns.distplot(df['Low'],ax=axs[1,1],kde=True, color='red')#, bins=100)
    plt.show()

def multipleLinearRegression():
    X = df[['Month','Year','Open', 'High', 'Low']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print(regressor.coef_)
    # print('Slope: ', np.asscalar(np.squeeze(model.coef_)))
    # The Intercept
    print('Intercept: ', regressor.intercept_)
    print('model score: ', regressor.score(X_train, y_train))
    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
    print(coeff_df)

    y_pred = regressor.predict(X_test)
    df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df2)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Root Squared Error:', metrics.r2_score(y_test, y_pred))


def decisionTree():
    dTDF = df["Close"]
    dTDF = pd.DataFrame(dTDF)
    future_days = predictionDays
    dTDF['Prediction'] = dTDF['Close'].shift(-future_days)
    X = np.array(dTDF.drop(['Prediction'], 1))[:-future_days]
    print(X)
    y = np.array(dTDF['Prediction'])[:-future_days]
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    tree = DecisionTreeRegressor().fit(x_train, y_train)
    x_future = dTDF.drop(['Prediction'], 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    print(x_future)
    tree_prediction = tree.predict(x_future)
    print(tree_prediction)
    predictions = tree_prediction
    valid = dTDF[X.shape[0]:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16, 8))
    plt.title("Decision Tree")
    plt.xlabel('Days')
    plt.ylabel('Close')
    plt.plot(dTDF['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(["Original", "Valid", 'Predicted'])
    plt.title('Decision Tree Regression', fontsize=20, fontname="Lucida Console")
    leg = plt.legend(loc='best')
    for text in leg.get_texts():
        plt.setp(text, color='white')
    ax = plt.axes()
    # Setting the background color
    ax.set_facecolor("#212946")
    plt.gca().xaxis.grid(True, color="#252e50")
    plt.gca().yaxis.grid(True, color="#252e50")
    plt.tight_layout()
    plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, tree_prediction))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, tree_prediction))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, tree_prediction)))
    print('Root Squared Error:', metrics.r2_score(y_test, tree_prediction))

def LR1(getFeature):
    global df_sampleLR
    df['Date'] = dates.date2num(df['Date'])
    # print("after num2date")
    # print(df['Date'])
    X = df['Date']
    X = X.values.reshape(-1, 1)
    y = df[getFeature]
    # print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # print(X_train.shape)
    model = LinearRegression()
    model.fit(X_train, y_train)
    # The coefficient
    # print(model.coef_)
    # print('Slope: ', np.asscalar(np.squeeze(model.coef_)))
    # The Intercept
    # print('Intercept: ', model.intercept_)
    InterceptLR = model.intercept_
    # print('model score: ', model.score(X_train, y_train))
    modelScoreLR = model.score(X_train, y_train)
    # Create test arrays
    X_test = df[['Date']]
    # print(X_test)  # ,'Month','Year','Open', 'High', 'Low', 'Traded Volume', 'Turnover']]
    y_test = df[getFeature]

    y_pred = model.predict(X_test)
    # print(y_pred)

    df['Date'] = dates.num2date(df['Date'])
    df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    # print(df1)
    df1 = df1.reset_index()
    df1['Date'] = (df['Date'])
    # print(df1[:predictionDays.get()])
    df_sampleLR = pd.DataFrame()
    df_sampleLR = df1[:predictionDays.get()]
    # print(df_sampleLR)
    df1[['Actual', 'Predicted']].plot()
    global fig
    fig = df_sampleLR[['Actual', 'Predicted']].plot()
    plt.title("Linear Regression")
    plt.show()
    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    MAELR = metrics.mean_absolute_error(y_test, y_pred)
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    MSELR = metrics.mean_squared_error(y_test, y_pred)
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    RMSELR= np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # print('Root Squared Error:', metrics.r2_score(y_test, y_pred))
    RsquareLR = metrics.r2_score(y_test, y_pred)
    AllValues = print(df_sampleLR,RMSELR, RsquareLR)
    print(AllValues)
    return RMSELR
def main():
    BGgraphic()
    checkButtonDescriptive()
    dateStartLabel()
    labelStock()
    stockListLabel()
    dateEndLabel()
    labelPrediction()
    labelPrice()
    radioButtonPriceValue()
    labelGraphs()
    labelPrediction()
    ma_ma1()
    radioButtonGraph()
    regression()
    buttonSubmit()
    exit()
    reset()
    window.mainloop()
# main()