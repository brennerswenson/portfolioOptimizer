import pandas as pd
# added extra line for api version compatability
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import statsmodels.api as sm
from scipy.optimize import minimize
from jupyterthemes import jtplot
import sys
import time
jtplot.style('grade3')


# FUNCTIONS
##############################################################################


def get_date(input_string):  # gets date from user
    try:
        date = pd.to_datetime(input(input_string))
        return date
    except ValueError:
        print('Invalid date format, please try again')
        print('\n')
        get_date(input_string)


def ticker_check(ticker):  # returns ticker if valid, else loops infinitely
    global all_tickers_list
    if ticker in all_tickers_list['Tickers'].values:
        print('\n')
        print('Ticker validated and stock added')
        print('\n')
        return ticker
    else:
        print('Invalid ticker, please try again')
        print('\n')
        ticker_check(
            input('Enter the next ticker you would like to add: ').upper())


def date_check():  # checks if date is valid
    global end
    global start
    try:
        if end < start:
            end = get_date(
                'End date cannot be less than start date. Please enter valid end date ')
            print('\n')
            date_check()
        else:
            pass
    except TypeError:
        start = get_date('Please enter the starting date again for the portfolio data. Format: YYYY-MM-DD '
                         )
        print('\n')
        end = get_date(
            'Please enter the ending date again for the portfolio data. Format: YYYY-MM-DD ')
        print('\n')
        date_check()


def ask_user_loop():  # main loop with user logic
    global stock_tickers
    global add_more_stocks
    if add_more_stocks == 'y':
        new_stock = input(
            'Enter the next ticker you would like to add: ').upper()
        if new_stock in stock_tickers:
            print('Duplicate stock, please enter a unique stock ticker')
            print('\n')
            ask_user_loop()
        else:
            stock_tickers.append(ticker_check(new_stock))
            add_more_stocks = input(
                'Would you like to add more stocks? y/n ').lower()
            print('\n')
            ask_user_loop()
    elif add_more_stocks == 'n':
        stock_tickers = list(filter(
            lambda ticker: ticker is not None, stock_tickers))  # filters out any None values that were added during the user input process

        print(bcolors.OKGREEN + str(stock_tickers) + bcolors.ENDC)

        print('\n')

        ticker_confirmation = input(
            'Does your portfolio look correct? y/n \n\nPress n to add more stocks or y to continue: ').lower()

        if ticker_confirmation == 'y':
            print('\n')
            pass
        else:
            add_more_stocks = 'y'
            print('\n')
            ask_user_loop()
    else:

        add_more_stocks = input(
            'Invalid input, would you like to add more stocks? y/n ').lower()
        print('\n')
        ask_user_loop()


def get_return_volatility_SR(weights):  # function that is minimized
    weights = np.array(weights)  # makes sure that input is a np array
    port_return = np.sum(log_ret.mean() * weights * 252)  # annualizes return
    # uses linear algebra to calculate portfolio variance
    port_volatility = np.sqrt(
        np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sharpe_ratio = port_return / port_volatility
    return np.array([port_return, port_volatility, sharpe_ratio])


# returns the negative sharpe ratio, as to use the minimizer function
def negative_sharpe(weights):
    return get_return_volatility_SR(weights)[2] * -1


def check_weight_sum(weights):
    return np.sum(weights) - 1  # needs to return zero


def minimize_volatility(weights):  # creates function to minimize stdev
    return get_return_volatility_SR(weights)[1]  # volatility is at index 1

# adds color options to terminal text


# terminal colors

class bcolors:

    HEADER = '\033[95m'

    OKBLUE = '\033[94m'

    OKGREEN = '\033[92m'

    YELLOW = '\033[93m'

    RED = '\033[91m'

    ENDC = '\033[0m'

    BOLD = '\033[1m'

    UNDERLINE = '\033[4m'


# END FUNCTION DEFINITIONS
#############################################################################


# load possible tickers, static file
all_tickers_list = pd.read_csv('Tickers.csv', header=0)

print('\n')
print('Welcome to Portfolio Optimizer, made by Brenner Swenson')
print('\n')

# start date
start = get_date(
    'Please enter the starting date for the portfolio data. Format: YYYY-MM-DD ')
print('\n')

# end date
end = get_date(
    'Please enter the ending date for the portfolio data. Format: YYYY-MM-DD ')
print('\n')


# checks date inputs
date_check()

# stock tickers in portfolio
stock_tickers = []
# first stock
stock_tickers.append(ticker_check(
    input('Enter the first ticker of your portfolio: ').upper()))


# second stock
stock_tickers.append(ticker_check(
    input('Enter the second ticker of your portfolio: ').upper()))


add_more_stocks = input('Would you like to add more stocks? y/n ').lower()
print('\n')

ask_user_loop()

time.sleep(1)

print('Requesting portfolio data from {} to {}...'.format(
    start, end))
print('\n')

stock_df = pd.DataFrame()  # creates empty DataFrame to add stocks to

# creates DataFrame with all stocks closing prices
for ticker in stock_tickers:
    try:
        temp = web.DataReader(ticker, 'morningstar', start, end, retry_count=1)[
            'Close']  # API call, filters for just the close column
        temp.index = pd.MultiIndex.droplevel(
            temp.index)  # drops multi level index
        stock_df[ticker] = temp
    except:
        # checks for errors from morningstar API if any
        print('{} information could not be loaded and was omitted from portfolio'.format(ticker))

# obtain market data for the period, S&P500 specifically
spy = pd.DataFrame()

# add close column to spy dataframe
spy['Close'] = web.DataReader(
    'SPY', 'morningstar', start, end, retry_count=1)['Close']

# remove SPY multi-index
spy.index = pd.MultiIndex.droplevel(
    spy.index)

# calculate SPY's hypothetical growth of $1000
spy['TOTAL'] = (spy['Close'] / spy['Close'][0]) * 1000

# prints beginning and end of data for user to see
print(stock_df.head())
print('\n')
print(stock_df.tail())
print('\n')
print('Data obtained successfully')
print('\n')

time.sleep(1)

# calculates log returns for entire dataframe
log_ret = np.log(stock_df / stock_df.shift(1))

# Optimizer constraints

# tells the statsmodels minimize function we are minimizing an equation
# then tells the function what the equation is
constraints = ({'type': 'eq', 'fun': check_weight_sum})

# creates dynamic length bound tuple
# makes sure that the weights are only between zero and 1, long only
bounds = []
for i in range(len(stock_tickers)):
    bounds.append((0, 1))
bounds = tuple(bounds)

# initial guess for weights, equally allocated weights

init_guess = []
for i in range(len(stock_tickers)):
    if i < len(stock_tickers) - 1:
        init_guess.append(1 / len(stock_tickers))
    else:
        init_guess.append(1 - sum(init_guess))


print('Optimizing...')
print('\n')
time.sleep(1)
# run optimizer and save the results to a variable
# uses sequential least squares programming (SLSQP)
optimized_results = minimize(negative_sharpe, init_guess,
                             method='SLSQP', bounds=bounds, constraints=constraints)

# reverse the sign of the function, as the optimizer minimizes the sharpe ratio
sharpe_ratio_result = -optimized_results.fun

# returning results, green text added
print("Optimal Sharpe Ratio of selected portfolio is: " + bcolors.OKGREEN + "{:.4f}".format(
    sharpe_ratio_result) + bcolors.ENDC)
print('\n')

print('Plotting returns and creating Efficient Frontier...')
print('\n')

# create a dictionary with the weights assigned to the tickers
weights_results_dict = dict(zip(stock_df.columns, optimized_results.x))

# creates columns in portfolio dataframe with
# hypothetical growth over requested time period for each stock
# also creates a simple moving average
opt_return_df = pd.DataFrame()
for ticker, weight in zip(stock_df.columns, optimized_results.x):
    opt_return_df[ticker + ' ' + 'allocated value'] = (
        stock_df[ticker] / stock_df.iloc[0][ticker]) * weight * 1000
opt_return_df['TOTAL'] = opt_return_df.sum(axis=1)
opt_return_df['TOTAL 3Mo SMA'] = opt_return_df['TOTAL'].rolling(62).mean()


# creates control df with result of equal asset allocation
equal_weight_return_df = pd.DataFrame()
for ticker, weight in zip(stock_df.columns, init_guess):
    equal_weight_return_df[ticker + ' ' + 'allocated value'] = (
        stock_df[ticker] / stock_df.iloc[0][ticker]) * weight * 1000
equal_weight_return_df['TOTAL'] = equal_weight_return_df.sum(axis=1)

# create axes, modify y axis formatting
fig, ax = plt.subplots(figsize=(16, 9))
ax.yaxis.set_major_formatter(
    tkr.FuncFormatter(lambda y,  p: format(int(y), ',')))

# plots all of the figures
spy['TOTAL'].plot(label='S&P500', lw=2, ls='-', c='y')
opt_return_df['TOTAL'].plot(
    label='Optimized weights', lw=2, ls='-', c='g',)
opt_return_df['TOTAL 3Mo SMA'].plot(
    label='Optimized 3 month SMA', lw=2, ls='--', c='r')
equal_weight_return_df['TOTAL'].plot(
    label='Equal allocations', lw=2, ls='-',)


plt.ylabel('Portfolio Value', fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.title('Hypothetical growth of $1,000 portfolio consisting of {} stocks'.format(
    len(stock_tickers)), fontsize=29)
plt.legend()
plt.draw()

# plot efficient frontier with given dataset

frontier_y_axis = np.linspace(0, 0.3, 200)

frontier_x_axis = []  # create empty list for stdev values to append to

for possible_return in frontier_y_axis:
    vol_constraints = ({'type': 'eq', 'fun': check_weight_sum}, {'type': 'eq',
                                                                 'fun': lambda weights: get_return_volatility_SR(weights)[0] - possible_return})
    # above line checks to make sure sum of weights is 1 and makes sure that the possible return being tested is what the SR function thinks is optimal
    result = minimize(minimize_volatility, init_guess,
                      method='SLSQP', bounds=bounds, constraints=vol_constraints)
    frontier_x_axis.append(result['fun'])  # appends the list of sharpe ratios


# create axes, modify x and y axis formatting

fig2, ax2 = plt.subplots(figsize=(16, 9))
ax2.yaxis.set_major_formatter(
    tkr.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax2.xaxis.set_major_formatter(
    tkr.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))


ax2.plot(frontier_x_axis, frontier_y_axis,
         'g--', lw=2.5, label='Efficient frontier')
plt.title('Efficient frontier of selected portfolio', fontsize=29)
plt.xlabel('Volatility', fontsize=15)
plt.ylabel('Return', fontsize=15)

# below plot takes return volatility function, passes in the optimized weights,
# and then plots the optimized sharpe ratio using those weights

# tuple unpacking to assign plotted sharpe ratio
scatter_x, scatter_y = (get_return_volatility_SR(
    optimized_results.x)[1], get_return_volatility_SR(optimized_results.x)[0])

# plots the highest sharpe ratio

plt.scatter(scatter_x,
            scatter_y, s=200, c='green', alpha=.35, edgecolors='black', label='Highest Sharpe Ratio')

plt.legend()

# Plots the optimized sharpe ratio annotation dynamically on the graph
plt.annotate(s='Optimized Sharpe Ratio: {:.4f}'.format(sharpe_ratio_result),
             xy=(scatter_x,
                 scatter_y), xytext=((scatter_x + (scatter_x / 20)), scatter_y), textcoords='data')
plt.draw()

# prints out a list of the optimized portfolio weights

print("Optimized portfolio weights: ")
print('')
for key, value in weights_results_dict.items():
    print(str(key) + ' : ' + bcolors.OKGREEN +
          '{:.2%}'.format(value) + bcolors.ENDC)
    print('-')
print('')
input('Optimization successful, press enter to exit and view graphs ')

plt.show()
