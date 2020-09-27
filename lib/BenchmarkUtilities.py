import numpy as np
import pandas as pd

# defined variables
ANNUAL_TRADING_DAYS = 252  # The number of trading days in a financial year
POSITION_BALANCE = 10000  # The amount for portfolio balance

def compute_daily_returns(asset_prices):
    """
    Generate Daily Returns: (new_price-old_price)/ old_price)
    :param asset_prices: pandas.DataFrame
    :return: daily simple returns of the assets in the dataframe
    """
    return asset_prices.pct_change()

def compute_covariance_matrix(returns):
    """
    Generate Covariance Matrix
    :param returns: pandas.DataFrame
    :return: covariance matrix
    """
    return returns.cov() * ANNUAL_TRADING_DAYS

def compute_expected_portfolio_variance(cov_matrix, weights):
    """
    Generate Expected Portfolio Variance
    :param cov_matrix: pandas.DataFrame
    :param weights: list of portfolio weights
    :return portfolio variance (float)
    """
    assert(len(cov_matrix.columns) == len(weights))
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def compute_expected_portfolio_volatility(portfolio_variance):
    """
    Generate Expected Portfolio Volatility
    :param portfolio_variance: pandas.DataFrame
    :return portfolio volatility (float)
    """
    return np.sqrt(portfolio_variance)

def compute_annual_return(returns, weights):
    """
    Generate Annual Portfolio Returns
    :param portfolio_variance: pandas.DataFrame
    :param weights: list of portfolio weights
    :return portfolio annual return (float)
    """
    return np.sum(returns.mean()*weights) * ANNUAL_TRADING_DAYS    

def get_normalized_returns(asset_prices):
    """
    Generate Normalized Returns Based on Asset Prices
    :param asset_prices: pandas.DataFrame
    :return normalized returns (pandas.DataFrame)
    """
    asset_norm_returns = pd.DataFrame()
    for asset_name, asset_adj_close in asset_prices.iteritems():
        asset_norm_returns[asset_name] = asset_adj_close / asset_adj_close.iloc[0]
    return asset_norm_returns

def build_asset_allocation_df(asset_norm_returns, weights, position_balance=POSITION_BALANCE):
    """
    Build Asset Allocation DataFrame
    - contains daily total position balance and daily return
    :param asset_norm_returns: pandas.DataFrame
    :param weights: list of portfolio weights
    :param position_balance: portfolio position balance
    :return (pandas.DataFrame)
    """
    asset_allocation = pd.DataFrame()
    for asset_name, weight in zip((asset_norm_returns.columns), weights):
        asset_allocation[asset_name] = asset_norm_returns[asset_name] * weight * position_balance
    asset_allocation['total_position'] = asset_allocation.sum(axis=1) # sum row by row
    asset_allocation['daily_return'] = asset_allocation['total_position'].pct_change(1)
    asset_allocation = asset_allocation.fillna(0.0) # first day in dataset should have a daily return of 0
    return asset_allocation

def compute_sharpe_ratio(asset_allocation):
    """
    Compute Sharpe Ratio
    :param: asset_allocation dataframe: (pandas.DataFrame)
    :return: (float)
    """
    sharpe_ratio = None
    try:
        sharpe_ratio = asset_allocation['daily_return'].mean() / asset_allocation['daily_return'].std()
    except Exception as e:
        raise Exception()
    return sharpe_ratio

def compute_annualized_sharpe_ratio(asset_allocation):
    """
    Compute Annualized Sharpe Ratio
    :param: asset_allocation dataframe: (pandas.DataFrame)
    :return: (float)
    """
    annualized_sharpe_ratio = None
    try:
        sharpe_ratio = compute_sharpe_ratio(asset_allocation)
        annualized_sharpe_ratio = (ANNUAL_TRADING_DAYS**0.5) * sharpe_ratio
    except Exception as e:
        raise Exception()
    return annualized_sharpe_ratio

def compute_rolling_sharpe_ratio(asset_allocation, window=30):
    """
    Compute Rolling Sharpe Ratio
    :param: asset_allocation dataframe: (pandas.DataFrame)
    :param: window: (datetime.timedelta)
    :return: (pandas.DataFrame)
    """
    rolling_returns = asset_allocation['daily_return'].rolling(window=window)
    rolling_sharpe_ratio = np.sqrt(window) * (rolling_returns.mean() / rolling_returns.std())
    rolling_sharpe_ratio = rolling_sharpe_ratio.to_frame(name='rolling_sharpe_ratio')
    return rolling_sharpe_ratio

def compute_rolling_returns(asset_allocation, returns_type, window=30):
    """
    Compute Rolling [mean, max, min] Returns
    :param: asset_allocation dataframe: (pandas.DataFrame)
    :param: returns_type: (string) ['mean', 'max', 'min']
    :param: window: (datetime.timedelta)
    :return: (pandas.DataFrame)
    """
    rolling_returns = asset_allocation['daily_return'].rolling(window=window)
    returns = None
    if returns_type == 'mean':
        returns = rolling_returns.mean()
    elif returns_type == 'max':
        returns = rolling_returns.max()
    elif returns_type == 'min':
        returns = rolling_returns.min()
    else:
        raise Exception("Rolling Type Not Valid")
    returns = returns.to_frame(name='rolling {} returns'.format(returns_type))
    return returns

def compute_rolling_volatility(asset_allocation, window=30):
    """
    Compute Rolling Volatility
    :param: asset_allocation dataframe: (pandas.DataFrame)
    :param: window: (datetime.timedelta)
    :return: (pandas.DataFrame)
    """
    rolling_returns = asset_allocation['daily_return'].rolling(window=window)
    rolling_volatility = rolling_returns.std()*(ANNUAL_TRADING_DAYS**0.5)
    rolling_volatility = rolling_volatility.to_frame(name='rolling volatility')
    return rolling_volatility
