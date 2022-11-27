import pandas as pd
import numpy as np

class Backtester():
    def __init__(self, changepoints, returns):
        self.changepoints = changepoints
        self.returns = returns.dropna()
        self.positions = None
        self.portfolio_returns = None
        self.portfolio_cumulative = None
        self.sharpe = None
    def find_min(self, x):
        if x in self.returns.index:
            return x
        else:
            days_diff = (x - self.returns.reset_index().Date).dt.days.abs()
            return self.returns.iloc[days_diff.argmin()].name
    def compute(self):
        cpds = [self.find_min(x) for x in self.changepoints]
        weight = np.sign(self.returns.head(63).mean()) #simple momentum position
        
        pos = []
        for i in self.returns.index:
            if i in cpds:
                weight *= -1
            pos.append(weight)
        self.positions = pos
        self.portfolio_returns = self.positions * self.returns
        self.portfolio_cumulative = (1 + self.portfolio_returns).cumprod()
        self.sharpe = (self.portfolio_returns.mean() * 252)/(self.portfolio_returns.std() * np.sqrt(252))
        print(f'Strategy Computed has IR {self.sharpe:.2f}')