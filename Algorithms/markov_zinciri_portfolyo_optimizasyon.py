import numpy as np
# . The MCMC method is a type of stochastic optimization algorithm that generates a sequence of portfolios (states) where each portfolio depends only on the previous one, which is the characteristic of a Markov chain.
# Asset returns
returns = {
    'Altın': [-0.0011, 0.0168, 0.0929, 0.2110, 0.7917],
    'Euro': [0.0024, 0.0116, 0.0635, 0.1209, 0.6257],
    'Dolar': [0.0055, 0.0246, 0.0753, 0.1388, 0.6361],
    'BIST100': [0.0226, 0.1517, 0.1889, 0.2073, 0.8469],
}

# Total investment
total_investment = int(input("Yatırım miktarını giriniz:"))

def calculate_portfolio_return(weights):
    total_return = 0
    for asset, weight in weights.items():
        total_return += weight * np.mean(returns[asset])
    return total_return

def calculate_portfolio_risk(weights):
    total_risk = 0
    for asset, weight in weights.items():
        total_risk += weight * np.std(returns[asset])
    return total_risk

def calculate_sharpe_ratio(weights, risk_free_rate=0):
    portfolio_return = calculate_portfolio_return(weights)
    portfolio_risk = calculate_portfolio_risk(weights)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    return sharpe_ratio

def calculate_acceptance_metric(weights):
    return np.exp(calculate_sharpe_ratio(weights)) # acceptance fonksiyon, her zaman pozitif
    # sharpe oranı tek başına her zaman pozitif dönmeyebilir!

def generate_new_portfolio(current_weights):
    new_weights = current_weights.copy()
    asset = np.random.choice(list(returns.keys()))
    new_weights[asset] = np.random.rand()
    total_weight = sum(new_weights.values())
    for key in new_weights:
        new_weights[key] /= total_weight
    return new_weights

# MCMC algorithm
# ilk portfolyo eşit ağırlıklı olacak
current_portfolio = {asset: 1/len(returns) for asset in returns.keys()}
best_portfolio = current_portfolio
best_sharpe_ratio = calculate_sharpe_ratio(best_portfolio)

for _ in range(1000):  # 1000 iterations
    new_portfolio = generate_new_portfolio(current_portfolio)
    acceptance_ratio = calculate_acceptance_metric(new_portfolio) / calculate_acceptance_metric(current_portfolio)
    # her portfolyo sadece ama sadece bir önceki portfolyoya bağlıdır.
    if np.random.rand() < acceptance_ratio:
        current_portfolio = new_portfolio
        current_sharpe_ratio = calculate_sharpe_ratio(current_portfolio)
        if current_sharpe_ratio > best_sharpe_ratio:
            best_portfolio = current_portfolio
            best_sharpe_ratio = current_sharpe_ratio

print("Best portfolio:", best_portfolio)
print("Best Sharpe Ratio:", best_sharpe_ratio)
print("Best portfolio return:", calculate_portfolio_return(best_portfolio))
