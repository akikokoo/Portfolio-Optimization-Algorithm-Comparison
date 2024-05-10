import numpy as np

# Sırayla son 1 hafta, 1 ay, 3 ay, 6 ay ve 1 yıl getirileri
returns = {
    'Altın': [-0.0011, 0.0168, 0.0929, 0.2110, 0.7917],  # Altın getirileri
    'Euro': [0.0024, 0.0116, 0.0635, 0.1209, 0.6257],   # Euro getirileri
    'Dolar': [0.0055, 0.0246, 0.0753, 0.1388, 0.6361],  # Dolar getirileri
    'BIST100': [0.0226, 0.1517, 0.1889, 0.2073, 0.8469],# BIST100 getirileri
 }

# Algoritma parametreleri
N = 100  # Popülasyon büyüklüğü
pc = 0.8  # Çaprazlama olasılığı
pm = 0.1  # Mutasyon olasılığı

# Maksimum yatırım miktarını belirleme6
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

# Uygunluk fonksiyonu
def fitness_function(weights):
    scaled_weights = {asset: weight * total_investment for asset, weight in weights.items()}
    return calculate_sharpe_ratio(scaled_weights)

# Popülasyonu oluşturma(eski popülasyon - rastgele)
def create_population(N):
    population = []
    for _ in range(N):
        chromosome = {}
        total_weight = 0
        for asset in returns.keys():
            weight = np.random.rand()
            chromosome[asset] = weight
            total_weight += weight
        # Normalizasyon işlemi
        for asset in returns.keys():
            chromosome[asset] /= total_weight
        population.append(chromosome)
    return population

# Çaprazlama
def crossover(parent1, parent2):
    child1 = {}
    child2 = {}
    crossover_point = np.random.randint(1, len(returns.keys()) - 1)
    asset_keys = list(returns.keys())
    for i in range(crossover_point):
        asset = asset_keys[i]
        child1[asset] = parent1[asset]
        child2[asset] = parent2[asset]
    for i in range(crossover_point, len(returns.keys())):
        asset = asset_keys[i]
        child1[asset] = parent2[asset]
        child2[asset] = parent1[asset]
    return child1, child2

# Mutasyon
def mutate(chromosome):
    mutated_chromosome = chromosome.copy()
    asset = np.random.choice(list(returns.keys()))
    mutated_chromosome[asset] = np.random.rand()
    total_weight = sum(mutated_chromosome.values())
    # Normalizasyon
    for key in mutated_chromosome:
        mutated_chromosome[key] /= total_weight
    return mutated_chromosome

# Genetik algoritma ana döngüsü
population = create_population(N)
for _ in range(N):  # N jenerasyon için döngü
    # Popülasyondaki her kromozom için uygunluk hesaplama ve bu değerlerden bir dizi oluşturma
    fitness_scores = [fitness_function(chromosome) for chromosome in population]

    # Yeni popülasyon oluşturma
    new_population = []
    while len(new_population) < N:
        # Çaprazlama
        parent1 = population[np.random.choice(N, p=fitness_scores/np.sum(fitness_scores))]
        parent2 = population[np.random.choice(N, p=fitness_scores/np.sum(fitness_scores))]
        if np.random.rand() < pc:
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        # Mutasyon
        if np.random.rand() < pm:
            child1 = mutate(child1)
        if np.random.rand() < pm:
            child2 = mutate(child2)

        new_population.extend([child1, child2])

    population = new_population[:N]  # Yeni popülasyonu güncelleme

# En iyi portföyü bulma
best_portfolio = max(population, key=fitness_function)
best_sharpe_ratio = calculate_sharpe_ratio(best_portfolio)
print("En iyi portföy:", best_portfolio)
print("En iyi Sharpe Oranı:", best_sharpe_ratio)
print("Best portfolio return:", calculate_portfolio_return(best_portfolio))
