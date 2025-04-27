import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import random
import time
import scipy.stats


#dovrebbe usare la gpu ma non la usa dato che ho tensorflow vecchio
#se lo disinstallo e metto su quello nuovo non funziona più keras
#no way out da questa situazione mi terrò la cpu
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"VIVA LA GPU: {physical_devices[0].name}")
else:
    print("Niente gpu amen")

class ImprovedAgentNetwork(tf.keras.Model):
    # network semplice con 3 strati ativati dalla relu e le policy gestite tramite softmax
    # rete neurale con architettura Actor-Critic
    # elabora lo stato dell'ambiente e produce una distribuzione di policy e un valore
    def __init__(self, state_size, action_size):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(64, activation='relu')
        
        self.value = tf.keras.layers.Dense(1)
        self.policy = tf.keras.layers.Dense(action_size, activation='softmax')
        
    def call(self, state, training=False):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        
        value = self.value(x)
        policy = self.policy(x)
        return policy, value
    
class ImprovedMARLAgricultureAgent:
    # Agente che utilizza memoria prioritaria, decay adattivo dell'epsilon e monitoraggio dell'equilibrio
    # inizializzazione
    def __init__(self, agent_id, state_size, action_size, learning_rate=0.0001, gamma=0.99):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.best_states = {}  
        self.best_equilibrium_value = 0.0
        
        self.initial_lr = learning_rate
        
        # learning rate scheduler
        self.lr_scheduler = ImprovedLearningRateScheduler(
            initial_lr=learning_rate, 
            min_lr=0.00001, 
            decay_factor=0.95,
            reward_threshold=50.0
        )
        
        # Epsilon decay adattiva
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.2
        
        # Network e optimizer
        self.network = ImprovedAgentNetwork(state_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, 
            clipnorm=0.5
        )
        
        # Experience replay
        self.memory = deque(maxlen=100000)
        self.priorities = deque(maxlen=100000)
        
        # tracciamento reward per lr
        self.cumulative_reward = 0
        
        # tracciamento equilibrium proximity per lr 
        self.equilibrium_proximity = 0.0
    
    def remember(self, state, action, reward, next_state, done, equilibrium_proximity=0.0, old_policy=None):
        self.cumulative_reward += reward
        self.equilibrium_proximity = equilibrium_proximity
        
        if equilibrium_proximity > 0.6:  
            state_key = tuple(np.round(state, 1))
            if state_key not in self.best_states or equilibrium_proximity > self.best_states[state_key]['eq']:
                self.best_states[state_key] = {
                    'action': action,
                    'eq': equilibrium_proximity,
                    'prob': min(0.9, equilibrium_proximity)  
                }
    
        if equilibrium_proximity > self.best_equilibrium_value:
            self.best_equilibrium_value = equilibrium_proximity
        
        # si dà priorità alle soluzioni con equilibrium proximity alto con i weight
        priority = abs(reward) + (equilibrium_proximity ** 2) * 15.0 + 1e-6
        
        self.memory.append((state, action, reward, next_state, done, old_policy))
        self.priorities.append(priority)
        
    def act(self, state, training=True):
        # decide l'azione da compiere in base allo stato attuale
        # bilancia esplorazione e sfruttamento basandosi sull'epsilon e sull'equilibrium proximity

        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        policy, _ = self.network(state_tensor)

        state_key = tuple(np.round(state, 1)) 
        if state_key in self.best_states and np.random.rand() < self.best_states[state_key]['prob']:
            return self.best_states[state_key]['action']
        
        # Riduzione esplorazione se alto equilibrium proximity
        effective_epsilon = self.epsilon * 0.1 if hasattr(self, 'equilibrium_proximity') and self.equilibrium_proximity > 0.6 and training else self.epsilon
        
        if training:
            # altrimenti esplora random
            if np.random.rand() <= effective_epsilon:
                if hasattr(self, 'high_equilibrium_actions') and len(self.high_equilibrium_actions) > 0 and np.random.rand() < self.equilibrium_proximity:
                    return np.random.choice(self.high_equilibrium_actions)
                return np.random.choice(self.action_size)

        # se exploitation abbassa equilibrium proximity per esplorare
        temperature = max(0.1, self.epsilon * (1.0 - self.equilibrium_proximity))
        
        policy_numpy = policy[0].numpy()
        policy_numpy = np.maximum(policy_numpy, 1e-10)
        
        scaled_policy = policy_numpy ** (1 / temperature)
        sum_scaled = np.sum(scaled_policy)
        scaled_policy = scaled_policy / sum_scaled if sum_scaled > 0 else np.ones_like(scaled_policy) / self.action_size
        
        return np.random.choice(self.action_size, p=scaled_policy)
    
    def replay(self, batch_size):

        # Addestra la rete neurale utilizzando un batch di esperienze
        # Implementa un algoritmo simile a PPO con clipping del rapporto di policy

        if len(self.memory) < batch_size:
            return 0, 0, 0
            
        # update lr
        new_lr = self.lr_scheduler.update(self.cumulative_reward, self.equilibrium_proximity)
        tf.keras.backend.set_value(self.optimizer.learning_rate, new_lr)

        # fa priority sampling
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        batch_indices = np.random.choice(
            len(self.memory), 
            batch_size, 
            replace=False, 
            p=probabilities
        )
        
        batch = [self.memory[i] for i in batch_indices]
        
        state_batch = np.array([experience[0] for experience in batch])
        action_batch = np.array([experience[1] for experience in batch])
        reward_batch = np.array([experience[2] for experience in batch])
        next_state_batch = np.array([experience[3] for experience in batch])
        done_batch = np.array([experience[4] for experience in batch])
        
        old_policy_batch = np.array([
            experience[5] if experience[5] is not None 
            else np.ones(self.action_size) / self.action_size 
            for experience in batch
        ])
        
        # tensori
        state_tensor = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        next_state_tensor = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        
        _, next_values = self.network(next_state_tensor)
        next_values = tf.squeeze(next_values)

        targets = reward_batch + self.gamma * next_values * (1 - done_batch)
        
        _, values = self.network(state_tensor)
        values = tf.squeeze(values)
        
        # advantages normalizzati
        advantages = targets - values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        advantages = np.clip(advantages, -5, 5)
        
        # backpropagation con gradiente
        with tf.GradientTape() as tape:
            policy, value = self.network(state_tensor, training=True)
            
            # entropia x esplorazione
            entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.math.log(policy + 1e-10), axis=1))
            
            actions_onehot = tf.one_hot(action_batch, depth=self.action_size)
            
            # policy ratio
            policy_ratio = tf.reduce_sum(actions_onehot * policy, axis=1)
            old_policy_ratio = tf.reduce_sum(actions_onehot * old_policy_batch, axis=1)
            ratio = policy_ratio / (old_policy_ratio + 1e-8)
            
            # PPO clip
            clipped_ratio = tf.clip_by_value(ratio, 0.8, 1.2)
            surrogate_loss = tf.minimum(
                advantages * ratio, 
                advantages * clipped_ratio
            )
            actor_loss = -tf.reduce_mean(surrogate_loss)
            
            # Huber loss per gli outlier
            critic_loss = tf.keras.losses.Huber()(targets, tf.squeeze(value))
            
            # loss con regolarizzazione entropia
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # applica gradienti
        grads = tape.gradient(loss, self.network.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        
        # Epsilon decay adattiva
        if self.cumulative_reward > 0:
            decay_rate = max(self.epsilon_min, self.epsilon_decay * (1.0 - 0.1 * min(1.0, self.equilibrium_proximity)))
            self.epsilon = max(self.epsilon_min, self.epsilon * decay_rate)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.equilibrium_proximity > 0.6:  # se vicino all'equilibrio fa decay più accentuato
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.8)

        self.cumulative_reward = 0
        
        return loss.numpy(), actor_loss.numpy(), critic_loss.numpy()

class ImprovedLearningRateScheduler:

    # Regola dinamicamente il tasso di apprendimento in base ai progressi dell'agente
    # Riduce il learning rate quando ci si avvicina all'equilibrio o si raggiungono ricompense elevate

    def __init__(self, initial_lr=0.0005, min_lr=0.00001, decay_factor=0.95, reward_threshold=50.0):
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.reward_threshold = reward_threshold
        self.best_reward = float('-inf')
        self.patience = 0
        self.max_patience = 5
        self.equilibrium_bonus_factor = 0.5  # bonus per equilibrium proximity
        
    def update(self, reward, equilibrium_proximity=0.0):
        # applica bonus
        adjusted_reward = reward * (1.0 + self.equilibrium_bonus_factor * (equilibrium_proximity ** 2))
        new_lr = self.current_lr

        if equilibrium_proximity > 0.5:
            # abbassa lr con decay
            new_lr = max(self.min_lr, self.current_lr * (self.decay_factor * 0.95))
        
        if adjusted_reward > self.best_reward:
            self.best_reward = adjusted_reward
            self.patience = 0
        else:
            self.patience += 1
            
        # abbassa lr se vicini all'equilibrio
        if (self.patience >= self.max_patience) or (adjusted_reward > self.reward_threshold and equilibrium_proximity > 0.5):
            new_lr = max(self.min_lr, self.current_lr * self.decay_factor)
            if equilibrium_proximity > 0.8:
                new_lr = max(self.min_lr, new_lr * self.decay_factor)
            self.current_lr = new_lr
            self.patience = 0

        # aumenta lr se lontani
        elif self.patience >= 2 and equilibrium_proximity < 0.2:
            self.current_lr = min(self.current_lr * 1.05, self.current_lr * 2)
            self.patience = 0
            
        return self.current_lr

class ImprovedEquilibriumDetector:

    # Rileva quando il sistema si avvicina a uno stato di equilibrio economico
    # Monitora la stabilità dei flussi e dei prezzi in un periodo

    def __init__(self, tolerance=1e-3, window_size=20, min_stability_period=5):
        self.tolerance = tolerance
        self.window_size = window_size
        self.min_stability_period = min_stability_period
        
        self.flow_history = deque(maxlen=window_size)
        self.price_history = deque(maxlen=window_size)
        self.stability_counters = {
            'flow': 0,
            'price': 0
        }
    
    def update(self, flows, prices):
        self.flow_history.append(flows)
        self.price_history.append(prices)
    
    def is_equilibrium(self):
        #fa track dell'equilibrio con periodi stabili x evitare oscillazioni
        if len(self.flow_history) < self.window_size:
            return False
        
        flow_changes = [np.max(np.abs(self.flow_history[i+1] - self.flow_history[i])) 
                        for i in range(len(self.flow_history)-1)]
        
        price_changes = [np.max(np.abs(self.price_history[i+1] - self.price_history[i])) 
                         for i in range(len(self.price_history)-1)]
        
        flow_stability = np.mean(flow_changes) < self.tolerance
        price_stability = np.mean(price_changes) < self.tolerance
        
        if flow_stability:
            self.stability_counters['flow'] += 1
        else:
            self.stability_counters['flow'] = 0
        
        if price_stability:
            self.stability_counters['price'] += 1
        else:
            self.stability_counters['price'] = 0
        
        return (
            flow_stability and 
            price_stability and
            self.stability_counters['flow'] >= self.min_stability_period and
            self.stability_counters['price'] >= self.min_stability_period
        )

# funzioni helper basiche

def quadratic_penalty(value, target, weight=10.0, max_penalty=100.0):
    penalty = weight * abs(value - target)
    return -min(max_penalty, penalty)

def normalize_reward(reward, min_reward=-100, max_reward=100):
    #clip della reward per evitare gradiente impazzito
    return np.clip(reward, min_reward, max_reward)

def improved_reward_shaping(flows, supply_prices, demand_prices, trans_costs, exchange_rates, capacities, production_caps=None, equilibrium_proximity=0.0):
    
    # Calcola la ricompensa complessiva del sistema considerando vincoli e condizioni di equilibrio
    # Penalizza violazioni di capacità e premia l'avvicinamento all'equilibrio di mercato
    # Compone le reward singolarmente e assegna bonus a  seconda del flow
    
    K, L, m, n = flows.shape
    
    # usa equilibrium_proximity per ridurre le penalità
    penalty_reduction = 0.5 * equilibrium_proximity
    
    # calcola supplies e demands
    supplies = np.zeros((K, m))
    demands = np.zeros((K, n))
    
    for k in range(K):
        for i in range(m):
            supplies[k, i] = np.sum(flows[k, :, i, :])
        for j in range(n):
            demands[k, j] = np.sum(flows[k, :, :, j])

    # aggiunge penalità per flow troppo bassi
    low_flow_penalty = 0
    for k in range(K):
        for l in range(L):
            for i in range(m):
                for j in range(n):
                    if 0 < flows[k, l, i, j] < 1.0:
                        low_flow_penalty -= min(10.0, 5.0 * (1.0 - flows[k, l, i, j]))

    # reward per equilibrium condition
    price_equilibrium_reward = 0
    for k in range(K):
        for l in range(L):
            for i in range(m):
                for j in range(n):
                    lhs = supply_prices[k, i] + trans_costs[k, l, i, j]
                    rhs = demand_prices[k, j] / exchange_rates[i, j]
                    flow = flows[k, l, i, j]
                    
                    if flow > 0.01:
                        # per flow positivi i due prezzi devono essere molto vicini -> clamp penalty
                        diff = abs(lhs - rhs)
                        price_equilibrium_reward -= min(50.0, 5.0 * diff)
                    else:
                        # per flow a zero (lhs >= rhs)
                        if lhs < rhs:
                            diff = abs(rhs - lhs)
                            price_equilibrium_reward -= min(25.0, 2.0 * diff)
                        else:
                            price_equilibrium_reward += 0.5

    # limiti di capacità
    capacity_reward = 0
    for l in range(L):
        for i in range(m):
            for j in range(n):
                total_flow = np.sum(flows[:, l, i, j])
                if total_flow > capacities[i, j, l]:
                    # penalità per eccesso
                    excess = total_flow - capacities[i, j, l]
                    capacity_reward -= min(100.0, 10.0 * excess)
                else:
                    # piccola reward per utilizzo giusto della capacità
                    capacity_reward += 1.0 * (total_flow / capacities[i, j, l])

    # limiti di produzione
    production_reward = 0
    if production_caps is not None:
        for i in range(m):
            total_production = np.sum(supplies[:, i])
            if total_production > production_caps[i]:
                excess = total_production - production_caps[i]
                production_reward -= min(100.0, 10.0 * excess)
            else:
                production_reward += 2.0 * (total_production / production_caps[i])

    # commodity ratio bonus
    commodity_ratio_bonus = 0
    for i in range(m):
        c1_flow = np.sum(flows[0, :, i, :])
        c2_flow = np.sum(flows[1, :, i, :])
        
        if c2_flow > c1_flow:
            ratio = c2_flow / (c1_flow + 1e-6)
            # log clippata per evitare esplosione
            commodity_ratio_bonus += min(10.0, 2.0 * np.log(1 + ratio))
        else:
            commodity_ratio_bonus -= 1.0

    # termini lagrangiani per binding
    lagrangian_term = 0
    for i in range(m):
        total_supply = np.sum(supplies[:, i])
        if production_caps is not None and abs(total_supply - production_caps[i]) < 0.1:
            lagrangian_term += 5.0

    # reward per market clearing (tutto soddisfatto)
    market_clearing = 0
    for k in range(K):
        total_supply = np.sum(supplies[k, :])
        total_demand = np.sum(demands[k, :])
        diff = abs(total_supply - total_demand)
        market_clearing -= min(50.0, 5.0 * diff)

    # somma componenti di reward
    total_reward = (
        price_equilibrium_reward * (1.0 - penalty_reduction) +
        capacity_reward * (1.0 - penalty_reduction) +
        production_reward * (1.0 - penalty_reduction) +
        commodity_ratio_bonus +
        lagrangian_term +
        market_clearing * (1.0 - penalty_reduction) +
        low_flow_penalty * (1.0 - penalty_reduction)
    )
    
    # bonus di equilibrio accentuato
    equilibrium_bonus = 200.0 * (equilibrium_proximity ** 2)
    
    return total_reward + equilibrium_bonus

class AgriculturalTradeMARL:

    # Modellazione flussi di merci tra mercati considerando prezzi, tassi di cambio e costi di trasporto

    def __init__(self, m, n, K, L, exchange_rates, capacities, production_caps=None, 
                 discretization=20, learning_rate=0.0003, gamma=0.95):
        self.m = m
        self.n = n
        self.K = K
        self.L = L
        self.exchange_rates = exchange_rates
        self.capacities = capacities
        self.production_caps = production_caps
        self.discretization = discretization
        self.equilibrium_tolerance = 1e-2 
        self.max_equilibrium_iterations = 200 
        self.equilibrium_detector = ImprovedEquilibriumDetector(tolerance=1e-2, window_size=10)
        
        # state space
        self.supply_state_size = K + n*K + K + n + 1
        self.demand_state_size = K + m*K + K + m + 1
        
        # action space
        self.supply_action_size = discretization * K * L * n
        self.demand_action_size = discretization
        
        # creazione agenti x supply e demand
        self.supply_agents = [
            ImprovedMARLAgricultureAgent(
                f"Supply_{i}", 
                self.supply_state_size, 
                self.supply_action_size,
                learning_rate,
                gamma
            ) for i in range(m)
        ]
        
        self.demand_agents = [
            ImprovedMARLAgricultureAgent(
                f"Demand_{j}", 
                self.demand_state_size, 
                self.demand_action_size,
                learning_rate,
                gamma
            ) for j in range(n)
        ]
        
        # Initialize
        self.reset()
    
    def reset(self):
        # store best solution
        if hasattr(self, 'flows'):
            current_proximity = self._calculate_equilibrium_proximity()
            if not hasattr(self, 'best_solution') or current_proximity > self.best_solution['proximity']:
                self.best_solution = {
                    'flows': np.copy(self.flows),
                    'proximity': current_proximity,
                    'supplies': np.copy(self.supplies) if hasattr(self, 'supplies') else None,
                    'demands': np.copy(self.demands) if hasattr(self, 'demands') else None,
                    'supply_prices': np.copy(self.supply_prices) if hasattr(self, 'supply_prices') else None,
                    'demand_prices': np.copy(self.demand_prices) if hasattr(self, 'demand_prices') else None
                }
    
        # inizializza con bias se abbiamo una soluzione buona salvata
        if hasattr(self, 'best_solution') and self.best_solution.get('flows') is not None:
            self.flows = self.best_solution['flows'] * (0.9 + 0.2 * np.random.rand())
        else:
            self.flows = np.random.uniform(0, 0.1, (self.K, self.L, self.m, self.n))
        
        # calcolo variabili
        self._update_state_variables()
        
        supply_states = self._get_supply_states()
        demand_states = self._get_demand_states()
        
        return supply_states, demand_states
    
    def _update_state_variables(self):
        # update di tutte le variabili basandosi sul flow corrente

        # calcolo supplies
        self.supplies = np.zeros((self.K, self.m))
        for k in range(self.K):
            for i in range(self.m):
                self.supplies[k, i] = np.sum(self.flows[k, :, i, :])
        
        # calcolo demands
        self.demands = np.zeros((self.K, self.n))
        for k in range(self.K):
            for j in range(self.n):
                self.demands[k, j] = np.sum(self.flows[k, :, :, j])
        
        # update dei prezzi
        self.supply_prices = self.supply_price_function(self.supplies)
        self.demand_prices = self.demand_price_function(self.demands)
        self.trans_costs = self.transportation_cost_function(self.flows)
    
    def _get_supply_states(self):

        # get dello stato delle supply attuali
        states = []
        for i in range(self.m):
            state = []
            
            for k in range(self.K):
                state.append(self.supplies[k, i])
            
            for j in range(self.n):
                for k in range(self.K):
                    state.append(self.demands[k, j])
            
            for k in range(self.K):
                state.append(self.supply_prices[k, i])
            
            for j in range(self.n):
                state.append(self.exchange_rates[i, j])
            
            remaining_capacity = np.sum(self.capacities[i, :, :]) - np.sum(self.flows[:, :, i, :])
            state.append(max(0, remaining_capacity))
            
            states.append(np.array(state, dtype=np.float32))
        
        return states
    
    def _get_demand_states(self):

        # analogo x demand
        states = []
        for j in range(self.n):
            state = []
            
            for k in range(self.K):
                state.append(self.demands[k, j])
            
            for i in range(self.m):
                for k in range(self.K):
                    state.append(self.supplies[k, i])
            
            for k in range(self.K):
                state.append(self.demand_prices[k, j])
            
            for i in range(self.m):
                state.append(self.exchange_rates[i, j])
            
            remaining_capacity = np.sum(self.capacities[:, j, :]) - np.sum(self.flows[:, :, :, j])
            state.append(max(0, remaining_capacity))
            
            states.append(np.array(state, dtype=np.float32))
        
        return states
    
    def supply_price_function(self, supplies):
        # funzioni di prezzo (modificabili in base all'esempio)
        supply_prices = np.zeros_like(supplies)
        
        # commodity 1, supply market 1
        supply_prices[0, 0] = 5 * supplies[0, 0] + 5
        
        # commodity 2, supply market 1
        supply_prices[1, 0] = supplies[1, 0] + 5
        
        return supply_prices
    
    def demand_price_function(self, demands):
        #analogo per le demand
        demand_prices = np.zeros_like(demands)
        
        # commodity 1, demand market 1
        demand_prices[0, 0] = -demands[0, 0] + 20
        
        # commodity 2, demand market 1
        demand_prices[1, 0] = -demands[1, 0] + 37
        
        return demand_prices
    
    def transportation_cost_function(self, flows):
        trans_costs = np.zeros_like(flows)
        
        # commodity 1, route 1, supply market 1, demand market 1
        trans_costs[0, 0, 0, 0] = flows[0, 0, 0, 0] + 1
        
        # commodity 2, Route 1, supply market 1, demand market 1
        trans_costs[1, 0, 0, 0] = flows[1, 0, 0, 0] + 2
        
        return trans_costs
    
    def _decode_supply_action(self, agent_idx, action_idx):
        # decompone azione passata
        remaining = action_idx
        
        # commodity
        k = remaining // (self.discretization * self.L * self.n)
        remaining = remaining % (self.discretization * self.L * self.n)
        
        # route
        l = remaining // (self.discretization * self.n)
        remaining = remaining % (self.discretization * self.n)
        
        # demand market
        j = remaining // self.discretization
        remaining = remaining % self.discretization
        
        # flow 
        flow_level = remaining
        
        # scala flow per  capacità
        max_flow = self.capacities[agent_idx, j, l]
        flow = (flow_level + 1) * max_flow / self.discretization
        
        return k, l, j, flow
    
    def _decode_demand_action(self, action_idx):
        # scaling
        demand_adjustment = (action_idx / self.discretization) * 2 - 1  # range: [-1, 1]
        return demand_adjustment
    
    def _update_best_solution(self, supply_actions=None, demand_actions=None):
        # update della soluzione se ha parametri migliori
        current_proximity = self._calculate_equilibrium_proximity()
        if not hasattr(self, 'best_solution') or current_proximity > self.best_solution['proximity']:
            self.best_solution = {
                'flows': np.copy(self.flows),
                'proximity': current_proximity,
                'supplies': np.copy(self.supplies),
                'demands': np.copy(self.demands),
                'supply_prices': np.copy(self.supply_prices),
                'demand_prices': np.copy(self.demand_prices)
            }
            
            if supply_actions is not None:
                self.best_solution['supply_actions'] = supply_actions.copy()
            if demand_actions is not None:
                self.best_solution['demand_actions'] = demand_actions.copy()
                
            print(f"New best solution found! Proximity: {current_proximity:.4f}")
            print(f"C1: {np.sum(self.flows[0, :, :, :]):.2f}, C2: {np.sum(self.flows[1, :, :, :]):.2f}")
            
        return current_proximity
    
    def _apply_constraints(self, new_flows):
        
        # applica limiti e constraints ai flow 

        # capacità route
        for l in range(self.L):
            for i in range(self.m):
                for j in range(self.n):
                    total_flow = np.sum(new_flows[:, l, i, j])
                    if total_flow > self.capacities[i, j, l]:
                        scale_factor = self.capacities[i, j, l] / total_flow
                        new_flows[:, l, i, j] = new_flows[:, l, i, j] * scale_factor
        
        # limiti di produzione
        if self.production_caps is not None:
            for i in range(self.m):
                supplies_i = np.sum([np.sum(new_flows[k, :, i, :]) for k in range(self.K)])
                if supplies_i > self.production_caps[i]:
                    scale_factor = self.production_caps[i] / supplies_i
                    for k in range(self.K):
                        new_flows[k, :, i, :] = new_flows[k, :, i, :] * scale_factor
        
        return new_flows
    
    def _update_agent_equilibrium_data(self, equilibrium_proximity, supply_actions, demand_actions):
        
        # fa update per azioni buone e fa store delle azioni con probabilità proporzionali all'equilibrium proximity
        if equilibrium_proximity < 0.5:
            return
        
        for i, agent in enumerate(self.supply_agents):
            action = supply_actions[i]
            if not hasattr(agent, 'high_equilibrium_actions'):
                agent.high_equilibrium_actions = []
                
            if np.random.rand() < equilibrium_proximity:
                agent.high_equilibrium_actions.append(action)
                if len(agent.high_equilibrium_actions) > 100:
                    agent.high_equilibrium_actions = agent.high_equilibrium_actions[-100:]
                    
            # fa update delle probabilità di transizione
            state_key = tuple(np.round(self._get_supply_states()[i], 1))
            if not hasattr(agent, 'equilibrium_state_actions'):
                agent.equilibrium_state_actions = {}
                
            if state_key not in agent.equilibrium_state_actions:
                agent.equilibrium_state_actions[state_key] = []
                
            agent.equilibrium_state_actions[state_key].append((action, equilibrium_proximity))
            
            if len(agent.equilibrium_state_actions[state_key]) > 20:
                agent.equilibrium_state_actions[state_key] = agent.equilibrium_state_actions[state_key][-20:]
                
        # analogoo per demand
        for j, agent in enumerate(self.demand_agents):
            action = demand_actions[j]
            if not hasattr(agent, 'high_equilibrium_actions'):
                agent.high_equilibrium_actions = []
                
            if np.random.rand() < equilibrium_proximity:
                agent.high_equilibrium_actions.append(action)
                if len(agent.high_equilibrium_actions) > 100:
                    agent.high_equilibrium_actions = agent.high_equilibrium_actions[-100:]
                    
            state_key = tuple(np.round(self._get_demand_states()[j], 1))
            if not hasattr(agent, 'equilibrium_state_actions'):
                agent.equilibrium_state_actions = {}
                
            if state_key not in agent.equilibrium_state_actions:
                agent.equilibrium_state_actions[state_key] = []
                
            agent.equilibrium_state_actions[state_key].append((action, equilibrium_proximity))
            if len(agent.equilibrium_state_actions[state_key]) > 20:
                agent.equilibrium_state_actions[state_key] = agent.equilibrium_state_actions[state_key][-20:]
        
    def step(self, supply_actions, demand_actions):

        # fa azione nell'env e osserve nuovi stati e rewards
        
        # processa supply actions e flow
        new_flows = np.copy(self.flows)
        for i, action in enumerate(supply_actions):
            k, l, j, flow = self._decode_supply_action(i, action)
            new_flows[k, l, i, j] = flow
        
        # applica constraints
        new_flows = self._apply_constraints(new_flows)
        
        # fa update delle variabili e calcola equilibrium proximity
        self.flows = new_flows
        self._update_state_variables()
        
        equilibrium_proximity = self._calculate_equilibrium_proximity()
        
        # fa update best solution se serve
        self._update_best_solution(supply_actions, demand_actions)
        
        self._update_agent_equilibrium_data(equilibrium_proximity, supply_actions, demand_actions)
        
        # calcola market efficiency  con reward di sistema
        market_efficiency = improved_reward_shaping(
            self.flows, 
            self.supply_prices, 
            self.demand_prices, 
            self.trans_costs, 
            self.exchange_rates,
            self.capacities,
            self.production_caps,
            equilibrium_proximity
        )

        # calcola reward individuali per agenti di supply e di demand
        individual_supply_rewards = self._calculate_supply_rewards(equilibrium_proximity)
        
        # Calculate individual rewards for demand agents
        individual_demand_rewards = self._calculate_demand_rewards(equilibrium_proximity)
        
        total_supply_reward = sum(individual_supply_rewards)
        total_demand_reward = sum(individual_demand_rewards)
        
        # distribuisce le reward per cooperativa
        distributed_supply_rewards = self._distribute_rewards(
            individual_supply_rewards, 
            total_supply_reward,
            total_demand_reward, 
            market_efficiency,
            "supply",
            equilibrium_proximity
        )
        
        distributed_demand_rewards = self._distribute_rewards(
            individual_demand_rewards, 
            total_demand_reward,
            total_supply_reward, 
            market_efficiency,
            "demand",
            equilibrium_proximity
        )
        
        # fa check equilibrio
        done = self._check_equilibrium()
        
        # nuovi stati
        next_supply_states = self._get_supply_states()
        next_demand_states = self._get_demand_states()

        return next_supply_states, next_demand_states, distributed_supply_rewards, distributed_demand_rewards, done, equilibrium_proximity

    def _calculate_supply_rewards(self, equilibrium_proximity):
        
        # calcola reward individuali x supply agent

        individual_supply_rewards = []
        for i in range(self.m):
            profit = 0
            for k in range(self.K):
                for l in range(self.L):
                    for j in range(self.n):
                        # calcola costo e revenue per il flow
                        revenue = self.demand_prices[k, j] * self.flows[k, l, i, j] * self.exchange_rates[i, j]
                        cost = (self.supply_prices[k, i] + self.trans_costs[k, l, i, j]) * self.flows[k, l, i, j]
                        
                        # applica limite capacità e penalità
                        capacity_used = np.sum(self.flows[:, l, i, j])
                        if capacity_used > self.capacities[i, j, l]:
                            capacity_penalty = quadratic_penalty(capacity_used, self.capacities[i, j, l], weight=10.0)
                        else:
                            capacity_penalty = 0
                        
                        profit += revenue - cost + capacity_penalty
            
            # fa ratio del flow x bonus
            c1_flow = np.sum(self.flows[0, :, i, :])
            c2_flow = np.sum(self.flows[1, :, i, :])
            commodity_ratio = c2_flow / (c1_flow + 1e-6)
            
            # applica bonus più alto per C2>C1 (unica parte guidata)
            flow_ratio_bonus = 0
            if c2_flow > c1_flow:
                flow_ratio_bonus = 3.0 * np.log(1 + commodity_ratio)
            else:
                flow_ratio_bonus = -2.0
            
            # bonus di esplorazione per commodity
            commodity_distribution = np.sum(self.flows[:, :, i, :], axis=(1, 2))
            entropy = scipy.stats.entropy(commodity_distribution + 1e-10)
            diversity_bonus = 0.1 * entropy
            
            # bonus x equilibrio
            equilibrium_bonus = 20.0 * equilibrium_proximity
            
            individual_supply_rewards.append(profit + diversity_bonus + flow_ratio_bonus + equilibrium_bonus)
        
        return individual_supply_rewards

    def _calculate_demand_rewards(self, equilibrium_proximity):
        
        individual_demand_rewards = []
        for j in range(self.n):
            consumer_surplus = 0
            for k in range(self.K):
                # calcola utility e spesa
                utility = (37 if k == 1 else 20) * self.demands[k, j] - 0.5 * self.demands[k, j]**2
                expenditure = self.demand_prices[k, j] * self.demands[k, j]
                consumer_surplus += utility - expenditure
            
            demand_penalty = 0
            if hasattr(self, 'target_demands'):
                for k in range(self.K):
                    demand_penalty += abs(self.demands[k, j] - self.target_demands[k, j]) * 5
            
            # bonus x equilibrio
            equilibrium_bonus = 10.0 * equilibrium_proximity**2
            
            individual_demand_rewards.append(consumer_surplus - demand_penalty + equilibrium_bonus)
        
        return individual_demand_rewards

    def _distribute_rewards(self, individual_rewards, total_own_rewards, total_other_rewards, 
                        market_efficiency, agent_type, equilibrium_proximity):
        
        # calcola reward distribuite tra i tipi e gli agenti
        # coefficienti di condivisione
        intra_type_sharing = 0.2
        inter_type_sharing = 0.1
        
        # numero agenti
        num_agents = self.m if agent_type == "supply" else self.n
        
        distributed_rewards = []
        for i in range(num_agents):
            own_reward = individual_rewards[i] * (1 - intra_type_sharing - inter_type_sharing)
            
            # parte dello stesso agente
            if num_agents > 1:
                share_from_same = (total_own_rewards - individual_rewards[i]) * intra_type_sharing / (num_agents - 1)
            else:
                share_from_same = 0
            
            # parte di agenti diversi
            share_from_other = total_other_rewards * inter_type_sharing / num_agents
            
            # calcola reward con la reward di sistema
            reward = own_reward + share_from_same + share_from_other + market_efficiency
            
            # bonus x equilibrium proximity
            if equilibrium_proximity > 0.6:
                reward *= (1 + 2 * equilibrium_proximity)
                
            # normalizza reward
            distributed_rewards.append(normalize_reward(reward))
        
        return distributed_rewards

    def save_best_solution(self):
        # salva soluzione migliore
        current_proximity = self._calculate_equilibrium_proximity()
        if not hasattr(self, 'best_solution') or current_proximity > self.best_solution['proximity']:
            self.best_solution = {
                'flows': np.copy(self.flows),
                'proximity': current_proximity,
                'supplies': np.copy(self.supplies),
                'demands': np.copy(self.demands),
                'supply_prices': np.copy(self.supply_prices),
                'demand_prices': np.copy(self.demand_prices)
            }
            print(f"New best solution found! Proximity: {self.best_solution['proximity']:.4f}")
            print(f"C1: {np.sum(self.flows[0, :, :, :]):.2f}, C2: {np.sum(self.flows[1, :, :, :]):.2f}")

    def train(self, episodes=500, batch_size=128, max_steps=300):

        # alterna episodi di esplorazione e sfruttamento delle soluzioni migliori
        # monitora metriche per equilibrium proximity e flussi

        rewards_history = {agent.agent_id: [] for agent in self.supply_agents + self.demand_agents}
        flows_history = []
        loss_history = {agent.agent_id: [] for agent in self.supply_agents + self.demand_agents}
        equilibrium_history = []
        
        # track di epsilon e lr
        epsilon_history = {agent.agent_id: [] for agent in self.supply_agents + self.demand_agents}
        lr_history = {agent.agent_id: [] for agent in self.supply_agents + self.demand_agents}
        
        # inizializza best solution
        if not hasattr(self, 'best_solution'):
            self.best_solution = {'proximity': 0, 'flows': None}
        
        start_time = time.time()
        
        for episode in range(episodes):
            episode_start = time.time()
            supply_states, demand_states = self.reset()
            episode_rewards = {agent.agent_id: 0 for agent in self.supply_agents + self.demand_agents}
            episode_losses = {agent.agent_id: [] for agent in self.supply_agents + self.demand_agents}
            
            # reset equilibrium detector
            self.equilibrium_detector = ImprovedEquilibriumDetector()
            
            # vede se fare episodio di exploitation
            exploit_episode = hasattr(self, 'best_solution') and self.best_solution['proximity'] > 0.6 and np.random.rand() < 0.5
            
            if exploit_episode:
                print(f"Episode {episode+1}: EXPLOITATION episode using best solution with proximity {self.best_solution['proximity']:.4f}")
                
                # inizializza stato da best solution
                self.flows = np.copy(self.best_solution['flows'])
                # ricalcolo
                self.supplies = np.zeros((self.K, self.m))
                for k in range(self.K):
                    for i in range(self.m):
                        self.supplies[k, i] = np.sum(self.flows[k, :, i, :])
                
                self.demands = np.zeros((self.K, self.n))
                for k in range(self.K):
                    for j in range(self.n):
                        self.demands[k, j] = np.sum(self.flows[k, :, :, j])
                        
                self.supply_prices = self.supply_price_function(self.supplies)
                self.demand_prices = self.demand_price_function(self.demands)
                self.trans_costs = self.transportation_cost_function(self.flows)
                
                # update states
                supply_states = self._get_supply_states()
                demand_states = self._get_demand_states()
                
                # abbassa exploration rate x episodi di exploitation 
                original_epsilons = []
                for agent in self.supply_agents + self.demand_agents:
                    original_epsilons.append(agent.epsilon)
                    agent.epsilon *= 0.17
            
            for step in range(max_steps):
                supply_actions = [agent.act(state) for agent, state in zip(self.supply_agents, supply_states)]
                demand_actions = [agent.act(state) for agent, state in zip(self.demand_agents, demand_states)]

                # prende policies prima di fare update
                supply_policies = []
                for i, agent in enumerate(self.supply_agents):
                    state_tensor = tf.convert_to_tensor(supply_states[i].reshape(1, -1), dtype=tf.float32)
                    policy, _ = agent.network(state_tensor)
                    supply_policies.append(policy[0].numpy())
                
                demand_policies = []
                for i, agent in enumerate(self.demand_agents):
                    state_tensor = tf.convert_to_tensor(demand_states[i].reshape(1, -1), dtype=tf.float32)
                    policy, _ = agent.network(state_tensor)
                    demand_policies.append(policy[0].numpy())
                
                # fa step tramite equilibrium proximity
                next_supply_states, next_demand_states, supply_rewards, demand_rewards, _, equilibrium_proximity = self.step(
                    supply_actions, demand_actions
                )
                
                if equilibrium_proximity > 0.7:
                    self.save_best_solution()

                equilibrium_history.append(equilibrium_proximity)
                
                # fa update dell'equilibrium detector
                self.equilibrium_detector.update(
                    self.flows, 
                    np.concatenate([self.supply_prices, self.demand_prices])
                )
                
                # remember delle azioni degli agenti
                for i, agent in enumerate(self.supply_agents):
                    agent.remember(
                        supply_states[i], 
                        supply_actions[i], 
                        supply_rewards[i], 
                        next_supply_states[i], 
                        self.equilibrium_detector.is_equilibrium(),
                        equilibrium_proximity,
                        supply_policies[i]
                    )
                    episode_rewards[agent.agent_id] += supply_rewards[i]
                
                for i, agent in enumerate(self.demand_agents):
                    agent.remember(
                        demand_states[i], 
                        demand_actions[i], 
                        demand_rewards[i], 
                        next_demand_states[i], 
                        self.equilibrium_detector.is_equilibrium(),
                        equilibrium_proximity,
                        demand_policies[i]
                    )
                    episode_rewards[agent.agent_id] += demand_rewards[i]
                
                # update states
                supply_states = next_supply_states
                demand_states = next_demand_states
                
                # training con replay
                for agent in self.supply_agents + self.demand_agents:
                    loss, actor_loss, critic_loss = agent.replay(batch_size)
                    episode_losses[agent.agent_id].append((loss, actor_loss, critic_loss))

                
                # early stopping se trova equilibrio
                if self.equilibrium_detector.is_equilibrium():
                    flows_history.append(np.copy(self.flows))
                    print(f"Equilibrium reached in Episode {episode+1}, Step {step+1}")
                    break
                
                if step % 10 == 0:  
                    print(f"Episode {episode+1}/{episodes}, Step {step+1}")
                    print(f"  Commodity 1 flow: {self.flows[0, 0, 0, 0]:.2f}, Commodity 2 flow: {self.flows[1, 0, 0, 0]:.2f}")
                    print(f"  Supply Agent Reward: {episode_rewards['Supply_0']:.2f}")
                    print(f"  Equilibrium Proximity: {equilibrium_proximity:.4f}")
            
            if exploit_episode:
                for i, agent in enumerate(self.supply_agents + self.demand_agents):
                    agent.epsilon = original_epsilons[i]

            # track rewards. loss e metriche
            for agent_id, reward in episode_rewards.items():
                rewards_history[agent_id].append(reward)
            
            for agent_id, losses in episode_losses.items():
                if losses:
                    avg_loss = np.mean([l[0] for l in losses if not np.isnan(l[0])])
                    loss_history[agent_id].append(avg_loss)
            
            # track epsilon e lr
            for agent in self.supply_agents + self.demand_agents:
                epsilon_history[agent.agent_id].append(agent.epsilon)
                lr_history[agent.agent_id].append(float(tf.keras.backend.get_value(agent.optimizer.learning_rate)))
            
            episode_time = time.time() - episode_start
            print(f"Episode {episode+1} completed in {episode_time:.2f} seconds")
            print(f"Final Equilibrium Proximity: {equilibrium_proximity:.4f}")
            print(f"Commodity 1 flow: {self.flows[0, 0, 0, 0]:.2f}, Commodity 2 flow: {self.flows[1, 0, 0, 0]:.2f}")
            print("-" * 50)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")

        # restore best solution se non trova di meglio
        if hasattr(self, 'best_solution') and self.best_solution['flows'] is not None:
            self.flows = self.best_solution['flows']
        
        return rewards_history, flows_history, loss_history, equilibrium_history, epsilon_history, lr_history

    def _calculate_equilibrium_proximity(self):

        # calcola quanto il sistema è vicino all'equilibrio economico 
        # analizza differenze di prezzo rispetto alle condizioni di equilibrio di mercato  (più i prezzi sono vicini meglio è)

        price_differentials = []

        for k in range(self.K):
            for l in range(self.L):
                for i in range(self.m):
                    for j in range(self.n):
                        lhs = (self.supply_prices[k, i] + 
                                self.trans_costs[k, l, i, j] * self.exchange_rates[i, j])
                        rhs = self.demand_prices[k, j]
                        
                        flow = self.flows[k, l, i, j]
                        
                        if flow > self.equilibrium_tolerance:
                            # prezzi uguali per flow positivi
                            price_diff = abs(lhs - rhs)
                            price_differentials.append(price_diff)
                        elif lhs < rhs - self.equilibrium_tolerance:
                            # prezzi supply + costi di trasporto non meno del prezzo di domanda 
                            price_differentials.append(abs(lhs - rhs))

        if not price_differentials:
            return 0.0

        avg_price_diff = np.mean(price_differentials)
        max_diff_allowed = 5.0  # threshold massima di differenza

        # inverso della differenza = proximity
        proximity = max(0.0, 1.0 - (avg_price_diff / max_diff_allowed))
        return proximity

    def _check_equilibrium(self):
        proximity = self._calculate_equilibrium_proximity()
        return proximity > 0.95  # equilibrio quando proximity > 95%

    def test(self, episodes=10, exploration_factor=0.2, verbose=True):
        # valuta le prestazioni degli agenti addestrati con diversi livelli di esplorazione
        # utilizza le soluzioni ottimali trovate durante l'addestramento come punto di partenza

        if not hasattr(self, 'best_solution') or self.best_solution['flows'] is None:
            print("Warning: No best solution found during training. Using random initialization.")
                
        results = []
        metrics = {
            'equilibrium_proximity': [],
            'commodity_flows': [],
            'price_differentials': []
        }

        for episode in range(episodes):
            
            if hasattr(self, 'best_solution') and self.best_solution['flows'] is not None:
                if np.random.rand() < exploration_factor:
                    # inizia con perturbazione random della best solution per exploration
                    self.flows = np.copy(self.best_solution['flows']) * (0.8 + 0.4 * np.random.rand())
                    if verbose:
                        print(f"Test Episode {episode+1}: Starting with perturbed best solution")
                else:
                    self.flows = np.copy(self.best_solution['flows'])
                    if verbose:
                        print(f"Test Episode {episode+1}: Starting with best solution")
                    
                # ricalcolo
                self.supplies = np.zeros((self.K, self.m))
                for k in range(self.K):
                    for i in range(self.m):
                        self.supplies[k, i] = np.sum(self.flows[k, :, i, :])
                
                self.demands = np.zeros((self.K, self.n))
                for k in range(self.K):
                    for j in range(self.n):
                        self.demands[k, j] = np.sum(self.flows[k, :, :, j])
                        
                self.supply_prices = self.supply_price_function(self.supplies)
                self.demand_prices = self.demand_price_function(self.demands)
                self.trans_costs = self.transportation_cost_function(self.flows)
                
                supply_states = self._get_supply_states()
                demand_states = self._get_demand_states()
            else:
                supply_states, demand_states = self.reset()
                if verbose:
                    print(f"Test Episode {episode+1}: Starting with random initialization")
                    
            done = False
            step = 0
            episode_metrics = {
                'steps': [],
                'equilibrium_proximity': [],
                'flows': [],
                'prices': []
            }
            
            # varia esplorazione
            for agent in self.supply_agents + self.demand_agents:
                agent.epsilon_backup = agent.epsilon
                
                agent.epsilon = exploration_factor
            
            while not done and step < 100:
                supply_actions = [agent.act(state, training=False) for agent, state in zip(self.supply_agents, supply_states)]
                demand_actions = [agent.act(state, training=False) for agent, state in zip(self.demand_agents, demand_states)]
                
                next_supply_states, next_demand_states, _, _, done, equilibrium_proximity = self.step(
                    supply_actions, demand_actions
                )
                
                
                episode_metrics['steps'].append(step)
                episode_metrics['equilibrium_proximity'].append(equilibrium_proximity)
                episode_metrics['flows'].append([self.flows[0, 0, 0, 0], self.flows[1, 0, 0, 0]])
                episode_metrics['prices'].append([
                    self.supply_prices[0, 0], 
                    self.supply_prices[1, 0],
                    self.demand_prices[0, 0],
                    self.demand_prices[1, 0]
                ])
                
                supply_states = next_supply_states
                demand_states = next_demand_states
                
                step += 1
            
            # restore dell'episolon originale
            for agent in self.supply_agents + self.demand_agents:
                agent.epsilon = agent.epsilon_backup
            
            results.append({
                'flows': self.flows.copy(),
                'supplies': self.supplies.copy(),
                'demands': self.demands.copy(),
                'supply_prices': self.supply_prices.copy(),
                'demand_prices': self.demand_prices.copy(),
                'trans_costs': self.trans_costs.copy(),
                'steps': step,
                'equilibrium_proximity': self._calculate_equilibrium_proximity(),
                'metrics': episode_metrics
            })
            
            
            metrics['equilibrium_proximity'].append(self._calculate_equilibrium_proximity())
            metrics['commodity_flows'].append([self.flows[0, 0, 0, 0], self.flows[1, 0, 0, 0]])
            
            if verbose:
                print(f"  Completed in {step} steps")
                print(f"  Commodity 1 flow: {self.flows[0, 0, 0, 0]:.2f} (Target: 2.00)")
                print(f"  Commodity 2 flow: {self.flows[1, 0, 0, 0]:.2f} (Target: 10.00)")
                print(f"  Equilibrium Proximity: {self._calculate_equilibrium_proximity():.4f}")
                print("-" * 40)

        
        results.append({'all_metrics': metrics})
        return results

    def plot_training_results(self, rewards_history, loss_history, equilibrium_history, epsilon_history=None, lr_history=None):
        fig = plt.figure(figsize=(20, 15))

        # 1. grafico delle ricompense degli agenti di offerta (supply)
        ax1 = fig.add_subplot(3, 2, 1)
        for agent_id, rewards in rewards_history.items():
            if 'Supply' in agent_id:
                ax1.plot(rewards, label=agent_id)
        ax1.set_title('Supply Agent Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. grafico delle ricompense degli agenti di domanda (demand)
        ax2 = fig.add_subplot(3, 2, 2)
        for agent_id, rewards in rewards_history.items():
            if 'Demand' in agent_id:
                ax2.plot(rewards, label=agent_id)
        ax2.set_title('Demand Agent Rewards')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. grafico delle perdite (loss) degli agenti
        ax3 = fig.add_subplot(3, 2, 3)
        for agent_id, losses in loss_history.items():
            if losses:
                ax3.plot(losses, label=agent_id)
        ax3.set_title('Agent Losses')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. grafico della vicinanza all'equilibrio
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(equilibrium_history, label='Proximity', color='green')
        ax4.axhline(y=0.95, color='r', linestyle='--', label='Equilibrium Threshold')
        ax4.set_title('Equilibrium Proximity')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Proximity (0-1)')
        ax4.set_ylim(0, 1)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. grafico di epsilon (esplorazione)
        if epsilon_history:
            ax5 = fig.add_subplot(3, 2, 5)
            for agent_id, epsilons in epsilon_history.items():
                if epsilons:
                    ax5.plot(epsilons, label=agent_id)
            ax5.set_title('Epsilon Decay')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Epsilon')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. grafico dell'adattamento del lr
        if lr_history:
            ax6 = fig.add_subplot(3, 2, 6)
            for agent_id, lrs in lr_history.items():
                if lrs:
                    ax6.plot(lrs, label=agent_id)
            ax6.set_title('Learning Rate Adaptation')
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Learning Rate')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_visualization.png', dpi=300)
        plt.close()

    def plot_test_results(self, test_results):
        
        metrics = test_results[-1]['all_metrics']

        fig = plt.figure(figsize=(20, 12))

        # 1. convergenza dei flussi
        ax1 = fig.add_subplot(2, 2, 1)
        commodity1_flows = [flow[0] for flow in metrics['commodity_flows']]  
        commodity2_flows = [flow[1] for flow in metrics['commodity_flows']]

        ax1.plot(commodity1_flows, 'b-o', label='Commodity 1 Flow')
        ax1.plot(commodity2_flows, 'g-o', label='Commodity 2 Flow')
        ax1.axhline(y=2.0, color='b', linestyle='--', alpha=0.5, label='Target C1')
        ax1.axhline(y=10.0, color='g', linestyle='--', alpha=0.5, label='Target C2')
        ax1.set_title('Commodity Flow Convergence')
        ax1.set_xlabel('Test Episode')
        ax1.set_ylabel('Flow Amount')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. equilibrium proximity
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(metrics['equilibrium_proximity'], 'r-o', label='Proximity')
        ax2.axhline(y=0.95, color='k', linestyle='--', label='Equilibrium Threshold')
        ax2.set_title('Equilibrium Proximity by Episode')
        ax2.set_xlabel('Test Episode')
        ax2.set_ylabel('Proximity Value')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. evoluzione dei flussi durante il test
        if len(test_results) > 1 and 'metrics' in test_results[0]:
            ax3 = fig.add_subplot(2, 2, 3)
        
            episode_metrics = test_results[0]['metrics']
            steps = episode_metrics['steps']
            
            c1_flows = [flow[0] for flow in episode_metrics['flows']]
            c2_flows = [flow[1] for flow in episode_metrics['flows']]
            
            ax3.plot(steps, c1_flows, 'b-', label='C1 Flow')
            ax3.plot(steps, c2_flows, 'g-', label='C2 Flow')
            ax3.axhline(y=2.0, color='b', linestyle='--', alpha=0.5)
            ax3.axhline(y=10.0, color='g', linestyle='--', alpha=0.5)
            ax3.set_title('Flow Evolution (Episode 1)')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Flow Value')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. evoluzione dei prezzi durante il test
        if len(test_results) > 1 and 'metrics' in test_results[0]:
            ax4 = fig.add_subplot(2, 2, 4)
            
            episode_metrics = test_results[0]['metrics']
            steps = episode_metrics['steps']
            
            if 'prices' in episode_metrics:
                supply_p1 = [price[0] for price in episode_metrics['prices']]
                supply_p2 = [price[1] for price in episode_metrics['prices']]
                demand_p1 = [price[2] for price in episode_metrics['prices']]
                demand_p2 = [price[3] for price in episode_metrics['prices']]
                
                ax4.plot(steps, supply_p1, 'b-', label='Supply P1')
                ax4.plot(steps, supply_p2, 'g-', label='Supply P2')
                ax4.plot(steps, demand_p1, 'b--', label='Demand P1')
                ax4.plot(steps, demand_p2, 'g--', label='Demand P2')
                ax4.set_title('Price Evolution (Episode 1)')
                ax4.set_xlabel('Step')
                ax4.set_ylabel('Price')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('test_visualization.png', dpi=300)
        plt.close()


def run_marl_example_1():

    print("MARL Example 1: Two Commodities, Single Route with High Capacity, Single Supply and Demand Market")
    print("=" * 80)
    
    # inizializzazione dei parametri
    m, n, K, L = 1, 1, 2, 1  # dimensioni dell'ambiente (mercati, prodotti, rotte)
    exchange_rates = np.ones((m, n))
    
    capacities = np.zeros((m, n, L))
    capacities[0, 0, 0] = 15.0  # capacità massima della rotta
    
    # creazione dell'ambiente di simulazione
    env = AgriculturalTradeMARL(
        m, n, K, L, exchange_rates, capacities, 
        discretization=100,
        learning_rate=0.0003,
        gamma=0.99
    )
    
    # creazione degli agenti di offerta
    env.supply_agents = [
        ImprovedMARLAgricultureAgent(
            f"Supply_{i}", 
            env.supply_state_size, 
            env.supply_action_size,
            learning_rate=0.0005,
            gamma=0.99
        ) for i in range(m)
    ]
    
    # creazione degli agenti di domanda
    env.demand_agents = [
        ImprovedMARLAgricultureAgent(
            f"Demand_{j}", 
            env.demand_state_size, 
            env.demand_action_size,
            learning_rate=0.0005,
            gamma=0.99
        ) for j in range(n)
    ]
    
    # addestramento degli agenti
    print("Training agents...")
    rewards_history, flows_history, loss_history, equilibrium_history, epsilon_history, lr_history = env.train(
        episodes=12,
        batch_size=128,
        max_steps=200
    )
    
    # visualizzazione dei risultati dell'addestramento
    env.plot_training_results(
        rewards_history, 
        loss_history, 
        equilibrium_history,
        epsilon_history,
        lr_history
    )
    
    # test degli agenti
    print("\nTesting agents with controlled exploration...")
    results = env.test(episodes=5, exploration_factor=0.2)
    
    # visualizzazione dei risultati del test
    env.plot_test_results(results)
    
    # ultimi risultati di test
    final_result = results[-2]  # -1 è il riepilogo delle metriche
    
    # stampa dei risultati finali dettagliati
    print("\nFinal Results:")
    print("-" * 40)
    print(f"Equilibrium Proximity: {final_result['equilibrium_proximity']:.4f}")
    print(f"Commodity 1 Flow: {final_result['flows'][0, 0, 0, 0]:.2f} (Target: 2.00)")
    print(f"Commodity 2 Flow: {final_result['flows'][1, 0, 0, 0]:.2f} (Target: 10.00)")
    
    print("\nPrices and Costs:")
    print("-" * 40)
    print(f"Supply price of commodity 1: {final_result['supply_prices'][0, 0]:.2f} (Target: 15.00)")
    print(f"Supply price of commodity 2: {final_result['supply_prices'][1, 0]:.2f} (Target: 15.00)")
    print(f"Transportation cost of commodity 1: {final_result['trans_costs'][0, 0, 0, 0]:.2f} (Target: 3.00)")
    print(f"Transportation cost of commodity 2: {final_result['trans_costs'][1, 0, 0, 0]:.2f} (Target: 12.00)")
    print(f"Demand price of commodity 1: {final_result['demand_prices'][0, 0]:.2f} (Target: 18.00)")
    print(f"Demand price of commodity 2: {final_result['demand_prices'][1, 0]:.2f} (Target: 27.00)")
    
    # categorie per valutare la qualità
    target_values = [2.0, 10.0, 15.0, 15.0, 18.0, 27.0, 3.0, 12.0]
    actual_values = [final_result['flows'][0, 0, 0, 0], 
                     final_result['flows'][1, 0, 0, 0],
                     final_result['supply_prices'][0, 0], 
                     final_result['supply_prices'][1, 0],
                     final_result['demand_prices'][0, 0], 
                     final_result['demand_prices'][1, 0],
                     final_result['trans_costs'][0, 0, 0, 0], 
                     final_result['trans_costs'][1, 0, 0, 0]]
    
    # calcolo dei punteggi di equilibrium proximity (scala 0-1)
    scores = []
    for i in range(len(target_values)):
        deviation = abs(actual_values[i] - target_values[i]) / target_values[i]
        score = max(0, 1 - min(deviation, 1))  # 1 = corrispondenza perfetta, 0 = deviazione del 100% o più
        scores.append(score)
    
    # calcolo e visualizzazione del punteggio di qualità della soluzione
    solution_quality = np.mean(scores)
    print(f"\nOverall Solution Quality Score: {solution_quality:.4f} (1.0 = perfect)")
    
    # soluzione
    return {
        'flows': final_result['flows'],
        'supplies': final_result['supplies'],
        'demands': final_result['demands'],
        'supply_prices': final_result['supply_prices'],
        'demand_prices': final_result['demand_prices'],
        'trans_costs': final_result['trans_costs'],
        'equilibrium_proximity': final_result['equilibrium_proximity'],
        'solution_quality': solution_quality
    }

if __name__ == "__main__":
    solution = run_marl_example_1()
    print("Agricultural Trade MARL simulation completed successfully!")