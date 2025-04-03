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
    def __init__(self, agent_id, state_size, action_size, learning_rate=0.0001, gamma=0.99):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # esplorazione con epsilon decay che dovrebbe essere adattiva, ma forse lo è troppo? 
        # considerando che continua a esplorare anche se sembra avere migliori risultati
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.1
        
        # adam come ottimizzatore e clip a 1.0 ->  provata anche a 2.0 ma il risultato non cambia
        self.network = ImprovedAgentNetwork(state_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, 
            clipnorm=1.0
        )
        
        # per experience replay
        self.memory = deque(maxlen=100000)
        self.priorities = deque(maxlen=100000)
        
    def act(self, state, training=True):
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        policy, _ = self.network(state_tensor)
        
        if training:
            # esplorazione random
            if np.random.rand() <= self.epsilon:
                return np.random.choice(self.action_size)
            
            # policy selection con boltzmann -> ho trovato in un paper che sarebbe stato meglio (non è stato vero ma è più veloce)
            temperature = max(0.5, self.epsilon)
            scaled_policy = policy[0].numpy() ** (1 / temperature)
            scaled_policy /= np.sum(scaled_policy)
            
            return np.random.choice(self.action_size, p=scaled_policy)
        
        return np.argmax(policy[0].numpy())
    
    def remember(self, state, action, reward, next_state, done, old_policy=None):
        #computazione priorità easy
        priority = abs(reward) + 1e-6
        
        self.memory.append((state, action, reward, next_state, done, old_policy))
        self.priorities.append(priority)
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0, 0, 0
        
        # fa sampling proporzionale per le priorità -> normale
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        batch_indices = np.random.choice(
            len(self.memory), 
            batch_size, 
            replace=False, 
            p=probabilities
        )
        
        batch = [self.memory[i] for i in batch_indices]
        
        # estrazione
        state_batch = np.array([experience[0] for experience in batch])
        action_batch = np.array([experience[1] for experience in batch])
        reward_batch = np.array([experience[2] for experience in batch])
        next_state_batch = np.array([experience[3] for experience in batch])
        done_batch = np.array([experience[4] for experience in batch])
        
        # nella prima stesura non lo avevo messo -> in teoria tratta la vecchia policy con i dati iniziali
        old_policy_batch = np.array([
            experience[5] if experience[5] is not None 
            else np.ones(self.action_size) / self.action_size 
            for experience in batch
        ])
        
        # TENSORI -> ogni volta che non funziona una conversione sparatoria al pc
        state_tensor = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        next_state_tensor = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        
        # nextvalues con lo squeeze altrimenti non vanno nei tensori dopo. ovviamente.
        _, next_values = self.network(next_state_tensor)
        next_values = tf.squeeze(next_values)
        
        targets = reward_batch + self.gamma * next_values * (1 - done_batch)
        
        #valori correnti
        _, values = self.network(state_tensor)
        values = tf.squeeze(values)
        
        # normalizzazione per gli advantages (forse è sbagliato qui? ma senza non funziona)
        advantages = targets - values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        advantages = np.clip(advantages, -5, 5)
        
        # training con loss del gradiente
        with tf.GradientTape() as tape:
            policy, value = self.network(state_tensor, training=True)
            
            #entropia per esplorazione migliore (forse eccessiva)
            entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.math.log(policy + 1e-10), axis=1))
            
            actions_onehot = tf.one_hot(action_batch, depth=self.action_size)
            
            # policy ratio per capire la differenza tra prima e dopo
            policy_ratio = tf.reduce_sum(actions_onehot * policy, axis=1)
            old_policy_ratio = tf.reduce_sum(actions_onehot * old_policy_batch, axis=1)
            ratio = policy_ratio / (old_policy_ratio + 1e-8)
            
            #clip ratio + loss
            clipped_ratio = tf.clip_by_value(ratio, 0.8, 1.2)
            surrogate_loss = tf.minimum(
                advantages * ratio, 
                advantages * clipped_ratio
            )
            actor_loss = -tf.reduce_mean(surrogate_loss)
            
            # huber loss per il critic ma cambia poco, si potrebbe misurare con qualunque altro tipo di loss e staremmo allo stesso punto
            critic_loss = tf.keras.losses.Huber()(targets, value)
        
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # gradiente
        grads = tape.gradient(loss, self.network.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        
        # epsilon  decay adattivo
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.numpy(), actor_loss.numpy(), critic_loss.numpy()

class AgriculturalTradeMARL:
    
    def __init__(self, m, n, K, L, exchange_rates, capacities, production_caps=None, 
                 discretization=50, learning_rate=0.0003, gamma=0.95):
        self.m = m
        self.n = n
        self.K = K
        self.L = L
        self.exchange_rates = exchange_rates
        self.capacities = capacities
        self.production_caps = production_caps
        self.discretization = discretization
        self.equilibrium_tolerance = 1e-3 
        self.max_equilibrium_iterations = 200 
        self.equilibrium_detector = ImprovedEquilibriumDetector() #alla fine. ho lasciato anche il metodo prima che usa la step ma tanto è inutile perché non ci arriva mai
        
        # size dello state space -> forse troppo grande (se diminuisco dovrebbe migliorare?).
        self.supply_state_size = K + n*K + K + n + 1
        self.demand_state_size = K + m*K + K + m + 1
        
        # action space decision -> più è alta la discretization più ci mette il codice ad eseguire
        self.supply_action_size = discretization * K * L * n
        self.demand_action_size = discretization
        
        #fa agenti per supply e demand
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
        
        #inizializza
        self.reset()
    
    def reset(self):
        self.flows = np.random.uniform(0, 0.1, (self.K, self.L, self.m, self.n))
        
        #calcola supplies
        self.supplies = np.zeros((self.K, self.m))
        for k in range(self.K):
            for i in range(self.m):
                self.supplies[k, i] = np.sum(self.flows[k, :, i, :])
        
        #calcola demands
        self.demands = np.zeros((self.K, self.n))
        for k in range(self.K):
            for j in range(self.n):
                self.demands[k, j] = np.sum(self.flows[k, :, :, j])
        
        #prezzi iniziali
        self.supply_prices = self.supply_price_function(self.supplies)
        self.demand_prices = self.demand_price_function(self.demands)
        self.trans_costs = self.transportation_cost_function(self.flows)
        
        supply_states = self._get_supply_states()
        demand_states = self._get_demand_states()
        
        return supply_states, demand_states
    
    def _get_supply_states(self):
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
        #funzioni di prezzo per  come sono definite nell'esempio uno
        supply_prices = np.zeros_like(supplies)
        
        #commodity 1, supply market 1
        supply_prices[0, 0] = 5 * supplies[0, 0] + 5
        
        #commodity 2, supply market 1
        supply_prices[1, 0] = supplies[1, 0] + 5
        
        return supply_prices
    
    def demand_price_function(self, demands):
        demand_prices = np.zeros_like(demands)
        
        #commodity 1, demand market 1
        demand_prices[0, 0] = -demands[0, 0] + 20
        
        #commodity 2, demand market 1
        demand_prices[1, 0] = -demands[1, 0] + 37
        
        return demand_prices
    
    def transportation_cost_function(self, flows):
        #ampliabile con gli altri esempi
        trans_costs = np.zeros_like(flows)
        
        # Commodity 1, Route 1, Supply market 1, Demand market 1
        trans_costs[0, 0, 0, 0] = flows[0, 0, 0, 0] + 1
        
        # Commodity 2, Route 1, Supply market 1, Demand market 1
        trans_costs[1, 0, 0, 0] = flows[1, 0, 0, 0] + 2
        
        return trans_costs
    
    def _decode_supply_action(self, agent_idx, action_idx):
        #decompone l'azione passata
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
        
        # flow level
        flow_level = remaining
        
        # (scala flow per capacità)
        max_flow = self.capacities[agent_idx, j, l]
        flow = (flow_level + 1) * max_flow / self.discretization
        
        return k, l, j, flow
    
    def _decode_demand_action(self, action_idx):
        #scaling altrimenti i  risultati non andavano
        demand_adjustment = (action_idx / self.discretization) * 2 - 1  # range: [-1, 1]
        return demand_adjustment
    
    def step(self, supply_actions, demand_actions):
        #processa l'azione del supply
        new_flows = np.copy(self.flows)
        for i, action in enumerate(supply_actions):
            k, l, j, flow = self._decode_supply_action(i, action)
            new_flows[k, l, i, j] = flow
        
        # applica bound della capacità della route
        for l in range(self.L):
            for i in range(self.m):
                for j in range(self.n):
                    total_flow = np.sum(new_flows[:, l, i, j])
                    if total_flow > self.capacities[i, j, l]:
                        scale_factor = self.capacities[i, j, l] / total_flow
                        new_flows[:, l, i, j] = new_flows[:, l, i, j] * scale_factor
        
        # applica bound della produzione
        if self.production_caps is not None:
            for i in range(self.m):
                supplies_i = np.sum([np.sum(new_flows[k, :, i, :]) for k in range(self.K)])
                if supplies_i > self.production_caps[i]:
                    scale_factor = self.production_caps[i] / supplies_i
                    for k in range(self.K):
                        new_flows[k, :, i, :] = new_flows[k, :, i, :] * scale_factor
        
        # update state
        self.flows = new_flows
        
        # calcola supplies per ogni m country
        self.supplies = np.zeros((self.K, self.m))
        for k in range(self.K):
            for i in range(self.m):
                self.supplies[k, i] = np.sum(self.flows[k, :, i, :])
        
        # calcola demands per ogni n country
        self.demands = np.zeros((self.K, self.n))
        for k in range(self.K):
            for j in range(self.n):
                self.demands[k, j] = np.sum(self.flows[k, :, :, j])
        
        # Update prezzi
        self.supply_prices = self.supply_price_function(self.supplies)
        self.demand_prices = self.demand_price_function(self.demands)
        self.trans_costs = self.transportation_cost_function(self.flows)
        
        # calcola market efficiency usando RewardShapingStrategy (è una variabile di aggiustamento per la reward)
        # il modello funziona meglio da quando l'ho messa ma è ancora oscillante
        market_efficiency = RewardShapingStrategy.calculate_market_efficiency(
            self.flows, 
            self.supply_prices, 
            self.demand_prices, 
            self.trans_costs, 
            self.exchange_rates
        )

        supply_rewards = []
        for i in range(self.m):
            profit = 0
            for k in range(self.K):
                for l in range(self.L):
                    for j in range(self.n):
                        # revenue - overproduzione 
                        revenue = self.demand_prices[k, j] * self.flows[k, l, i, j] * self.exchange_rates[i, j]
                        cost = (self.supply_prices[k, i] + self.trans_costs[k, l, i, j]) * self.flows[k, l, i, j]
                        
                        # penalità se utilizzata la capacità
                        capacity_used = np.sum(self.flows[:, l, i, j])
                        capacity_penalty = max(0, capacity_used - self.capacities[i, j, l]) * 10
                        
                        profit += revenue - cost - capacity_penalty
            
            # bonus per esplorare entrambe le commodity
            commodity_distribution = np.sum(self.flows[:, :, i, :], axis=(1, 2))
            entropy = scipy.stats.entropy(commodity_distribution + 1e-10)
            diversity_bonus = 0.1 * entropy
            
            # aggiunge market efficiency come bonus/penalità
            supply_rewards.append(profit + diversity_bonus + market_efficiency)

        demand_rewards = []
        for j in range(self.n):
            consumer_surplus = 0
            for k in range(self.K):
                # quadratica per approssimazione
                utility = (37 if k == 1 else 20) * self.demands[k, j] - 0.5 * self.demands[k, j]**2
                expenditure = self.demand_prices[k, j] * self.demands[k, j]
                consumer_surplus += utility - expenditure
            
            # penalità se non si soddisfa la domanda
            demand_penalty = 0
            if hasattr(self, 'target_demands'):
                for k in range(self.K):
                    demand_penalty += abs(self.demands[k, j] - self.target_demands[k, j]) * 5
            
            
            demand_rewards.append(consumer_surplus - demand_penalty + market_efficiency)
        
        # vede se  c'è l'equilibrio
        done = self._check_equilibrium() #c'è il duplicato dell'improved, ma onestamente al momento non serve.
        
        #prende i nuovi stati
        next_supply_states = self._get_supply_states()
        next_demand_states = self._get_demand_states()
        
        return next_supply_states, next_demand_states, supply_rewards, demand_rewards, done
    
    def _check_equilibrium(self, tolerance=None):
        #relativa perché anche se funzionasse non lo saprei dato che le run si bloccano a valori troppo bassi
        if tolerance is None:
            tolerance = self.equilibrium_tolerance
        
        price_differentials = []
        flow_conditions = []
        
        for k in range(self.K):
            for l in range(self.L):
                for i in range(self.m):
                    for j in range(self.n):
                        lhs = (self.supply_prices[k, i] + 
                               self.trans_costs[k, l, i, j] * self.exchange_rates[i, j])
                        rhs = self.demand_prices[k, j]
                        
                        flow = self.flows[k, l, i, j]
                        
                        
                        if flow > tolerance:
                            # se flow positivo ->  prezzi molto vicini (ma questo lo so dalla soluzione)
                            price_diff = abs(lhs - rhs)
                            price_differentials.append(price_diff)
                            flow_conditions.append(flow)
                        elif lhs < rhs - tolerance:
                            price_differentials.append(abs(lhs - rhs))
                            flow_conditions.append(0)
        
        avg_price_diff = np.mean(price_differentials) if price_differentials else 0
        max_price_diff = np.max(price_differentials) if price_differentials else 0
        
        return (avg_price_diff < tolerance and 
                max_price_diff < tolerance * 2 and 
                len(price_differentials) > 0)

    
    def train(self, episodes=500, batch_size=128, max_steps=300):
            rewards_history = {agent.agent_id: [] for agent in self.supply_agents + self.demand_agents}
            flows_history = []
            loss_history = {agent.agent_id: [] for agent in self.supply_agents + self.demand_agents}
            
            start_time = time.time()
            
            for episode in range(episodes):
                episode_start = time.time() #prima calcolavo il tempo necessario, ora non serve più perché ci mette poco
                supply_states, demand_states = self.reset()
                episode_rewards = {agent.agent_id: 0 for agent in self.supply_agents + self.demand_agents}
                episode_losses = {agent.agent_id: [] for agent in self.supply_agents + self.demand_agents}
                
                # reset detector
                self.equilibrium_detector = ImprovedEquilibriumDetector()
                
                for step in range(max_steps):
                    supply_actions = [agent.act(state) for agent, state in zip(self.supply_agents, supply_states)]
                    demand_actions = [agent.act(state) for agent, state in zip(self.demand_agents, demand_states)]
                    
                    next_supply_states, next_demand_states, supply_rewards, demand_rewards, _ = self.step(
                        supply_actions, demand_actions
                    )
                    
                    # update equilibrium detector
                    self.equilibrium_detector.update(
                        self.flows, 
                        np.concatenate([self.supply_prices, self.demand_prices])
                    )
                    
                    
                    # fa tracking della reward e degli state
                    for i, agent in enumerate(self.supply_agents):
                        agent.remember(supply_states[i], supply_actions[i], supply_rewards[i], 
                                    next_supply_states[i], self.equilibrium_detector.is_equilibrium())
                        episode_rewards[agent.agent_id] += supply_rewards[i]
                    
                    for i, agent in enumerate(self.demand_agents):
                        agent.remember(demand_states[i], demand_actions[i], demand_rewards[i], 
                                    next_demand_states[i], self.equilibrium_detector.is_equilibrium())
                        episode_rewards[agent.agent_id] += demand_rewards[i]
                    
                    # update statee
                    supply_states = next_supply_states
                    demand_states = next_demand_states
                    
                    # tracking loss e training
                    for agent in self.supply_agents + self.demand_agents:
                        loss, actor_loss, critic_loss = agent.replay(batch_size)
                        episode_losses[agent.agent_id].append((loss, actor_loss, critic_loss))
                    
                    # early  stopping (non so manco se funziona, mai arrivata all'equilibrio)
                    if self.equilibrium_detector.is_equilibrium():
                        flows_history.append(np.copy(self.flows))
                        print(f"Equilibrium reached in Episode {episode+1}, Step {step+1}")
                        break                      
                        
                        
                    print(f"Episode {episode+1}/{episodes}, Step {step+1}")
                    print(f"  Commodity 1 flow: {self.flows[0, 0, 0, 0]:.2f}, Commodity 2 flow: {self.flows[1, 0, 0, 0]:.2f}")
                    print(f"  Supply Agent Reward: {episode_rewards['Supply_0']:.2f}")
                
                    for agent_id, reward in episode_rewards.items():
                        rewards_history[agent_id].append(reward)
                    
                    for agent_id, losses in episode_losses.items():
                        if losses:  #vede le loss
                            avg_loss = np.mean([l[0] for l in losses if not np.isnan(l[0])])
                            loss_history[agent_id].append(avg_loss)
                    
                    break
            
                total_time = time.time() - start_time
                print(f"\nTraining completed in {total_time:.2f} seconds")
    
            
            return rewards_history, flows_history, loss_history
    
    def test(self, episodes=10):
        #testing -> funzione analoga alla training senza esplorazione
        results = []
        
        for episode in range(episodes):
            supply_states, demand_states = self.reset()
            done = False
            step = 0
            
            while not done and step < 100:
                supply_actions = [agent.act(state, training=False) for agent, state in zip(self.supply_agents, supply_states)]
                demand_actions = [agent.act(state, training=False) for agent, state in zip(self.demand_agents, demand_states)]
                
                
                next_supply_states, next_demand_states, supply_rewards, demand_rewards, done = self.step(
                    supply_actions, demand_actions
                )
                
                
                supply_states = next_supply_states
                demand_states = next_demand_states
                
                step += 1
            
            # Store dell'ultimo state
            results.append({
                'flows': self.flows.copy(),
                'supplies': self.supplies.copy(),
                'demands': self.demands.copy(),
                'supply_prices': self.supply_prices.copy(),
                'demand_prices': self.demand_prices.copy(),
                'trans_costs': self.trans_costs.copy(),
                'steps': step
            })
            
            print(f"Test Episode {episode+1}: Equilibrium reached in {step} steps")
            print(f"Commodity 1 flow: {self.flows[0, 0, 0, 0]:.2f} (Target: 2.00)")
            print(f"Commodity 2 flow: {self.flows[1, 0, 0, 0]:.2f} (Target: 10.00)")
        
        return results
    
class ImprovedEquilibriumDetector:
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
        if len(self.flow_history) < self.window_size:
            return False
        
        flow_changes = [np.max(np.abs(self.flow_history[i+1] - self.flow_history[i])) 
                        for i in range(len(self.flow_history)-1)]
        
        price_changes = [np.max(np.abs(self.price_history[i+1] - self.price_history[i])) 
                         for i in range(len(self.price_history)-1)]
        
        flow_stability = np.mean(flow_changes) < self.tolerance
        price_stability = np.mean(price_changes) < self.tolerance
        
        # fa track dei periodi stabili nel training per evitare l'oscillazione (ma non la evita davvero)
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


class RewardShapingStrategy:
    @staticmethod
    def calculate_market_efficiency(flows, supply_prices, demand_prices, trans_costs, exchange_rates):
        total_efficiency = 0
        K, L, m, n = flows.shape
        capacities = np.zeros((m, n, L))
        capacities[0, 0, 0] = 15.0
        
        for k in range(K):
            for l in range(L):
                for i in range(m):
                    for j in range(n):
                        #usa  la condizione del  paper per l'equilibrio -> ma a questo punto è come se calcolassi comunque l'iterativa, solo che mi serve da confronto
                        price_diff = (supply_prices[k, i] + trans_costs[k, l, i, j]) * exchange_rates[i, j] - demand_prices[k, j]
                        flow = flows[k, l, i, j]
                        
                        # per flow positivi differenza molto piccola
                        # per flow a zero differenza molto alta
                        if flow > 0.01:  # piccola threshold
                            #penalizza i flow molto bassi
                            total_efficiency -= abs(price_diff) * 10.0
                        else:
                            # per flow vicini a zero, abbiamo supply price + transport cost >= demand price
                            # se è vero il contrario viola le condizioni di equilibrio
                            if price_diff < 0:
                                total_efficiency -= abs(price_diff) * 5.0
        
        # bound di capacità
        for l in range(L):
            for i in range(m):
                for j in range(n):
                    total_flow = np.sum(flows[:, l, i, j])
                    if total_flow > capacities[i, j, l]:
                        total_efficiency -= (total_flow - capacities[i, j, l]) * 20.0
        
        return total_efficiency


def run_marl_example_1():
    print("MARL Example 1: Two Commodities, Single Route with High Capacity, Single Supply and Demand Market")
    print("=" * 80)
    
    m, n, K, L = 1, 1, 2, 1
    exchange_rates = np.ones((m, n))
    
    capacities = np.zeros((m, n, L))
    capacities[0, 0, 0] = 15.0
    
    env = AgriculturalTradeMARL(
        m, n, K, L, exchange_rates, capacities, 
        discretization=100,  #forse cambiare i parametri è la soluzione
        learning_rate=0.0003,
        gamma=0.99
    )
    
    env.supply_agents = [
        ImprovedMARLAgricultureAgent(
            f"Supply_{i}", 
            env.supply_state_size, 
            env.supply_action_size,
            learning_rate=0.0005,
            gamma=0.99
        ) for i in range(m)
    ]
    
    env.demand_agents = [
        ImprovedMARLAgricultureAgent(
            f"Demand_{j}", 
            env.demand_state_size, 
            env.demand_action_size,
            learning_rate=0.0005,
            gamma=0.99
        ) for j in range(n)
    ]
    
    print("Training agents...")
    rewards_history, flows_history, loss_history = env.train(
        episodes=2000,  #meglio provarla con pochi episodi -> per 100 con 500 di max_steps ci mette poco
        batch_size=64, #ho provato con 32, 64 e 128 ma non cambia niente
        max_steps=1000  #alzare a 1000 forse è meglio
    )
    
    # plot vari
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for agent_id, rewards in rewards_history.items():
        if 'Supply' in agent_id:
            plt.plot(rewards, label=agent_id)
    plt.title('Supply Agent Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    for agent_id, rewards in rewards_history.items():
        if 'Demand' in agent_id:
            plt.plot(rewards, label=agent_id)
    plt.title('Demand Agent Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    for agent_id, losses in loss_history.items():
        if losses: 
            plt.plot(losses, label=agent_id)
    plt.title('Agent Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    
    #test
    print("\nTesting agents...")
    results = env.test(episodes=5)
    
    #convergenza da test
    plt.figure(figsize=(10, 6))
    commodity1_flows = [result['flows'][0, 0, 0, 0] for result in results]
    commodity2_flows = [result['flows'][1, 0, 0, 0] for result in results]
    
    plt.plot(commodity1_flows, label='Commodity 1 Flow')
    plt.plot(commodity2_flows, label='Commodity 2 Flow')
    plt.axhline(y=2.0, color='r', linestyle='--', label='Target Commodity 1 Flow')
    plt.axhline(y=10.0, color='g', linestyle='--', label='Target Commodity 2  Flow')

    plt.title('Commodity Flows in Test Episodes')
    plt.xlabel('Test Episode')
    plt.ylabel('Flow Amount')
    plt.legend()
    plt.grid(True)
    plt.savefig('flow_convergence.png')
    
    #risultati finali
    final_result = results[-1]
    
    print("\nFinal Results:")
    print("--------------")
    print(f"Commodity 1 Flow: {final_result['flows'][0, 0, 0, 0]:.2f} (Target: 2.00)")
    print(f"Commodity 2 Flow: {final_result['flows'][1, 0, 0, 0]:.2f} (Target: 10.00)")
    
    print("\nPrices and Costs:")
    print("----------------")
    print(f"Supply price of commodity 1: {final_result['supply_prices'][0, 0]:.2f} (Target: 15.00)")
    print(f"Supply price of commodity 2: {final_result['supply_prices'][1, 0]:.2f} (Target: 15.00)")
    print(f"Transportation cost of commodity 1: {final_result['trans_costs'][0, 0, 0, 0]:.2f} (Target: 3.00)")
    print(f"Transportation cost of commodity 2: {final_result['trans_costs'][1, 0, 0, 0]:.2f} (Target: 12.00)")
    print(f"Demand price of commodity 1: {final_result['demand_prices'][0, 0]:.2f} (Target: 18.00)")
    print(f"Demand price of commodity 2: {final_result['demand_prices'][1, 0]:.2f} (Target: 27.00)")


if __name__ == "__main__":
    run_marl_example_1()



