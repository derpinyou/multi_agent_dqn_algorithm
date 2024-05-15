import math
import numpy as np
import pandas as pd

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


class MarketPlacesEnv():

    def __init__(
            self,
            households_num=5,
            firms_num=2,
            marketplaces_num=2,
            distributors_num=1,
            distributor_technology=0.5,
            firms_technology_shocks=lambda x: np.random.normal(0, 1),
            marketplace_technology_shocks=lambda x: np.random.normal(0, 0.25),
            episodes_horizon=100,
            default_time_capacity=100

    ):
        self.households_num = households_num
        self.savings_distr = [0 for j in range(households_num)]
        self.households_consumption_online = [[[0, 0], [0, 0]] for j in range(households_num)]  ### [[т1, т2][т1 , т2]]
        self.households_consumption_offline = [[0, 0] for j in range(households_num)]  ### [т1, т2]
        self.households_demand = [[1/6] * 6 for j in range(households_num)]
        self.skill_levels = [[1, 1.2][::int(math.copysign(1, j % 2 - 0.5))] for j in range(households_num)]
        self.firm_chosen = [k % 2 for k in range(households_num)]
        self.isoelasticity = [[0.33, 0.5] for j in range(households_num)]
        self.discount_factor = [0.99 - 0.01 * j for j in range(households_num)]
        self.default_time_capacity = default_time_capacity

        self.episodes_horizon = episodes_horizon
        self.current_episode = 0

        self.marketplaces_num = marketplaces_num
        self.marketplace_technology_shocks = marketplace_technology_shocks
        self.marketplace_technologies = [j + 1 for j in range(marketplaces_num)]
        self.marketplace_rates = [(j + 1) * 10 for j in range(marketplaces_num)]
        self.marketplace_stock = [[0, 0] for j in range(marketplaces_num)]

        self.distributors_num = distributors_num
        self.distributor_technology = distributor_technology
        self.distributor_rate = 30
        self.distributor_stock = [0, 0]

        self.firms_num = firms_num
        self.firms_inventory = [0, 0]
        self.firms_distribution_attrition = [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]
        self.firms_online_prices = [[10, 10], [10, 10]]  ### [t1, t2],[t1, t2]
        self.firms_offline_prices = [12, 12]
        self.firms_technology_shocks = firms_technology_shocks
        self.firms_technologies = [j + 1 for j in range(firms_num)]
        self.firms_wages = [20, 30]

    def marketplace_reward(self, agent_idx=0):
        """
        calculate marketmaker reward over action
        :param self:
        :param agent_idx: marketplace's index
        :return: delta reward
        """

        prices_online = self.firms_online_prices[agent_idx]
        cons_onl = self.households_consumption_online
        bought_amount = [cons_onl[k][agent_idx] for k in range(len(cons_onl))]
        bought_amount = [
            sum([bought_amount[j][k] for j in range(len(bought_amount))])
            for k in range(len(prices_online))
        ]

        income = sum([prices_online[j] * bought_amount[j] for j in range(len(prices_online))])

        rate = self.marketplace_rates[agent_idx] / 100

        return income * rate

    def household_reward(self, agent_idx=0):
        """
        calculate household reward over action
        :param self
        :param agent_idx: household's index
        :return: delta reward
        """

        consumed_online = self.households_consumption_online[agent_idx]
        consumed_offline = self.households_consumption_offline[agent_idx]

        costs_offline = 1 / self.distributor_technology
        costs_online = [1 / tech for tech in self.marketplace_technologies]

        time_wasted_offline = sum([costs_offline * el for el in consumed_offline])
        time_wasted_online = sum([
            sum([costs_online[k] * consumed_online[k][j] for j in range(len(consumed_online[k]))])
            for k in range(len(costs_online))
        ])

        total_time_spent = time_wasted_offline + time_wasted_online

        gamma = self.isoelasticity[agent_idx]

        consumed_total = [
            sum([consumed_offline[k] + consumed_online[j][k] for j in range(len(consumed_online))])
            for k in range(len(consumed_offline))
        ]

        utility_from_goods = sum([
            consumed_total[k] ** (gamma[k]) / (1 - gamma[k])
            for k in range(len(consumed_total))
        ])
        utility_from_rest = 0.03 * (self.default_time_capacity - total_time_spent)

        return utility_from_goods + utility_from_rest

    def firm_reward(self, agent_idx=0):
        """
        calculate firm reward over action
        :param self:
        :param agent_idx: firm's agent index
        :return: delta reward
        """

        # Выручка
        price_offline = self.firms_offline_prices[agent_idx]
        consumption_offline = self.households_consumption_offline
        total_consumed_offline = sum([consumption_offline[j][agent_idx] for j in range(len(consumption_offline))])

        prices_online = self.firms_online_prices
        prices_online = [prices_online[j][agent_idx] for j in range(len(prices_online))]
        consumption_online = self.households_consumption_online
        consumption_online = [
            sum([consumption_online[j][k][agent_idx] for j in range(len(consumption_online))])
            for k in range(len(prices_online))
        ]

        online_gross_income = sum([
            consumption_online[j] * prices_online[j] * (1 - self.marketplace_rates[j] / 100)
            for j in range(len(prices_online))
        ])
        offline_income = total_consumed_offline * price_offline

        # Зарплаты и премии
        skills = self.skill_levels
        skills = np.array([el[agent_idx] for el in skills])
        indices_of_workers = np.where(np.array(self.firm_chosen) == agent_idx)
        wages_paid = sum(skills[indices_of_workers]) * self.firms_wages[agent_idx]

        # Косты коммуникация с маркетплейсом
        communication_costs = sum([
            self.marketplace_technologies[j] * consumption_online[j] for j in range(len(consumption_online))
        ])

        # Косты производства
        production_costs = (total_consumed_offline + sum(consumption_online)) * self.firms_technologies[agent_idx]

        # Косты поставщику
        delivery_costs = self.distributor_technology * total_consumed_offline
        
        return online_gross_income + offline_income - \
               wages_paid - communication_costs - \
               delivery_costs - production_costs

    def distributor_reward(self):
        """
        calculate distributor reward over action
        :param self:
        :return: delta reward
        """

        prices_offline = self.firms_offline_prices

        consumption_offline = self.households_consumption_offline
        each_good_consumed = [
            sum([consumption_offline[j][k] for j in range(len(consumption_offline))])
            for k in range(len(prices_offline))
        ]

        income = sum([prices_offline[j] * each_good_consumed[j] for j in range(len(prices_offline))])

        current_rate = self.distributor_rate / 100

        return current_rate * income

    def household_step(self, action_, agent_idx=0, regime='choose_firm'):
        """
        env dynamics due to some household action based on
        individual bounded observation space
        :param self:
        """

        if regime == 'choose_firm':
            if action_ == 2:
                if self.firm_chosen[agent_idx] == 0:
                    self.firm_chosen[agent_idx] = 1
                else:
                    self.firm_chosen[agent_idx] = 0

        elif regime == 'redistribute_demand':
            if action_ == 1:
                self.households_demand[agent_idx][0] += 0.1
                demand_mask = self.households_demand[agent_idx]
                demand_mask = [demand_mask[j] / sum(demand_mask) for j in range(len(demand_mask))]
                self.households_demand[agent_idx] = demand_mask
            elif action_ == 2:
                self.households_demand[agent_idx][1] += 0.1
                demand_mask = self.households_demand[agent_idx]
                demand_mask = [demand_mask[j] / sum(demand_mask) for j in range(len(demand_mask))]
                self.households_demand[agent_idx] = demand_mask
            elif action_ == 3:
                self.households_demand[agent_idx][2] += 0.1
                demand_mask = self.households_demand[agent_idx]
                demand_mask = [demand_mask[j] / sum(demand_mask) for j in range(len(demand_mask))]
                self.households_demand[agent_idx] = demand_mask
            elif action_ == 4:
                self.households_demand[agent_idx][3] += 0.1
                demand_mask = self.households_demand[agent_idx]
                demand_mask = [demand_mask[j] / sum(demand_mask) for j in range(len(demand_mask))]
                self.households_demand[agent_idx] = demand_mask
            elif action_ == 5:
                self.households_demand[agent_idx][4] += 0.1
                demand_mask = self.households_demand[agent_idx]
                demand_mask = [demand_mask[j] / sum(demand_mask) for j in range(len(demand_mask))]
                self.households_demand[agent_idx] = demand_mask
            else:
                self.households_demand[agent_idx][5] += 0.1
                demand_mask = self.households_demand[agent_idx]
                demand_mask = [demand_mask[j] / sum(demand_mask) for j in range(len(demand_mask))]
                self.households_demand[agent_idx] = demand_mask

            current_money = np.max([0.1, self.savings_distr[agent_idx]])
            demand = self.households_demand[agent_idx]
            current_money_per_good = [current_money * el for el in demand]

            prices = self.firms_online_prices[0] + \
                     self.firms_online_prices[1] + \
                     [el * (1 + self.distributor_rate / 100) for el in self.firms_offline_prices]

            goods_amounts_req = [current_money_per_good[j] / prices[j] for j in range(len(prices))]
            inventories = self.marketplace_stock[0] + self.marketplace_stock[1] + self.distributor_stock
            goods_amounts = [
                goods_amounts_req[j]
                if goods_amounts_req[j] <= inventories[j]
                else inventories[j]
                for j in range(len(inventories))
            ]
            goods_amounts = [
                0 
                if goods_amounts[j] <= 0
                else goods_amounts[j] 
                for j in range(len(goods_amounts))
            ]
            money_spent = sum([goods_amounts[j] * prices[j] for j in range(len(prices))])

            self.households_consumption_online[agent_idx] = [goods_amounts[:2], goods_amounts[2:4]]
            self.households_consumption_offline[agent_idx] = goods_amounts[4:]
            self.savings_distr[agent_idx] -= money_spent

            self.marketplace_stock[0] = [
                self.marketplace_stock[0][j] - goods_amounts[j]
                for j in range(len(self.marketplace_stock[0]))
            ]
            self.marketplace_stock[1] = [
                self.marketplace_stock[1][j] - goods_amounts[j+2]
                for j in range(len(self.marketplace_stock[1]))
            ]
            self.distributor_stock = [
                self.distributor_stock[j] - goods_amounts[j+2]
                for j in range(len(self.distributor_stock))
            ]

    def firm_step(self, action_, regime='wage', agent_idx=0):
        """
        env dynamics due to some firm action based on
        individual bounded observation space
        :param self:
        """

        if regime == 'wage':
            if action_ == 2:
                self.firms_wages[agent_idx] *= 1.1
            elif action_ == 3:
                self.firms_wages[agent_idx] *= 0.9
            self.firms_wages[agent_idx] = np.min([self.firms_wages[agent_idx], 100])

            to_whom = np.where(np.array(self.firm_chosen) == agent_idx)[0]
            for ind in to_whom:
                self.savings_distr[ind] += self.firms_wages[agent_idx] * \
                                           self.skill_levels[ind][agent_idx]

        elif regime == 'produce':
            workerks_idx = np.where(np.array(self.firm_chosen) == agent_idx)
            total_productivity = np.sum(np.array(self.skill_levels)[workerks_idx])
            self.firms_inventory[agent_idx] += total_productivity

        elif regime == 'redistribute_inventories':
            if action_ == 1:
                self.firms_distribution_attrition[agent_idx][0] += 0.1
                attr_mask = self.firms_distribution_attrition[agent_idx]
                attr_mask = [attr_mask[j] / sum(attr_mask) for j in range(len(attr_mask))]
                self.firms_distribution_attrition[agent_idx] = attr_mask
            elif action_ == 2:
                self.firms_distribution_attrition[agent_idx][1] += 0.1
                attr_mask = self.firms_distribution_attrition[agent_idx]
                attr_mask = [attr_mask[j] / sum(attr_mask) for j in range(len(attr_mask))]
                self.firms_distribution_attrition[agent_idx] = attr_mask
            else:
                self.firms_distribution_attrition[agent_idx][2] += 0.1
                attr_mask = self.firms_distribution_attrition[agent_idx]
                attr_mask = [attr_mask[j] / sum(attr_mask) for j in range(len(attr_mask))]
                self.firms_distribution_attrition[agent_idx] = attr_mask

            goods_to_deliver = [self.firms_inventory[agent_idx] * el for el in attr_mask]
            self.marketplace_stock[0][agent_idx] += goods_to_deliver[0]
            self.marketplace_stock[1][agent_idx] += goods_to_deliver[1]
            self.distributor_stock[agent_idx] += goods_to_deliver[2]
            self.firms_inventory[agent_idx] = 0

        elif regime == 'prices_online_1':
            # 1 -> 0%
            # 2 -> +10%
            # 3 -> -10%
            if action_ == 2:
                self.firms_online_prices[0][agent_idx] *= 1.1
            if action_ == 3:
                self.firms_online_prices[0][agent_idx] *= 0.9

        elif regime == 'prices_online_2':
            # 1 -> 0%
            # 2 -> +10%
            # 3 -> -10%
            if action_ == 2:
                self.firms_online_prices[1][agent_idx] *= 1.1
            if action_ == 3:
                self.firms_online_prices[1][agent_idx] *= 0.9

        elif regime == 'price_offline':
            # 1 -> 0%
            # 2 -> +10%
            # 3 -> -10%
            if action_ == 2:
                self.firms_offline_prices[agent_idx] *= 1.1
            elif action_ == 3:
                self.firms_offline_prices[agent_idx] *= 0.9

        #elif regime == 'channel_online_1':
        #    ### 1 -> 0%
        #    ### 2 -> 25%
        #    ### 3 -> 50%
        #    if action_ == 2:
        #        self.marketplace_stock[0][agent_idx] += self.firms_inventory * 0.25
        #        self.firms_inventory[agent_idx] *= 0.75
        #    if action_ == 3:
        #        self.marketplace_stock[0][agent_idx] += self.firms_inventory * 0.5
        #        self.firms_inventory[agent_idx] *= 0.5

        #elif regime == 'channel_online_2':
        #    ### 1 -> 0%
        #    ### 2 -> 25%
        #    ### 3 -> 50%
        #    if action_ == 2:
        #        self.marketplace_stock[1][agent_idx] += self.firms_inventory * 0.25
        #        self.firms_inventory[agent_idx] *= 0.75
        #    if action_ == 3:
        #        self.marketplace_stock[1][agent_idx] += self.firms_inventory * 0.5
        #        self.firms_inventory[agent_idx] *= 0.5

        #elif regime == 'channel_offline':
        #    self.distributor_stock[agent_idx] = self.firms_inventory[agent_idx]
        #    self.firms_inventory[agent_idx] = 0


    def marketplace_step(self, action_, agent_idx=0):
        """
        env dynamics due to some marketplace action based on
        individual bounded observation space
        :param self:
        :param action_: action num
        :param agent_idx: mp agent index

        a = 1: Hold curren rate
        a = 2: Increase rate by 1%
        a = 3: Decrease rate by 1%
        """

        if action_ == 2:
            self.marketplace_rates[agent_idx] += 1
        elif action_ == 3:
            self.marketplace_rates[agent_idx] -= 1

    def distributor_step(self, action_):
        """
        env dynamics due to the only distributors action based on
        individual bounded observation space
        :param self:
        :param action_: action num

        a = 1: Hold curren rate
        a = 2: Increase rate by 1%
        a = 3: Decrease rate by 1%
        """

        if action_ == 2:
            self.distributor_rate += 1
        elif action_ == 3:
            self.distributor_rate -= 1

    def stochastic_step(self,):
        """
        change state in response to action if available
        :param self:
        :return:
        """

        self.marketplace_technologies[0] = np.min([3, np.max([
            0.5,
            self.marketplace_technologies[0] + self.marketplace_technology_shocks(0)
        ])])
        self.marketplace_technologies[1] = np.min([3, np.max([
            0.5,
            self.marketplace_technologies[1] + self.marketplace_technology_shocks(0)
        ])])
        self.firms_technologies[0] = np.min([3, np.max([
            0.5,
            self.firms_technologies[0] + self.firms_technology_shocks(0)
        ])])
        self.firms_technologies[1] = np.min([3, np.max([
            0.5,
            self.firms_technologies[1] + self.firms_technology_shocks(0)
        ])])

        return

    def get_household_state(self, agent_idx=0, regime='choose_firm'):
        """
        return current state for household
        :param self:
        :param agent_idx: agents index
        :param regime: regime mode for state diff
        :return:
        """

        regimes_dict = {
            'choose_firm':0,
            'redistribute_demand':1
        }

        return flatten([
            self.current_episode,
            self.episodes_horizon,
            self.firms_online_prices,
            self.firms_offline_prices,
            self.firms_wages,
            self.skill_levels[agent_idx],
            self.marketplace_technologies,
            self.distributor_technology,
            self.marketplace_stock,
            self.distributor_stock,
            self.savings_distr[agent_idx],
            self.households_demand[agent_idx],
            self.isoelasticity[agent_idx],
            self.firm_chosen[agent_idx],
            self.distributor_rate,
            regimes_dict[regime]
        ])

    def get_marketplace_state(self, agent_idx=0):
        """
        return current state for marketplace
        :param self:
        :param agent_idx: agents index
        :return:
        """

        return flatten([
            self.current_episode,
            self.episodes_horizon,
            self.firms_online_prices,
            self.firms_offline_prices,
            self.marketplace_rates,
            self.marketplace_stock[agent_idx],
            self.marketplace_technologies,
            self.households_consumption_online[agent_idx],
            self.distributor_technology,
        ])


    def get_distributor_state(self):
        """
        return current state for distributor
        :param self:
        :return:
        """

        return flatten([
            self.current_episode,
            self.episodes_horizon,
            self.firms_online_prices,
            self.firms_offline_prices,
            self.marketplace_rates,
            self.marketplace_technologies,
            self.households_consumption_offline,
            self.distributor_technology,
            self.distributor_rate,
            self.distributor_stock
        ])

    def get_firm_state(self, agent_idx=0, regime='wage'):
        """
        return current state for firm
        :param self:
        :param agent_idx: agents index
        :param regime: regime mode state for diff
        :return:
        """

        regimes_dict = {
            'wage': 0,
            'produce': 1,
            'redistribute_inventories':2,
            'prices_online_1':3,
            'prices_online_2':4,
            'price_offline':5
        }

        return flatten([
            self.current_episode,
            self.episodes_horizon,
            [elem[agent_idx] for elem in self.households_consumption_offline],
            [[elem[0][agent_idx], elem[1][agent_idx]] for elem in self.households_consumption_online],
            self.firms_online_prices,
            self.firms_offline_prices[agent_idx],
            self.firms_wages[agent_idx],
            self.skill_levels,
            self.firm_chosen,
            self.firms_inventory[agent_idx],
            self.marketplace_technologies,
            self.marketplace_rates,
            self.marketplace_stock,
            self.distributor_technology,
            self.distributor_stock,
            self.distributor_rate,
            regimes_dict[regime]
        ])
        
    def reset(self):
        """
        reset a session to its initial state
        :param self:
        :return: initial state
        """

        self.households_num = 5
        self.savings_distr = [0 for j in range(5)]
        self.households_consumption_online = [[[0, 0], [0, 0]] for j in range(5)]  ### [[т1, т2][т1 , т2]]
        self.households_consumption_offline = [[0, 0] for j in range(5)]  ### [т1, т2]
        self.households_demand = [[1/6] * 6 for j in range(5)]
        self.skill_levels = [[1, 1.2][::int(math.copysign(1, j % 2 - 0.5))] for j in range(5)]
        self.firm_chosen = [k % 2 for k in range(5)]
        self.isoelasticity = [[0.33, 0.5] for j in range(5)]
        self.discount_factor = [0.99 - 0.01 * j for j in range(5)]
        self.default_time_capacity = 100

        self.episodes_horizon = 100
        self.current_episode = 0

        self.marketplaces_num = 2
        self.marketplace_technology_shocks = lambda x: np.random.normal(0, 0.25)
        self.marketplace_technologies = [j + 1 for j in range(self.marketplaces_num)]
        self.marketplace_rates = [(j + 1) * 10 for j in range(self.marketplaces_num)]
        self.marketplace_stock = [[0, 0] for j in range(self.marketplaces_num)]

        self.distributors_num = 1
        self.distributor_technology = 0.5
        self.distributor_rate = 30
        self.distributor_stock = [0, 0]

        self.firms_num = 2
        self.firms_inventory = [0, 0]
        self.firms_distribution_attrition = [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]
        self.firms_online_prices = [[10, 10], [10, 10]]  ### [t1, t2],[t1, t2]
        self.firms_offline_prices = [12, 12]
        self.firms_technology_shocks = lambda x: np.random.normal(0, 1)
        self.firms_technologies = [j + 1 for j in range(2)]
        self.firms_wages = [20, 30]

        return
