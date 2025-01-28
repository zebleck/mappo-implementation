import numpy as np


class SimpleUltimatumEnv:
    """
    A simple implementation of the Ultimatum Game where:
    - One agent (proposer) offers a split of 100 points
    - Other agent (responder) accepts or rejects the offer
    - If accepted, both get their proposed split
    - If rejected, both get 0
    """

    def __init__(self):
        self.total_amount = 100
        self.n_agents = 2
        self.current_proposer = 0  # Track who is proposer (roles can switch)

    def reset(self):
        self.current_proposer = 1 - self.current_proposer  # Alternate roles
        self.current_step = 0
        self.current_offer = None

        # Observations: [is_proposer, offer_made]
        observations = []
        for i in range(self.n_agents):
            is_proposer = 1.0 if i == self.current_proposer else 0.0
            observations.append(np.array([is_proposer, -1.0]))  # -1 means no offer yet

        return observations

    def step(self, actions):
        """
        For proposer: action is amount to offer (0-100)
        For responder: action is 1 (accept) or 0 (reject)
        """
        self.current_step += 1
        rewards = np.zeros(self.n_agents)
        done = False

        responder = 1 - self.current_proposer

        # Handle proposer's offer
        if self.current_step == 1:
            self.current_offer = min(
                max(int(actions[self.current_proposer]), 0), self.total_amount
            )
            observations = []
            for i in range(self.n_agents):
                is_proposer = 1.0 if i == self.current_proposer else 0.0
                observations.append(np.array([is_proposer, float(self.current_offer)]))
            return observations, rewards, done

        # Handle responder's decision
        if self.current_step == 2:
            done = True
            if actions[responder] == 1:  # Accept
                rewards[self.current_proposer] = self.total_amount - self.current_offer
                rewards[responder] = self.current_offer
            # else: rewards stay 0 for rejection

        observations = []
        for i in range(self.n_agents):
            is_proposer = 1.0 if i == self.current_proposer else 0.0
            observations.append(np.array([is_proposer, float(self.current_offer)]))

        return observations, rewards, done

    def get_observation_size(self):
        return 2  # [is_proposer, offer_made]

    def get_action_size(self, is_proposer=True):
        return self.total_amount + 1 if is_proposer else 2
