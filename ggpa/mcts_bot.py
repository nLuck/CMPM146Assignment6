from __future__ import annotations
import math
from copy import deepcopy
import time
from agent import Agent
from battle import BattleState
from card import Card
from action.action import EndAgentTurn, PlayCard
from game import GameState
from ggpa.ggpa import GGPA
from config import Verbose
import random

from status_effecs import StatusEffectRepo


# You only need to modify the TreeNode!
class TreeNode:
    # You can change this to include other attributes. 
    # param is the value passed via the -p command line option (default: 0.5)
    # You can use this for e.g. the "c" value in the UCB-1 formula
    def __init__(self, param, parent=None):
        self.children = {}
        self.parent = parent
        self.results = []    # list of rollout scores
        self.visits = 0      # number of times this node was visited
        self.param = param
        self.action = None   # action that led to this node
    
    # REQUIRED function
    # Called once per iteration
    def step(self, state):
        self.select(state)
        
    # REQUIRED function
    # Called after all iterations are done; should return the 
    # best action from among state.get_actions()
    def get_best(self, state):
        valid_actions = state.get_actions()
        valid_actions_keys = {a.key() for a in valid_actions}
        
        if state.verbose != Verbose.NO_LOG:
            print("Child action keys:", list(self.children.keys()))
            print("Valid action keys:", [a.key() for a in valid_actions])
            print("Current hand:", [(c.name, c.upgrade_count) for c in state.hand])
        
        best_action = None
        best_score = float('-inf')
        for action_key, child in self.children.items():
            if action_key in valid_actions_keys and child.visits > 0:
                avg_score = sum(child.results) / child.visits
                if avg_score > best_score:
                    best_score = avg_score
                    best_action = next(a for a in state.get_actions() if a.key() == action_key)
        
        if best_action is None:
            if state.verbose != Verbose.NO_LOG:
                print("No valid child actions found")
            for action in valid_actions:
                if isinstance(action.to_action(state), EndAgentTurn):
                    return action
            return random.choice(valid_actions)
        
        return best_action
        
    # REQUIRED function (implementation optional, but *very* helpful for debugging)
    # Called after all iterations when the -v command line parameter is present
    def print_tree(self, indent = 0):
        # print tree for debugging
        indent_str = " " * indent
        avg_score = sum(self.results) / self.visits if self.visits > 0 else 0
        action_str = self.action.key() if self.action else "Root"
        print(f"{indent_str}{action_str}: visits={self.visits}, avg_score={avg_score:.4f}")
        for child in self.children.values():
            child.print_tree(indent + 2)


    # RECOMMENDED: select gets all actions available in the state it is passed
    # If there are any child nodes missing (i.e. there are actions that have not 
    # been explored yet), call expand with the available options
    # Otherwise, pick a child node according to your selection criterion (e.g. UCB-1)
    # apply its action to the state and recursively call select on that child node.
    def select(self, state):
        # if node not fully expanded, expand it
        available_actions = state.get_actions()
        valid_action_keys = {a.key() for a in available_actions}
        
        if state.verbose != Verbose.NO_LOG:
            print("select: Available action keys:", [a.key() for a in available_actions])
            print("select: Current hand:", [(c.name, c.upgrade_count) for c in state.hand])
        
        self.children = {k: v for k, v in self.children.items() if k in valid_action_keys}
        
        unexplored = [a for a in available_actions if a.key() not in self.children]
        if unexplored:
            self.expand(state, unexplored)
            return
        
        if not self.children:
            if state.verbose != Verbose.NO_LOG:
                print("select: no children to select")
            return
        
        total_visits = self.visits
        ucb_scores = []
        probabilities = []
        for child in self.children.values():
            if child.visits == 0:
                ucb_score = float('inf')
            else:
                avg_score = sum(child.results) / child.visits
                exploration = self.param * math.sqrt(math.log(total_visits) / child.visits)
                ucb_score = avg_score + exploration
            ucb_scores.append(ucb_score)
        
        total_ucb = sum(ucb_scores)
        if total_ucb == 0 or math.isinf(total_ucb):
            probabilities = [1 / len(self.children) for _ in self.children]
        else:
            probabilities = [score / total_ucb for score in ucb_scores]
        
        # select a child randomly based on UCB-1 probablities
        child_key = random.choices(list(self.children.keys()), weights=probabilities, k=1)[0]
        child = self.children[child_key]
        
        if child.action.key() not in valid_action_keys:
            if state.verbose != Verbose.NO_LOG:
                print(f"select: invalid child action {child.action.key()}")
                return
        
        next_state = state.copy_undeterministic()
        next_state.step(child.action)
        child.select(next_state)

    # RECOMMENDED: expand takes the available actions, and picks one at random,
    # adds a child node corresponding to that action, applies the action ot the state
    # and then calls rollout on that new node
    def expand(self, state, available):
        # pick random unexplored action and create new child node
        action = random.choice(available)
        new_state = state.copy_undeterministic()
        new_state.step(action)
        child = TreeNode(self.param, parent=self)
        child.action = action
        self.children[action.key()] = child
        
        score = child.rollout(new_state)
        child.backpropagate(score)

    # RECOMMENDED: rollout plays the game randomly until its conclusion, and then 
    # calls backpropagate with the result you get 
    def rollout(self, state):
        current_state = state.copy_undeterministic()
        while not current_state.ended():
            actions = current_state.get_actions()
            preferred = []
            for action in actions:
                if action.card:
                    card_name = action.card[0]
                    if card_name in ["Bludgeon", "Thunderclap", "Bash"] and current_state.mana >= 2:
                        preferred.append(action)
                    elif card_name == "Thunderclap" or card_name == "Inflame":
                        preferred.append(action)
                    elif card_name == "Defend" and current_state.health() < 0.4:
                        preferred.append(action)
                    elif card_name != "Offering" or current_state.health() > 0.5:
                        preferred.append(action)
            if preferred and random.random() < 0.8:
                action = random.choice(preferred)
            else:
                action = random.choice(actions)
            current_state.step(action)
        return self.score(current_state)
        
    # RECOMMENDED: backpropagate records the score you got in the current node, and 
    # then recursively calls the parent's backpropagate as well.
    # If you record scores in a list, you can use sum(self.results)/len(self.results)
    # to get an average.
    def backpropagate(self, result):
        self.results.append(result)
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(result)
        
    # RECOMMENDED: You can start by just using state.score() as the actual value you are 
    # optimizing; for the challenge scenario, in particular, you may want to experiment
    # with other options (e.g. squaring the score, or incorporating state.health(), etc.)
    def score(self, state): 
        if state.ended() and state.get_end_result() == 1:
            return 3.0
        health = state.health()
        score = state.score()
        health_weight = 0.6 if health < 0.3 else 0.4 if health < 0.5 else 0.2
        health_penalty = -0.3 if health < 0.1 else -0.1 if health < 0.2 else 0.0
        status_bonus = 0.0
        for enemy in state.enemies:
            if enemy.status_effect_state.has(StatusEffectRepo.VULNERABLE):
                status_bonus += 0.15
        if state.player.status_effect_state.has(StatusEffectRepo.STRENGTH):
            status_bonus += 0.15
        bludgeon_bonus = 0.2 if any(a.card and a.card[0] in ["Bludgeon", "SearingBlow"] and state.mana >= 3 for a in state.get_actions()) else 0.0
        return (1 - health_weight) * score + health_weight * health + health_penalty + status_bonus + bludgeon_bonus
        
        
# You do not have to modify the MCTS Agent (but you can)
class MCTSAgent(GGPA):
    def __init__(self, iterations: int, verbose: bool, param: float):
        self.iterations = iterations
        self.verbose = verbose
        self.param = param

    # REQUIRED METHOD
    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        actions = battle_state.get_actions()
        if len(actions) == 1:
            return actions[0].to_action(battle_state)
    
        t = TreeNode(self.param)
        start_time = time.time()

        for i in range(self.iterations):
            sample_state = battle_state.copy_undeterministic()
            t.step(sample_state)
        
        best_action = t.get_best(battle_state)
        if self.verbose:
            t.print_tree()
        
        if best_action is None:
            print("WARNING: MCTS did not return any action")
            return random.choice(self.get_choose_card_options(game_state, battle_state)) # fallback option
        return best_action.to_action(battle_state)
    
    # REQUIRED METHOD: All our scenarios only have one enemy
    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        return agent_list[0]
    
    # REQUIRED METHOD: Our scenarios do not involve targeting cards
    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return card_list[0]
