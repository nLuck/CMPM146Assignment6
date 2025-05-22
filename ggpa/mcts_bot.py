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
        
        best_action = None
        best_score = float('-inf')
        for action_key, child in self.children.items():
            if action_key in valid_actions_keys and child.visits > 0:
                avg_score = sum(child.results) / child.visits
                if avg_score > best_score:
                    best_score = avg_score
                    best_action = next(a for a in state.get_actions() if a.key() == action_key)
        
        if best_action is None:
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

        self.children = {k: v for k, v in self.children.items() if k in valid_action_keys}
        
        unexplored = [a for a in available_actions if a.key() not in self.children]
        if unexplored:
            self.expand(state, unexplored)
            return
        
        if not self.children:
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
    def rollout(self, state: BattleState):
        current_state = state.copy_undeterministic()
        
        while not current_state.ended():
            enemy_intent = current_state.enemies[0].get_intention(current_state.game_state, current_state) if current_state.enemies else None
            intent_damage = self.get_intent_damage(enemy_intent)
            actions = current_state.get_actions()
            preferred = []
            for action in actions:
                score = 0
                if isinstance(action, EndAgentTurn):
                    if current_state.health() <= intent_damage and intent_damage > 0:
                        score = -10
                    else:
                        score = 0
                elif action.card is not None:
                    card_name = action.card[0]
                    if card_name == "Bludgeon":
                        score += 32 * (2 if current_state.enemies[0].status_effect_state.has(StatusEffectRepo.VULNERABLE) > 0 else 1)
                    elif card_name == "SearingBlow":
                        score += 12 * (2 if current_state.enemies[0].status_effect_state.has(StatusEffectRepo.VULNERABLE) > 0 else 1)
                    elif card_name == "Bash":
                        score += 8 + 10
                    elif card_name == "Thunderclap":
                        score += 4 + 10
                    elif card_name == "Strike":
                        score += 6 * (2 if current_state.enemies[0].status_effect_state.has(StatusEffectRepo.VULNERABLE) > 0 else 1)
                    elif card_name == "Offering":
                        if current_state.health() > 6 + intent_damage:
                            score += 10
                        else:
                            score = -5
                    elif card_name == "Defend":
                        if intent_damage >= current_state.health():
                            score += 5
                    elif card_name == "Inflame":
                        score += 10
                    elif card_name == "PommelStrike":
                        score += 9 + 2
                else:
                    score = 0
                preferred.append((action, score))
            
            max_score = max(score for _, score in preferred)
            best_actions = [action for action, score in preferred if score == max_score]
            action = random.choice(best_actions)
            
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
        return (1 - health_weight) * score + health_weight * health + health_penalty
    
    def get_intent_damage(self, intent):
        if not intent:
            return 0
        from action.agent_targeted_action import DealAttackDamage
        if isinstance(intent, DealAttackDamage):
            return intent.values[0].peek()
        return 0
        
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
