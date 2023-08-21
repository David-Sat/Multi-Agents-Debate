"""
MAD: Multi-Agent Debate with Large Language Models
Copyright (C) 2023  The MAD Team

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import os
import json
import random
# random.seed(0)
from code.utils.agent import Agent

with open("./code/configs/keys.json") as json_data_file:
    key_config = json.load(json_data_file)

openai_api_key = key_config["OPENAI_API_KEY"]


with open('./code/configs/config.json') as f:
    config = json.load(f)



class DebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature:float, openai_api_key: str, sleep_time: float) -> None:
        """Create a player in the debate

        Args:
            model_name(str): model name
            name (str): name of this player
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            openai_api_key (str): As the parameter name suggests
            sleep_time (float): sleep because of rate limits
        """
        super(DebatePlayer, self).__init__(model_name, name, temperature, sleep_time)
        self.openai_api_key = openai_api_key


class Debate:
    def __init__(self,
            model_name: str='gpt-3.5-turbo-16k', 
            temperature: float=0, 
            num_players: int=3, 
            openai_api_key: str=None,
            config: dict=None,
            max_round: int=3,
            sleep_time: float=0
        ) -> None:
        """Create a debate

        Args:
            model_name (str): openai model name
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            num_players (int): num of players
            openai_api_key (str): As the parameter name suggests
            max_round (int): maximum Rounds of Debate
            sleep_time (float): sleep because of rate limits
        """

        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.openai_api_key = openai_api_key
        self.config = config
        self.max_round = max_round
        self.sleep_time = sleep_time

        self.answers = []
        self.expert_prompts = []

        self.init_prompt()

        # creat&init agents
        self.create_agents()
        self.init_agents()


    def init_prompt(self):
        def prompt_replace(key):
            self.config[key] = self.config['general_prompts'][key].replace("##debate_topic##", self.config["debate_topic"])
        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("judge_prompt_last2")
        self.expert_prompts = self.generate_expert_prompts().get('experts', [])


    def create_agents(self):
        expert_names = [expert['field'] for expert in self.expert_prompts]
        expert_names.append('Moderator')
        self.agents = [DebatePlayer(model_name=self.model_name, name=name, temperature=self.temperature, openai_api_key=self.openai_api_key, sleep_time=self.sleep_time) for name in expert_names]

        self.players = self.agents[:self.num_players]
        self.moderator = self.agents[-1]  
    
    
    def generate_expert_prompts(self):
        creator_player = DebatePlayer(model_name='gpt-4', name='Creator', temperature=self.temperature, openai_api_key=self.openai_api_key, sleep_time=self.sleep_time)

        creator_player.set_meta_prompt(self.config['general_prompts']['creator_meta_prompt'].replace('##debate_topic##', self.config["debate_topic"]))
        creator_player.set_meta_prompt(
            self.config['general_prompts']['creator_prompt']
            .replace('##debate_topic##', self.config["debate_topic"])
            .replace('##num_players##', str(self.num_players))
        )

        creator_player.add_event(self.config['general_prompts']['creator_prompt']
            .replace('##debate_topic##', self.config["debate_topic"])
            .replace('##num_players##', str(self.num_players))
            )
        ans = creator_player.construct_prompts()
        creator_player.add_memory(ans)
        
        #ans = eval(ans)
        print(ans)
        return ans


    def init_agents(self):

        expert_prompts_list = self.expert_prompts

        for idx, player in enumerate(self.players):
            if idx < len(expert_prompts_list):
                prompt_data = expert_prompts_list[idx]

                player.set_meta_prompt(prompt_data['prompt'])
                player.set_base_debate_prompt(prompt_data['debate_prompt'])

        self.moderator.set_meta_prompt(self.config['general_prompts']['moderator_meta_prompt'])
        
        
        # start: first round debate, state opinions
        print(f"===== Debate Round-1 =====\n")

        for player in self.players:
            answer = player.ask()
            player.add_memory(answer)
            self.config['base_answer'] = answer
            self.answers.append(self.config['argument'].replace('##player##', player.name).replace('##answer##', answer))

        
        self.moderator.add_event(self.config['general_prompts']['moderator_prompt'].replace('##mod_info##', '\n'.join(self.answers[-(self.num_players):])).replace('##round##', 'first'))
        self.mod_ans = self.moderator.ask()
        self.moderator.add_memory(self.mod_ans)
        self.mod_ans = eval(self.mod_ans)

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]

    def print_answer(self):
        print("\n\n===== Debate Done! =====")
        print("\n----- Debate Topic -----")
        print(self.config["debate_topic"])
        print("\n----- Debate Summary -----")
        print(self.config["summary"])
        print("\n----- Debate Answer -----")
        print(self.config["debate_answer"])
        print("\n----- Debate Reason -----")
        print(self.config["reasons"])

    def broadcast(self, msg: str):
        """Broadcast a message to all players. 
        Typical use is for the host to announce public information

        Args:
            msg (str): the message
        """
        # print(msg)
        for player in self.players:
            player.add_event(msg)

    def speak(self, speaker: str, msg: str):
        """The speaker broadcasts a message to all other players. 

        Args:
            speaker (str): name of the speaker
            msg (str): the message
        """
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        # print(msg)
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def ask_and_speak(self, player: DebatePlayer):
        ans = player.ask()
        player.add_memory(ans)
        self.speak(player.name, ans)


    def run(self):

        for round in range(self.max_round - 1):

            if self.mod_ans["debate_answer"] != '':
                break
            else:
                self.answers.append("")


                print(f"===== Debate Round-{round+2} =====\n")

                for player in self.players:
                    player.add_debate_prompt('\n'.join(self.answers[-(self.num_players-1):]))
                    answer = player.ask()
                    player.add_memory(answer)
                    self.answers.append(self.config['argument'].replace('##player##', player.name).replace('##answer##', answer))


                self.moderator.add_event(self.config['general_prompts']['moderator_prompt'].replace('##mod_info##', '\n'.join(self.answers[-(self.num_players):])).replace('##round##', self.round_dct(round+2)))
                self.mod_ans = self.moderator.ask()
                self.moderator.add_memory(self.mod_ans)
                self.mod_ans = eval(self.mod_ans)

        if self.mod_ans["debate_answer"] != '':
            self.config.update(self.mod_ans)
            self.config['success'] = True

        # ultimate deadly technique.
        else:
            print("judge time")
            judge_player = DebatePlayer(model_name=self.model_name, name='Judge', temperature=self.temperature, openai_api_key=self.openai_api_key, sleep_time=self.sleep_time)


            judge_player.set_meta_prompt(self.config['general_prompts']['moderator_meta_prompt'].replace('##debate_topic##', self.config["debate_topic"]))

            # extract answer candidates
            print("extract answer candidates")
            print(self.config['general_prompts']['judge_prompt_last1'].replace('##mod_info##', '\n'.join(self.answers[-(self.num_players):])))
            judge_player.add_event(self.config['general_prompts']['judge_prompt_last1'].replace('##mod_info##', '\n'.join(self.answers[-(self.num_players):])))
            ans = judge_player.ask()
            judge_player.add_memory(ans)

            # select one from the candidates
            judge_player.add_event(self.config['judge_prompt_last2'])
            ans = judge_player.ask()
            judge_player.add_memory(ans)
            
            ans = eval(ans)
            if ans["debate_answer"] != '':
                self.config['success'] = True
                # save file
            self.config.update(ans)
            self.players.append(judge_player)

        self.print_answer()


if __name__ == "__main__":

    current_script_path = os.path.abspath(__file__)
    MAD_path = current_script_path.rsplit("/", 1)[0]

    while True:
        debate_topic = ""
        while debate_topic == "":
            debate_topic = input(f"\nEnter your debate topic: ")
            
        config = json.load(open(f"./code/configs/config.json", "r"))
        config['debate_topic'] = debate_topic

        debate = Debate(num_players=4, openai_api_key=openai_api_key, config=config, temperature=0, sleep_time=0)
        debate.run()

