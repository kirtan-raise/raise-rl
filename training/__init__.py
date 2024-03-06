from gymnasium.envs.registration import register

register(
    id="Ur20BracketInsert-V0",
    entry_point="training.env.bracket_insertion:Ur20BracketInsertEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)