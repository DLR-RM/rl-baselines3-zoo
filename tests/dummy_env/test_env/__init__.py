from gymnasium.envs.registration import register

register(
    id="TestEnv-v0",
    entry_point="test_env.test_env:TestEnv",
)
