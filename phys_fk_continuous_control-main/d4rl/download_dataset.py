import gym
import d4rl
import os

datasets = [
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "walker2d-medium-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-expert-v2"
]

# Ensure the directory exists
os.makedirs(os.path.expanduser("./datasets"), exist_ok=True)

for env_name in datasets:
    print(f"--- Processing {env_name} ---")
    try:
        env = gym.make(env_name)
        # This call triggers the download if the file is missing or truncated
        dataset = env.get_dataset()
        print(f"Successfully cached {env_name}\n")
    except Exception as e:
        print(f"Failed to download {env_name}: {e}\n")

print("All downloads complete. You can now safely launch parallel jobs.")