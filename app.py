import numpy as np
from unityagents import UnityEnvironment

ENV_FILE = './Banana_Linux/Banana.x86_64'

# Create unity environment
env = UnityEnvironment(file_name=ENV_FILE)

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

print("Number of agents: ", len(env_info.agents))

action_size = brain.vector_action_space_size
print("Number of actions: ", action_size)

state = env_info.vector_observations[0]
print("State looks like: ", state)
state_size = len(state)
print("State size: ", state_size)

# Take random action
env_info = env.reset(train_mode=False)[brain_name]
state = env_info.vector_observations[0]
score = 0
while True:
    action = np.random.randint(action_size)
    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    score += reward
    state = next_state
    if done:
        break
print(f"Score: {score}")

# Close the environment
env.close()
