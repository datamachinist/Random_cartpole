import gym
# https://github.com/openai/gym/wiki/Leaderboard

"""
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 
The system is controlled by applying a force of +1 or -1 to the cart. 
The pendulum starts upright, and the goal is to prevent it from falling over. 
A reward of +1 is provided for every timestep that the pole remains upright. 
The episode ends when the pole is more than 15 degrees from vertical, 
or the cart moves more than 2.4 units from the center.
"""

env = gym.make('CartPole-v0')
print(env.action_space)       # there are 2 possible actions: push to the left (0) or to the right (1)
print(env.observation_space)  # there are 4 possible states: [Position Velocity Angle Velocity_at_tip]

# show the observation's bound
print(env.observation_space.high)
print(env.observation_space.low)
# position: min = -4.8, max = 4.8
# velocity: min = -3e38, max = 3e38
# angle: min = -0.4, max = +0.4
# velocity at tip: min = -3e38, max = 3e38

def policy():
    action = env.action_space.sample()  # take a random action: either 0 or 1
    return action


for i_episode in range(20):      # 5 episodes
    state = env.reset()
    rewards = []
    
    for t in range(100):        # 100 timesteps
        env.render()
        state, reward, done, info = env.step(policy())  # implement the action and get the observation and reward
        rewards.append(reward)
        
        if done:
            cumulative_reward = sum(rewards)
            print("episode {} finished after {} timesteps. Total reward: {}".format(i_episode, t+1, cumulative_reward))  # either the pole is > 15 deg from vertical or the cart move by > 2.4 unit from the centre
            break
    
env.close()
