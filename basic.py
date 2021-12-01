import gym
env = gym.make('BipedalWalker-v3')
for episode in range(100):
    observation = env.reset()  # initialise the environment for each episode

    # now loop and render the environment
    for i in range(10000):
        env.render()

    # sample random actions from the environment's action space
    action = env.action_space.sample()

    # for each action step, record observation, reward, done and info
    observation, reward, done, info = env.step(action)

    if done:
        print("{} timesteps taken for the Episode".format(i+1))
        break
