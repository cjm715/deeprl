from deeprl.vpg import vpg, PolicyNet
import gym

env = gym.make('CartPole-v0')

mean_return_list, trained_policy, _ = vpg(env, num_iter=100)

plt.plot(mean_return_list)
plt.savefig('vpg_returns.png', format='png', dpi=300)

state = env.reset()
for t in range(1000):
    action, _ = trained_policy.draw_action(state)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break
env.close()
