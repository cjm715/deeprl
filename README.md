# Deep Reinforcement Learning Algorithms

## Introduction

This repository is a small library for simple implementations of the following deep reinforcement learning algorithms written in pytorch.

This was built mainly for learning purposes. It may also help others who are learning deep RL and looking for simple implementations. I am following [OpenAI's Spinning Up](https://spinningup.openai.com/en/latest/index.html) documentation for guidance on algorithm details. I also used the [Udacity's Deep Reinforcement Learning Nanodegree Repository](https://github.com/udacity/deep-reinforcement-learning) as a reference.

Here are the following algorithms implemented so far:
- Vanilla Policy Gradient (VPG)

<!-- - Trust Region Policy Optimization (TRPO)
- Proximal Policy Optimization (PPO)
- Deep Deterministic Policy Gradient (DDPG)
- Twin Delayed DDPG 
- Soft Actor-Critic -->

## Getting Started

Make sure to have to have the following packages installed:
- [OpenAI gym](https://github.com/openai/gym)
- matplotlib
- numpy
- [pytorch](https://github.com/pytorch/pytorch)

Download the repository by clicking the green button on this page. 

## Instructions

To see a demonstration of any algorithm run the command `python [algo]_demo.py` from root of this repository. This will train an agent, save output files, and render the environment under control by the trained agent.