# RLwithUnity

This project include some state-of-art or classic RL(reinforcement learning) algorithms used for training agents by interactive with Unity through [ml-agents](https://github.com/Unity-Technologies/ml-agents) v0.6.0.

Although the newest version of this plugin(ml-agents) is now v0.7.0, I think there is little difference between v0.6.0 and v0.7.0, Maybe when v0.8.0 is released, I will update this plugin in this project.

The Algorithms in this repository are writer totally separated, 'cause I want each algorithm being different with others, what I mean is that I just wanna each algorithm has its own `.py` file and don't have to switch to another file to find out the implementation which one may confused.

This framework implements training mechanism conversion between On-Policy and Off-Policy for Actor-Critic architecture algorithms. Just need to set the value of varibale `use_replay_buffer` in `config_file.py`.

I am very appreciate to my best friend - [BlueFisher](https://github.com/BlueFisher) - who always look down on my coding style and competence(Although he **is** right, at least for now, but will be **was**).

Any questions about this project or errors about my bad grammer, plz let me know in [this](https://github.com/StepNeverStop/RLwithUnity/issues).