# RLwithUnity

This project include some state-of-art or classic RL(reinforcement learning) algorithms used for training agents by interactive with Unity through [ml-agents](https://github.com/Unity-Technologies/ml-agents) v0.8.1.

The Algorithms in this repository are writed totally separated, 'cause I want each algorithm being different with others, what I mean is that I just wanna each algorithm has its own `.py` file and don't have to switch to another file to find out the implementation which one may confused. And those algorithms will never be encapsulated into a base algorithm model.

This framework implements training mechanism conversion between On-Policy and Off-Policy for Actor-Critic architecture algorithms. Just need to set the value of varibable `use_replay_buffer` in `config_file.py`(True for off-policy and False for on-policy).

You can just run each algorithm in this repository by `python simple_run.py`. I don't put any record function in it(like excel, mongo, logger, checkpoint, summary...). 

I am very appreciate to my best friend - [BlueFisher](https://github.com/BlueFisher) - who always look down on my coding style and competence(Although he **is** right, at least for now, but will be **was**).

Any questions about this project or errors about my bad grammer, plz let me know in [this](https://github.com/StepNeverStop/RLwithUnity/issues).