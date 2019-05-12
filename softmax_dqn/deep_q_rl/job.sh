#!/bin/bash

# running s-dqn on seaquest with tau=5
nohup ./run_nature.py --rom seaquest --temp 5.0 &> soft_dqn_nature_seaquest_temp5.log&

# running s-ddqn on seaquest with tau=5
nohup ./run_nature.py --rom seaquest --double-learning --temp 5.0 &> soft_ddqn_nature_seaquest_temp5.log&
