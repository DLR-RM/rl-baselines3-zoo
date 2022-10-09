from sbx import SAC, TQC, DroQ

import rl_zoo3
import rl_zoo3.enjoy
from rl_zoo3.enjoy import enjoy

rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["droq"] = DroQ
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.enjoy.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

if __name__ == "__main__":
    enjoy()
