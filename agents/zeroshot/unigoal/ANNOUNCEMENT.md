This directory contains an adapted version of [UniGoal](https://github.com/bagh2178/UniGoal), a third-party Vision-Language Navigation algorithm originally designed for the Habitat simulator.
The original UniGoal codebase has been significantly modified to integrate into the ZSIBOT_VLN framework and to support real-robot and realistic simulation execution.

Major adaptations include:
* Decoupling from Habitat and re-integration into ZsiBot I/O
* Incorporating realistic robot body geometry and full 6-DoF poses instead of treating the robot as a point
* A simplified and streamlined mapping module
* A fallback strategy that recursively revisits previous goals when frontiers fall outside the local planning window
* General cleanups and structural adjustments

Note: UniGoal is included here solely as a third-party baseline to demonstrate that the ZsiBot_VLN framework can run VLN methods. We do not claim any performance guarantees, improvements over the original method, or official support for UniGoal itself.
