# Third-party notices

## NDM Gym, BlinkMimic and InstantTrack

The WMBPTT speed options (`wiki/code-opt-ndm-port.md` in the parent repository)
port code from [NDM Gym](https://github.com/Frank-ZY-Dou/ndm-gym) @ `05ea9f3`,
which ports it from [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) @
`d264658`, itself derived from
[InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack). All are under the
MIT License below. Each ported file carries the notice.

| file here | source in NDM Gym |
|---|---|
| `rlopt/agent/wm_bptt/muon.py` | `ndm_gym/core/muon.py`, without its RigidFormer `matrices` argument |
| `rlopt/agent/wm_bptt/replay.py` (`set_terminal`, `rollout_mask`, `use_device_head`, `uniform_rows`) | the row's true `s1`, the window mask and the device write head (`use_device_ptr`, `_randint`) of `ReplayBuffer` in `ndm_gym/buffer.py` |
| `rlopt/agent/wm_bptt/speed_options.py` | adapted parts of `ndm_gym/algos/common.py`, `ndm_gym/algos/blinkmimic.py` (`Graphed`, as `GraphedUpdate`), `ndm_gym/core/normalizer.py`, `ndm_gym/train.py` (`update_graphs`) |

```
MIT License

Copyright (c) 2026 NDM Gym Contributors
Copyright (c) 2026 ETH Zurich, Chong Zhang
Copyright (c) 2026 Fuyi Chen
Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
