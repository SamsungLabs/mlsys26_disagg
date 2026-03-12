#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
import pandas as pd
import numpy as np
from tqdm import tqdm

out_dir_root = 'outputs'
graph_format = 'png'


def run(d, N, slow_frac_max, s1_max, s2_max, srv_bw, show_print=False):

    # conditions:
    # total bw = n1*s1 + n2*s2 <= srv_bw
    # s1 <= s1_max
    # s2 <= s2_max
    
    assert s2_max <= s1_max, "s2_max must be <= s1_max" 

    # Old way: average bw per slowest client
    s_srv_max = srv_bw / N
    s = min(s2_max, s_srv_max)
    t_av = d / s
    if show_print:
        print('Time for average slow speed:', t_av)
        print('Average slow speed:', s)

    # 1st group initial selection
    slow_frac_step = 0.05
    slow_frac_range = np.arange(0.05, slow_frac_max, slow_frac_step)    # legal range for 1st group

    # s1_step = 500
    # s1_range = range(s1_step, s1_max + s1_step, s1_step)    # legal range for 1st group

    tt = {'total_time':[], 'speed_upper_phase_1':[], 'speed_upper_phase_2':[], 'speed_lower_phase_1':[], 'speed_lower_phase_2':[], 
          'time_phase_1':[], 'time_phase_2':[], 'slow_frac': [], 'num_fast_clients': [], 'num_slow_clients': []}

    # sample 1st group and calc 2nd
    for sf in tqdm(slow_frac_range, desc="Finding best s1 for slow_frac={sf:.2f}...", disable=True):
        # 1st and 2nd group sizes
        n1 = int((1-sf)*N)
        n2 = int(sf*N)
        
        # we need to make sure we also cover scenario where both server_1 and server_2 have avg speeds
        # and the scenario for minimum phase 1 speeds at 10 kBps

        s1_step = 500 #500
        s1_range = [10] + [s] + list(range(s1_step, s1_max + s1_step, s1_step))    # legal range for 1st group
        s1_range = list(reversed(s1_range)) # s1 is evaluated from fast to slow
        
        s2_step = 50 #500
        s2_range = [10] + [s] + list(range(s2_step, s2_max + s2_step, s2_step))    # legal range for 1st group
        
        t_best = np.iinfo('int32').max
        t1_best = None
        t2_best = None
        s1_1_best = None
        s1_2_best = None
        s2_1_best = None
        s2_2_best = None

        for s2_1 in s2_range:   # loop over possible speeds for slow clients in phase 1
            for s1_1 in s1_range:
                # test available bw according to server
                if s1_1*n1 + s2_1*n2 > srv_bw:
                    # skip this combination as server can't support it
                    continue
                
                if s1_1 == s2_1:
                    # both will finish at the same time
                    t1 = d / s1_1
                    t2 = 0
                    s1_2 = 0
                    s2_2 = 0

                elif s1_1 > s2_1:
                    # n1 (fast clients) finish first, then n2 can use full server bandwidth
                    t1 = d / s1_1
                    
                    # data processed for each slow client in phase 1
                    d_2_1 = s2_1 * t1
                    
                    # remaining data for each slow client after phase 1
                    d_2_2 = d - d_2_1

                    # phase 2 speed for fast clients (0 = done)
                    s1_2 = 0
                    
                    # phase 2 speed for slow clients
                    s2_2 = min(s2_max, srv_bw / n2)
                    
                    # time to process remaining data for each slow client in phase 2
                    t2 = d_2_2 / s2_2

                else:
                    # n2 (slow clients) finish first, then n1 can use full server bandwidth
                    t1 = d / s2_1
                    
                    # data processed for each fast client in phase 1
                    d_1_1 = s1_1 * t1
                    
                    # remaining data for each fast client after phase 1
                    d_1_2 = d - d_1_1
                    
                    # phase 2 speed for fast clients
                    s1_2 = min(s1_max, srv_bw / n1)

                    # phase 2 speed for slow clients (0 = done)
                    s2_2 = 0
                    
                    # time to process remaining data for each fast client in phase 2
                    t2 = d_1_2 / s1_2
                
                # total time
                t = t1 + t2

                if t < t_best:
                    if show_print:
                        print("old_t_best:", t_best, "new_t_best:", t)
                    t_best = t
                    t1_best = t1
                    t2_best = t2
                    s1_1_best = s1_1
                    s1_2_best = s1_2
                    s2_1_best = s2_1
                    s2_2_best = s2_2
        
        tt['total_time'].append(t_best)           # total time
        tt['speed_upper_phase_1'].append(s1_1_best)         # 1st group speed (1st phase)
        tt['speed_upper_phase_2'].append(s1_2_best)         # 1st group speed (2nd phase)
        tt['speed_lower_phase_1'].append(s2_1_best)     # 2nd group speed (1st phase)
        tt['speed_lower_phase_2'].append(s2_2_best)     # 2nd group speed (2nd phase)
        tt['time_phase_1'].append(t1_best)         # 1st phase time
        tt['time_phase_2'].append(t2_best)         # 2nd phase time
        tt['slow_frac'].append(sf)
        tt['num_fast_clients'].append(n1)
        tt['num_slow_clients'].append(n2)

    return tt


if __name__ == '__main__':

    d = 10_000              # data to transfer in KB, either download-upload
    N = 10_000                # number of clients or aggregators
    slow_frac_max = 1.0         # the fraction of slow-speed clients
    s1_max = 20_000         # the 1st group fast speed limit (KB/s), (down or up)
    s2_max = 2_000         # the 2nd group slow speed limit (KB/s), (down or up)
    srv_bw = 25_000_000     # the server bandwidth (KB/s)
    
    print("Running heterogeneity test with parameters:\n")
    print(f"d={d}, N={N}, slow_frac_max={slow_frac_max}, s1_max={s1_max}, s2_max={s2_max}, srv_bw={srv_bw}")

    tt = run(d, N, slow_frac_max, s1_max, s2_max, srv_bw, show_print=False)

    df = pd.DataFrame.from_dict(tt)
    print(df)
