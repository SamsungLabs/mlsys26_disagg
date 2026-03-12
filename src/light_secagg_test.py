#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
import disagg_test

if __name__ == '__main__':
    disagg_test.USE_LIGHT_SEC_AGG = True
    disagg_test.Client = disagg_test.LightSecAggClient
    disagg_test.Server = disagg_test.LightSecAggServer
    disagg_test.init_parameters['name'] = 'LightSecAgg'
    disagg_test.common.run_with_temp_folder(disagg_test.init_parameters, 
                                            disagg_test.run_simulation)
