#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
import secagg_plus_test

if __name__ == '__main__':
    secagg_plus_test.USE_SECAGG_PLUS = False
    secagg_plus_test.init_parameters['name'] = 'SecAgg'
    secagg_plus_test.common.run_with_temp_folder(secagg_plus_test.init_parameters, 
                                            secagg_plus_test.run_simulation)
