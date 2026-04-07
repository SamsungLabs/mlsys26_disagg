#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================

from setuptools import setup, find_packages

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="DisAgg",
    version="1.0.0",
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={'': 'src'},
    packages=find_packages("src"),
    # List all single modules in src here:
    py_modules=["disagg_test", "light_secagg_test", "opa_test",
                "secagg_plus_test", "secagg_test", "dataset.build_dataset"],
    python_requires=">=3.9, <3.11",
    install_requires=[
        "torch==2.4.0",
        "torchvision==0.19.0",
        "datasets==3.1.0",
        "evaluate==0.4.6",
        "peft==0.18.1",
        "scipy==1.15.3",
        "tqdm==4.67.1",
        "cryptography==43.0.3",
        "pycryptodomex==3.23.0",
        "numpy>=1.23.5,<2.0",
        "scikit_learn==1.7.2",
        "transformers==4.57.5",
        "timm==1.0.25",
        "networkx==3.4.2",
        "sympy==1.14.0",
        "pandas==2.3.3",
    ],
)
