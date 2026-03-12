#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================

from collections import defaultdict
from typing import Dict, Union, cast
from uuid import UUID

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    ParameterFormat,
    PrivateFormat,
    PublicFormat,
    load_pem_parameters,
    load_pem_private_key,
    load_pem_public_key,
)


class DiffieHellman:
    def __init__(
        self,
        parameters: Union[bytes, dh.DHParameters],
    ):
        if not isinstance(parameters, bytes):
            self.parameters = serialize_dh(parameters)
        else:
            self.parameters = parameters
        self.private_keys: Dict[str, dh.DHPrivateKey] = {}
        self.public_keys: Dict[str, dh.DHPublicKey] = {}
        self.shared_keys: Dict[str, Dict[UUID, bytes]] = defaultdict(dict)
        self.derived_keys: Dict[str, Dict[UUID, bytes]] = defaultdict(dict)

    def generate_private_key(self, name: str) -> dh.DHPrivateKey:
        self.private_keys[name] = deserialize_dh(self.parameters).generate_private_key()
        return self.private_keys[name]

    def generate_public_key(self, name: str) -> dh.DHPublicKey:
        if name not in self.private_keys:
            raise ValueError(f"Private key is not generated yet for {name}.")
        self.public_keys[name] = self.private_keys[name].public_key()
        return self.public_keys[name]

    def generate_shared_key(
        self, peer_public_key: dh.DHPublicKey, name: str, client_id: UUID
    ) -> bytes:
        if name not in self.private_keys:
            raise ValueError("Private key is not generated yet.")
        self.shared_keys[name][client_id] = self.private_keys[name].exchange(
            peer_public_key
        )
        return self.shared_keys[name][client_id]

    def derive_keys(self, name: str, client_id: UUID) -> bytes:
        """This function derives a key from the shared key using HKDF.
        At the moment, we are using HKDF, but other choices are possible.
        Therefore the selection of the key derivation function will have to be dynamic
        in the future.

        For most applications the shared_key should be passed to a key derivation function.
        This allows mixing of additional information into the key, derivation of multiple keys,
        and destroys any structure that may be present.
        https://cryptography.io/en/latest/hazmat/primitives/asymmetric/dh/"""

        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"handshake data",
        ).derive(self.shared_keys[name][client_id])
        self.derived_keys[name][client_id] = derived_key
        return derived_key

    def _reset_keys(self):
        self.private_keys = {}
        self.public_keys = {}
        self.shared_keys = {}

    def get_public_key(self, name: str) -> dh.DHPublicKey:
        return self.public_keys[name]

    def get_private_key(self, name: str) -> dh.DHPrivateKey:
        return self.private_keys[name]

    def get_shared_key(self, name: str, client_id: UUID) -> bytes:
        return self.shared_keys[name][client_id]

    def get_all_shared_keys(self, name: str) -> Dict[UUID, bytes]:
        return self.shared_keys[name]

    def get_derived_key(self, name: str, client_id: UUID) -> bytes:
        return self.derived_keys[name][client_id]

    def get_all_derived_keys(self, name: str) -> Dict[UUID, bytes]:
        return self.derived_keys[name]

    def serialize(self):
        self.private_keys = {
            name: serialize_dh(private_key)
            for name, private_key in self.private_keys.items()
        }
        self.public_keys = {
            name: serialize_dh(public_key)
            for name, public_key in self.public_keys.items()
        }

    def deserialize(self):
        self.private_keys = {
            name: deserialize_private_key(private_key)
            for name, private_key in self.private_keys.items()
        }
        self.public_keys = {
            name: deserialize_public_key(public_key)
            for name, public_key in self.public_keys.items()
        }


def serialize_dh(
    dh_params: Union[dh.DHParameters, dh.DHPrivateKey, dh.DHPublicKey]
) -> bytes:
    if isinstance(dh_params, bytes):
        return dh_params
    elif isinstance(dh_params, dh.DHParameters):
        return dh_params.parameter_bytes(Encoding.PEM, ParameterFormat.PKCS3)
    elif isinstance(dh_params, dh.DHPrivateKey):
        return dh_params.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    elif isinstance(dh_params, dh.DHPublicKey):
        return dh_params.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)


def deserialize_public_key(dh_params: bytes) -> dh.DHPublicKey:
    return cast(dh.DHPublicKey, load_pem_public_key(dh_params))


def deserialize_dh(dh_params: bytes) -> dh.DHParameters:
    return load_pem_parameters(dh_params)


def deserialize_private_key(dh_params: bytes) -> dh.DHPrivateKey:
    return cast(dh.DHPrivateKey, load_pem_private_key(dh_params, None))
