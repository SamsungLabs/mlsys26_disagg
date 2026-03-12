#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np


class ParameterType(Enum):
    WEIGHTS = "weights"
    DELTAS = "deltas"
    GRADIENTS = "gradients"
    METRICS = "metrics"
    VALUES = "values"


# pylint: disable=invalid-name
npArray = np.ndarray[Any, np.dtype[Any]]


SignalType = Union[
    npArray,
    Sequence[Union[npArray, int, float, bool]],
    Sequence[Sequence[Union[int, float, bool]]],
    int,
    float,
    bool,
    Mapping[Any, Union[npArray, int, float, bool]],
    "Parameters",
]

ParamShape = Mapping[str, Sequence[int]]
npArrayDict = Dict[str, npArray]
npArrays = List[npArray]


@dataclass
class Parameters:
    tensors: npArrayDict
    param_type: ParameterType = ParameterType.VALUES

    def __init__(
        self,
        data: Optional[SignalType] = None,
        param_type: ParameterType = ParameterType.VALUES,
    ):
        self.tensors = _convert_to_tensors(data)
        self.param_type = param_type

    def __post_init__(self):
        for key, value in self.tensors.items():
            if not isinstance(key, str):
                raise ValueError("The keys must be strings")
            if not isinstance(value, np.ndarray):
                raise ValueError("The values must be numpy arrays")

    @classmethod
    def map(cls, func: Callable[..., npArray], *parameters: "Parameters") -> "Parameters":
        parameters = list(parameters)
        for idx, parameter in enumerate(parameters):
            if not isinstance(parameter, Parameters):
                parameters[idx] = Parameters(parameter)
            cls.assert_same_structure(parameters[0], parameter)

        composed_func_args: Dict[str, npArrays] = {}
        for key in parameters[0]:
            composed_func_args[key] = [parameter[key] for parameter in parameters]

        return Parameters(
            {
                key: func(*composed_params)
                for key, composed_params in composed_func_args.items()
            },
            parameters[0].param_type,
        )

    @classmethod
    def apply(
        cls, func: Callable[[npArrays], npArray], parameters: Sequence["Parameters"]
    ) -> "Parameters":
        if not isinstance(parameters, Sequence):
            raise ValueError("The parameters must be a type of Sequence")
        for parameter in parameters:
            if not isinstance(parameter, Parameters):
                raise ValueError("The parameters must be of type Parameters")
            cls.assert_same_structure(parameters[0], parameter)

        composed_func_args: Dict[str, npArrays] = {}
        for key in parameters[0]:
            composed_func_args[key] = [parameter[key] for parameter in parameters]

        return Parameters(
            {
                key: func(composed_params)
                for key, composed_params in composed_func_args.items()
            },
            parameters[0].param_type,
        )

    @classmethod
    def assert_same_structure(
        cls, root_params: "Parameters", checking_params: "Parameters"
    ):
        for key in root_params.sorted_keys:
            if key not in checking_params.sorted_keys:
                raise ValueError("The parameters have different structures")

        if root_params.param_type != checking_params.param_type:
            raise ValueError("The parameters are of different types")

    @classmethod
    def zeros(
        cls, shape: ParamShape, param_type: ParameterType = ParameterType.VALUES
    ) -> "Parameters":
        zero_tensors = {}
        for key in shape.keys():
            zero_tensors[key] = np.zeros(shape[key])
        return Parameters(zero_tensors, param_type)

    @classmethod
    def from_flat_array(
        cls, data: npArray, shape: ParamShape, param_type: ParameterType = ParameterType.VALUES
    ) -> "Parameters":
        tensors = {}
        start_index = 0
        for key in shape.keys():
            chunk_size = 1
            for s in shape[key]:
                chunk_size = chunk_size * s
            chunk = data[start_index : start_index + chunk_size].reshape(shape[key])
            tensors[key] = chunk
            start_index = start_index + chunk_size
        return Parameters(tensors, param_type)


    @classmethod
    def convert_to(
        cls, data: "Parameters", output_type: Optional[SignalType]
    ) -> SignalType:
        if (
            isinstance(output_type, (int, float, bool))
            or output_type is None
            or np.isscalar(data)
        ):
            return _convert_to_scalar(data)
        elif isinstance(output_type, np.ndarray):
            return _convert_to_numpy(data)
        elif isinstance(output_type, Parameters):
            return _convert_to_parameters(data)
        elif isinstance(output_type, dict):
            return _convert_to_dict(data, output_type)
        elif isinstance(output_type, Sequence):
            return _convert_to_sequence(data, output_type)
        else:
            raise ValueError(f"Data type {type(data)} not supported")

    @property
    def sorted_keys(self) -> List[str]:
        return sorted(self.tensors.keys())

    @property
    def shape(self) -> ParamShape:
        return {key: self.tensors[key].shape for key in self.sorted_keys}

    @property
    def flat(self) -> npArray:
        return np.concatenate([self.tensors[key].flatten() for key in self.sorted_keys])

    def tolist(self) -> npArrays:
        """Returns the tensors as a list of numpy arrays."""
        return [self.tensors[key] for key in self.sorted_keys]

    def update_all_tensors(self, tensors: npArrays):
        """Updates all the tensors with the given list of numpy arrays."""
        if len(tensors) != len(self.sorted_keys):
            raise ValueError("The number of tensors does not match the number of keys")
        self.tensors = {key: tensor for key, tensor in zip(self.sorted_keys, tensors)}

    def items(self):
        return self.tensors.items()

    def __getitem__(self, key: Union[str, int]) -> npArray:
        if isinstance(key, int):
            return self.tensors[self.sorted_keys[key]]

        if key not in self.tensors:
            raise KeyError(f"Key '{key}' not found in the parameters")
        return self.tensors[key]

    def __setitem__(self, key: str, value: npArray):
        self.tensors[key] = value

    def __mul__(self, multiplier: SignalType) -> "Parameters":
        multiplier = Parameters(multiplier)

        if len(multiplier) == 1:
            return Parameters(
                {key: tensor * multiplier[0] for key, tensor in self.tensors.items()},
                self.param_type,
            )
        if len(multiplier) != len(self):
            raise ValueError("The number of tensors does not match the number of keys")

        return Parameters(
            {key: tensor * multiplier[key] for key, tensor in self.tensors.items()},
            self.param_type,
        )

    def __truediv__(self, scalar: Union[float, int]) -> "Parameters":
        if not isinstance(scalar, (float, int)):
            raise ValueError("The scalar parameter must be a float or int")
        return Parameters(
            {key: tensor / scalar for key, tensor in self.tensors.items()},
            self.param_type,
        )

    def __iadd__(self, tensors: "Parameters") -> "Parameters":
        if set(tensors.sorted_keys).difference(set(self.sorted_keys)):
            raise ValueError(
                "The layer names in the tensors and model weights are different"
            )

        for key in tensors.sorted_keys:
            self.tensors[key] += tensors[key]

        return self

    def __add__(self, tensors: Union[int, float, "Parameters"]) -> "Parameters":
        if isinstance(tensors, Parameters):
            if set(tensors.sorted_keys).difference(set(self.sorted_keys)):
                raise ValueError(
                    "The layer names in the tensors and model weights are different"
                )

            new_sum = {}
            for key in tensors.sorted_keys:
                new_sum[key] = self.tensors[key] + tensors[key]
        else:
            new_sum = {}
            for key in self.sorted_keys:
                new_sum[key] = self.tensors[key] + tensors
        return Parameters(new_sum, self.param_type)

    def __sub__(self, tensors: Union[int, float, "Parameters"]) -> "Parameters":
        if isinstance(tensors, Parameters):
            if set(tensors.sorted_keys).difference(set(self.sorted_keys)):
                raise ValueError(
                    "The layer names in the tensors and model weights are different"
                )

            new_sum: npArrayDict = {}
            for key in tensors.sorted_keys:
                new_sum[key] = self.tensors[key] - tensors[key]
        else:
            new_sum = {}
            for key in self.sorted_keys:
                new_sum[key] = self.tensors[key] - tensors

        return Parameters(new_sum, self.param_type)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Parameters):
            other = Parameters(other)
        for tensor_name in self.sorted_keys:
            if not np.array_equal(self[tensor_name], other[tensor_name]):
                return False
        return True

    def __iter__(self):
        return iter(self.sorted_keys)

    def __len__(self):
        return len(self.tensors)


def _convert_to_tensors(data: Optional[SignalType]) -> npArrayDict:
    if isinstance(data, (int, float, bool)) or np.isscalar(data):
        return _convert_from_scalar(data)
    elif isinstance(data, Parameters):
        return _convert_from_parameters(data)
    elif isinstance(data, dict):
        return _convert_from_dict(data)
    elif isinstance(data, Sequence):
        return _convert_from_sequence(data)
    elif data is None:
        return _convert_from_none(data)
    elif isinstance(data, np.ndarray):
        return {"0": data}
    else:
        raise ValueError(f"Data type {type(data)} not supported")


def _convert_from_scalar(data: Union[int, float, bool, npArray]) -> npArrayDict:
    "Not real scalar, but a single element."
    return {"": np.array(data).reshape((1,))}


def _convert_from_dict(
    data: Dict[str, Union[Sequence[Union[int, float, bool]], int, float, bool, npArray]]
) -> npArrayDict:
    casted_data: Dict[str, npArray] = {}
    for key, value in data.items():
        if isinstance(value, (int, float, bool, Sequence)) or np.isscalar(value):
            casted_data[key] = np.array(value).reshape((-1,))
        elif isinstance(value, np.ndarray):
            casted_data[key] = value
        else:
            raise ValueError(f"Data type {type(value)} not supported")

    return casted_data


def _convert_from_sequence(data: Sequence[Union[npArray, int, float]]) -> npArrayDict:
    np_dict: npArrayDict = {}
    for i, value in enumerate(data):
        np_dict[f"{i:03d}"] = np.array(value)
    return np_dict


def _convert_from_none(_: None) -> npArrayDict:
    return {"": np.empty((1,))}


def _convert_from_parameters(data: Parameters) -> npArrayDict:
    return data.tensors


def _convert_to_scalar(data: Parameters) -> Union[int, float, bool]:
    return type(data[0][0])(data[0][0])


def _convert_to_numpy(data: Parameters) -> npArray:
    return data[0]


def _convert_to_dict(
    data: Parameters, output_data: Dict[str, Union[npArray, int, float]]
) -> Dict[str, Union[npArray, int, float]]:
    casted_data: Dict[str, Union[npArray, int, float]] = {}
    for key, value in data.items():
        if isinstance(output_data[key], (int, float)):
            casted_data[key] = float(value[0])
        else:
            casted_data[key] = value
    return casted_data


def _convert_to_sequence(
    data: Parameters, output_type: Sequence[Union[npArray, int, float]]
) -> Sequence[Union[npArray, int, float]]:
    values: List[Union[npArray, int, float]] = []
    for i, value in enumerate(output_type):
        if isinstance(value, (int, float)):
            values.append(float(data[i][0]))
        else:
            values.append(data[i])
    return type(output_type)(values)  # type: ignore


def _convert_to_parameters(data: Parameters) -> Parameters:
    return data
