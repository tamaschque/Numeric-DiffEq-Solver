import os
import json
import numpy as np

class ComplexEncoder(json.JSONEncoder):
    """Encoder for JSON to serialize complex numbers."""
    def default(self, obj):
        if isinstance(obj, (complex, np.complexfloating)):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        return super().default(obj)


def complex_hook(d):
    """Decoder for reading encoded complex numbers from the cache."""
    if "__complex__" in d:
        return complex(d["real"], d["imag"])
    return d


def ensure_json_ext(filename):
    ext = os.path.splitext(filename)[-1].lower()
    if ext == "":
        return filename + ".json"
    elif ext == ".json":
        return filename
    else:
        raise ValueError(f"Invalid filepath. Expected filetype \"json\", got \"{ext}\".")


def cache(filename: str, t_values=None, y_values=None, **options):
    filename = ensure_json_ext(filename)

    t_values = t_values.tolist() if isinstance(t_values, np.ndarray) else t_values
    y_values = y_values.tolist() if isinstance(y_values, np.ndarray) else y_values

    data = {
        "options": options,
        "t_values": t_values,
        "y_values": y_values
    }

    with open(filename, "w") as file:
        json.dump(data, file, cls=ComplexEncoder)


def load_chached_result(filename: str, *options: str):
    filename = ensure_json_ext(filename)

    with open(filename, "r") as file:
        data = json.load(file, object_hook=complex_hook)

    t_values = np.array(data["t_values"])
    y_values = np.array(data["y_values"])
    cache_opt = data["options"]

    return_opt = {opt: cache_opt[opt] for opt in options}

    if return_opt:
        return t_values, y_values, return_opt
    else:
        return t_values, y_values


def matching_cache_options(filename: str, **params):
    filename = ensure_json_ext(filename)

    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return False

    with open(filename, "r") as file:
        try:
            data = json.load(file, object_hook=complex_hook)
        except json.decoder.JSONDecodeError:
            return False

        options = data["options"]

        for key, value in params.items():
            if key in options.keys():
                if value != options[key]:
                    return False

            else:
                return False

    return True
