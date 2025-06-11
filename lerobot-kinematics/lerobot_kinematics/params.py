_params = {
    "unicode": True,
}


def set_param(param, value):
    _params[param] = value

def get_param(param):
    return _params[param]