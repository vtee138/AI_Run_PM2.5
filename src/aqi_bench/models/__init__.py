from .baselines import Persistence, SeasonalNaive, RidgeDirect
from .trees import LightGBMDirect, XGBoostDirect
from .deep import DeepMultiHorizon

REGISTRY = {
    "persistence": Persistence,
    "snaive": SeasonalNaive,
    "ridge": RidgeDirect,
    "lgbm": LightGBMDirect,
    "xgb": XGBoostDirect,
    "lstm": lambda **kw: DeepMultiHorizon(arch="lstm", **kw),
    "gru": lambda **kw: DeepMultiHorizon(arch="gru", **kw),
    "tcn": lambda **kw: DeepMultiHorizon(arch="tcn", **kw),
    "transformer": lambda **kw: DeepMultiHorizon(arch="transformer", **kw),
}
TABULAR = {"persistence", "snaive", "ridge", "lgbm", "xgb"}
DEEP = {"lstm", "gru", "tcn", "transformer"}
STOCHASTIC = {"lgbm", "xgb"} | DEEP
