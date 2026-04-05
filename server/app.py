# server/app.py
from openenv.core.env_server import create_fastapi_app, create_app
from models import BudgetAction, BudgetObservation
from tasks.task1_budget import BudgetEnvironment
# env = CalcEnvironment()
# app = create_fastapi_app(env, CalcAction, CalcObservation)
_global_env_instance = BudgetEnvironment()

def get_calc_environment():
    """Always returns the same instance so state is preserved."""
    return _global_env_instance

# Pass the factory function
app = create_app(
    get_calc_environment, 
    BudgetAction, 
    BudgetObservation,
    env_name="adops_env"
)
# app = create_fastapi_app(CalcEnvironment, CalcAction, CalcObservation)