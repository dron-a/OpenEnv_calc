# server/app.py
import os
from openenv.core.env_server import create_fastapi_app, create_app
from models import AdPlatformAction, AdPlatformObservation
from server.environment import AdPlatformEnvironment
# env = CalcEnvironment()
# app = create_fastapi_app(env, CalcAction, CalcObservation)
task_name = os.getenv("TASK")

def create_ad_platform_environment():
    """Always returns the same instance so state is preserved."""
    return AdPlatformEnvironment(task = task_name)

# Pass the factory function
app = create_app(
    create_ad_platform_environment, 
    AdPlatformAction, 
    AdPlatformObservation,
    env_name="adops_env"
)
# app = create_fastapi_app(CalcEnvironment, CalcAction, CalcObservation)