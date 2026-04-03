from openenv.core.env_server import create_app
from models import AdSpendAction, AdObservation
from server.environment import AdBiddingEnv

_global_env_instance = AdBiddingEnv()

def get_ad_environment():
    """Always returns the same instance so state is preserved."""
    return _global_env_instance

# Pass the factory function to OpenEnv
app = create_app(
    get_ad_environment,
    AdSpendAction,
    AdObservation,
    env_name="ad_bidding_env"
)