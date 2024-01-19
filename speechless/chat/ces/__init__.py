from .common import Manager
from .manager.sub_proc import SubProcessManager


def code_execution_service_factory(env_dir: str) -> Manager:
    return SubProcessManager(env_dir=env_dir)
