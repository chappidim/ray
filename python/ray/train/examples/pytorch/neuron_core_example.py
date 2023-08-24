import ray
from ray.train.backend import Backend, BackendConfig
from ray.train._internal.utils import get_address_and_port
from ray.train.torch.config import TorchConfig
from ray.train._internal.worker_group import WorkerGroup
from ray.train.torch import TorchTrainer

import os
import torch
import uuid

ray.init()


class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Linear(4, 4)
        self.nl1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(4, 2)
        self.nl2 = torch.nn.Tanh()

    def forward(self, x):
        x = self.nl1(self.layer1(x))
        return self.nl2(self.layer2(x))


def log(txt):
    rank = os.environ.get("RANK", "unknown")
    print(f"{rank}: {txt}", flush=True)


def train_func():
    import torch_xla.core.xla_model as xm

    log("before 1st rendezvous")
    xm.rendezvous("first")
    device = xm.xla_device()
    for c in range(10):
        ones = torch.ones((2, 3))
        xones = ones.to(device)
        result = xm.all_reduce("sum", xones)
        xm.mark_step()
        result_cpu = result.cpu()
        expected = torch.ones((2, 3)) * int(os.environ.get("WORLD_SIZE", 0))
        log(f"result: {c}: {result}  result.size(): {result.size()}")
        assert torch.all(result_cpu == expected), f"ERROR: {result_cpu} != {expected}"
    log("before final rendezvous")
    xm.rendezvous("last")
    log("done!")


class TorchXLAConfig(TorchConfig):
    """Configuration for torch xla process group setup."""

    @property
    def backend_cls(self):
        return TorchXLABackend


def kill_xrt_server():
    import subprocess

    subprocess.call(["pkill", "-f", "xrt_run_server"])


class TorchXLABackend(Backend):
    random_uuid: str = str(uuid.uuid4())

    def on_start(self, worker_group: WorkerGroup, backend_config: TorchXLAConfig):
        """Logic ran right before training is started."""
        worker_group.execute(kill_xrt_server)
        print(ray.available_resources())
        master_addr, master_port = worker_group.execute_single(0, get_address_and_port)
        print(master_addr)
        print(master_port)
        print(worker_group.num_workers)

        def set_env_vars(addr, port):
            os.environ["MASTER_ADDR"] = addr
            os.environ["MASTER_PORT"] = str(port)
            # To trigger the xrt server
            os.environ["TORCHELASTIC_RUN_ID"] = self.random_uuid
            print("Set Master Addr and Port")

        worker_group.execute(set_env_vars, addr=master_addr, port=master_port)
        set_env_vars(master_addr, master_port)

    def on_training_start(
        self, worker_group: WorkerGroup, backend_config: BackendConfig
    ):
        def setup_xla_torch_process_group():
            try:
                import torch_xla.core.xla_model as xm  # noqa
                import torch_xla.distributed.xla_backend  # noqa
                import torch.distributed as dist

                dist.init_process_group("xla")
            except ImportError:
                raise ImportError(
                    "torch_xla must be installed to use torch_xla backend."
                )

        def set_xla_env_vars():
            from ray.air import session

            os.environ["LOCAL_RANK"] = str(session.get_local_rank())
            os.environ["RANK"] = str(session.get_world_rank())
            os.environ["LOCAL_WORLD_SIZE"] = str(session.get_local_world_size())
            os.environ["WORLD_SIZE"] = str(session.get_world_size())
            # os.environ["NODE_RANK"] = str(session.get_node_rank())
            os.environ["GROUP_RANK"] = str(session.get_node_rank())
            os.environ["GROUP_WORLD_SIZE"] = str(
                session.get_world_size() / session.get_local_world_size()
            )
            os.environ["ROLE_RANK"] = str(session.get_world_rank())
            os.environ["ROLE_NAME"] = "default"
            os.environ["ROLE_WORLD_SIZE"] = str(session.get_world_size())
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
            # On disabling, we could reach the device(w=2, c=1) as XRT
            # server shares the cores. As alternative, we could share
            # the cores to all actors within node.
            # os.environ.pop("NEURON_RT_VISIBLE_CORES", None)

            # EFA
            os.environ["FI_PROVIDER"] = "efa"
            os.environ["FI_EFA_USE_DEVICE_RDMA"] = "1"
            os.environ["FI_EFA_FORK_SAFE"] = "1"

        worker_group.execute(set_xla_env_vars)
        worker_group.execute(setup_xla_torch_process_group)

    def on_shutdown(self, worker_group: WorkerGroup, backend_config: BackendConfig):
        worker_group.execute(kill_xrt_server)


train_dataset = ray.data.from_items([1, 2, 3])
assert train_dataset.count() == 3
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    torch_config=TorchXLAConfig(),
    scaling_config=ray.air.config.ScalingConfig(
        num_workers=2, resources_per_worker={"neuron_cores": 32}
    ),
    datasets={"train": train_dataset},
)
result = trainer.fit()
print(result)
