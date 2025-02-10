import io
import os
import pickle
import socket

from mpi4py import MPI
import torch as torch
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = torch.cuda.device_count()

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    comm = MPI.COMM_WORLD
    backend = "gloo" if not torch.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")

    torch.cuda.set_device(MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE)

def is_master():
    return MPI.COMM_WORLD.Get_rank() == 0

def get_rank():
    return MPI.COMM_WORLD.Get_rank()

def get_local_rank():
    return MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_local_rank()}")
    return torch.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return torch.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

def broadcast(data = None):
    """
    Broadcast an object from rank 0 to all other ranks.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        data = pickle.dumps(data)
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return pickle.loads(data)

def get_hostname():
    return socket.gethostbyname(socket.getfqdn())