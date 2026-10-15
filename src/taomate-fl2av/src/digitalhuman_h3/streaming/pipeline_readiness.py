"""CPU rendezvous with a common post-warmup model-input timestamp."""
import json
import os
import socket
import struct
import time


def await_pipeline_ready(role):
    if os.environ.get("H3_PROGRESSIVE_PP_SYNC_START") != "1":
        return time.time()
    import torch

    group = torch.distributed.new_group(backend="gloo")
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    released = [None]
    if torch.distributed.get_rank() == 0:
        started = time.time()
        host = os.environ["H3_PROGRESSIVE_PP_READY_HOST"]
        port = int(os.environ["H3_PROGRESSIVE_PP_READY_PORT"])
        if role == "stage2":
            peer_count = 4 if os.environ.get("H3_PROGRESSIVE_LIVE_TEACHER") == "1" else 3
            with socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("0.0.0.0", port))
                listener.listen(peer_count)
                listener.settimeout(300)
                peers = []
                try:
                    for _ in range(peer_count):
                        peer, _ = listener.accept()
                        peers.append(peer)
                    released[0] = time.time()
                    payload = struct.pack("!d", released[0])
                    for peer in peers:
                        peer.sendall(payload)
                finally:
                    for peer in peers:
                        peer.close()
        else:
            deadline = time.monotonic() + 300
            while True:
                try:
                    peer = socket.create_connection((host, port), timeout=2)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.05)
            with peer:
                peer.settimeout(300)
                payload = bytearray()
                while len(payload) < 8:
                    part = peer.recv(8 - len(payload))
                    if not part:
                        raise RuntimeError("Progressive coordinator exited before start")
                    payload.extend(part)
                released[0] = struct.unpack("!d", payload)[0]
        print(json.dumps(dict(
            status="pipeline_loading_ready", role=role, transport="tcp",
            ready_at=started, released_at=released[0],
            loading_wait_seconds=released[0] - started,
        )), flush=True)
    torch.distributed.broadcast_object_list(released, src=0, group=group)
    torch.distributed.destroy_process_group(group)
    return released[0]
