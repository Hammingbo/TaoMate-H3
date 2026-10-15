"""Stage-local Progressive PP transport; only the final stage publishes media.

Stages are independent TP/U process groups. Leaders transfer video activations
and two-frame clean tails; audio milestones are already identical at every
stage. No layer K/V is sent between machines.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import queue
import socket
import struct
import threading
import time


def _read(sock, count):
    result = bytearray(count)
    view = memoryview(result)
    offset = 0
    while offset < count:
        received = sock.recv_into(view[offset:])
        if not received:
            raise EOFError("Progressive stage peer closed")
        offset += received
    return result


class _Sender:
    def __init__(self, address, records):
        self.address, self.records = address, records
        self.jobs = queue.Queue()
        self.error = None
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def send(self, tensor, key):
        if self.error is not None:
            raise RuntimeError("Progressive stage send failed") from self.error
        # Copy owns its storage until the asynchronous socket send completes.
        self.jobs.put((tensor.detach().float().cpu().contiguous().numpy(), key, None, None))

    def send_ready_array(self, array, key, ready, on_complete=None):
        """Queue owned pinned host storage; wait for its CUDA copy off-thread."""
        if self.error is not None:
            raise RuntimeError("Progressive stage send failed") from self.error
        self.jobs.put((array, key, ready, on_complete))

    def _run(self):
        connection = None
        try:
            while True:
                job = self.jobs.get()
                if job is None:
                    break
                array, key, ready, on_complete = job
                if ready is not None:
                    ready.synchronize()
                started = time.time()
                if connection is None:
                    deadline = time.monotonic() + 300
                    while connection is None:
                        try:
                            connection = socket.create_connection(self.address, timeout=5)
                            connection.settimeout(300)
                            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        except OSError:
                            if time.monotonic() >= deadline:
                                raise
                            time.sleep(0.1)
                header = json.dumps({"shape": list(array.shape), "key": key}).encode()
                connection.sendall(struct.pack("!I", len(header)) + header)
                connection.sendall(memoryview(array).cast("B"))
                self.records.append({"kind": "send", "key": key, "start": started,
                                     "end": time.time(), "bytes": array.nbytes})
                if on_complete is not None:
                    on_complete()
        except BaseException as error:
            self.error = error
        finally:
            if connection is not None:
                connection.close()

    def close(self):
        if self.error is not None:
            raise RuntimeError("Progressive sender failed") from self.error
        self.jobs.put(None)
        self.worker.join(timeout=310)
        if self.worker.is_alive():
            raise TimeoutError("Progressive sender did not drain")
        if self.error is not None:
            raise RuntimeError("Progressive sender failed") from self.error


class _Receiver:
    def __init__(self, port):
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("0.0.0.0", port))
        self.listener.listen(1)
        self.listener.settimeout(300)
        self.connection = None

    def receive(self, key):
        import numpy as np
        import torch
        if self.connection is None:
            self.connection, _ = self.listener.accept()
            self.connection.settimeout(300)
        size = struct.unpack("!I", _read(self.connection, 4))[0]
        header = json.loads(_read(self.connection, size))
        if header["key"] != key:
            raise RuntimeError(f"Progressive packet order differs: {header['key']} != {key}")
        shape = tuple(header["shape"])
        return torch.from_numpy(np.frombuffer(_read(self.connection, int(np.prod(shape)) * 4),
                                              dtype=np.float32).reshape(shape))

    def close(self):
        if self.connection is not None:
            self.connection.close()
        self.listener.close()


class ProgressivePipelineStage:
    def __init__(self, *, cross_request_visual_handoff=True):
        import torch
        self.torch = torch
        self.cross_request_visual_handoff = cross_request_visual_handoff
        self.stage = int(os.environ["H3_PROGRESSIVE_PP_STAGE"])
        self.rank = torch.distributed.get_rank()
        self.device = torch.device("cuda", torch.cuda.current_device())
        hosts = os.environ["H3_PROGRESSIVE_PP_HOSTS"].split(",")
        port = int(os.environ.get("H3_PROGRESSIVE_PP_PORT", "29740"))
        self.records, self.cuda_records = [], []
        self.async_send = os.environ.get("H3_PROGRESSIVE_PP_ASYNC_SEND", "0") == "1"
        self.copy_records = []
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.async_send and self.rank == 0 else None
        self.senders, self.receivers = {}, {}
        if self.stage not in (0, 1, 2) or len(hosts) != 3:
            raise ValueError("Progressive PP needs three stage hosts and stage 0/1/2")
        if self.rank == 0:
            if self.stage > 0:
                self.receivers["input"] = _Receiver(port + self.stage)
            if self.stage < 2:
                self.receivers["tail"] = _Receiver(port + 10 + self.stage)
                self.senders["next"] = _Sender((hosts[self.stage + 1], port + self.stage + 1), self.records)
            else:
                for stage in (0, 1):
                    self.senders[f"tail{stage}"] = _Sender((hosts[stage], port + 10 + stage), self.records)
                vae_host = os.environ.get("H3_PROGRESSIVE_PP_VAE_HOST")
                if vae_host:
                    self.senders["vae"] = _Sender((vae_host, port + 30), self.records)
        from .pipeline_readiness import await_pipeline_ready
        self.input_ready_epoch = await_pipeline_ready(f'stage{self.stage}')
        self.anchor = torch.cuda.Event(enable_timing=True)
        self.anchor.record()
        self.anchor.synchronize()
        self.anchor_epoch = time.time()

    @contextmanager
    def compute(self, label, **metadata):
        start, end = (self.torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        try:
            yield
        finally:
            end.record()
            self.cuda_records.append((label, start, end, metadata))

    def receive(self, channel, request, phase, shape):
        started = time.time()
        key = [request, phase]
        if self.rank == 0:
            tensor = self.receivers[channel].receive(key)
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"Progressive {channel} shape differs: {tensor.shape} != {shape}")
            tensor = tensor.to(self.device)
        else:
            tensor = self.torch.empty(shape, dtype=self.torch.float32, device=self.device)
        self.torch.distributed.broadcast(tensor, src=0)
        self.records.append({"kind": "receive_and_broadcast", "channel": channel,
                             "key": key, "start": started, "end": time.time()})
        return tensor

    def send(self, channel, tensor, request, phase):
        if self.rank == 0:
            started = time.time()
            if self.async_send:
                # The denoiser can mutate its source on the next iteration.
                # Snapshot on its current stream, then copy owned storage on
                # the transfer stream without blocking Python on CUDA.
                owned = tensor.detach().float().clone(memory_format=self.torch.contiguous_format)
                host = self.torch.empty(owned.shape, dtype=owned.dtype, device="cpu", pin_memory=True)
                self.copy_stream.wait_stream(self.torch.cuda.current_stream(self.device))
                begin, ready = (self.torch.cuda.Event(enable_timing=True) for _ in range(2))
                with self.torch.cuda.stream(self.copy_stream):
                    begin.record()
                    host.copy_(owned, non_blocking=True)
                    ready.record()
                owned.record_stream(self.copy_stream)
                self.senders[channel].send_ready_array(host.numpy(), [request, phase], ready)
                self.copy_records.append((channel, begin, ready, [request, phase], host.numel() * host.element_size()))
            else:
                self.senders[channel].send(tensor, [request, phase])
            self.records.append({"kind": "d2h_enqueue", "channel": channel,
                                 "key": [request, phase], "start": started, "end": time.time(),
                                 "async_copy": self.async_send})

    def feedback(self, video, *, request, phase, last_phase, needs_next_tail, frame_rows):
        if needs_next_tail or (phase == last_phase and self.cross_request_visual_handoff):
            for stage in (0, 1):
                self.send(f"tail{stage}", video[-2 * frame_rows:], request, phase)

    def publish_phase(self, video, audio, request, phase):
        if os.environ.get("H3_PROGRESSIVE_PP_VAE_HOST"):
            self.send("vae", video, request, [phase, "video"])
            self.send("vae", audio, request, [phase, "audio"])

    def finish_low_phase(self, *, phase, request, last_phase, needs_next_tail,
                         frame_rows, teacher_audio):
        # Unowned high rows are only shape placeholders inside a stage worker.
        # They are never decoded/published; real tail rows drive native handoff.
        video = self.torch.zeros((phase.video_latent_count * frame_rows, 96),
                                 device=self.device, dtype=self.torch.float32)
        if needs_next_tail or (phase.index == last_phase and self.cross_request_visual_handoff):
            tail = self.receive("tail", request, phase.index, (2 * frame_rows, 96))
            video[-2 * frame_rows:].copy_(tail)
        return video, teacher_audio

    def finish(self, directory):
        for sender in self.senders.values():
            sender.close()
        self.torch.cuda.synchronize(self.device)
        for label, start, end, metadata in self.cuda_records:
            self.records.append({"kind": "cuda_compute", "label": label,
                                 "start": self.anchor_epoch + self.anchor.elapsed_time(start) / 1000,
                                 "end": self.anchor_epoch + self.anchor.elapsed_time(end) / 1000,
                                 **metadata})
        for channel, start, end, key, size in self.copy_records:
            self.records.append({"kind": "cuda_transfer", "label": "d2h_async",
                                 "channel": channel, "key": key, "bytes": size,
                                 "start": self.anchor_epoch + self.anchor.elapsed_time(start) / 1000,
                                 "end": self.anchor_epoch + self.anchor.elapsed_time(end) / 1000})
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"stage{self.stage}_rank{self.rank}.json").write_text(json.dumps({
            "stage": self.stage, "rank": self.rank, "world_size": self.torch.distributed.get_world_size(),
            "input_ready_epoch": self.input_ready_epoch,
            "events": self.records, "cuda_clock": "device elapsed mapped to host epoch at synchronized anchor",
        }, indent=2))
        for receiver in self.receivers.values():
            receiver.close()
