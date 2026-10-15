"""Request-ready FL2VA teacher transport and first-stage deferred consumption."""
from collections.abc import Mapping
from pathlib import Path
import json
import os
import socket
import struct
import time

from .base10_hiar import (
    CompletedBase10TeacherArtifact, base10_rollover_teacher_producer_topology,
    build_base10_hiar_schedule,
)


def teacher_port(port):
    return port + int(os.environ.get('H3_INFER_PORT_OFFSET', '0'))


class LiveProgressiveTeacher(CompletedBase10TeacherArtifact):
    live = True

    @classmethod
    def for_case(cls, root, *, case_id, prompts, seeds, geometry):
        schedule = build_base10_hiar_schedule(tuple(prompts), tuple(seeds))
        # This is the launch contract, not a synthetic completion marker.
        return cls(root=Path(root), case_id=case_id, schedule=schedule,
                   completion={
                       'schema': 'digitalhuman_h3.inference.base10_hiar_teacher_set.v1',
                       'strategy': 'previous_clean_audio_tail_reference_then_new_noise',
                       'audio_noise_seed_sequence': [8301+i for i in range(len(prompts))],
                   }, producer_topology=base10_rollover_teacher_producer_topology(),
                   producer_geometry=dict(geometry))

    def load_request(self, torch, *, request_index, device):
        return StreamingTeacherRequest(self, torch, request_index, device)


class LiveTeacherPublisher:
    def __init__(self, artifact):
        from digitalhuman_h3.streaming.progressive_pipeline import _Sender
        self.artifact = artifact
        self.events = []
        hosts = os.environ['H3_PROGRESSIVE_TEACHER_TCP_HOSTS'].split(',')
        self.senders = [_Sender((host, teacher_port(30040)+i), self.events) for i,host in enumerate(hosts)]
        audio_host = os.environ.get('H3_PROGRESSIVE_TEACHER_AUDIO_VAE_HOST')
        self.audio_sender = _Sender((audio_host, teacher_port(30058)), self.events) if audio_host else None

    def publish_milestone(self, request_index, milestone_index, rows):
        for sender in self.senders:
            sender.send(rows, {'request_index':request_index, 'milestone':milestone_index})
        if milestone_index == 2 and self.audio_sender is not None:
            self.audio_sender.send(rows, request_index)

    def publish(self, torch, request_index):
        payload = self.artifact.load_request(torch, request_index=request_index, device='cpu')
        rows = torch.stack([payload['milestones'][i] for i in range(3)])
        for sender in self.senders:
            sender.send(rows, {'request_index':request_index, 'receipt':payload['receipt']})

    def close(self):
        for sender in self.senders:
            sender.close()
        if self.audio_sender is not None:
            self.audio_sender.close()


class StreamingTeacherRequest(Mapping):
    """Consume exact Base states independently; completion is bookkeeping only."""
    streaming_milestones = True

    def __init__(self, artifact, torch, request_index, device):
        self.artifact, self.torch = artifact, torch
        self.request_index, self.device = request_index, device
        self.rows, self.receipt, self.active_fields = {}, None, {}
        self.milestones = StreamingMilestones(self)

    def active(self, *, official_audio_latents, active_audio_latents,
               transport_prefix_audio_latents):
        if active_audio_latents != self.artifact.schedule.requests[self.request_index].audio_latent_count:
            raise ValueError('Live teacher active audio geometry differs')
        self.active_fields = dict(
            official_audio_latents_per_channel=official_audio_latents,
            active_audio_latents_per_channel=active_audio_latents,
            transport_prefix_audio_latents_per_channel=transport_prefix_audio_latents,
            active_audio_slice_per_channel=[0, active_audio_latents],
        )
        return self

    def receive(self):
        from digitalhuman_h3.streaming.progressive_pipeline import _read
        import numpy as np
        torch, artifact = self.torch, self.artifact
        started = time.time()
        rank = torch.distributed.get_rank()
        headers = [None]
        if rank == 0:
            receiver = getattr(artifact, '_receiver', None)
            if receiver is None:
                with socket.socket() as listener:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    listener.bind(('0.0.0.0', teacher_port(30040)+int(os.environ['H3_PROGRESSIVE_PP_STAGE'])))
                    listener.listen(1)
                    listener.settimeout(300)
                    receiver, _ = listener.accept()
                receiver.settimeout(300)
                object.__setattr__(artifact, '_receiver', receiver)
            size = struct.unpack('!I', _read(receiver, 4))[0]
            header = json.loads(_read(receiver, size))
            shape = tuple(header['shape'])
            array = np.frombuffer(_read(receiver, int(np.prod(shape))*4), dtype=np.float32).copy().reshape(shape)
            rows = torch.from_numpy(array).to(self.device)
            headers[0] = header
        torch.distributed.broadcast_object_list(headers, src=0)
        header = headers[0]
        key = header['key']
        if key['request_index'] != self.request_index:
            raise ValueError('Live teacher packet request differs')
        milestone = key.get('milestone')
        expected = (2*artifact.schedule.requests[self.request_index].audio_latent_count, 32)
        if tuple(header['shape']) != (expected if milestone is not None else (3, *expected)):
            raise ValueError('Live teacher packet shape differs')
        if rank != 0:
            rows = torch.empty(header['shape'], device=self.device, dtype=torch.float32)
        torch.distributed.broadcast(rows, src=0)
        if milestone is not None:
            self.rows[milestone] = rows
        else:
            self.receipt = key['receipt']
        events = getattr(artifact, '_timing_events', None)
        if events is None:
            events = []
            object.__setattr__(artifact, '_timing_events', events)
        events.append(dict(request=self.request_index, milestone=milestone, start=started,
                           end=time.time(), kind='teacher_milestone_receive_and_broadcast'))
        if self.receipt is not None and self.request_index == len(artifact.schedule.requests)-1:
            path = os.environ.get('H3_PIPELINE_WARMUP_DIR')
            if path:
                stage = os.environ['H3_PROGRESSIVE_PP_STAGE']
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path)/f'teacher_wait_stage{stage}_rank{rank}.json').write_text(json.dumps(events, indent=2))

    def __getitem__(self, key):
        if key == 'milestones':
            return self.milestones
        if key != 'receipt':
            raise KeyError(key)
        while self.receipt is None:
            self.receive()
        return {**self.receipt, **self.active_fields}

    def __iter__(self):
        return iter(('milestones', 'receipt'))

    def __len__(self):
        return 2


class StreamingMilestones(Mapping):
    def __init__(self, teacher):
        self.teacher = teacher

    def __getitem__(self, key):
        if key not in (0, 1, 2):
            raise KeyError(key)
        while key not in self.teacher.rows:
            self.teacher.receive()
        return self.teacher.rows[key]

    def __iter__(self):
        return iter((0, 1, 2))

    def __len__(self):
        return 3


def wait_teacher_before_vae(torch, group=None):
    """Wait on CPU until the teacher yields the shared decoder GPUs."""
    if os.environ.get('H3_PROGRESSIVE_TEACHER_PRIORITY') != '1':
        return
    started = time.time()
    if torch.distributed.get_rank() == 0:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', teacher_port(30049)))
            listener.listen(1)
            listener.settimeout(300)
            peer, _ = listener.accept()
            with peer:
                peer.recv(1)
    torch.distributed.barrier(group=group)
    if torch.distributed.get_rank() == 0:
        print(json.dumps({'status':'teacher_gpu_handoff_to_vae',
                          'wait_seconds':time.time()-started}), flush=True)


def release_teacher_gpus_to_vae():
    if os.environ.get('H3_PROGRESSIVE_TEACHER_PRIORITY') != '1':
        return
    deadline = time.monotonic()+300
    while True:
        try:
            peer = socket.create_connection(('127.0.0.1', teacher_port(30049)), timeout=2)
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.05)
    with peer:
        peer.sendall(b'G')


_teacher_cpu_handoff_group = None


def prepare_teacher_cpu_handoff(torch):
    global _teacher_cpu_handoff_group
    if _teacher_cpu_handoff_group is None:
        _teacher_cpu_handoff_group = torch.distributed.new_group(backend='gloo')
    return _teacher_cpu_handoff_group


def teacher_wait_vae_credit(torch):
    """VAE grants a whole-GPU wave boundary; no polling or GPU busy wait."""
    started = time.time()
    if torch.distributed.get_rank() == 0:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', teacher_port(30048)))
            listener.listen(1)
            listener.settimeout(300)
            peer, _ = listener.accept()
            with peer:
                peer.recv(1)
    # NCCL would keep seven GPUs spinning while rank0 waits on the CPU socket,
    # stealing compute from the VAE processes that own the handoff interval.
    torch.distributed.barrier(group=prepare_teacher_cpu_handoff(torch))
    if torch.distributed.get_rank() == 0:
        print(json.dumps(dict(status='teacher_vae_credit', wait_seconds=time.time()-started)), flush=True)


def signal_teacher_credit(torch):
    if os.environ.get('H3_PROGRESSIVE_LIVE_TEACHER') != '1':
        return
    if torch.distributed.get_rank() == 0:
        deadline = time.monotonic()+300
        while True:
            try:
                peer = socket.create_connection(('127.0.0.1', teacher_port(30048)), timeout=2)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.05)
        with peer:
            peer.sendall(b'G')


def vae_grant_teacher_credit(torch, group):
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    signal_teacher_credit(torch)
    wait_teacher_before_vae(torch, group)


class DeferredTeacher(Mapping):
    """The first denoise needs noise only; consume milestones in its callback."""
    def __init__(self, resolve):
        self.resolve = resolve
        self.value = None
        self.milestones = DeferredMilestones(self)

    def materialize(self):
        if self.value is None:
            self.value = self.resolve()
        return self.value

    def __getitem__(self, key):
        if key == 'milestones':
            return self.milestones
        return self.materialize()[key]

    def __iter__(self):
        return iter(('milestones', 'receipt'))

    def __len__(self):
        return 2


class DeferredMilestones(Mapping):
    def __init__(self, teacher):
        self.teacher = teacher

    def __getitem__(self, key):
        return self.teacher.materialize()['milestones'][key]

    def __iter__(self):
        return iter((0, 1, 2))

    def __len__(self):
        return 3
