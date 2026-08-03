"""hal/shm_backend.py — 실HW 백엔드. libbipedshm.so(shm_bridge) 를 ctypes 로 로드.

★Pi(Emb) 전용. hal/build_bridge.sh 로 .so 를 먼저 빌드하고 config shm.lib 경로 지정.
데스크톱에선 로드 실패(라이브러리 없음) → app 이 MockBackend 로 폴백(MOCK=1).
"""
from __future__ import annotations
import ctypes as C
import numpy as np
from backend import Backend, RawState

f32p = C.POINTER(C.c_float)
i32p = C.POINTER(C.c_int)


def _p(arr):
    return arr.ctypes.data_as(f32p) if arr is not None else None


def _ip(arr):
    return arr.ctypes.data_as(i32p) if arr is not None else None


class ShmBackend(Backend):
    def __init__(self, lib_path: str, n_channel: int, recv_wait_ms: int = 2000):
        super().__init__(n_channel)
        self.recv_wait_ms = int(recv_wait_ms)
        self.lib = C.CDLL(lib_path)          # 실패 시 OSError → 호출부가 처리
        sig = self.lib
        sig.bridge_n_channel.restype = C.c_int
        sig.bridge_init.restype = C.c_int; sig.bridge_init.argtypes = [C.c_int]
        sig.bridge_read.restype = C.c_int; sig.bridge_read.argtypes = [f32p] * 7 + [i32p, i32p]
        sig.bridge_write_pos.restype = C.c_int; sig.bridge_write_pos.argtypes = [f32p, f32p, f32p, C.c_int]
        sig.bridge_write_mit.restype = C.c_int
        sig.bridge_write_mit.argtypes = [f32p, f32p, f32p, f32p, f32p, C.c_int]
        sig.bridge_enable.restype = C.c_int; sig.bridge_enable.argtypes = [C.c_int]
        # 브리지 채널 수와 config 일치 확인
        nb = int(sig.bridge_n_channel())
        if nb != n_channel:
            print(f"[ShmBackend] ⚠ 브리지 채널 {nb} ≠ config {n_channel} → 브리지값 사용")
            self.n_channel = nb
        n = self.n_channel
        self._q = np.zeros(n, np.float32); self._dq = np.zeros(n, np.float32)
        self._tau = np.zeros(n, np.float32); self._cur = np.zeros(n, np.float32)
        self._rpy = np.zeros(3, np.float32); self._acc = np.zeros(3, np.float32)
        self._gyro = np.zeros(3, np.float32); self._conn = np.zeros(n, np.int32)
        self._stat = np.zeros(n, np.int32)

    def init(self) -> None:
        if self.lib.bridge_init(self.recv_wait_ms) != 0:
            raise RuntimeError("bridge_init 실패 — 임베디드 모터 컨트롤러 미기동(모터 상태 미수신)")
        print(f"[ShmBackend] init OK · n_channel={self.n_channel}")

    def read(self) -> RawState:
        m = self.lib.bridge_read(_p(self._q), _p(self._dq), _p(self._tau), _p(self._cur),
                                 _p(self._rpy), _p(self._acc), _p(self._gyro),
                                 _ip(self._conn), _ip(self._stat))
        return RawState(q_deg=self._q.astype(float), dq_dps=self._dq.astype(float),
                        tau_nm=self._tau.astype(float), imu_rpy_deg=self._rpy.astype(float),
                        imu_acc=self._acc.astype(float), imu_gyro=self._gyro.astype(float),
                        connected=self._conn.astype(bool), status=self._stat.astype(int),
                        updated=int(m) if m >= 0 else 0)

    def write_pos(self, q_des_deg, kp, kd) -> None:
        q = np.asarray(q_des_deg, np.float32); kp = np.asarray(kp, np.float32); kd = np.asarray(kd, np.float32)
        self.lib.bridge_write_pos(_p(q), _p(kp), _p(kd), len(q))

    def write_mit(self, q_des_deg, dq_des_dps, tau_ff_nm, kp, kd) -> None:
        a = [np.asarray(x, np.float32) for x in (q_des_deg, dq_des_dps, tau_ff_nm, kp, kd)]
        self.lib.bridge_write_mit(_p(a[0]), _p(a[1]), _p(a[2]), _p(a[3]), _p(a[4]), len(a[0]))

    def enable(self, on: bool) -> None:
        self.lib.bridge_enable(1 if on else 0)
