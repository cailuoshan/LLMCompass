from enum import Enum, auto
from math import ceil


class TopologyType(Enum):
    RING = auto()
    FC = auto()


class LinkModule:
    def __init__(
        self,
        bandwidth_per_direction: float,  # B/s
        bandwidth_both_direction: float,  # B/s
        latency: float,  # s
        flit_size: int,  # B
        max_payload_size: int,  # B
        header_size: int,  # B
    ) -> None:
        self.bandwidth_per_direction = bandwidth_per_direction
        self.bandwidth_both_direction = bandwidth_both_direction
        self.latency = latency
        self.flit_size = flit_size
        self.max_payload_size = max_payload_size
        self.header_size = ceil(header_size / flit_size) * flit_size


link_module_dict = {
    "NVLinkV3": LinkModule(25e9, 50e9, 8.92e-6, 16, 256, 16),
    "TPUv3Link": LinkModule(81.25e9 / 2, 81.25e9, 150e-6, 16, 256, 16),
    "XSNoC": LinkModule(
        bandwidth_per_direction=128e9,   # 64 B/cycle @ 2GHz = 128GB/s
        bandwidth_both_direction=256e9,  # 全双工
        latency=12e-9,                   # ~3 cycle per hop, ~8 hops per req = 24 cycles = 12 ns
        flit_size=32,
        max_payload_size=64,
        header_size=16
    ),
}
# we cannot find a way to measure TPU p2p latency, we also don't know TPU packet format


class InterConnectModule:
    def __init__(
        self,
        device_count: int,
        topology,
        link_module: LinkModule,
        link_count_per_device: int,
        internal_link_bandwidth_per_direction: float = float("inf"),
    ) -> None:
        self.device_count = device_count
        self.topology = topology
        self.link_module = link_module
        self.link_count_per_device = link_count_per_device
        self.internal_link_bandwidth_per_direction = (
            internal_link_bandwidth_per_direction
        )
        pass


# we treat the 2D torus interconnect of 8 TPU cores as 2 rings + internal link
interconnect_module_dict = {
    "NVLinkV3_FC_4": InterConnectModule(
        4, TopologyType.FC, link_module_dict["NVLinkV3"], 12
    ),
    "TPUv3Link_8": InterConnectModule(
        4, TopologyType.RING, link_module_dict["TPUv3Link"], 2, 162.5e9
    ),
    # 8 核 CPU 片上 NoC，拓扑先当作 FC（每核都可以和任意核高带宽通信），TODO: 后续可以换成 RING 或者自己扩展一个 MESH 类型
    "XSNoC_8": InterConnectModule(
        8, TopologyType.FC, link_module_dict["XSNoC"], 
        link_count_per_device=1,  # 一核视为一条“总出口带宽”的逻辑 link
        internal_link_bandwidth_per_direction=512e9,  # L2/LLC → NoC → mem 的等效路径带宽
    ),
}

# class InterConnectTorusModule:
#     def __init__(self) -> None:
#         pass


# class InterConnectUniRingModule:
#     def __init__(
#         self,
#         device_count,
#         link: LinkModule,
#     ) -> None:
#         pass


# class InterConnectMeshModule:
#     def __init__(self) -> None:
#         pass


# class InterConnectFCModule:
#     def __init__(self) -> None:
#         pass
