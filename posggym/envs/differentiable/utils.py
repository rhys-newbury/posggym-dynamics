from vmas.simulator.core import AgentState, World
import torch
from vmas.simulator.sensors import Lidar, Sensor
from abc import abstractmethod


def clone_tensors(obj: AgentState):
    cloned_attrs = {}
    for attr_name, attr_value in obj.__dict__.items():
        if isinstance(attr_value, torch.Tensor):
            cloned_attrs[attr_name] = attr_value.clone()
        else:
            cloned_attrs[attr_name] = attr_value
    return cloned_attrs


class AgentStateWrapper(AgentState):
    @property
    def pos_safe(self) -> torch.Tensor:
        if self._pos is None:
            raise AttributeError("pos is none")

        return self._pos

    @property
    def rot_safe(self) -> torch.Tensor:
        if self._rot is None:
            raise AttributeError("rot is none")

        return self._rot

    @property
    def vel_safe(self) -> torch.Tensor:
        if self._vel is None:
            raise AttributeError("vel is none")

        return self._vel

    @property
    def ang_vel_safe(self) -> torch.Tensor:
        if self._ang_vel is None:
            raise AttributeError("ang_vel is none")

        return self._ang_vel


def clone_state(state: AgentStateWrapper):
    a_s = AgentStateWrapper()
    t = clone_tensors(state)
    # import  pdb; pdb.set_trace()
    a_s._batch_dim = t["_batch_dim"]  # pyright: ignore
    a_s._device = t["_device"]  # pyright: ignore
    a_s.pos = t["_pos"]
    a_s.ang_vel = t["_ang_vel"]
    a_s.force = t["_force"]
    a_s.pos = t["_pos"]
    a_s.rot = t["_rot"]
    a_s.torque = t["_torque"]
    a_s.vel = t["_vel"]

    return a_s


class POSGGymSensor(Sensor):
    @abstractmethod
    def measure(self, world: World):
        raise NotImplementedError


class POSGGymLidar(Lidar, POSGGymSensor):
    def measure(self, world: World):
        dists = []
        assert self.agent is not None
        for angle in self._angles:
            dists.append(
                world.cast_ray(
                    self.agent,
                    angle,
                    max_range=self._max_range,
                    entity_filter=self.entity_filter,
                )
            )
        measurement = torch.stack(dists, dim=1)
        self._last_measurement = measurement.swapaxes(1, 0)
        return measurement
