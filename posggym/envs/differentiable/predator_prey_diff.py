from posggym.core import DefaultEnv
import posggym.model as M
from vmas.simulator.core import Agent, Landmark, Line, Sphere, World, Box
import torch
from vmas.simulator.utils import Color
from typing import List, Dict, Optional, Tuple, NamedTuple, cast
from vmas.simulator.utils import (
    TorchUtils,
    X,
    Y,
)
from posggym.utils import seeding
import numpy as np
from gymnasium import spaces
from ctypes import byref
import timeit
import math
import random
from utils import clone_state, POSGGymLidar, AgentStateWrapper, POSGGymSensor
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.dynamics.holonomic import Holonomic

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PPObs = torch.Tensor
PPAction = torch.Tensor


class PPState(NamedTuple):
    """A state in the Continuous Predator-Prey Environment."""

    predator_states: Dict[str, AgentStateWrapper]
    prey_states: Dict[str, AgentStateWrapper]
    prey_caught: List[bool]


class PPAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.caught = False

        self._state = AgentStateWrapper()
        self.rew = torch.Tensor()

    def set_caught(self, caught):
        self.caught = caught

    @property
    def state(self) -> AgentStateWrapper:
        return self._state

    @property
    def sensors(self) -> List[POSGGymSensor]:
        return self._sensors


class PPWorld(World):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agents: List[PPAgent] = []

    def add_agent(self, agent: PPAgent):
        super().add_agent(agent)

    def update_state(self, state: PPState):
        for a in self.agents:
            if a.name.startswith("adversary"):
                a_state = state.prey_states[a.name]
            else:
                a_state = state.predator_states[a.name]

            a.set_pos(a_state.pos_safe.clone(), 0)
            a.set_vel(a_state.vel_safe.clone(), 0)
            a.set_rot(a_state.rot_safe.clone(), 0)
            a.set_ang_vel(a_state.ang_vel_safe.clone(), 0)

    def get_state(self) -> PPState:
        return PPState(
            {x.name: clone_state(x.state) for x in self.predator},
            {x.name: clone_state(x.state) for x in self.prey},
            [x.caught for x in self.prey],
        )

    @property
    def agents(self) -> List[PPAgent]:
        return self._agents

    @property
    def prey(self) -> List[PPAgent]:
        return [x for x in self.agents if x.name.startswith("adversary")]

    @property
    def predator(self) -> List[PPAgent]:
        return [x for x in self.agents if x.name.startswith("agent")]


class PredatorPreyDiffModel(M.POSGModel[PPState, torch.Tensor, torch.Tensor]):
    R_MAX = 1

    def __init__(
        self,
        num_predators: int = 2,
        num_prey: int = 8,
        cooperative: bool = False,
        prey_strength: Optional[int] = None,
        obs_dist: float = 10,
        n_sensors: int = 8,
    ):
        assert 1 < num_predators <= 8
        assert num_prey > 0
        assert obs_dist > 0

        self.num_predators = num_predators
        self.num_prey = num_prey
        self.num_landmarks = 2
        self.num_agents = self.num_predators + self.num_predators
        self.cooperative = cooperative
        self.prey_strength = prey_strength
        self.obs_dist = obs_dist
        self.n_sensors = n_sensors
        self.prey_obs_dist = 1.0
        self.adversaries_share_rew = True
        self.shape_agent_rew = True
        self.shape_adversary_rew = True
        self.agents_share_rew = False
        self.prey_share_rew = True
        self.observe_same_team = True
        self.observe_pos = True
        self.observe_vel = True
        self.bound = 1.0
        self.respawn_at_catch = False
        self.per_prey_reward = self.R_MAX / self.num_prey
        self.prey_capture_dist = 0.1

        def _pos_space(n_agents: int):
            # x, y, angle, vx, vy, vangle
            # stacked n_agents time
            # shape = (n_agents, 6)
            size, angle = 5, 2 * math.pi
            low = np.array([-1, -1, -angle, -1, -1, -angle], dtype=np.float32)
            high = np.array(
                [size, size, angle, 1.0, 1.0, angle],
                dtype=np.float32,
            )
            return spaces.Box(
                low=np.tile(low, (n_agents, 1)), high=np.tile(high, (n_agents, 1))
            )

        self.state_space = spaces.Tuple(
            (
                # state of each predator
                _pos_space(self.num_predators),
                # state of each prey
                _pos_space(self.num_prey),
                # prey caught/not
                spaces.MultiBinary(self.num_prey),
            )
        )

        self.action_spaces = {
            i: spaces.Box(np.array([-1, -1]), np.array([1, 1]), seed=42)
            for i in self.possible_agents
        }

        self.observation_spaces = {
            i: spaces.Box(
                low=np.array([0.0] * self.n_sensors * 3),
                high=np.array([self.obs_dist] * self.n_sensors * 3),
            )
            for i in self.possible_agents
        }
        self.initialise()

    @property
    def reward_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {i: (0.0, self.R_MAX) for i in self.possible_agents}

    def get_agents(self, state: PPState) -> List[str]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> PPState:
        return self.reset_world_at()

    def initialise(self) -> PPState:
        self.world = PPWorld(
            batch_dim=1,
            device=torch.device("cuda"),
            x_semidim=self.bound,
            y_semidim=self.bound,
            substeps=10,
            collision_force=500,
        )

        # set any world properties first
        num_agents = self.num_predators + self.num_predators
        self.adversary_radius = 0.075

        # Add agents
        def gen_dynamics():
            return random.choice(
                (Holonomic(), DiffDrive(self.world, integration="rk4"))
            )

        for i in range(num_agents):
            adversary = i < self.num_predators
            name = f"adversary_{i}" if adversary else f"agent_{i - self.num_predators}"
            agent = PPAgent(
                name=name,
                collide=True,
                shape=Sphere(radius=self.adversary_radius if adversary else 0.05),
                u_multiplier=3.0 if adversary else 4.0,
                max_speed=1.0 if adversary else 1.3,
                color=Color.BLUE if adversary else Color.GREEN,
                adversary=adversary,
                dynamics=Holonomic() if adversary else gen_dynamics(),
                sensors=(
                    [
                        POSGGymLidar(
                            self.world,
                            n_rays=self.n_sensors,
                            max_range=self.obs_dist,
                            render_color=Color.GREEN,
                            entity_filter=lambda e: e.name.startswith("landmark"),
                        ),
                        POSGGymLidar(
                            self.world,
                            n_rays=self.n_sensors,
                            max_range=self.obs_dist,
                            render_color=Color.RED,
                            entity_filter=lambda e: e.name.startswith("adversary"),
                        ),
                        POSGGymLidar(
                            self.world,
                            n_rays=self.n_sensors,
                            max_range=self.obs_dist,
                            render_color=Color.BLUE,
                            entity_filter=lambda e: e.name.startswith("agent"),
                        ),
                    ]
                    if not adversary
                    else []
                ),
            )
            self.world.add_agent(agent)

        for i in range(4):
            self.world.add_landmark(
                Landmark(
                    name=f"landmark-wall{i}",
                    collide=True,
                    shape=Line(length=self.bound * 2),
                    color=Color.WHITE,
                )
            )
        # Add landmarks
        for i in range(self.num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=True,
                shape=Box(width=0.2, length=0.2),
                color=Color.BLACK,
            )
            self.world.add_landmark(landmark)

        all_wall_pos: List[Tuple[List[float], float]] = [
            ([0, self.bound], 0),
            ([0, -self.bound], 0),
            ([-self.bound, 0], np.pi / 2),
            ([self.bound, 0], np.pi / 2),
        ]

        for idx, (pos, rot) in enumerate(all_wall_pos):
            self.world.landmarks[idx].set_pos(
                torch.tensor(
                    pos,
                    device=self.world.device,
                ),
                batch_index=None,  # type: ignore
            )
            self.world.landmarks[idx].set_rot(
                torch.tensor(
                    [rot],
                    device=self.world.device,
                ),
                batch_index=None,  # type: ignore
            )

        for landmark in self.world.landmarks[4:]:
            landmark.set_pos(
                torch.zeros(
                    (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -(self.bound - 0.1),
                    self.bound - 0.1,
                ),
                batch_index=None,  # type: ignore
            )

        return self.reset_world_at()

    def reset_world_at(self) -> PPState:
        predator_states, prey_states, prey_caught = {}, {}, [False, False]
        for p in self.world.predator:
            state = AgentStateWrapper()
            state.batch_dim = self.world._batch_dim  # pyright: ignore
            state.device = self.world._device  # pyright: ignore

            state.pos = torch.zeros(
                (self.world.batch_dim, self.world.dim_p),
                device="cuda",
                dtype=torch.float32,
            ).uniform_(
                -self.bound,
                self.bound,
            )
            state.pos.requires_grad = True

            state.vel = torch.zeros(
                (self.world.batch_dim, self.world.dim_p),
                device="cuda",
                dtype=torch.float32,
                requires_grad=True,
            )
            state.rot = torch.zeros(
                self.world.batch_dim,
                1,
                device="cuda",
                dtype=torch.float32,
                requires_grad=True,
            )
            state.ang_vel = torch.zeros(
                self.world.batch_dim,
                1,
                device="cuda",
                dtype=torch.float32,
                requires_grad=True,
            )
            predator_states[p.name] = state

        for p in self.world.prey:
            state = AgentStateWrapper()
            state.batch_dim = self.world._batch_dim  # pyright: ignore
            state.device = self.world._device  # pyright: ignore

            state.pos = torch.zeros(
                (self.world.batch_dim, self.world.dim_p),
                device="cuda",
                dtype=torch.float32,
            ).uniform_(
                -self.bound,
                self.bound,
            )
            state.pos.requires_grad = True
            state.vel = torch.zeros(
                (self.world.batch_dim, self.world.dim_p),
                device="cuda",
                dtype=torch.float32,
                requires_grad=True,
            )
            state.rot = torch.zeros(
                self.world.batch_dim,
                1,
                device="cuda",
                dtype=torch.float32,
                requires_grad=True,
            )
            state.ang_vel = torch.zeros(
                self.world.batch_dim,
                1,
                device="cuda",
                dtype=torch.float32,
                requires_grad=True,
            )
            prey_states[p.name] = state

        return PPState(predator_states, prey_states, prey_caught)

    def is_collision(self, agent1: Agent, agent2: Agent):
        delta_pos = agent1.state.pos - agent2.state.pos  # type: ignore
        dist = torch.linalg.vector_norm(delta_pos, dim=-1)
        dist_min = agent1.shape.radius + agent2.shape.radius  # type: ignore
        return dist < dist_min

    # return all adversarial agents
    def prey(self, world: PPWorld):
        return [agent for agent in world.agents if agent.adversary]

    def _get_prey_move_angles(self, state: PPState) -> torch.Tensor:
        prey_actions = []
        active_prey = self.num_prey - sum(state.prey_caught)

        pred_states = torch.cat([x.pos_safe for x in state.predator_states.values()])
        prey_states = torch.cat([x.pos_safe for x in state.prey_states.values()])
        torch.tensor(state.prey_caught)

        pred_dists = torch.linalg.norm(prey_states - pred_states, axis=1)
        prey_dists = torch.linalg.norm(
            prey_states[:, None, :] - prey_states[None, :, :], dim=1
        )
        print("pred_dists: ", pred_dists, "prey_dists: ", prey_dists)

        for i, (prey, caught) in enumerate(
            zip(state.prey_states.values(), state.prey_caught)
        ):
            if caught:
                # prey stays in same position
                prey_actions.append(0.0)
                continue

            pred_dists = torch.linalg.norm(prey.pos - pred_states, axis=1)

            min_pred_dist, pred_idx = pred_dists.min(dim=0)
            if min_pred_dist <= self.prey_obs_dist or active_prey == 1:
                pred_state = pred_states[pred_idx]

                angle = torch.atan2(
                    prey.pos_safe[0, 1] - pred_state[1],
                    prey.pos_safe[0, 0] - pred_state[0],
                )
                prey_actions.append(angle)
                continue

            prey_positions = prey_states

            not_current_mask = torch.ones(len(state.prey_states), dtype=torch.bool)
            not_current_mask[i] = False

            # Compute distances
            prey_dists = torch.linalg.norm(
                prey_positions[not_current_mask] - prey_positions[i], dim=1
            )

            _, other_prey_idx = prey_dists.squeeze().min(dim=0)

            other_prey_state = np.array(list(state.prey_states.values()), dtype=object)[
                not_current_mask
            ][other_prey_idx]

            angle = torch.atan2(
                prey.pos_safe[0, 1] - other_prey_state.pos[0, 1],
                prey.pos_safe[0, 0] - other_prey_state.pos[0, 0],
            )
            prey_actions.append(angle)

        return torch.stack(prey_actions)

    def reward(self, agent: PPAgent):
        is_first = agent == self.world.predator[0]

        if is_first:
            # import pdb; pdb.set_trace()
            for a in self.world.predator:
                a.rew = self.agent_reward(a)

            self.agents_rew = torch.stack(
                [a.rew for a in self.world.predator], dim=-1
            ).sum(-1)

        if self.agents_share_rew:
            return self.agents_rew
        else:
            return agent.rew

    def agent_reward(self, agent: PPAgent):
        # Agents are negatively rewarded if caught by adversaries
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        adversaries = self.world.prey
        if self.shape_agent_rew:
            # reward can optionally be shaped
            # (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * torch.linalg.vector_norm(
                    agent.state.pos_safe - adv.state.pos, dim=-1
                )
        if agent.collide:
            for a in adversaries:
                # pass
                rew[self.is_collision(a, agent)] -= 10

        return rew

    def adversary_reward(self, agent: PPAgent):
        # Adversaries are rewarded for collisions with agents
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        agents = self.world.predator
        if self.shape_adversary_rew:  # reward can optionally be shaped
            # (decreased reward for increased distance from agents)
            rew -= (
                0.1
                * torch.min(
                    torch.stack(
                        [
                            torch.linalg.vector_norm(
                                a.state.pos_safe - agent.state.pos,
                                dim=-1,
                            )
                            for a in agents
                        ],
                        dim=-1,
                    ),
                    dim=-1,
                )[0]
            )
            print("adv rew: ", rew)
        if agent.collide:
            for ag in agents:
                rew[self.is_collision(ag, agent)] += 10
        return rew

    def observation(self, name: str):
        timeit.default_timer()
        world_agent = [x for x in self.world.agents if name == x.name][0]
        lidar_1_measures = torch.cat(
            tuple(s.measure(self.world) for s in world_agent.sensors)
        )
        timeit.default_timer()
        # print(t2 - t1)
        return lidar_1_measures

    def sample_initial_obs(self, state: PPState) -> Dict[str, torch.Tensor]:
        obs = {}
        for name, agent in state.predator_states.items():
            observation = TorchUtils.recursive_clone(self.observation(name))
            obs.update({name: observation})
        return obs

    def info(self, world: PPWorld, agent: Agent):
        return {}

    def done(self, world: PPWorld):
        agents: List[PPAgent] = world.agents  # type: ignore
        return torch.Tensor([x.caught for x in agents])

    def get_from_scenario(
        self,
    ):
        obs, rewards, infos, dones = {}, {}, {}, {}

        for agent in self.world.agents:
            if agent.name.startswith("agent"):
                observation = TorchUtils.recursive_clone(self.observation(agent.name))
                obs.update({agent.name: observation})

        # import pdb; pdb.set_trace()

        for agent in self.world.predator:
            reward = self.reward(agent).clone()
            rewards.update({agent.name: reward})

        for agent in self.world.agents:
            info = TorchUtils.recursive_clone(self.info(self.world, agent))
            infos.update({agent.name: info})

        dones = [False for _ in self.world.agents]
        truncated = [False for _ in self.world.agents]

        return [obs, rewards, dones, truncated, infos]

    @property
    def possible_agents(self):
        return tuple((str(x) for x in range(self.num_predators)))

    def render(self):
        pass

    def step(
        self, state: PPState, actions: Dict[str, torch.Tensor]
    ) -> M.JointTimestep[PPState, torch.Tensor]:
        self.world.update_state(state)

        prey_actions = self._get_prey_move_angles(state)
        prey_actions_ = [
            torch.stack([torch.cos(angle), torch.sin(angle)]).detach()
            for angle in prey_actions
        ]

        for idx, agent in enumerate(self.world.predator):
            if isinstance(actions[str(idx)], np.ndarray):
                action = torch.from_numpy(actions[str(idx)])
            else:
                action = actions[str(idx)]

            agent.action.u = torch.unsqueeze(action, dim=0)
            agent.dynamics.process_action()

        for act, agent in zip(prey_actions_, self.world.prey):
            agent.action.u = torch.unsqueeze(act, dim=0)
            agent.state.force = agent.action.u
        self.world.step()

        next_state = self.world.get_state()

        obs, rewards, terminated, truncated, infos = self.get_from_scenario()
        all_done = all(terminated or truncated)

        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, infos
        )

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random(seed=42)
        return self._rng


class PredatorPrey(DefaultEnv[PPState, PPObs, PPAction]):
    def __init__(self):
        model = PredatorPreyDiffModel()
        self.viewer = None
        self.visible_display = None
        self.device = "cuda"
        super().__init__(model)

    def render(
        self,
        mode="human",
        env_index=0,
        agent_index_focus: Optional[int] = None,
        visualize_when_rgb: bool = False,
    ):
        """
        Render function for environment using pyglet
        From VMAS
        """

        viewer_size = (700, 700)

        model = cast(PredatorPreyDiffModel, self.model)

        shared_viewer = agent_index_focus is None
        aspect_ratio = viewer_size[0] / viewer_size[1]

        headless = mode == "rgb_array" and not visualize_when_rgb
        # First time rendering
        if self.visible_display is None:
            self.visible_display = not headless
            self.headless = headless
        # All other times headless should be the same
        else:
            assert self.visible_display is not headless

        # First time rendering
        if self.viewer is None:
            try:
                import pyglet
            except ImportError:
                raise ImportError(
                    "Cannot import pyglet: you can install"
                    "pyglet directly via 'pip install pyglet'."
                )

            try:
                # Try to use EGL
                pyglet.lib.load_library("EGL")

                # Only if we have GPUs
                from pyglet.libs.egl import egl, eglext

                num_devices = egl.EGLint()
                eglext.eglQueryDevicesEXT(0, None, byref(num_devices))
                assert num_devices.value > 0

            except (ImportError, AssertionError):
                self.headless = False
            pyglet.options["headless"] = self.headless

            self._init_rendering()

        if 1.2 <= 0:
            raise ValueError("Scenario viewer zoom must be > 0")
        zoom = 1.2

        if aspect_ratio < 1:
            cam_range = torch.tensor([zoom, zoom / aspect_ratio], device=self.device)
        else:
            cam_range = torch.tensor([zoom * aspect_ratio, zoom], device=self.device)

        if shared_viewer:
            # zoom out to fit everyone
            all_poses = torch.stack(
                [agent.state.pos_safe[env_index] for agent in model.world.agents],
                dim=0,
            )
            max_agent_radius = max(
                [agent.shape.circumscribed_radius() for agent in model.world.agents]
            )
            viewer_size_fit = (
                torch.stack(
                    [
                        torch.max(torch.abs(all_poses[:, X] - 0)),
                        torch.max(torch.abs(all_poses[:, Y] - 0)),
                    ]
                )
                + 2 * max_agent_radius
            )

            viewer_size = torch.maximum(
                viewer_size_fit / cam_range,
                torch.tensor(zoom, device=self.device),
            )
            cam_range *= torch.max(viewer_size)
            assert self.viewer is not None

            self.viewer.set_bounds(
                -cam_range[X] + 0,
                cam_range[X] + 0,
                -cam_range[Y] + 0,
                cam_range[Y] + 0,
            )

        for entity in model.world.entities:
            assert self.viewer is not None
            self.viewer.add_onetime_list(entity.render(env_index=env_index))

        # render to display or array
        assert self.viewer is not None

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def _init_rendering(self):
        from vmas.simulator import rendering

        self.viewer = rendering.Viewer(
            *(700, 700), visible=self.visible_display or False
        )
        model = cast(PredatorPreyDiffModel, self.model)

        self.text_lines = []
        idx = 0
        if model.world.dim_c > 0:
            for agent in model.world.agents:
                if not agent.silent:
                    text_line = rendering.TextLine(y=idx * 40)
                    self.viewer.geoms.append(text_line)
                    self.text_lines.append(text_line)
                    idx += 1


if __name__ == "__main__":
    env = PredatorPrey()
    states = []

    for step in range(50):
        a = {i: torch.Tensor(env.action_spaces[i].sample()) for i in env.agents}
        for action in a.values():
            action.requires_grad_(True)

        print(a)
        if step == 0:
            first_action = a

        # shouldn't affect grads?
        state = env.model.sample_initial_state()
        state = env.model.step(state, a).state
        state = env.model.step(state, a).state
        state = env.model.step(state, a).state
        state = env.model.step(state, a).state

        obs, rews, dones, _, _, info = env.step(a)
        env.render()

    loss = obs["agent_0"].mean() + rews["agent_0"].mean()

    grad = torch.autograd.grad(loss, first_action["0"], allow_unused=True)
    print(env._state[0]["agent_0"]._pos)
    print(grad)
