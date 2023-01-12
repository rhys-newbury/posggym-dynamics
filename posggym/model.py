"""The model data structure."""
from __future__ import annotations

import abc
import enum
import dataclasses
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    List,
    Optional,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)
import random

import numpy as np
from gymnasium import spaces

from posggym import error
from posggym.utils import seeding


if TYPE_CHECKING:
    from posggym.envs.registration import EnvSpec


AgentID = Union[int, str]
StateType = TypeVar("StateType")
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class Outcome(enum.Enum):
    """Final POSG episode Outcome for an agent."""

    LOSS = -1
    DRAW = 0
    WIN = 1
    NA = None

    def __str__(self):
        return self.name


@dataclasses.dataclass(order=True)
class JointTimestep(Generic[StateType, ObsType]):
    """Stores values returned by model after a single step.

    Supports iteration.

    A dataclass is used instead of a Namedtuple so that generic typing is seamlessly
    supported.

    """
    state: StateType
    observations: Dict[AgentID, ObsType]
    rewards: Dict[AgentID, SupportsFloat]
    terminated: Dict[AgentID, bool]
    truncated: Dict[AgentID, bool]
    all_done: bool
    info: Dict[AgentID, Dict]

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)


class POSGModel(abc.ABC, Generic[StateType, ObsType, ActType]):
    """A Partially Observable Stochastic Game model.

    This class defines functions and attributes necessary for a generative POSG
    model for use in simulation-based planners (e.g. MCTS).

    The API includes the following,

    TODO Update this

    Attributes
    ----------
    :attr:possible_agents : Tuple[AgentID]
        All agents that may appear in the environment
    observation_first : bool
        whether the environment is observation (True) or action (False) first.
        See the POSGModel.observation_first property function for details.
    is_symmetric : bool
        whether the environment is symmetric, that is whether all agents are
        identical irrespective of their ID (i.e. same actions, observation, and
        reward spaces and dynamics)
    state_space : gym.space.Space | None
        the space of all possible environment states. Note that an explicit
        state space definition is not needed by many simulation-based
        algorithms (including RL and MCTS) and can be hard to define so
        implementing this property should be seen as optional. In cases where
        it is not implemented it should be None.
    action_spaces : Tuple[gym.space.Space, ...]
        the action space for each agent (A_0, ..., A_n)
    observation_spaces : Tuple[gym.space.Space, ...]
        the observation space for each agent (O_0, ..., O_n)
    reward_ranges : Tuple[Tuple[Reward, Reward], ...]
        the minimum and maximim possible step reward for each agent
    rng : posggym.utils.seeding.RNG
        the model's internal random number generator (RNG).

    Methods
    -------
    get_agents :
        returns the IDs of agents currently active for a given state
    step :
        the generative step function
        G(s, a) -> (s', o, r, dones, all_done, outcomes)
    seed :
        set the seed for the model's RNG
    sample_initial_state :
        samples an initial state from the initial belief.
    sample_initial_obs :
        samples an initial observation from a state. This function should only
        be used in observation first environments.
    sample_agent_initial_state :
        sample an initial state for an agent given the agent's initial observation.
        This function is only applicable in observation first environments. It is
        also optional to implement, since agent observation conditioned initial states
        can be generated by filtering the outputs from the sample_initial_state and
        sample_initial_obs methods. However, implementing this method can speed up an
        agent's initial belief update for problems with a large number of possible
        starting states.

    """

    # EnvSpec used to instantiate env instance this model is for
    # This is set when env is made using posggym.make function
    spec: Optional["EnvSpec"] = None

    # All agents that may appear in the environment
    possible_agents: Tuple[AgentID, ...]
    # State space
    state_space: spaces.Space | None = None
    # Action space for each agent
    action_spaces: Dict[AgentID, spaces.Space]
    # Observation space for each agent
    observation_spaces: Dict[AgentID, spaces.Space]

    # Random number generator, created as needed.
    _rng: seeding.RNG | None = None

    @property
    @abc.abstractmethod
    def observation_first(self) -> bool:
        """Get whether environment is observation or action first.

        "Observation first" environments start by providing the agents with an
        observation from the initial belief before any action is taken. Most
        Reinforcement Learning algorithms typically assume this setting.

        "Action first" environments expect the agents to take an action from the initial
        belief before providing an observation. Many planning algorithms use this
        paradigm.

        Note
        ----
        "Action first" environments can always be converted into "Observation first"
          by introducing a dummy initial observation. Similarly, "Action first"
          algorithms can be made compatible with "Observation first" environments by
          introducing a single dummy action for the first step only.

        Returns
        -------
        bool
          ``True`` if environment is observation first, ``False`` if environment is
          action first.

        """

    @property
    @abc.abstractmethod
    def is_symmetric(self) -> bool:
        """Get whether environment is symmetric.

        An environment is "symmetric" if the ID of an agent in the environment does not
        affect the agent in anyway (i.e. all agents have the same action and observation
        spaces, same reward functions, and there are no differences in initial
        conditions all things considered). Classic examples include Rock-Paper-Scissors,
        Chess, Poker. In "symmetric" environments the same "policy" should do equally
        well independent of the ID of the agent the policy is used for.

        If an environment is not "symmetric" then it is "asymmetric", meaning that
        there are differences in agent properties based on the agent's ID. In
        "asymmetric" environments there is no guarantee that the same "policy" will
        work for different agent IDs. Examples include Pursuit-Evasion games, any
        environments where action and/or observation space differs by agent ID.

        Returns
        -------
        bool
          ``True`` if environment is symmetric, ``False`` if environment is asymmetric.

        """

    @property
    def reward_ranges(self) -> Dict[AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        r"""Get the minimum and maximum possible rewards for each agent.

        Returns
        -------
        Dict[AgentID, Tuple[SupportsFloat, SupportsFloat]]
            Dictionary mapping agent ID to a tuple with the minimum and maximum
            possible rewards for an agent over an episode. The default reward range
            is set to :math:`(-\infty,+\infty)` for each agent.

        """
        return {i: (-float("inf"), float("inf")) for i in self.possible_agents}

    @abc.abstractmethod
    def get_agents(self, state: StateType) -> List[AgentID]:
        """Get list of IDs for all agents that are active in given state.

        These list of active agents may change depending on state.

        For any environment where the number of agents remains constant during AND
        across episodes. This will be :attr:`possible_agents`, independent of state.

        Returns
        -------
        List[AgentID]
          List of IDs for all agents that active in given state,

        """

    @abc.abstractmethod
    def sample_initial_state(self) -> StateType:
        """Sample an initial state.

        Returns
        -------
        StateType
          An initial state.

        """

    def sample_initial_obs(self, state: StateType) -> Dict[AgentID, ObsType]:
        """Sample initial agent observations given an initial state.

        This method must be implemented for `observation_first` models.

        Arguments
        ---------
        state : StateType
          The initial state.

        Returns
        -------
        Dict[AgentID, ObsType]
          A mapping from AgentID to initial observation.

        Raises
        ------
        AssertionError
          If this method is called for action first model.

        """
        if self.observation_first:
            raise NotImplementedError
        raise AssertionError(
            "Model is action_first so expects agents to perform an action "
            "before the initial observations are generated. This is done "
            "using the step() function."
        )

    @abc.abstractmethod
    def step(
        self, state: StateType, actions: Dict[AgentID, ActType]
    ) -> JointTimestep[StateType, ObsType]:
        """Perform generative step.

        For custom environments that have win/loss or success/fail conditions, you are
        encouraged to include this information in the `info` property of the returned
        value. We suggest using the the "outcomes" key with an instance of the
        ``Outcome`` class for values.

        """

    @property
    @abc.abstractmethod
    def rng(self) -> seeding.RNG:
        """Return the model's internal random number generator (RNG).

        Initializes RNG with a random seed if not yet initialized.

        `posggym` models and environments support the use of both the python built-in
        random library and the numpy library, unlike `gymnasium` which only explicitly
        supports the numpy library. Support for the built-in library is included as it
        can be 2-3X faster than the numpy library when drawing single samples, providing
        a significant speed-up for many environments.

        Theres also nothing stopping users from using other RNG libraries so long as
        they implement the model API. However, explicit support in the form of tests
        and type hints is only provided for the random and numpy libraries.

        Returns
        -------
        rng : model's internal random number generator.

        """

    def seed(self, seed: Optional[int] = None):
        """Set the seed for the model RNG.

        Also handles seeding for the action, observation, and (if it exists) state
        spaces.

        """
        if isinstance(self.rng, random.Random):
            self._rng, seed = seeding.std_random(seed)
        elif isinstance(self.rng, np.random.Generator):
            self._rng, seed = seeding.np_random(seed)
        else:
            raise error.UnseedableEnv(
                "{self.__class__.__name__} unseedable. Please ensure the model has "
                "implemented the rng property. The model class must also overwrite "
                "the `seed` method if it uses a RNG not from the `random` or "
                "`numpy.random` libraries."
            )

        seed += 1
        for act_space in self.action_spaces.values():
            act_space.seed(seed)
            seed += 1

        for obs_space in self.observation_spaces.values():
            obs_space.seed(seed)
            seed += 1

        if self.state_space is not None:
            self.state_space.seed(seed)

    def sample_agent_initial_state(self, agent_id: AgentID, obs: ObsType) -> StateType:
        """Sample an initial state for an agent given it's initial observation.

        Only applicable in observation first environments.

        It is optional to implement but can helpful in environments that are used for
        planning where there are a huge number of possible initial states.

        """
        if self.observation_first:
            raise NotImplementedError
        raise AssertionError(
            "The `sample_agent_initial_state` method is not supported for action first "
            "environments. Use the `sample_initial_state` method instead."
        )


class POSGFullModel(POSGModel[StateType, ObsType, ActType], abc.ABC):
    """A Fully definte Partially Observable Stochastic Game model.

    This class includes implementions for all components of a POSG, including:

    Functions
    ---------
    get_initial_belief_dist : the initial belief distribution
    transition_fn : the transition function T(s, a, s')
    observation_fn : the observation function Z(o, s', a)
    reward_fn : the reward function R(s, a)

    This is in addition to the functions and properties defined in the
    POSGModel class.

    """

    @abc.abstractmethod
    def get_initial_belief_dist(self) -> Dict[StateType, float]:
        """Get initial belief distribution: S -> prob map."""

    @abc.abstractmethod
    def transition_fn(
        self, state: StateType, actions: Dict[AgentID, ActType], next_state: StateType
    ) -> float:
        """Transition function Pr(next_state | state, action)."""

    @abc.abstractmethod
    def observation_fn(
        self,
        obs: Dict[AgentID, ObsType],
        next_state: StateType,
        actions: Dict[AgentID, ActType],
    ) -> float:
        """Observation function Pr(obs | next_state, action)."""

    @abc.abstractmethod
    def reward_fn(
        self, state: StateType, actions: Dict[AgentID, ActType]
    ) -> Dict[AgentID, SupportsFloat]:
        """Reward Function R: S X (a_0, ..., a_n) -> (r_0, ..., r_n)."""
