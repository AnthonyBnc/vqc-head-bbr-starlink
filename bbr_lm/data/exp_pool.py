"""Experience pool: states, target gains, rewards and phases of each interval."""

from bbr_lm.bbr import validate_phase, validate_phase_action


class ExperiencePool:
    """
    Experience pool for collecting trajectories.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.phases = []
        self.sample_ids = []
        self.metadata = {}

    def add(self, state, action, reward, done, phase, sample_id=None):
        phase = validate_phase(phase)
        validate_phase_action(phase, action)
        self.states.append(state)  # Sometimes state is also called obs (observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.phases.append(phase)
        self.sample_ids.append(sample_id)

    def __len__(self):
        return len(self.states)

    def __getstate__(self):
        """
        Custom method to serialize the state without relying on the class or module.
        """
        # Return only the necessary data for pickling
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'phases': self.phases,
            'sample_ids': self.sample_ids,
            'metadata': self.metadata,
        }

    def __setstate__(self, state):
        """
        Custom method to restore the state when unpickling, without requiring the class/module.
        """
        self.states = state['states']
        self.actions = state['actions']
        self.rewards = state['rewards']
        self.dones = state['dones']
        self.phases = state.get('phases')
        self.sample_ids = state.get('sample_ids', [None] * len(self.states))
        self.metadata = state.get('metadata', {})


# Experience pools are stored with pickle, which records the module path of
# the class. The pools used in the paper were written from ``utils.exp_pool``,
# and the pipeline checks their SHA-256 values. Keeping that module path makes
# old pools loadable and keeps newly built pools byte-identical to them.
ExperiencePool.__module__ = "utils.exp_pool"
