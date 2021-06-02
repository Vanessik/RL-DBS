from gym import Space
from gym.utils import seeding
import numpy as np
from gym.spaces import Box

class Bbox(Box):
    """
    Applying Box space to our limuted with 3 actions space.
    Action space is a Rn box with dependency between actions such, that a1+a2+a1*a3<=T - period of oscillator.
    """
    def __init__(self, low, high, act_limits, osc_period=6280, shape=None, dtype=np.float32):
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = np.dtype(dtype)

        # determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
            assert np.isscalar(low) or low.shape == shape, "low.shape doesn't match provided shape"
            assert np.isscalar(high) or high.shape == shape, "high.shape doesn't match provided shape"
        elif not np.isscalar(low):
            shape = low.shape
            assert np.isscalar(high) or high.shape == shape, "high.shape doesn't match low.shape"
        elif not np.isscalar(high):
            shape = high.shape
            assert np.isscalar(low) or low.shape == shape, "low.shape doesn't match high.shape"
        else:
            raise ValueError("shape must be provided or inferred from the shapes of low or high")

        if np.isscalar(low):
            low = np.full(shape, low, dtype=dtype)

        if np.isscalar(high):
            high = np.full(shape, high, dtype=dtype)

        self.shape = shape
        super(Box, self).__init__(self.shape, self.dtype)
        self.low = low
        self.high = high
        self.limits_low, self.limits_high = act_limits[0], act_limits[1]
        self.osc_period = osc_period
        
        def _get_precision(dtype):
            if np.issubdtype(dtype, np.floating):
                return np.finfo(dtype).precision
            else:
                return np.inf
        low_precision = _get_precision(self.low.dtype)
        high_precision = _get_precision(self.high.dtype)
        dtype_precision = _get_precision(self.dtype)
        if min(low_precision, high_precision) > dtype_precision:
            logger.warn("Box bound precision lowered by casting to {}".format(self.dtype))
        self.low = self.low.astype(self.dtype)
        self.high = self.high.astype(self.dtype)

        # Boolean arrays which indicate the interval type for each coordinate
        self.bounded_below = -np.inf < self.low
        self.bounded_above = np.inf > self.high
#         super(Box, self).__init__(self.shape, self.dtype)
#         super(Bbox, self).__init__(self.shape, self.dtype)

    def is_bounded(self, manner="both"):
        below = np.all(self.bounded_below)
        above = np.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self):
        """
        Generates a single random sample inside of the Box.
        In creating a sample of the box, each coordinate is sampled according to
        the form of the interval:
        * [a, b] : uniform distribution
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """
        high = self.high if self.dtype.kind == 'f' \
                else self.high.astype('int64') + 1
        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        period_osc = self.osc_period
#         print('period', period_osc)
        cur_vec_a = []
        denorm_vec_a = []
#         sample 1st action
        a1 = np.random.uniform(low=-1, high=1)
        cur_vec_a.append(a1)
        a1_de = self.denormalize(a1, 0)
#         period_osc -= a1_de
#         sample 2nd action
        max_a2 = period_osc-(self.limits_low[2]+1)*a1_de
        max_norm = self.normalize(max_a2, 1)
#         print('a1', a1_de)
#         print('a2max', max_a2)
        a2 = np.random.uniform(low=-1, high=max_norm)
        cur_vec_a.append(a2)
        a2_de = self.denormalize(a2, 1)
#         period_osc = period_osc - 2*a1_de - a2_de
        period_osc = period_osc - a1_de - a2_de
#         sample 3rd action
#         print('a2', a2_de)
        max_a3 = int(np.clip(period_osc/a1_de,self.limits_low[2], self.limits_high[2]))
#         print('os', period_osc)
        max_norm = self.normalize(max_a3, 2)
#         print(max_a3)
        a3 = np.random.uniform(low=-1, high=max_norm)
        a3_de = self.denormalize(a3, 2)
        cur_vec_a.append(a3)
        
        assert a1_de+a2_de+a1_de*a3_de<=self.osc_period
        sample = np.array(cur_vec_a)
        sample_de = np.array([a1_de, a2_de, a3_de])

        if self.dtype.kind == 'i':
            sample = np.floor(sample)
#         print(sample.shape)

        return np.array([sample.astype(self.dtype)])
    
    def denormalize(self, a, num):
        de_act = self.limits_low[num] + (self.limits_high[num]-self.limits_low[num])*(a+1)/2 
        return int(de_act)
    
    def normalize(self, a, num):
        norm_act = 2*(a-self.limits_low[num])/(self.limits_high[num]-self.limits_low[num])-1
        return norm_act

    
    
class BBbox(Box):
    """
    Applying Box space to our limuted with 3 actions space.
    Action space is a Rn box with dependency between actions such, that a1+a2+a1*a3<=T - period of oscillator.
    """
    def __init__(self, low, high, act_limits, osc_period=6280, shape=None, dtype=np.float32):
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = np.dtype(dtype)
        print('ggooos')
        # determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
            assert np.isscalar(low) or low.shape == shape, "low.shape doesn't match provided shape"
            assert np.isscalar(high) or high.shape == shape, "high.shape doesn't match provided shape"
        elif not np.isscalar(low):
            shape = low.shape
            assert np.isscalar(high) or high.shape == shape, "high.shape doesn't match low.shape"
        elif not np.isscalar(high):
            shape = high.shape
            assert np.isscalar(low) or low.shape == shape, "low.shape doesn't match high.shape"
        else:
            raise ValueError("shape must be provided or inferred from the shapes of low or high")

        if np.isscalar(low):
            low = np.full(shape, low, dtype=dtype)

        if np.isscalar(high):
            high = np.full(shape, high, dtype=dtype)

        self.shape = shape
        super(Box, self).__init__(self.shape, self.dtype)
        self.width_p = 400
        self.low = low
        self.high = high
        self.limits_low, self.limits_high = act_limits[0], act_limits[1]
        self.osc_period = osc_period
        
        def _get_precision(dtype):
            if np.issubdtype(dtype, np.floating):
                return np.finfo(dtype).precision
            else:
                return np.inf
        low_precision = _get_precision(self.low.dtype)
        high_precision = _get_precision(self.high.dtype)
        dtype_precision = _get_precision(self.dtype)
        if min(low_precision, high_precision) > dtype_precision:
            logger.warn("Box bound precision lowered by casting to {}".format(self.dtype))
        self.low = self.low.astype(self.dtype)
        self.high = self.high.astype(self.dtype)

        # Boolean arrays which indicate the interval type for each coordinate
        self.bounded_below = -np.inf < self.low
        self.bounded_above = np.inf > self.high
#         super(Box, self).__init__(self.shape, self.dtype)
#         super(Bbox, self).__init__(self.shape, self.dtype)

    def is_bounded(self, manner="both"):
        below = np.all(self.bounded_below)
        above = np.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self):
        """
        Generates a single random sample inside of the Box.
        In creating a sample of the box, each coordinate is sampled according to
        the form of the interval:
        * [a, b] : uniform distribution
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """
        high = self.high if self.dtype.kind == 'f' \
                else self.high.astype('int64') + 1
        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        period_osc = self.osc_period
#         print('period', period_osc)
        cur_vec_a = []
        denorm_vec_a = []
#         sample 1st action
#         a1 = np.random.uniform(low=-1, high=1)
#         cur_vec_a.append(a1)
        a1_de = self.width_p
#         sample 2nd action
        max_a2 = period_osc-(self.limits_low[1]+1)*a1_de
        max_norm = self.normalize(max_a2, 0)
        a2 = np.random.uniform(low=-1, high=max_norm)
        cur_vec_a.append(a2)
        a2_de = self.denormalize(a2, 0)
#         period_osc = period_osc - 2*a1_de - a2_de
        period_osc = period_osc - a1_de - a2_de
#         sample 3rd action
#         print('a2', a2_de)
        max_a3 = int(np.clip(period_osc/a1_de,self.limits_low[1], self.limits_high[1]))
#         print('os', period_osc)
        max_norm = self.normalize(max_a3, 1)
#         print(max_a3)
        a3 = np.random.uniform(low=-1, high=max_norm)
        a3_de = self.denormalize(a3, 1)
        cur_vec_a.append(a3)
        print(a1_de, a2_de, a3_de)
        assert a1_de+a2_de+a1_de*a3_de<=self.osc_period
        sample = np.array(cur_vec_a)
        sample_de = np.array([a1_de, a2_de, a3_de])

        if self.dtype.kind == 'i':
            sample = np.floor(sample)
#         print(sample.shape)

        return np.array([sample.astype(self.dtype)])
    
    def denormalize(self, a, num):
        de_act = self.limits_low[num] + (self.limits_high[num]-self.limits_low[num])*(a+1)/2 
        return int(de_act)
    
    def normalize(self, a, num):
        norm_act = 2*(a-self.limits_low[num])/(self.limits_high[num]-self.limits_low[num])-1
        return norm_act
